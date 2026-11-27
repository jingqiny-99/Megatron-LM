# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GPU tests of the loop parameter store under DDP, the distributed optimizer and finalize.

Runs under torchrun with CUDA and a world size divisible by 4: the GTP-remat grid is initialized
with ``gtp_remat_size = 4`` (shard axis z = 4, replicate W / 4; 2 on one 8-GPU node), the model
is built with the gtp axes of its process-group collection nulled (no ``GTPShardedParam``: the
grid works on TE 2.16, stage-6 design E7 / E8), the store is installed before the DDP wrap, and
DDP, the distributed optimizer (Adam, grad clipping) and ``finalize_model_grads`` get the full
collection. Steps are driven through the store's API (``param_store_utils.StoreDriver``: the CUDA
side streams, events, coalesced NCCL gathers / reduce-scatters, lookahead and carry) on a local
dense loop model whose fc1 has 182 rows (two pad rows on the last shard rank).

Gates: per-sample losses of step 1 bitwise across Layer window, Resident + sharded grads,
Resident and no store; Layer window == Resident + sharded grads bitwise over three optimizer steps
(losses, grad norm, final weights); the grad norm (shards counted once, replicated parameters
once per shard axis) equal to the no-store run within the stage-2 rule; later losses close to it;
shard pad rows exactly zero after every finalize; the replicate-group parameter hash check passes
(E18); the store's reduce counts exactly once per (item, unit). The same Layer window and
Resident + sharded grads runs with ``side_stream_delay_cycles`` (the all-gather and
reduce-scatter streams spin before every collective, so a dropped event wait reads a poisoned or
stale buffer) stay bitwise equal to the undelayed runs. A bf16 arm (``Float16Module``, bf16
shards, fp32 ``main_grad`` and fp32 distributed-optimizer masters: the production configuration)
repeats the step-1 loss, Layer window == Resident + sharded grads, pad-row, hash and reduce-count
gates, with the grad norm within 1e-3 of its no-store run.
"""

import copy
import os

import pytest
import torch
import torch.distributed as dist

from megatron.core.distributed import (
    DistributedDataParallel,
    DistributedDataParallelConfig,
    finalize_model_grads,
)
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection, resolve_gtp_remat_group
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.loop.hybrid_stack import LoopHybridStack
from megatron.core.transformer.loop.param_store import (
    LoopParamStore,
    LoopParamStoreConfig,
    ResidencyMode,
    StoreGroups,
    StoreModes,
)
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import check_param_hashes_across_dp_replicas
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.loop import param_store_utils as psu
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    VOCAB,
    hybrid_loop_config,
    local_model_spec,
)

Z = 4
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or WORLD < Z or WORLD % Z,
    reason=f"the store DDP tests need CUDA and a world size divisible by {Z}",
)

DENSE = "*-*-*-*-"
SEQ = 16
CORE = 4
COUNT = 2  # samples per EU per step
DEPTH = 2
STEPS = 3
LR = 1e-3
DELAY = 2_000_000  # side-stream spin cycles (~1 ms) before every store collective


@pytest.fixture(scope="module", autouse=True)
def _grid():
    Utils.initialize_model_parallel(1, 1, gtp_remat_size=Z)
    model_parallel_cuda_manual_seed(123)
    try:
        yield
    finally:
        Utils.destroy_model_parallel()


def _collections():
    full = ProcessGroupCollection.use_mpu_process_groups()
    model_pgc = copy.copy(full)
    model_pgc.gtp_remat = None
    model_pgc.expt_gtp_remat = None
    return full, model_pgc


def _build(model_pgc, bf16=False):
    precision = dict(bf16=True, params_dtype=torch.bfloat16) if bf16 else {}
    config = hybrid_loop_config(
        DENSE,
        2,
        2,
        ffn_hidden_size=91,
        loop_iterations=DEPTH,
        loop_schedule="rsbf",
        calculate_per_token_loss=True,
        **precision,
    )
    torch.manual_seed(1234)
    model = HybridModel(
        config=config,
        hybrid_stack_spec=local_model_spec(LoopHybridStack),
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
        hybrid_layer_pattern=DENSE,
        position_embedding_type="none",
        pg_collection=model_pgc,
    ).cuda()
    if bf16:
        return Float16Module(config, model), model
    return model, model


def _train(modes_text, bf16=False, **store_config):
    """Three optimizer steps; returns per-step losses and grad norms, the final full weights,
    and the store (None without one). ``bf16``: the model in ``Float16Module``, bf16 params."""
    full, model_pgc = _collections()
    assert resolve_gtp_remat_group(model_pgc, is_expert=False) is None
    model, hybrid = _build(model_pgc, bf16)
    assert not any(getattr(p, "is_gtp_weight_remat", False) for p in model.parameters())
    store = None
    plan = psu.core_plan_of("block", CORE)
    if modes_text is not None:
        groups = StoreGroups(shard=full.gtp_remat, domain=full.dp_gtp_remat, replicate=full.dp)
        store = LoopParamStore.install(
            model, LoopParamStoreConfig(shard_size=Z, min_numel=1, **store_config), groups
        )
        expected = torch.bfloat16 if bf16 else torch.float32
        assert store.dtype == expected and all(t.shard.dtype == expected for t in store.tensors())
    ddp = DistributedDataParallel(
        model.config,
        DistributedDataParallelConfig(
            overlap_grad_reduce=False,
            use_distributed_optimizer=True,
            grad_reduce_in_fp32=True,
            average_in_collective=False,
        ),
        model,
        pg_collection=full,
    )
    if store is not None:
        store.configure_modes(StoreModes.parse(modes_text), core_plan=plan)
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam",
            lr=LR,
            use_distributed_optimizer=True,
            clip_grad=1.0,
            bf16=bf16,
            params_dtype=torch.bfloat16 if bf16 else torch.float32,
        ),
        [ddp],
        use_gloo_process_groups=False,
        pg_collection=full,
    )
    if store is not None:
        assert all(t.shard.main_grad.dtype == torch.float32 for t in store.tensors())
    driver = psu.StoreDriver(hybrid, store, depth=DEPTH, core_plan=plan)
    rank = dist.get_rank()
    losses, norms, pads_zero, reduces_ok = [], [], True, True
    for step in range(STEPS):
        ddp.zero_grad_buffer()
        optimizer.zero_grad()
        samples = psu.make_samples(rank, COUNT, vocab=VOCAB, seq=SEQ, seed=41 + step)
        driver.run_step([samples], COUNT)
        losses.append(torch.stack([s.loss for s in samples]))
        num_tokens = torch.tensor(COUNT * SEQ, dtype=torch.int, device="cuda")
        finalize_model_grads([ddp], num_tokens, pg_collection=full)
        if store is not None:
            for name, pad in psu.shard_pad_rows(store).items():
                pads_zero &= bool((pad == 0).all())
            reduces_ok &= _exactly_once(store)
        success, grad_norm, _ = optimizer.step()
        assert success
        norms.append(float(grad_norm))
    hashes_ok = check_param_hashes_across_dp_replicas([ddp])
    weights = _full_weights(hybrid, store, full)
    return dict(
        losses=losses,
        norms=norms,
        weights=weights,
        pads_zero=pads_zero,
        reduces_ok=reduces_ok,
        hashes_ok=hashes_ok,
        store=store,
    )


def _exactly_once(store):
    count = COUNT * DEPTH
    for unit in store.units:
        mode = store.mode_of(unit)
        got = {kind: n for (kind, key), n in store.stats.reduces.items() if key == unit.key}
        if mode is ResidencyMode.RESIDENT:
            if got or store.stats.resident_reduces[unit.key] != 1:
                return False
            continue
        expected = {
            "core": {"core_b": count},
            "prelude": {"prelude_b": COUNT},
            "coda": {"coda": COUNT},
        }.get(unit.group, {"prelude_b" if "embedding" in unit.key else "coda": COUNT})
        if got != expected:
            return False
    return True


def _full_weights(model, store, full):
    out = {}
    stored = set()
    if store is not None:
        for tensor in store.tensors():
            shard = tensor.shard.detach()
            parts = [torch.empty_like(shard) for _ in range(Z)]
            dist.all_gather(parts, shard.contiguous(), group=full.gtp_remat)
            out[tensor.name] = torch.cat(parts)[: tensor.rows].float().cpu()
            stored.add(tensor.name)
    for name, param in model.named_parameters():
        if "loop_param_store." not in name:
            out[name] = param.detach().float().cpu()
    return out


@pytest.fixture(scope="module")
def runs():
    return {
        "none": _train(None),
        "resident": _train("all=r"),
        "rsg": _train("all=rsg"),
        "lw": _train(
            "core=lw,edges=lw,emb_head=rsg",
            poison_released=True,
            verify_gathers=True,
            check_visits=True,
            timing=True,
        ),
        "lw_regather": _train("core=lw,edges=lw,emb_head=rsg,core_regather=on,edge_regather=on"),
        "lw_delayed": _train(
            "core=lw,edges=lw,emb_head=rsg", poison_released=True, side_stream_delay_cycles=DELAY
        ),
        "rsg_delayed": _train("all=rsg", poison_released=True, side_stream_delay_cycles=DELAY),
    }


@pytest.fixture(scope="module")
def bf16_runs():
    return {
        "none": _train(None, bf16=True),
        "resident": _train("all=r", bf16=True),
        "rsg": _train("all=rsg", bf16=True),
        "lw": _train(
            "core=lw,edges=lw,emb_head=rsg", bf16=True, poison_released=True, check_visits=True
        ),
    }


def test_step_one_losses_are_bitwise_in_every_mode(runs):
    reference = runs["none"]["losses"][0]
    for name, run in runs.items():
        assert torch.equal(run["losses"][0], reference), name


def _bitwise_over_steps(a, b):
    for step in range(STEPS):
        assert torch.equal(a["losses"][step], b["losses"][step]), step
        assert a["norms"][step] == b["norms"][step], step
    for name, weight in a["weights"].items():
        assert torch.equal(weight, b["weights"][name]), name


@pytest.mark.parametrize("window", ["lw", "lw_regather"])
def test_layer_window_equals_sharded_grads_bitwise_over_steps(runs, window):
    _bitwise_over_steps(runs[window], runs["rsg"])


@pytest.mark.parametrize("name", ["lw", "rsg"])
def test_delayed_side_streams_change_nothing(runs, name):
    """Every gather, bind, scratch reuse and reduce is event-ordered: delaying the side streams
    (with released slots and reduced scratch entries NaN-poisoned) leaves the run bitwise."""
    _bitwise_over_steps(runs[f"{name}_delayed"], runs[name])


def test_bf16_store_runs(bf16_runs):
    reference = bf16_runs["none"]
    for name, run in bf16_runs.items():
        assert torch.equal(run["losses"][0], reference["losses"][0]), name
        if name != "none":
            assert run["pads_zero"] and run["hashes_ok"] and run["reduces_ok"], name
            assert abs(run["norms"][0] - reference["norms"][0]) <= 1e-3 * reference["norms"][0]
    _bitwise_over_steps(bf16_runs["lw"], bf16_runs["rsg"])


def test_grad_norm_and_losses_match_the_storeless_run(runs):
    reference = runs["none"]
    for name in ("resident", "rsg", "lw"):
        run = runs[name]
        assert abs(run["norms"][0] - reference["norms"][0]) <= 1e-5 * reference["norms"][0], name
        for step in range(1, STEPS):
            assert torch.allclose(
                run["losses"][step], reference["losses"][step], rtol=1e-3, atol=0
            ), (name, step)
        for weight_name, weight in run["weights"].items():
            assert torch.allclose(
                weight, reference["weights"][weight_name], rtol=0, atol=2 * LR * STEPS
            ), (name, weight_name)


def test_pad_rows_hashes_and_reduce_counts(runs):
    for name in ("resident", "rsg", "lw", "lw_regather"):
        run = runs[name]
        assert run["pads_zero"], name
        assert run["hashes_ok"], name
        assert run["reduces_ok"], name
    lw = runs["lw"]["store"]
    assert lw.stats.prefetches > 0
    # timing=True: one exposure record per window bind, read at end_step.
    assert lw.stats.exposure_ms and all(ms >= 0.0 for ms in lw.stats.exposure_ms)
    assert runs["none"]["hashes_ok"]
