# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Every consumer of the row-tail padding of SBHD document packing excludes it.

Packed rows (``GPTDatasetConfig.sbhd_document_packing``) end in padding: ``padding_mask`` is
True on the padded tokens and the loss mask is 0 on the labels in the padding. The consumers:

* the LM loss (``pretrain_hybrid.loss_func``): masked per-token losses change neither the sum
  nor the token count (CPU);
* the exit criterion (contract rule 8, ``valid_token_mean``): padded values, non-finite ones
  included, never reach ``exit`` / ``crit_state`` / ``aux_loss``; through the RSBF executor
  (dense, MTP and mHC stacks; the online halting and the padding-depth criteria) two batches
  that differ only in their padded tokens give bitwise equal depths, criterion states and
  losses, at one EU and over a spawned Gloo world of two EUs where samples move (CPU);
* the MoE router statistics and aux losses (aux, sequence aux, z-loss, expert-bias counts):
  equal to the router run on the valid tokens alone, with garbage logits in the padding (CPU;
  the dense-index / HybridEP count path is covered by tests/unit_tests/transformer/moe/
  test_routers.py), and on GPU through a DSv4-like loop model (expert-bias counts equal the
  routing of the valid rows);
* the DSA indexer loss: the MTP depth runs its layer in the indexer-loss row-mask scope of its
  rolled padding mask, taken before the sequence-parallel scatter (CPU spies); on GPU every
  CSA indexer loss of a DSv4-like loop model (prelude, core, coda and the MTP depth, eager and
  RSBF) receives the real rows of its sample's (rolled) padding mask.

Unpacked data is unchanged: the where-based criterion reduction equals the stage-4
multiplicative one bitwise on finite inputs, and the stage-4 online-criterion step (no padding
mask) is reproduced bitwise with the stage-4 reduction swapped in.
"""

import contextlib
from functools import partial
from types import SimpleNamespace

import pytest
import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import multi_token_prediction as mtp_module
from megatron.core.transformer.experimental_attention_variant.dsa_masking import (
    indexer_loss_padding_mask_in_scope,
)
from megatron.core.transformer.loop.program import SampleKey, valid_token_mean
from megatron.core.transformer.moe import router as router_module
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.transformer.loop import hybrid_loop_utils, rsbf_test_criteria
from tests.unit_tests.transformer.loop import test_loop_rsbf_executor as base
from tests.unit_tests.transformer.loop.hybrid_loop_utils import (
    CUDA,
    SEQ,
    VOCAB,
    device,
    model_parallel_once,
)
from tests.unit_tests.transformer.loop.multi_eu_utils import (
    SPAWN_SKIP_REASON,
    cpu_worker_setup,
    cpu_worker_teardown,
    run_spawned,
    spawn_tests_enabled,
)
from tests.unit_tests.transformer.loop.rsbf_test_criteria import (
    HaltingHead,
    OnlineHaltingCriterion,
    PaddingDepthCriterion,
)

cpu_mtp = hybrid_loop_utils.cpu_mtp
mhc_on_this_device = hybrid_loop_utils.mhc_on_this_device
hybrid = base.hybrid
deterministic_algorithms = base.deterministic_algorithms

gpu = pytest.mark.skipif(not CUDA, reason="TE / MoE / DSv4 layers need CUDA")
spawned = pytest.mark.skipif(not spawn_tests_enabled(), reason=SPAWN_SKIP_REASON)


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    with model_parallel_once():
        try:
            yield
        finally:
            base.get_loop_rsbf_schedule().reset()


def _poison_padding(batches, seed=99):
    """The batches with every padded token (and its label) replaced by other token ids."""
    generator = torch.Generator().manual_seed(seed)
    poisoned = []
    for batch in batches:
        batch = dict(batch)
        padding = batch["padding_mask"]
        if padding is not None:
            for name in ("tokens", "labels"):
                values = batch[name].clone()
                other = torch.randint(1, VOCAB, values.shape, generator=generator).to(values)
                values[padding] = (values[padding] + other[padding]) % VOCAB
                batch[name] = values
        poisoned.append(batch)
    return poisoned


# ---------------------------------------------------------------------------
# LM loss
# ---------------------------------------------------------------------------


def test_lm_loss_excludes_the_padded_labels(hybrid):
    """Packed rows (MockGPTDataset, S 4096: every mock document fits): the loss mask is 0 on
    the padding; per-token losses of 1e30 there change neither the loss nor the token
    count."""
    from megatron.core.datasets.gpt_dataset import (
        GPTDatasetConfig,
        MockGPTDataset,
        MockGPTLowLevelDataset,
    )
    from megatron.core.datasets.utils import Split
    from megatron.core.tokenizers import MegatronTokenizer

    tokenizer = MegatronTokenizer.from_pretrained(
        metadata_path={"library": "null-text"}, vocab_size=8192
    )
    config = GPTDatasetConfig(
        random_seed=1234,
        sequence_length=4096,
        split="1,1,1",
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        create_attention_mask=False,
        tokenizer=tokenizer,
        sbhd_document_packing=True,
    )
    dataset = MockGPTDataset(
        MockGPTLowLevelDataset(tokenizer),
        None,
        torch.arange(400, dtype=torch.int32).numpy(),
        16,
        Split.train,
        config,
    )
    batch = torch.utils.data.default_collate([dataset[i] for i in range(8)])
    padding, loss_mask = batch["padding_mask"], batch["loss_mask"]
    assert padding.any() and not loss_mask[padding].any()
    # Row content in tokens (up to S + 1 with the extra token): a full row has no padded input
    # position and S loss labels, so the count comes from the packed pieces, not the mask.
    contents = [sum(piece[2] for piece in dataset.packed_row_pieces(i)) for i in range(8)]
    assert all(0 < content <= 4097 for content in contents)
    assert padding.sum(1).tolist() == [max(4096 - content, 0) for content in contents]
    expected_tokens = torch.tensor(sum(min(content - 1, 4096) for content in contents))
    losses = torch.rand(8, 4096, generator=torch.Generator().manual_seed(0))
    loss, num_tokens, report = hybrid.loss_func(loss_mask, losses)
    poisoned = torch.where(loss_mask == 0, torch.full_like(losses, 1e30), losses)
    loss_p, num_tokens_p, _ = hybrid.loss_func(loss_mask, poisoned)
    assert torch.equal(loss, loss_p) and torch.equal(num_tokens, num_tokens_p)
    assert num_tokens.item() == expected_tokens.item()
    assert report["lm loss"][1].item() == expected_tokens.item()


# ---------------------------------------------------------------------------
# Exit criterion
# ---------------------------------------------------------------------------


def _stage4_stream_contraction(x, weight, valid):
    """``rsbf_test_criteria.stream_contraction`` at vendor 0acc168de (multiplicative mask)."""
    sequence, rows = x.shape[:2]
    streams = x.float().reshape(sequence, rows, *weight.shape)
    if valid is None:
        mean = streams.mean(0)
    else:
        mask = valid.t().float()[:, :, None, None]
        mean = (streams * mask).sum(0) / mask.sum(0).clamp_min(1.0)
    return (mean * weight.float()).sum((-2, -1))


def test_valid_token_mean_selects_padded_values_away():
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(6, 3, 2, 4, generator=generator, requires_grad=True)
    valid = torch.tensor(
        [[True] * 6, [True, True, True, False, False, False], [False] * 6], dtype=torch.bool
    )
    clean = valid_token_mean(x, valid)
    poisoned_x = x.detach().clone()
    tail = (~valid).t()
    poisoned_x[tail] = (
        torch.tensor([float("inf"), float("-inf"), float("nan")])
        .repeat(int(tail.sum()) // 3 + 1)[: int(tail.sum())]
        .view(-1, 1, 1)
        .expand(-1, 2, 4)
    )
    poisoned_x.requires_grad_()
    poisoned = valid_token_mean(poisoned_x, valid)
    assert torch.isfinite(poisoned).all()
    assert torch.equal(poisoned, clean)
    torch.testing.assert_close(clean[0], x[:, 0].mean(0))
    torch.testing.assert_close(clean[1], x[:3, 1].mean(0))
    assert torch.equal(clean[2], torch.zeros(2, 4))
    poisoned.sum().backward()
    assert torch.isfinite(poisoned_x.grad).all() and not poisoned_x.grad[tail].any()
    assert torch.equal(valid_token_mean(x, None), x.float().mean(0))
    with pytest.raises(ValueError, match="bool"):
        valid_token_mean(x, valid.float())
    with pytest.raises(ValueError, match=r"\[c, S\]"):
        valid_token_mean(x, valid.t())


@pytest.mark.parametrize("padded", [False, True])
def test_where_reduction_is_the_stage4_reduction_on_finite_inputs(padded):
    """Unpacked (and padded, finite) inputs: values and gradients bitwise as at stage 4."""
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(SEQ, 2, 4 * 8, generator=generator)
    weight = torch.randn(4, 8, generator=generator)
    valid = None
    if padded:
        valid = torch.ones(2, SEQ, dtype=torch.bool)
        valid[1, -5:] = False
    results = []
    for function in (_stage4_stream_contraction, rsbf_test_criteria.stream_contraction):
        xs, ws = x.clone().requires_grad_(), weight.clone().requires_grad_()
        out = function(xs, ws, valid)
        (out * torch.tensor([0.3, -1.7])).sum().backward()
        results.append((out.detach(), xs.grad, ws.grad))
    for old, new in zip(*results):
        assert torch.equal(old, new)


class _Recording:
    """Wraps a criterion: records every step's criterion state and checks that the ``valid``
    mask a step sees is the sample's own (PaddingDepthCriterion: target = padding + 1)."""

    def __init__(self, criterion):
        self.criterion = criterion
        self.states = []

    def __getattr__(self, name):
        return getattr(self.criterion, name)

    def init_state(self, injection, valid):
        return self.criterion.init_state(injection, valid)

    def known_depths(self, keys):
        return self.criterion.known_depths(keys)

    def step(self, state, crit_state, r, valid, depth):
        out = self.criterion.step(state, crit_state, r, valid, depth)
        if isinstance(self.criterion, PaddingDepthCriterion):
            padded = torch.zeros_like(crit_state[:, 0]) if valid is None else (~valid).sum(1)
            assert torch.equal(crit_state[:, 0], padded.float() + 1), "a step saw another mask"
        self.states.append(out.crit_state.detach().clone())
        return out


def _criterion(kind, model):
    if kind == "online":
        stack = model.decoder
        head = HaltingHead(
            stack.loop_num_streams, model.config.hidden_size, device=device(), seed=3
        )
        return OnlineHaltingCriterion(head, tau=1.4, carry=0.05)
    return PaddingDepthCriterion()


def _as_criterion(recording):
    """A real ``ExitCriterion`` subclass instance forwarding to the recording wrapper (the
    program checks the type)."""
    criterion = recording.criterion

    class Forwarding(type(criterion)):
        def init_state(self, injection, valid):
            return recording.init_state(injection, valid)

        def step(self, state, crit_state, r, valid, depth):
            return recording.step(state, crit_state, r, valid, depth)

    forwarding = Forwarding.__new__(Forwarding)
    forwarding.__dict__.update(criterion.__dict__)
    return forwarding


@pytest.mark.parametrize("kind", ["online", "padding_depth"])
@pytest.mark.parametrize("name", ["dense", "mtp", "mhc"])
def test_executor_ignores_the_padded_tokens(request, hybrid, deterministic_algorithms, name, kind):
    """One EU, padded rows (the last quarter of every odd row): a batch and the same batch with
    other padded tokens give bitwise equal depths, criterion states, losses and gradients."""
    model = base._model(name, request)
    batches = base._batches(6, padded=True, seed=11)
    runs = []
    for arm in (batches, _poison_padding(batches)):
        recording = _Recording(_criterion(kind, model))
        extra = (recording.head,) if kind == "online" else ()
        reports, executor, grads = base._rsbf(
            model, hybrid, arm, criterion=_as_criterion(recording), extra_modules=extra
        )
        depths = [executor.stats.depths[SampleKey(i, 0)] for i in range(6)]
        runs.append((base._lm_losses(reports), depths, recording.states, grads))
    (losses, depths, states, grads), (losses_p, depths_p, states_p, grads_p) = runs
    base._assert_losses_equal(losses_p, losses)
    assert depths_p == depths
    assert len(states_p) == len(states) and all(map(torch.equal, states_p, states))
    base._assert_grads_equal(grads_p, grads)
    if kind == "padding_depth":
        # Padded rows ask for SEQ // 4 + 1 rounds (forced at R_max), the others for 1.
        assert depths == [1, base.R_MAX] * 3


def test_unpacked_online_step_is_the_stage4_step(
    request, hybrid, deterministic_algorithms, monkeypatch
):
    """Per-row exit on unpacked data unchanged (bitwise vs stage 4): the stage-4 online
    criterion case (mHC, no padding mask) with the stage-4 multiplicative reduction swapped in
    gives the same depths, losses and gradients as with ``valid_token_mean``."""
    model = base._model("mhc", request)
    batches = base._batches(6, seed=11)
    runs = []
    for contraction in (_stage4_stream_contraction, rsbf_test_criteria.stream_contraction):
        monkeypatch.setattr(rsbf_test_criteria, "stream_contraction", contraction)
        criterion = base._halting(model)
        reports, executor, grads = base._rsbf(
            model, hybrid, batches, criterion=criterion, extra_modules=(criterion.head,)
        )
        depths = [executor.stats.depths[SampleKey(i, 0)] for i in range(6)]
        runs.append((base._lm_losses(reports), depths, grads))
    assert len(set(runs[0][1])) > 1, runs[0][1]
    base._assert_losses_equal(runs[1][0], runs[0][0])
    assert runs[1][1] == runs[0][1]
    base._assert_grads_equal(runs[1][2], runs[0][2])


def _moved_worker(rank, world):
    """W = 2, two samples per EU: EU 0 holds the unpadded samples (depth 1), EU 1 the padded
    ones (depth R_max under the padding-depth criterion), so the plan moves EU 1's samples to
    EU 0 after round 1; every step checks it sees its sample's own mask. The clean and the
    poisoned batch give bitwise equal losses and depths."""
    from megatron.core import parallel_state
    from tests.unit_tests.transformer.loop import test_loop_rsbf_multi_eu as multi

    cpu_worker_setup()
    try:
        domain = parallel_state.get_data_parallel_group()
        status = parallel_state.get_data_parallel_group_gloo()
        model = multi._model("mtp")
        rows = multi._rows(4, padded=True, seed=5)
        out = []
        for arm in ("clean", "poisoned"):
            if arm == "poisoned":
                poisoned = _poison_padding([dict(rows)])[0]
                rows = {**rows, "tokens": poisoned["tokens"], "labels": poisoned["labels"]}
            mine = multi._microbatches(rows, mbs=1, world=world, rank=rank)
            recording = _Recording(PaddingDepthCriterion())
            reports, executor, _ = multi._step(
                model,
                mine,
                _as_criterion(recording),
                mbs=1,
                domain=domain,
                status=status,
                check_plan=True,
            )
            depths = {key.global_index: d for key, d in executor.stats.depths.items()}
            out.append(
                (
                    multi._losses(executor, reports),
                    depths,
                    dict(executor.stats.moves),
                    len(recording.states),
                )
            )
        (losses, depths, moves, steps), (losses_p, depths_p, _, steps_p) = out
        assert losses.keys() == losses_p.keys()
        assert all(torch.equal(losses[k], losses_p[k]) for k in losses)
        assert depths == depths_p == {0: 1, 1: base.R_MAX, 2: 1, 3: base.R_MAX}
        assert steps == steps_p
        return moves
    finally:
        cpu_worker_teardown()


@spawned
def test_moved_samples_carry_their_padding_mask():
    moves = run_spawned(_moved_worker, 2, timeout=600.0)
    assert moves[0] == moves[1] and moves[0]["sample"] > 0, moves


# ---------------------------------------------------------------------------
# MoE router statistics and aux losses (CPU: TopKRouter.routing on given logits)
# ---------------------------------------------------------------------------


@pytest.fixture
def router_records(monkeypatch):
    # The bare ``backward`` below runs outside a schedule: with no scale set,
    # ``MoEAuxLossAutoScaler.backward`` creates a 0-d one, which a later scheduled step's [1]
    # scale cannot be copied into (``set_loss_scale``). Start and leave the class attribute as
    # the scheduled tests expect (as test_routers.py's teardown does).
    monkeypatch.setattr(MoEAuxLossAutoScaler, "main_loss_backward_scale", None)
    records = []

    def record(name, value, *args, **kwargs):
        records.append((name, value.detach().clone()))

    monkeypatch.setattr(
        router_module, "get_moe_metrics_tracker", lambda: SimpleNamespace(record=record)
    )
    return records


def _router():
    config = TransformerConfig(
        num_layers=2,
        hidden_size=16,
        num_attention_heads=4,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_router_load_balancing_type=["aux_loss", "seq_aux_loss"],
        moe_aux_loss_coeff=[1e-2, 1e-2],
        moe_z_loss_coeff=1e-3,
        use_cpu_initialization=True,
        add_bias_linear=False,
    )
    router = TopKRouter(config, pg_collection=ProcessGroupCollection.use_mpu_process_groups())
    router.set_layer_number(1)
    return router.train()


@pytest.mark.parametrize("batch", [1, 2])
def test_router_losses_and_counts_ignore_the_padding(router_records, batch):
    """Row-tail padding with garbage logits (x 1e3): z-loss, aux loss and (MBS 1) sequence aux
    loss equal the router's values on the valid tokens alone; the padded logits get no
    gradient from them; expert-bias counts are the valid rows' routes. ``routing`` takes the
    ``[S, b]`` mask that ``MoELayer.route`` transposes from the batch-first one."""
    router = _router()
    generator = torch.Generator().manual_seed(0)
    content = 10
    logits = torch.randn(SEQ, batch, 8, generator=generator)
    logits[content:] = 1e3 * torch.randn(SEQ - content, batch, 8, generator=generator)
    padding = torch.zeros(batch, SEQ, dtype=torch.bool)
    padding[:, content:] = True
    padded_logits = logits.clone().requires_grad_()
    probs, routing_map = router.routing(padded_logits, padding_mask=padding.t())
    padded = dict(router_records)
    (probs.view(SEQ, batch, 8)[:content] * 0).sum().backward()
    assert not padded_logits.grad[content:].any()
    router_records.clear()
    valid_logits = logits[:content].clone().requires_grad_()
    valid_probs, valid_map = router.routing(valid_logits)
    reference = dict(router_records)
    (valid_probs * 0).sum().backward()
    names = ["z_loss", "load_balancing_loss"] + (["seq_load_balancing_loss"] if batch == 1 else [])
    for name in names:
        torch.testing.assert_close(padded[name], reference[name], rtol=1e-6, atol=0)
    torch.testing.assert_close(
        padded_logits.grad[:content], valid_logits.grad, rtol=1e-5, atol=1e-9
    )
    assert torch.equal(routing_map.view(SEQ, batch, 8)[:content].reshape(-1, 8), valid_map)
    # Expert-bias counts (the same counting function, its bool-map path) over the padded map.
    counter = SimpleNamespace(enable_expert_bias=True, local_tokens_per_expert=torch.zeros(8))
    TopKRouter._apply_expert_bias(counter, routing_map, padding_mask=padding.t().reshape(-1))
    assert torch.equal(counter.local_tokens_per_expert, valid_map.sum(0).float())


# ---------------------------------------------------------------------------
# The MTP depth's indexer-loss row mask
# ---------------------------------------------------------------------------


def _rolled(mask):
    """The MTP depth-1 padding mask: rolled left by one, the vacated position padding."""
    return torch.cat([mask[:, 1:], torch.ones_like(mask[:, :1])], dim=1)


@contextlib.contextmanager
def _mtp_layer_spy(model):
    """Records the indexer-loss mask in scope at every MTP transformer-layer call."""
    seen = []
    layer = model.mtp.layers[0].mtp_model_layer
    handle = layer.register_forward_pre_hook(
        lambda module, args: seen.append(indexer_loss_padding_mask_in_scope())
    )
    try:
        yield seen
    finally:
        handle.remove()


@pytest.mark.parametrize("arm", ["eager", "rsbf"])
def test_the_mtp_depth_scopes_its_rolled_padding_mask(request, hybrid, arm):
    """Eager loop model and RSBF (sweeps and the CODA item): the MTP layer runs inside the
    indexer-loss scope of its sample's rolled padding mask (the MTP block itself still sees
    the unrolled one, test_csa_indexer_row_mask.py)."""
    model = base._model("mtp", request)
    batches = base._batches(len(base.DEPTHS), padded=True)
    masks = [row["padding_mask"] for row in base._rows(batches)]
    with _mtp_layer_spy(model) as seen:
        if arm == "eager":
            base._eager(model, hybrid, batches, base.DEPTHS)
        else:
            base._rsbf(model, hybrid, batches, base.DEPTHS)
    assert len(seen) >= len(masks)
    expected = [_rolled(mask) for mask in masks]
    for mask in seen:
        assert mask is not None and any(torch.equal(mask, e) for e in expected)
    for e in expected:
        assert any(torch.equal(mask, e) for mask in seen)
    assert indexer_loss_padding_mask_in_scope() is None


def test_no_padding_mask_enters_no_mtp_scope(request, hybrid, monkeypatch):
    """Without a padding mask the MTP depth asks for no rolled mask and enters no scope (the
    pre-stage-5 call)."""
    model = base._model("mtp", request)
    batches = base._batches(2)
    calls = []
    real = mtp_module.MultiTokenPredictionLayer._get_embeddings

    def spy(self, *args, **kwargs):
        calls.append(kwargs.get("return_rolled_padding_mask"))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(mtp_module.MultiTokenPredictionLayer, "_get_embeddings", spy)
    with _mtp_layer_spy(model) as seen:
        base._eager(model, hybrid, batches, [2, 2])
    assert seen and all(mask is None for mask in seen)
    assert calls and not any(calls)


@pytest.mark.parametrize("sp_sharded_mask", [False, True])
def test_rolled_mask_is_taken_before_the_sequence_parallel_scatter(monkeypatch, sp_sharded_mask):
    """``_get_embeddings(return_rolled_padding_mask=True)`` under sequence parallelism: the
    seventh value is the full-length rolled mask, the third the SP-local scattered one."""
    tp_group = object()
    layer = SimpleNamespace(
        config=SimpleNamespace(
            sequence_parallel=True, attention_cp_layout="zigzag", mtp_detach_heads=False
        ),
        cp_group=None,
        tp_group=tp_group,
    )
    tokens = torch.arange(4).view(1, 4)
    full_mask = torch.tensor([[False, False, True, True]])
    rolled_full = torch.tensor([[False, True, True, True]])
    padding_mask = full_mask[:, :2].clone() if sp_sharded_mask else full_mask.clone()

    monkeypatch.setattr(mtp_module, "resolve_cp_group", lambda group, params: None)
    monkeypatch.setattr(
        mtp_module,
        "gather_from_sequence_parallel_region",
        lambda mask, tensor_parallel_output_grad, group: full_mask.transpose(0, 1),
    )
    monkeypatch.setattr(
        mtp_module, "scatter_to_sequence_parallel_region", lambda mask, group: mask[:2]
    )
    monkeypatch.setattr(
        mtp_module,
        "roll_tensor",
        lambda fields, **kwargs: [torch.tensor([[1, 2, 3, 0]]), rolled_full.clone()],
    )

    def embedding(input_ids, position_ids):
        return torch.zeros(2, 1, 4)

    embedding.add_position_embedding = False
    embedding.scatter_to_sequence_parallel = True
    kwargs = dict(
        input_ids=tokens,
        position_ids=tokens,
        embedding=embedding,
        hidden_states=torch.zeros(2, 1, 4),
        padding_mask=padding_mask,
    )
    plain = mtp_module.MultiTokenPredictionLayer._get_embeddings(layer, **kwargs)
    assert len(plain) == 6
    result = mtp_module.MultiTokenPredictionLayer._get_embeddings(
        layer, return_rolled_padding_mask=True, **kwargs
    )
    assert len(result) == 7
    assert torch.equal(result[6], rolled_full)
    assert torch.equal(result[2], rolled_full[:, :2])
    for a, b in zip(plain, result[:6]):
        assert (a is None and b is None) or torch.equal(a, b)


# ---------------------------------------------------------------------------
# GPU: a DSv4-like loop model (CSA indexer loss, MoE router with expert bias, MTP)
# ---------------------------------------------------------------------------


def _dsv4_batches(count, seq=256):
    """``count`` rows with row-tail padding of different lengths (row 0 unpadded)."""
    batches = base._batches(count, seq=seq, dev=torch.device("cuda"))
    for index, batch in enumerate(batches):
        content = seq - 37 * index
        padding = torch.arange(seq, device="cuda").unsqueeze(0) >= content
        batch["padding_mask"] = padding
        batch["loss_mask"] = batch["loss_mask"] * (torch.arange(seq, device="cuda") < content - 1)
    return batches


@contextlib.contextmanager
def _indexer_rows_spy(model):
    """Records (in the MTP block, query_valid_rows) for every CSA indexer-loss call."""
    from unittest.mock import patch

    from megatron.core.transformer.experimental_attention_variant import csa as csa_module

    calls, inside = [], [False]
    real = csa_module.indexer_loss_query_valid_rows

    def spy(b, sq, dev):
        rows = real(b, sq, dev)
        calls.append((inside[0], None if rows is None else rows.clone()))
        return rows

    handles = [
        model.mtp.register_forward_pre_hook(lambda module, args: inside.__setitem__(0, True)),
        model.mtp.register_forward_hook(lambda module, args, out: inside.__setitem__(0, False)),
    ]
    try:
        with patch.object(csa_module, "indexer_loss_query_valid_rows", spy):
            yield calls
    finally:
        for handle in handles:
            handle.remove()


@gpu
@pytest.mark.parametrize("arm", ["eager", "rsbf"])
def test_dsv4_indexer_losses_get_the_real_rows(hybrid, deterministic_algorithms, arm):
    """Every CSA indexer-loss call of the DSv4-like loop model (prelude, core, coda: the
    sample's real rows; the MTP depth: the real rows of its rolled mask), at full depth."""
    from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker, set_moe_metrics_tracker

    depths = (2, 2, 2)
    with base._hadamard_fallback():
        model = base._dsv4_model()
        batches = _dsv4_batches(len(depths))
        set_moe_metrics_tracker(MoEMetricsTracker())
        masks = [batch["padding_mask"] for batch in batches]
        with _indexer_rows_spy(model) as calls:
            if arm == "eager":
                base._eager(model, hybrid, batches, depths)
            else:
                base._rsbf(model, hybrid, batches, depths)
    from megatron.core.transformer.experimental_attention_variant.csa import (
        CompressedSparseAttention,
    )

    mtp_has_csa = any(isinstance(m, CompressedSparseAttention) for m in model.mtp.modules())
    assert mtp_has_csa == any(in_mtp for in_mtp, _ in calls)
    assert any(not in_mtp for in_mtp, _ in calls)
    for in_mtp, rows in calls:
        assert rows is not None
        expected = [~(_rolled(mask) if in_mtp else mask) for mask in masks]
        assert any(torch.equal(rows, e) for e in expected), in_mtp
    for mask in masks:
        assert any(torch.equal(rows, ~mask) for in_mtp, rows in calls if not in_mtp)
        if mtp_has_csa:
            assert any(torch.equal(rows, ~_rolled(mask)) for in_mtp, rows in calls if in_mtp)


@gpu
def test_dsv4_expert_bias_counts_are_the_valid_routes(
    hybrid, deterministic_algorithms, monkeypatch
):
    """Eager DSv4-like loop model with padded rows: every router's expert-bias token counts
    equal the routes of its valid tokens (a spy on ``TopKRouter.routing``), and changing the
    padded tokens changes neither the counts nor the loss."""
    from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker, set_moe_metrics_tracker

    depths = (2, 2, 2)
    expected = {}
    real = TopKRouter.routing
    # The test's own masks, flattened in the router's [S, b] order (b = 1): each batch's mask
    # and its MTP-rolled form. ``keep`` comes from these, not from what the router received,
    # so a router that lost the mask (None) or got a wrong one fails here.
    test_masks, matched = [], []

    def spy(self, logits, *args, padding_mask=None, **kwargs):
        probs, routing_map = real(self, logits, *args, padding_mask=padding_mask, **kwargs)
        assert padding_mask is not None, "the router got no padding mask"
        hits = [
            index
            for index, mask in enumerate(test_masks)
            if torch.equal(padding_mask.reshape(-1).bool(), mask)
        ]
        assert len(hits) == 1, "the router's padding mask is none of the batch masks"
        matched.append(hits[0])
        num_experts = self.config.num_moe_experts
        if routing_map.dtype == torch.bool:
            routes = routing_map
        else:
            valid_route = routing_map >= 0
            routes = (
                torch.zeros(
                    routing_map.shape[0], num_experts, dtype=torch.int64, device=routing_map.device
                )
                .scatter_add_(1, routing_map.long().clamp_min(0), valid_route.long())
                .bool()
            )
        keep = ~test_masks[hits[0]]
        counts = (routes & keep.unsqueeze(-1)).sum(0)
        expected[id(self)] = expected.get(id(self), 0) + counts
        return probs, routing_map

    monkeypatch.setattr(TopKRouter, "routing", spy)
    with base._hadamard_fallback():
        model = base._dsv4_model()
        batches = _dsv4_batches(len(depths))
        assert all(batch["padding_mask"].shape[0] == 1 for batch in batches)
        test_masks.extend(batch["padding_mask"].t().reshape(-1) for batch in batches)
        test_masks.extend(_rolled(batch["padding_mask"]).t().reshape(-1) for batch in batches)
        runs = []
        for arm in (batches, _poison_padding(batches)):
            for module in model.modules():
                if getattr(module, "local_tokens_per_expert", None) is not None:
                    module.local_tokens_per_expert.zero_()
            expected.clear()
            set_moe_metrics_tracker(MoEMetricsTracker())
            losses, _ = base._eager(model, hybrid, arm, depths)
            counts = {}
            for module in model.modules():
                if getattr(module, "local_tokens_per_expert", None) is not None:
                    counts[id(module)] = module.local_tokens_per_expert.clone()
                    assert torch.equal(
                        module.local_tokens_per_expert.long(), expected[id(module)].long()
                    )
            runs.append((losses, counts))
    # Every batch's own (unrolled) mask reached a router.
    assert set(range(len(batches))) <= set(matched)
    assert runs[0][1].keys() == runs[1][1].keys() and runs[0][1]
    for key, counts in runs[0][1].items():
        assert torch.equal(runs[1][1][key], counts)
    for a, b in zip(runs[0][0], runs[1][0]):
        torch.testing.assert_close(a, b, rtol=1e-2, atol=0)
