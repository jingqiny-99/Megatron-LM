# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The indexer-loss row mask of loop stacks (M8 glue; plan: frozen rows and row-tail padding
must not train the indexer).

``indexer_loss_padding_mask(mask)`` scopes a ``[b, sq]`` padding mask (True = padding) over
layer calls; the SBHD ``CompressedSparseAttention`` passes its complement as
``query_valid_rows`` to ``FusedDSAIndexerLoss`` on the unfused path (masked rows drop out of
the KL sum and its gradient) and refuses the fused training path with a mask in scope (its
kernel has no row mask). ``LoopHybridStack._run_range`` enters the scope with its layer
context's padding mask (row-tail padding; in per-row-depth rounds also the exited rows), the
RSBF CODA item and the eager loop model around the head (the MTP layers).

CPU: the scope helpers and where the loop stacks enter the scope (eager and RSBF; padded rows,
exited rows, MTP). GPU: an unfused CSA layer (ratio 4, indexer loss on) with row-tail padding:
the masked loss equals the loss with explicit ``query_valid_rows``, the padded tokens' inputs
change neither the loss nor the indexer gradients, no scope is the dev path bitwise, and the
fused training path raises with a mask in scope.
"""

import contextlib
from unittest.mock import patch

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import dsa_masking
from megatron.core.transformer.experimental_attention_variant.dsa_masking import (
    indexer_loss_padding_mask,
    indexer_loss_padding_mask_in_scope,
    indexer_loss_query_valid_rows,
)
from megatron.core.transformer.forward_replay import in_forward_replay, statistics_round
from tests.unit_tests.transformer.loop import hybrid_loop_utils
from tests.unit_tests.transformer.loop import test_loop_rsbf_executor as base
from tests.unit_tests.transformer.loop.hybrid_loop_utils import CUDA, SEQ, model_parallel_once

cpu_mtp = hybrid_loop_utils.cpu_mtp
hybrid = base.hybrid

gpu = pytest.mark.skipif(not CUDA, reason="CSA layers (TE linears) need CUDA")


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    with model_parallel_once():
        try:
            yield
        finally:
            base.get_loop_rsbf_schedule().reset()


# ---------------------------------------------------------------------------
# The scope
# ---------------------------------------------------------------------------


def test_scope_gives_the_real_rows_and_resets():
    assert indexer_loss_padding_mask_in_scope() is None
    assert indexer_loss_query_valid_rows(2, 4, torch.device("cpu")) is None
    mask = torch.tensor([[False, False, True, True], [False, False, False, False]])
    with indexer_loss_padding_mask(mask):
        assert indexer_loss_padding_mask_in_scope() is mask
        rows = indexer_loss_query_valid_rows(2, 4, torch.device("cpu"))
        assert torch.equal(rows, ~mask)
        # None hides an enclosing mask.
        with indexer_loss_padding_mask(None):
            assert indexer_loss_query_valid_rows(2, 4, torch.device("cpu")) is None
        assert indexer_loss_padding_mask_in_scope() is mask
        with pytest.raises(ValueError, match="shape mismatch"):
            indexer_loss_query_valid_rows(2, 8, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="boom"):
        with indexer_loss_padding_mask(mask):
            raise RuntimeError("boom")
    assert indexer_loss_padding_mask_in_scope() is None
    with indexer_loss_padding_mask(mask.int()):
        with pytest.raises(ValueError, match="must be bool"):
            indexer_loss_query_valid_rows(2, 4, torch.device("cpu"))


# ---------------------------------------------------------------------------
# Where the loop stacks enter the scope (CPU, local layers)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _mask_spy(model):
    """Pre-hooks on every decoder sublayer and on the MTP block recording (where, round,
    replay, the padding mask in scope) per call."""
    calls = []

    def hook(where):
        return lambda module, args: calls.append(
            (where, statistics_round(), in_forward_replay(), indexer_loss_padding_mask_in_scope())
        )

    handles = [
        layer.register_forward_pre_hook(hook(index))
        for index, layer in enumerate(model.decoder.layers)
    ]
    handles.append(model.mtp.register_forward_pre_hook(hook("mtp")))
    try:
        yield calls
    finally:
        for handle in handles:
            handle.remove()


def _padding_of(batches):
    """Every sample's ``[1, S]`` padding mask (rows of one-row microbatches)."""
    return [row["padding_mask"] for row in base._rows(batches)]


@pytest.mark.parametrize("arm", ["eager", "rsbf"])
def test_loop_layers_and_the_mtp_see_the_padding_mask(request, hybrid, arm):
    """Every sublayer call and the MTP block run inside the sample's padding mask (row-tail
    padding of the odd rows), in the eager loop model and in every RSBF cell and item (the
    sweeps, the replays, the CODA item's head)."""
    model = base._model("mtp", request)
    batches = base._batches(len(base.DEPTHS), padded=True)
    masks = _padding_of(batches)
    assert masks[1][0, -1] and not masks[0].any()
    with _mask_spy(model) as calls:
        if arm == "eager":
            base._eager(model, hybrid, batches, base.DEPTHS)
        else:
            base._rsbf(model, hybrid, batches, base.DEPTHS)
    assert calls and any(where == "mtp" for where, *_ in calls)
    seen = [mask for *_, mask in calls]
    assert all(mask is not None and mask.shape == (1, SEQ) for mask in seen)
    # Each call's mask is one sample's mask; every sample's mask reaches the MTP once.
    for mask in seen:
        assert any(torch.equal(mask, expected) for expected in masks)
    mtp_masks = [mask for where, _, _, mask in calls if where == "mtp"]
    assert len(mtp_masks) == len(base.DEPTHS)
    for expected in masks:
        assert any(torch.equal(mask, expected) for mask in mtp_masks)
    assert indexer_loss_padding_mask_in_scope() is None


def test_eager_exited_rows_are_masked_in_later_rounds(request, hybrid):
    """The eager per-row-depth loop (MBS 2, depths (1, 3)): in rounds 2 and 3 the core layers
    see row 0 (exited after round 1) masked in full, plus row 1's padding; in round 1 and in
    the prelude / coda only the padding."""
    model = base._model("mtp", request)
    roles = model.decoder.loop_roles
    batch = base._batches(2, 2, padded=True)[0]
    padding = batch["padding_mask"]
    model.decoder.set_loop_depths(torch.tensor([1, 3], device=padding.device))
    with _mask_spy(model) as calls:
        model(
            batch["tokens"],
            batch["position_ids"],
            None,
            labels=batch["labels"],
            loss_mask=batch["loss_mask"],
            padding_mask=padding,
        )
    exited = padding.clone()
    exited[0] = True
    rounds = set()
    for where, loop_round, replay, mask in calls:
        assert not replay
        if where != "mtp" and roles.core_start <= where < roles.core_end:
            rounds.add(loop_round[0])
            expected = padding if loop_round[0] == 1 else exited
        else:
            assert loop_round is None
            expected = padding
        assert torch.equal(mask, expected), (where, loop_round)
    assert rounds == {1, 2, 3}


# ---------------------------------------------------------------------------
# GPU: the unfused CSA layer under the scope
# ---------------------------------------------------------------------------


def _csa(seed=123):
    """An SBHD CSA layer, compress ratio 4 (compressor + indexer), indexer loss coefficient 1,
    unfused kernels (``dsa_kernel_backend='none'``), fp32 queries."""
    from megatron.core.models.common.embeddings import RotaryEmbedding
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.experimental_attention_variant.csa import (
        CompressedSparseAttention,
    )
    from tests.unit_tests.transformer.experimental_attention_variant import (
        test_attention_variant_csa as csa_tests,
    )

    config = csa_tests._make_mla_config(
        csa_compress_ratios=[4, 128, 4, 128],
        csa_window_size=8,
        dsa_indexer_topk=8,
        dsa_indexer_loss_coeff=1.0,
    )
    pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
    rotary = RotaryEmbedding(
        config.qk_pos_emb_head_dim,
        rotary_percent=config.rotary_percent,
        rotary_base=config.rotary_base,
        cp_group=pg_collection.cp,
    )
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    layer = CompressedSparseAttention(
        config=config,
        submodules=csa_tests._make_csa_submodules(),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        pg_collection=pg_collection,
        rotary_pos_emb=rotary,
        compress_ratio=4,
    ).cuda()
    layer.train()
    assert layer.indexer is not None and not layer.use_fused_kernels
    return layer


def _inputs(layer, seq, batch, seed=7):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    config = layer.config
    heads, dim = config.num_attention_heads, config.v_head_dim

    def randn(*shape, dtype=torch.float32):
        return torch.randn(*shape, generator=generator, device="cuda", dtype=dtype)

    return dict(
        query=randn(seq, batch, heads, dim),
        key=randn(seq, batch, 1, dim),
        x=randn(seq, batch, config.hidden_size, dtype=torch.bfloat16),
        qr=randn(seq, batch, config.q_lora_rank, dtype=torch.bfloat16),
    )


@contextlib.contextmanager
def _hadamard_fallback():
    with base._hadamard_fallback():
        yield


def _run(layer, inputs, mask=None):
    """One forward + backward of ``layer``; returns the indexer loss (captured from the
    ``FusedDSAIndexerLoss`` call), the ``query_valid_rows`` it received and the indexer
    parameters' gradients."""
    from megatron.core.transformer.experimental_attention_variant import csa as csa_module

    captured = {}
    real = csa_module.FusedDSAIndexerLoss.apply

    def spy(*args):
        captured["rows"] = args[14]
        topk, loss = real(*args)
        captured["loss"] = loss.detach().clone()
        return topk, loss

    layer.zero_grad(set_to_none=True)
    query = inputs["query"].clone().requires_grad_()
    key = inputs["key"].clone().requires_grad_()
    scope = contextlib.nullcontext() if mask is None else indexer_loss_padding_mask(mask)
    with patch.object(csa_module.FusedDSAIndexerLoss, "apply", spy), scope:
        output = layer(
            query=query,
            key=key,
            value=key.detach(),
            attention_mask=None,
            x=inputs["x"],
            qr=inputs["qr"],
        )
    output.float().sum().backward()
    grads = {
        name: param.grad.detach().clone()
        for name, param in layer.indexer.named_parameters()
        if param.grad is not None
    }
    assert grads
    return captured["loss"], captured["rows"], grads


@gpu
def test_unfused_csa_masks_padded_rows_of_the_indexer_loss():
    """Row 1's last quarter is padding: the loss is the one computed with explicit
    ``query_valid_rows``; changing the padded tokens' inputs changes neither the loss nor the
    indexer gradients; without a mask (no scope, or a None scope) the layer runs the dev path
    (``query_valid_rows`` None) bitwise."""
    seq, batch = 256, 2
    with _hadamard_fallback():
        layer = _csa()
        inputs = _inputs(layer, seq, batch)
        mask = torch.zeros(batch, seq, dtype=torch.bool, device="cuda")
        mask[1, -seq // 4 :] = True
        dev_loss, dev_rows, dev_grads = _run(layer, inputs)
        assert dev_rows is None
        none_loss, none_rows, none_grads = _run(layer, inputs, mask=None)
        assert none_rows is None and torch.equal(none_loss, dev_loss)
        for name, grad in dev_grads.items():
            assert torch.equal(none_grads[name], grad), name

        loss, rows, grads = _run(layer, inputs, mask=mask)
        assert torch.equal(rows, ~mask)
        assert not torch.equal(loss, dev_loss)

        # The masked loss equals the layer's loss computed with explicit real rows.
        from megatron.core.transformer.experimental_attention_variant import csa as csa_module

        real = csa_module.FusedDSAIndexerLoss.apply

        def explicit(*args):
            args = list(args)
            args[14] = ~mask
            return real(*args)

        with patch.object(csa_module.FusedDSAIndexerLoss, "apply", explicit):
            explicit_loss, _, explicit_grads = _run(layer, inputs)
        assert torch.equal(loss, explicit_loss)
        for name, grad in grads.items():
            assert torch.equal(explicit_grads[name], grad), name

        # Garbage in the padded tokens' inputs reaches neither the loss nor the gradients
        # (row-tail padding: only padded queries see the padded keys).
        perturbed = {name: value.clone() for name, value in inputs.items()}
        for name in ("query", "key", "x", "qr"):
            perturbed[name][-seq // 4 :, 1] = 10 * torch.randn_like(perturbed[name][-seq // 4 :, 1])
        perturbed_loss, _, perturbed_grads = _run(layer, perturbed, mask=mask)
        torch.testing.assert_close(perturbed_loss, loss, rtol=1e-6, atol=0)
        for name, grad in grads.items():
            torch.testing.assert_close(
                perturbed_grads[name], grad, rtol=1e-5, atol=1e-6 * grad.abs().max().item()
            )


@gpu
def test_fused_csa_training_path_refuses_a_row_mask():
    """The fused SBHD training path has no row mask: with a padding mask in scope and an
    indexer-loss coefficient it raises before any kernel runs."""
    seq, batch = 256, 2
    with _hadamard_fallback():
        layer = _csa()
        layer.use_fused_kernels = True
        inputs = _inputs(layer, seq, batch)
        mask = torch.zeros(batch, seq, dtype=torch.bool, device="cuda")
        with indexer_loss_padding_mask(mask), pytest.raises(NotImplementedError, match="row mask"):
            layer(
                query=inputs["query"],
                key=inputs["key"],
                value=inputs["key"],
                attention_mask=None,
                x=inputs["x"],
                qr=inputs["qr"],
            )
    assert dsa_masking.indexer_loss_padding_mask_in_scope() is None
