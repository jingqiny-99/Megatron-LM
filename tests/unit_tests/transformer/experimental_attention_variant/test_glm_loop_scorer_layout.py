# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Authenticated SBHD zigzag scoring and non-reentrant GLM checkpoint selection."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from megatron.core.transformer.experimental_attention_variant import (
    dsa,
    dsa_cudnn_kernels as kernels,
)
from megatron.core.transformer.loop.flops import native_dsa_forward_work
from megatron.core.transformer.loop.glm_dsa import GLMDSALoopStack
from megatron.core.transformer.loop.glm_ring_stack import GLMRingStack


def _bounds(seq, rank, device):
    chunks = torch.arange(seq, device=device).chunk(4)
    rows = torch.cat((chunks[rank], chunks[3 - rank]))
    return torch.zeros_like(rows), rows + 1


@pytest.mark.parametrize("rank", [0, 1])
def test_sbhd_uses_one_document_mxfp8_metadata(rank, monkeypatch):
    seq, top = 16, 3
    starts, ends = _bounds(seq, rank, "cpu")
    q = torch.zeros(1, seq // 2, 1, 4)
    k = torch.zeros(1, seq, 4)
    w = torch.ones(1, seq // 2, 1)
    options = kernels.IndexerScorerOptions(
        precision="mxfp8", scorer_backend="deepgemm", topk_backend="deepselect"
    )
    calls = []

    def packed(
        q,
        k,
        w,
        topk,
        scores,
        lo,
        hi,
        cu_q,
        cu_k,
        mq,
        mk,
        cp,
        cp_rank,
        local_start,
        local_len,
        *,
        deterministic,
        scorer_options,
    ):
        assert cp_rank == rank and cp == 2 and local_start == 0 and local_len == seq // 2
        assert cu_q is cu_k and cu_q.tolist() == [0, seq] and mq == mk == seq
        assert torch.equal(lo, starts) and torch.equal(hi, ends) and scorer_options is options
        calls.append(True)
        indices = torch.arange(topk)[None, None, :].expand(1, seq // 2, topk)
        return indices, None

    monkeypatch.setattr(kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(kernels, "_supports_indexer_q_causal_offsets", lambda: True)
    monkeypatch.setattr(kernels, "_indexer_topk_packed_thd", packed)
    indices, lengths, scores = kernels._indexer_topk_bshd(
        q,
        k,
        w,
        top,
        starts,
        ends,
        return_scores=False,
        packed_cp_size=2,
        local_packed_cp_rank=rank,
        local_packed_cp_query_len=seq // 2,
        sbhd_cp_zigzag=True,
        scorer_options=options,
    )
    assert calls == [True] and scores is None
    torch.testing.assert_close(lengths.flatten(), ends.int().clamp(max=top))
    assert (indices < ends[None, :, None]).all()


@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize("reuse", [False, True])
def test_selector_backward_and_checkpoint_work(train, reuse, monkeypatch):
    from megatron.core.models.hybrid.layers.utils import Symbols

    # Isolate the physical body's selector. The parent emitted a body plus a tail term.
    config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=2,
        q_lora_rank=6,
        kv_lora_rank=4,
        qk_head_dim=3,
        v_head_dim=3,
        qk_pos_emb_head_dim=2,
        dsa_indexer_topk=3,
        dsa_indexer_head_dim=4,
        dsa_indexer_n_heads=2,
        dsa_indexer_loss_coeff=float(train),
        dsa_indexer_reuse_checkpoint_topk=reuse,
        loop_iterations=2,
    )
    attention, selection = native_dsa_forward_work(config, 8, 32)
    module = SimpleNamespace(
        self_attention=SimpleNamespace(core_attention=SimpleNamespace(indexer=1))
    )

    class Body(GLMDSALoopStack):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.config = config
            self.layers = [module]
            self._loop_flop_rows = [(Symbols.DS_ATTENTION, config, None)]
            self._cross_loop_attention_layers = {0: (1, True)}

        def _ring_global_sequence(self, local):
            return local * 4

    def parent(self, lap, checkpoint, sequence, batch):
        yield attention + (selection if lap == 1 else 0), 2, True
        yield 13, 2, False

    monkeypatch.setattr(GLMRingStack, "_ring_lap_work", parent)
    body = Body()
    terms = list(body._ring_lap_work(1, True, 8, 1))
    assert sum(w for w, _, _ in terms) == attention + selection + 13
    assert sum(w * f for w, f, _ in terms) == 2 * (attention + selection * train + 13)
    projection = 2 * 8 * (6 * 2 * 4 + 8 * (4 + 2))
    assert sum(w for w, _, replay in terms if replay) == attention + (
        projection * train if reuse else selection
    )
    assert list(body._ring_lap_work(2, True, 8, 1)) == [(attention, 2, True), (13, 2, False)]


@pytest.mark.parametrize("train", [True, False], ids=["trainable", "frozen"])
def test_mxfp8_on_glm_nonreentrant_replay(train, monkeypatch):
    from megatron.core import tensor_parallel
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.loop_transformer_block import loop_checkpoint_context_fn
    from tests.unit_tests.test_utilities import Utils
    from tests.unit_tests.transformer.experimental_attention_variant import (
        test_dsa_checkpoint_topk_reuse as reuse,
        test_dsa_deepgemm_layer_gpu as gpu,
    )

    gpu._require_deepgemm_layer(1, monkeypatch)
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    try:
        gpu._warm_up_this_rank(indexer_heads=64)
        model_parallel_cuda_manual_seed(1234)
        torch.manual_seed(1234)
        dsa.DSAIndexerLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))
        groups = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        config = reuse._make_config(tp=1, cp=1, deterministic=False)
        gpu._use_deepgemm(config)
        if not train:
            config.dsa_indexer_loss_coeff = 0
        layer = reuse._make_attention(config, groups)
        inputs = gpu._alias_padded_cu_seqlens(
            reuse._make_inputs(config, tp_rank=0, cp_rank=0, cp=1)
        )
        spy = gpu._DeepGemmSpy(monkeypatch)
        gpu._PinnedTopk(monkeypatch)  # Select identical members among equally scored BF16 keys.
        step = (
            (lambda: reuse._checkpointed_step(layer, inputs, grad_output_seed=7))
            if train
            else (lambda: reuse._frozen_step(layer, inputs, grad_output_seed=7, checkpoint=True))
        )

        # Native reentrant checkpoint provides an independently scheduled reference.
        config.dsa_indexer_reuse_checkpoint_topk = False
        reference = step()
        assert spy.launches == 2

        config.dsa_indexer_reuse_checkpoint_topk = True
        spy.reset()
        monkeypatch.setattr(reuse, "_checkpoint_function", lambda f, *_args: f)

        def local_checkpoint(f, distribute, *args):
            owner = object()

            def contexts():
                fwd, replay = loop_checkpoint_context_fn("test_glm_dsa")()

                @contextmanager
                def phase(cm, replay):
                    with cm, dsa.dsa_nonreentrant_checkpoint_scope(owner, replay=replay):
                        yield

                return phase(fwd, False), phase(replay, True)

            with set_checkpoint_early_stop(False):
                return checkpoint(f, *args, use_reentrant=False, context_fn=contexts)

        monkeypatch.setattr(tensor_parallel, "checkpoint", local_checkpoint)
        actual = step()
        assert spy.launches == 1 and spy.recomputes == int(train)
        torch.testing.assert_close(actual[0], reference[0], atol=0, rtol=0)
        if train:
            torch.testing.assert_close(actual[1], reference[1], atol=0, rtol=0)
        assert actual[-1].keys() == reference[-1].keys()
        for key, value in actual[-1].items():
            reuse._assert_similar(value, reference[-1][key], label=key, eps=1e-4)
    finally:
        dsa.DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        Utils.destroy_model_parallel()


def test_sbhd_deepgemm_deepselect_matches_one_document(monkeypatch):
    from megatron.core.transformer.experimental_attention_variant import dsa_deepselect_topk
    from tests.unit_tests.transformer.experimental_attention_variant import (
        test_dsa_cudnn_mxfp8_scorer_gpu as inputs,
        test_dsa_deepgemm_layer_gpu as gpu,
    )

    gpu._require_deepgemm_layer(1, monkeypatch)
    if torch.cuda.get_device_properties(0).total_memory < 200 * 1024**3:
        pytest.skip("wide SBHD dispatch requires a GB300")
    gpu._warm_up_this_rank(indexer_heads=32)
    spy = gpu._DeepGemmSpy(monkeypatch)
    n_select = []
    topk = dsa_deepselect_topk.deepselect_topk

    def select(*args, **kwargs):
        n_select.append(True)
        return topk(*args, **kwargs)

    monkeypatch.setattr(dsa_deepselect_topk, "deepselect_topk", select)
    seq = 65536
    q, k, w = inputs._inputs(seq // 2, seq, 54)
    starts, ends = _bounds(seq, 0, q.device)
    options = kernels.IndexerScorerOptions(
        precision="mxfp8", scorer_backend="deepgemm", topk_backend="deepselect"
    )
    common = dict(
        q_bshd=q,
        k_bsd=k,
        w_bsh=w,
        topk=2048,
        varlen_starts=starts,
        varlen_ends=ends,
        return_scores=False,
        scorer_options=options,
        packed_cp_size=2,
        local_packed_cp_rank=0,
        local_packed_cp_query_len=seq // 2,
    )
    cu = torch.tensor([0, seq], device=q.device, dtype=torch.int32)
    thd = kernels._indexer_topk_bshd(
        **common,
        use_local_indexer_varlen=True,
        single_packed_thd_sequence=True,
        packed_cu_seqlens_q=cu,
        packed_cu_seqlens_k=cu,
        packed_max_seqlen_q=seq,
        packed_max_seqlen_k=seq,
    )
    launches = spy.launches
    selections = len(n_select)
    assert launches > 0 and selections > 0 and set(spy.reasons) == {None}
    spy.reset()
    n_select.clear()
    sbhd = kernels._indexer_topk_bshd(**common, sbhd_cp_zigzag=True)
    assert spy.launches == launches and len(n_select) == selections and set(spy.reasons) == {None}
    assert torch.equal(sbhd[0], thd[0]) and torch.equal(sbhd[1], thd[1])
    print(
        f"SBHD CP2 matched the one-document THD selections on {seq // 2} rows; "
        f"{launches} DeepGEMM and {selections} DeepSelect launches per path."
    )


def test_loop_carrier_and_mtp_record_ownership():
    from megatron.core.transformer.loop.glm_dsa import GLMLoopDSAttention, _INDEX_SHARE

    layer = object.__new__(GLMLoopDSAttention)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace()
    pack = SimpleNamespace()
    carriers = [SimpleNamespace() for _ in range(3)]
    records = []
    mtp = layer._get_mtp_sharing_payload(pack, None, create=True)
    mtp.topk_by_layer[3] = torch.tensor([7])
    for n, carrier in enumerate(carriers):
        token = _INDEX_SHARE.set(carrier)
        try:
            record = layer._get_dsa_index_sharing_payload(pack, None)
            record.topk_by_layer[3] = torch.tensor([n])
            assert (
                layer._get_dsa_index_sharing_payload(pack, None).topk_by_layer
                is record.topk_by_layer
            )
            assert layer._get_mtp_sharing_payload(pack, None, create=False) is mtp
            records.append(record)
        finally:
            _INDEX_SHARE.reset(token)
    assert [r.topk_by_layer[3].item() for r in records] == [0, 1, 2]
    assert mtp.topk_by_layer[3].item() == 7


def test_pipeline_prewarm_physical_mtp_and_loop_indexers():
    from megatron.core.pipeline_parallel.parallel_prewarm import _discover_local_layers
    from megatron.core.transformer.loop.glm_dsa import (
        GLMLoopDSAttention,
        _INDEX_SHARE,
        loop_hybrid_glm_dsa_stack_spec,
    )

    spec = loop_hybrid_glm_dsa_stack_spec(None)

    def attention(submodules):
        return submodules.dsa_layer.submodules.self_attention.submodules.core_attention.module

    assert attention(spec.submodules) is GLMLoopDSAttention
    assert attention(spec.submodules.mtp_stack_submodules) is dsa.DSAttention
    s0, s1 = GLMDSALoopStack.__new__(GLMDSALoopStack), GLMDSALoopStack.__new__(GLMDSALoopStack)
    for stack in (s0, s1):
        torch.nn.Module.__init__(stack)
    l0, l1, d, e = range(4)
    s0.layers = [l0]
    s1.layers = [l1]
    mtp = SimpleNamespace(is_hybrid_mtp=True, mtp_model_layer=SimpleNamespace(layers=[d, e]))
    c0 = SimpleNamespace(decoder=s0)
    c1 = SimpleNamespace(decoder=s1, mtp=SimpleNamespace(layers=[mtp]))
    assert _discover_local_layers([c0, c1]) == [
        (l0, False, c0),
        (l1, False, c1),
        (d, True, c1),
        (e, True, c1),
    ]
    requests = [{}, {}]
    with s0.layer_prewarm_context(requests[0]):
        first = _INDEX_SHARE.get()
        first.topk = torch.tensor([3])
    assert _INDEX_SHARE.get() is None
    with s0.layer_prewarm_context(requests[0]):
        assert _INDEX_SHARE.get() is first
    with s1.layer_prewarm_context(requests[1]):
        assert _INDEX_SHARE.get() is not first and not hasattr(_INDEX_SHARE.get(), 'topk')
    assert _INDEX_SHARE.get() is None
