# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the loop parameter store's weight binding (``loop/param_store_binding.py``).

Single linears, no store and no collective: a "gather" is a copy of the full weights into a slot
region. The first part runs on CPU and CUDA: the environment probe, placeholders and hiding, the
``.data`` binding against plain parameters for every consumer kind (including the saved-alias
consumers of E14 under an address-stable re-gather), the three gradient-capture classes with fake
autograd Functions mimicking TE (lazy ``main_grad``, E4) and Megatron's fused linear (forward
capture, E15), a tied weight's fused GEMM plus lookup against DDP's hook arithmetic
(``zero_out_wgrad``), overwrite / zeroing / pad tails, the use counter (recompute included) and
the per-segment check of overwrite targets (unwritten, raw-consumed), and the
``DistributedWeight`` binding against a fake TE 2.20 consumer routed through local copies of TE's
dispatchers. The second part needs CUDA and Transformer
Engine: TE linears and Megatron's fused local linear with rebound weights (forward, backward, wgrad
accumulation), and the cast after the gather under FP8 block scaling / MXFP8 where the device
supports it. Model parallelism (TP = 1) is initialized once for the module.
"""

import math
import weakref

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer.loop import param_store_binding as pb
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.transformer.loop.hybrid_loop_utils import model_parallel_once

CUDA = torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])
NAN = float("nan")


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    with model_parallel_once():
        yield


@pytest.fixture(params=DEVICES)
def dev(request):
    return torch.device(request.param)


# ---------------------------------------------------------------------------------------------
# Helpers: consumers and fake autograd Functions
# ---------------------------------------------------------------------------------------------


def _is_dw(weight):
    """TE 2.20 ``is_distributed_weight``: the marker, on a ``torch.Tensor`` only."""
    flag = bool(getattr(weight, "is_distributed_weight", False))
    if flag and not isinstance(weight, torch.Tensor):
        raise TypeError("DistributedWeight implementers must be torch.Tensor subclasses")
    return flag


# Local copies of TE 2.20's dispatchers (transformer_engine/pytorch/distributed_weight.py): list
# shaped, the leader is weights[0], and a single wgrad is handed over bare (not as a list).


def _te_materialize_weight_for_forward(weights):
    weights = list(weights) if isinstance(weights, (list, tuple)) else [weights]
    if _is_dw(weights[0]):
        out = weights[0].materialize_group_for_forward()
        return list(out) if isinstance(out, (list, tuple)) else [out]
    return weights


def _te_materialize_weight_for_backward(weights):
    weights = list(weights) if isinstance(weights, (list, tuple)) else [weights]
    if _is_dw(weights[0]):
        out = weights[0].materialize_group_for_backward()
        return list(out) if isinstance(out, (list, tuple)) else [out]
    return weights


def _te_finalize_weight_grads(weights, wgrads):
    weights = list(weights) if isinstance(weights, (list, tuple)) else [weights]
    if _is_dw(weights[0]):
        out = weights[0].finalize_group_grads(wgrads if len(wgrads) > 1 else wgrads[0])
        return list(out) if isinstance(out, (list, tuple)) else [out]
    return list(wgrads)


def _te_dummy_wgrad(origin, zero):
    """TE's ``get_dummy_wgrad`` on the origin's (shard) shape; "empty" is NaN here (stricter)."""
    fill = 0.0 if zero else NAN
    return torch.full(tuple(origin.shape), fill, dtype=origin.dtype, device=origin.device)


def _wgrad(grad_out, inp):
    """dW = dY^T X over every leading dim, in fp32 (a fused-wgrad GEMM's output dtype)."""
    g2 = grad_out.reshape(-1, grad_out.shape[-1]).float()
    x2 = inp.reshape(-1, inp.shape[-1]).float()
    return g2.t() @ x2


class FakeTELinearFn(torch.autograd.Function):
    """TE's ``_Linear`` weight / ``main_grad`` contract (TE 2.16 and the 2.20 protocol), in torch.

    Forward: a distributed weight is materialized through TE's dispatcher;
    ``overwrite_main_grad`` is read; the ``main_grad`` getter is captured lazily
    (``weight.grad_buffer`` for a distributed weight). Backward (linear.py:1170-1185, 1573-1575,
    1654, 1699-1719 of TE 2.20): ``main_grad`` is fetched (and re-assigned) *now*; the wgrad GEMM
    accumulates unless overwrite or distributed; a distributed weight's wgrad goes through
    ``finalize_weight_grads`` (bare tensor); with fused wgrad and ``grad_added_to_main_grad`` the
    returned gradient is TE's dummy on the origin's shape (zero under ``zero_out_wgrad``).
    """

    @staticmethod
    def forward(ctx, inp, weight, fuse_wgrad):
        dist = _is_dw(weight)
        mat = _te_materialize_weight_for_forward(weight)[0] if dist else weight
        ctx.save_for_backward(inp, weight)
        ctx.fuse = bool(fuse_wgrad and weight.requires_grad)
        if ctx.fuse:
            ctx.origin = weakref.ref(weight)
            ctx.overwrite = bool(getattr(weight, "overwrite_main_grad", False))
            ctx.main_grad_func = weight.grad_buffer if dist else (lambda: weight.main_grad)
        return inp @ mat.t()

    @staticmethod
    def backward(ctx, grad_out):
        inp, saved = ctx.saved_tensors
        dist = _is_dw(saved)
        origin = None
        if ctx.fuse:
            origin = ctx.origin()
            main_grad = ctx.main_grad_func()
            if not dist:
                origin.main_grad = main_grad
        mat = _te_materialize_weight_for_backward(saved)[0] if dist else saved
        dgrad = grad_out @ mat
        if ctx.fuse:
            accumulate = False if dist else not ctx.overwrite
            if accumulate:
                main_grad.add_(_wgrad(grad_out, inp))
            else:
                main_grad.copy_(_wgrad(grad_out, inp))
            wgrad = main_grad
        else:
            wgrad = (grad_out.reshape(-1, grad_out.shape[-1]).t()) @ inp.reshape(-1, inp.shape[-1])
        if dist:
            wgrad = _te_finalize_weight_grads(saved, [wgrad])[0]
        if ctx.fuse:
            if hasattr(origin, "grad_added_to_main_grad"):
                origin.grad_added_to_main_grad = True
                wgrad = _te_dummy_wgrad(origin, getattr(origin, "zero_out_wgrad", False))
            else:
                wgrad = None
        return dgrad, wgrad, None


class FakeTELinear(nn.Module):
    """A TE-like linear: weight [out, in], ``fuse_wgrad_accumulation``, ``FakeTELinearFn``."""

    def __init__(self, in_f, out_f, *, fuse, dtype=torch.float32, device="cpu"):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f, dtype=dtype, device=device) * 0.1)
        self.fuse_wgrad_accumulation = fuse

    def forward(self, x):
        return FakeTELinearFn.apply(x, self.weight, self.fuse_wgrad_accumulation)


class FakeMegatronFusedFn(torch.autograd.Function):
    """Megatron's fused local linear (E15): ``main_grad`` captured at forward, accumulated.

    The returned gradient is a dummy (zero under ``zero_out_wgrad``, layers.py:843-860).
    """

    @staticmethod
    def forward(ctx, inp, weight):
        ctx.main_grad = weight.main_grad if hasattr(weight, "main_grad") else None
        ctx.save_for_backward(inp, weight)
        return inp @ weight.t()

    @staticmethod
    def backward(ctx, grad_out):
        inp, weight = ctx.saved_tensors
        weight.main_grad = ctx.main_grad
        weight.main_grad.add_(_wgrad(grad_out, inp))
        if not hasattr(weight, "grad_added_to_main_grad"):
            return grad_out @ weight, None
        weight.grad_added_to_main_grad = True
        return grad_out @ weight, _te_dummy_wgrad(weight, getattr(weight, "zero_out_wgrad", False))


class FakeMegatronFused(nn.Module):
    def __init__(self, in_f, out_f, device="cpu"):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f, device=device) * 0.1)

    def forward(self, x):
        return FakeMegatronFusedFn.apply(x, self.weight)


class SaveParamFn(torch.autograd.Function):
    """A custom Function that saves the leaf parameter itself (follows a ``.data`` rebind)."""

    @staticmethod
    def forward(ctx, inp, weight):
        ctx.save_for_backward(inp, weight)
        return inp @ weight.t()

    @staticmethod
    def backward(ctx, grad_out):
        inp, weight = ctx.saved_tensors
        g2 = grad_out.reshape(-1, grad_out.shape[-1])
        return grad_out @ weight, g2.t() @ inp.reshape(-1, inp.shape[-1])


IN_F, OUT_F = 12, 10
# Consumers whose saved tensors alias the forward-time storage (E14) vs the leaf parameter.
ALIAS_KINDS = ("nn_linear", "linear_3d", "einsum_view")
PARAM_KINDS = ("save_param_fn", "megatron_cpl")
KINDS = ALIAS_KINDS + PARAM_KINDS


class RawConsumer(nn.Module):
    """A raw ``weight`` consumed by ``F.linear`` 3-D, einsum on a view, or ``SaveParamFn``."""

    def __init__(self, kind, device):
        super().__init__()
        self.kind = kind
        self.weight = nn.Parameter(torch.randn(OUT_F, IN_F, device=device) * 0.1)

    def forward(self, x):
        if self.kind == "linear_3d":
            return F.linear(x, self.weight)
        if self.kind == "einsum_view":  # the DSv4 linear_o_group_proj pattern
            return torch.einsum("...d,rd->...r", x, self.weight.view(OUT_F, IN_F))
        return SaveParamFn.apply(x, self.weight)


def _cpl_config(**over):
    kwargs = dict(num_layers=1, hidden_size=16, num_attention_heads=1, use_cpu_initialization=True)
    kwargs.update(over)
    return TransformerConfig(**kwargs)


def _consumer(kind, device):
    """(module owning ``weight``, callable x -> y, input shape)."""
    torch.manual_seed(1)
    if kind == "nn_linear":
        mod = nn.Linear(IN_F, OUT_F, bias=False).to(device)
        return mod, mod, (7, IN_F)
    if kind == "megatron_cpl":
        mod = ColumnParallelLinear(
            IN_F,
            OUT_F,
            config=_cpl_config(),
            init_method=nn.init.normal_,
            bias=False,
            gather_output=False,
        ).to(device)
        return mod, (lambda x: mod(x)[0]), (5, 2, IN_F)
    mod = RawConsumer(kind, device)
    return mod, mod, (5, 2, IN_F)


def _reference(kind, device, weight, x, grad_out):
    """Output, dgrad and wgrad of the same consumer with an ordinary parameter."""
    mod, fn, _ = _consumer(kind, device)
    with torch.no_grad():
        mod.weight.copy_(weight)
    x = x.detach().clone().requires_grad_()
    y = fn(x)
    y.backward(grad_out)
    return y.detach(), x.grad, mod.weight.grad.float()


def _take(module, *, name="u.weight", z=1, **kwargs):
    return pb.make_store_tensor(name, module, "weight", shard_size=z, **kwargs)


def _slot(tensors, device, dtype=torch.float32):
    numel = pb.layout_tensors(tensors)
    return torch.empty(numel, dtype=dtype, device=device)


def _scratch(tensor, device, fill=NAN):
    return torch.full((tensor.numel_padded,), fill, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------------------------
# Environment probe
# ---------------------------------------------------------------------------------------------


def test_environment_probe_records_facts():
    env = pb.probe_store_environment()
    assert env.te_version == pb.te_version()
    assert env.distributed_weight == pb.te_distributed_weight_available()
    assert (env.distributed_weight_reason is None) == env.distributed_weight
    assert env.cuda == CUDA
    assert env.default_binding == "data"
    summary = env.summary()
    for key in ("transformer_engine", "DistributedWeight protocol", "FP8 block scaling", "MXFP8"):
        assert key in summary
    # The stage report reads these lines from the per-rank log (design S6.0 (d)).
    print("loop param store environment:\n" + summary)


def test_distributed_weight_reason_names_the_te_version(monkeypatch):
    monkeypatch.setattr(pb, "te_version", lambda: "2.16.0+4220403e")
    assert not pb.te_distributed_weight_available()  # the version gate, before any import
    assert pb.distributed_weight_unavailable_reason() == (
        "TE 2.16.0+4220403e has no DistributedWeight protocol (needs TE >= 2.19, e.g. the "
        "uv.lock pin 2.20.2); use binding='data'"
    )
    monkeypatch.setattr(pb, "te_version", lambda: None)
    assert not pb.te_distributed_weight_available()
    assert "not installed" in pb.distributed_weight_unavailable_reason()
    monkeypatch.setattr(pb, "te_version", lambda: "2.20.2+44f31bbe")
    monkeypatch.setattr(pb.importlib.util, "find_spec", lambda name: object())
    assert pb.te_distributed_weight_available()
    assert pb.distributed_weight_unavailable_reason() is None
    monkeypatch.setattr(pb.importlib.util, "find_spec", lambda name: None)
    assert not pb.te_distributed_weight_available()


# ---------------------------------------------------------------------------------------------
# Placeholders, hiding, layout
# ---------------------------------------------------------------------------------------------


def test_placeholder_is_logical_nan_and_write_protected(dev):
    p = nn.Parameter(torch.randn(4, 3, device=dev))
    ph = pb.logical_placeholder((4, 3), p.dtype, dev)
    p.data = ph
    assert p.shape == (4, 3) and p.numel() == 12
    assert torch.isnan(p).all()
    assert pb.is_placeholder(p.data)
    with pytest.raises(RuntimeError, match="single memory location"):
        with torch.no_grad():
            p.add_(1.0)
    # One element per (dtype, device), shared by every placeholder.
    assert pb.logical_placeholder((7, 2, 5), p.dtype, dev).data_ptr() == ph.data_ptr()
    with pytest.raises(ValueError):
        pb.logical_placeholder((2,), torch.int64, dev)


def test_hide_parameter_keeps_attribute_and_logical_name():
    model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
    weight = model[0].weight
    assert pb.hide_parameter(model[0], "weight") is weight
    assert model[0].weight is weight
    names = [n for n, _ in model.named_parameters()]
    assert "0.weight" not in names and "0.bias" in names
    assert "0.weight" not in model.state_dict()
    logical = dict(pb.named_logical_parameters(model))
    assert logical["0.weight"] is weight
    assert set(logical) == {"0.weight", "0.bias", "1.weight", "1.bias"}
    assert dict(pb.named_logical_parameters(model, prefix="m"))["m.0.weight"] is weight
    assert set(dict(pb.named_logical_parameters(model[0], recurse=False))) == {"bias", "weight"}
    # The forward still reads the hidden parameter.
    x = torch.randn(5, 4)
    assert torch.equal(model[0](x), F.linear(x, weight, model[0].bias))
    # The store's mirror subtree is skipped.
    mirror = nn.Module()
    mirror.shard = nn.Parameter(torch.zeros(2, 4))
    object.__setattr__(mirror, pb.MIRROR_FLAG_ATTR, True)
    model.add_module("loop_param_store", mirror)
    assert "loop_param_store.shard" in dict(model.named_parameters())
    assert "loop_param_store.shard" not in dict(pb.named_logical_parameters(model))
    assert pb.restore_parameter(model[0], "weight") is weight
    assert "0.weight" in dict(model.named_parameters())
    with pytest.raises(ValueError):
        pb.hide_parameter(model[0], "missing")


def test_layout_regions_are_aligned_disjoint_and_padded():
    mods = [nn.Linear(6, 5, bias=False), nn.Linear(3, 7, bias=False), nn.Linear(4, 4, bias=False)]
    for mod in mods:
        mod.to(torch.bfloat16)
    tensors = [_take(m, name=f"m{i}.weight", z=3) for i, m in enumerate(mods)]
    assert [t.rows_padded for t in tensors] == [6, 9, 6]
    assert [t.pad_length for t in tensors] == [1, 2, 2]
    assert [t.shard_shape for t in tensors] == [(2, 6), (3, 3), (2, 4)]
    numel = pb.layout_tensors(tensors, align_bytes=256)
    step = 256 // 2
    ends = []
    for t in tensors:
        assert t.offset % step == 0
        ends.append((t.offset, t.offset + t.numel_padded))
    assert all(a[1] <= b[0] for a, b in zip(ends, ends[1:]))
    assert numel % step == 0 and numel >= ends[-1][1]
    flat = torch.zeros(numel, dtype=torch.bfloat16)
    assert pb.tensor_view(flat, tensors[1]).shape == (7, 3)
    assert pb.tensor_region(flat, tensors[1], padded=True).numel() == 27
    with pytest.raises(ValueError):
        pb.tensor_region(flat[:10], tensors[2])
    full = torch.arange(15.0).view(5, 3)
    shards = [pb.shard_slice(full, 2, r) for r in range(2)]
    assert torch.equal(torch.cat(shards)[:5], full) and torch.equal(shards[1][-1], torch.zeros(3))


# ---------------------------------------------------------------------------------------------
# The data binding with every consumer kind
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_data_binding_matches_ordinary_parameter(dev, kind):
    mod, fn, shape = _consumer(kind, dev)
    full = mod.weight.detach().clone()
    t = _take(mod)
    assert t.grad_capture == "hook" and t.weight is t.compute
    slot = _slot([t], dev)
    pb.unbind_weight(t)
    assert pb.is_placeholder(mod.weight.data) and mod.weight.shape == full.shape
    pb.tensor_view(slot, t).copy_(full)  # the "gather"
    pb.bind_weight(t, pb.tensor_view(slot, t))
    assert mod.weight.data_ptr() == slot.data_ptr()
    target = pb.bind_grad(t, _scratch(t, dev), zero=True)
    x = torch.randn(shape, device=dev, requires_grad=True)
    g = torch.randn(*shape[:-1], OUT_F, device=dev)
    y = fn(x)
    y.backward(g)
    ref_y, ref_dx, ref_dw = _reference(kind, dev, full, x, g)
    assert torch.equal(y, ref_y) and torch.equal(x.grad, ref_dx)
    assert torch.equal(target, ref_dw)
    assert mod.weight.grad is None


@pytest.mark.parametrize("kind", KINDS)
def test_regather_into_the_same_slot_is_exact(dev, kind):
    """Released and NaN-filled, the slot reused by another unit, re-gathered: exact (E14)."""
    mod, fn, shape = _consumer(kind, dev)
    other, _, _ = _consumer("nn_linear", dev)
    with torch.no_grad():
        other.weight.mul_(3.0)
    full, other_full = mod.weight.detach().clone(), other.weight.detach().clone()
    t, u = _take(mod), _take(other, name="v.weight")
    slot_a = _slot([t], dev)
    view = pb.tensor_view(slot_a, t)
    view.copy_(full)
    pb.bind_weight(t, view)
    x = torch.randn(shape, device=dev, requires_grad=True)
    g = torch.randn(*shape[:-1], OUT_F, device=dev)
    y = fn(x)
    pb.unbind_weight(t)
    slot_a.fill_(NAN)  # poison_released
    pb.tensor_view(slot_a, u).copy_(other_full)  # another unit takes the slot ...
    pb.bind_weight(u, pb.tensor_view(slot_a, u))
    with torch.no_grad():
        other(torch.randn(3, IN_F, device=dev))
    pb.unbind_weight(u)
    slot_a.fill_(NAN)
    view.copy_(full)  # ... and the unit is re-gathered into its forward slot
    pb.bind_weight(t, view)
    target = pb.bind_grad(t, _scratch(t, dev), zero=True)
    y.backward(g)
    ref_y, ref_dx, ref_dw = _reference(kind, dev, full, x, g)
    assert torch.equal(y, ref_y) and torch.equal(x.grad, ref_dx) and torch.equal(target, ref_dw)


@pytest.mark.parametrize("kind", KINDS)
def test_regather_into_another_slot_breaks_alias_consumers(dev, kind):
    """Documents E14: only consumers that saved the leaf parameter follow a moved binding."""
    mod, fn, shape = _consumer(kind, dev)
    full = mod.weight.detach().clone()
    t = _take(mod)
    slot_a, slot_b = _slot([t], dev), _slot([t], dev)
    pb.tensor_view(slot_a, t).copy_(full)
    pb.bind_weight(t, pb.tensor_view(slot_a, t))
    x = torch.randn(shape, device=dev, requires_grad=True)
    g = torch.randn(*shape[:-1], OUT_F, device=dev)
    y = fn(x)
    pb.unbind_weight(t)
    slot_a.fill_(NAN)
    pb.tensor_view(slot_b, t).copy_(full)
    pb.bind_weight(t, pb.tensor_view(slot_b, t))
    target = pb.bind_grad(t, _scratch(t, dev), zero=True)
    y.backward(g)
    _, ref_dx, ref_dw = _reference(kind, dev, full, x, g)
    assert torch.equal(target, ref_dw)  # dW = dY^T X never reads the weight
    if kind in ALIAS_KINDS:
        assert torch.isnan(x.grad).all()
    else:
        assert torch.equal(x.grad, ref_dx)


# ---------------------------------------------------------------------------------------------
# Gradient capture classes, overwrite, zeroing, pad tails, hooks
# ---------------------------------------------------------------------------------------------


def _fake_te_reference(weight, xs, gs, *, fuse=True):
    ref = FakeTELinear(IN_F, OUT_F, fuse=fuse, device=weight.device)
    with torch.no_grad():
        ref.weight.copy_(weight)
    ref.weight.main_grad = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)
    ref.weight.grad_added_to_main_grad = False
    dxs = []
    for x, g in zip(xs, gs):
        x = x.detach().clone().requires_grad_()
        ref(x).backward(g)
        dxs.append(x.grad)
    if fuse:
        return ref.weight.main_grad, dxs
    return ref.weight.grad.float(), dxs


def test_lazy_main_grad_bound_after_forward(dev):
    """E4: TE reads ``main_grad`` at backward, so the target may be bound after the forward."""
    mod = FakeTELinear(IN_F, OUT_F, fuse=True, device=dev)
    full = mod.weight.detach().clone()
    t = _take(mod, te_linear=True)
    assert t.grad_capture == "lazy" and t.fused_wgrad  # FakeTELinear is not a TE module class
    assert pb.classify_grad_capture(mod, "weight") == "hook"
    slot = _slot([t], dev)
    pb.tensor_view(slot, t).copy_(full)
    pb.bind_weight(t, pb.tensor_view(slot, t))
    assert not hasattr(mod.weight, "main_grad")
    x = torch.randn(4, 3, IN_F, device=dev, requires_grad=True)
    g = torch.randn(4, 3, OUT_F, device=dev)
    ref_dw, (ref_dx,) = _fake_te_reference(full, [x], [g])
    # overwrite_main_grad is read at forward: set by the mode before the step's first forward.
    pb.set_overwrite(t, True)
    y = mod(x)
    target = pb.bind_grad(t, _scratch(t, dev, fill=5.0))  # dirty entry: only the pad tail zeroed
    y.backward(g)
    assert torch.equal(target, ref_dw) and torch.equal(x.grad, ref_dx)
    assert mod.weight.grad is None and mod.weight.grad_added_to_main_grad is False
    pb.unbind_grad(t)
    assert not hasattr(mod.weight, "main_grad") and mod.weight.overwrite_main_grad is True
    # Set after the forward, the flag is not seen: the dirty entry is accumulated into (E4).
    pb.set_overwrite(t, False)
    y = mod(x)
    pb.set_overwrite(t, True)
    target = pb.bind_grad(t, _scratch(t, dev, fill=5.0))
    y.backward(g)
    assert torch.equal(target, ref_dw + 5.0)


def test_lazy_resident_accumulator_over_microbatches(dev):
    mod = FakeTELinear(IN_F, OUT_F, fuse=True, device=dev)
    full = mod.weight.detach().clone()
    t = _take(mod, te_linear=True)
    slot = _slot([t], dev)
    pb.tensor_view(slot, t).copy_(full)
    pb.bind_weight(t, pb.tensor_view(slot, t))
    acc = _scratch(t, dev)
    target = pb.bind_grad(t, acc, zero=True)
    xs = [torch.randn(4, IN_F, device=dev) for _ in range(3)]
    gs = [torch.randn(4, OUT_F, device=dev) for _ in range(3)]
    for i, (x, g) in enumerate(zip(xs, gs)):
        if i:
            assert pb.bind_grad(t, acc, zero=False) is not target  # rebinding keeps the sum
        mod(x.requires_grad_()).backward(g)
    ref_dw, _ = _fake_te_reference(full, xs, gs)
    assert torch.equal(pb.tensor_view(acc, t), ref_dw)


def test_forward_capture_target_is_bound_at_forward_and_zeroed(dev):
    """E15: Megatron's fused linear captures ``main_grad`` at forward and always accumulates."""
    mod = FakeMegatronFused(IN_F, OUT_F, device=dev)
    full = mod.weight.detach().clone()
    t = _take(mod, grad_capture="forward")
    slot = _slot([t], dev)
    pb.tensor_view(slot, t).copy_(full)
    pb.bind_weight(t, pb.tensor_view(slot, t))
    x = torch.randn(6, IN_F, device=dev)
    g = torch.randn(6, OUT_F, device=dev)
    dirty = _scratch(t, dev, fill=3.0)  # a pool entry holding another unit's gradient
    target = pb.bind_grad(t, dirty, zero=True)
    y = mod(x)
    y.backward(g)
    assert torch.equal(target, _wgrad(g, x))
    # Bound after the forward, the target is not the one written (why the store binds at forward).
    early = pb.bind_grad(t, _scratch(t, dev), zero=True)
    y = mod(x)
    late = pb.bind_grad(t, _scratch(t, dev), zero=True)
    y.backward(g)
    assert torch.equal(early, _wgrad(g, x)) and torch.equal(late, torch.zeros_like(late))


def test_overwrite_only_for_lazy_single_use_tensors():
    hook_t = _take(nn.Linear(IN_F, OUT_F, bias=False), name="a.weight")
    fwd_t = _take(FakeMegatronFused(IN_F, OUT_F), name="b.weight", grad_capture="forward")
    multi_t = _take(
        FakeTELinear(IN_F, OUT_F, fuse=True), name="c.weight", te_linear=True, single_use=False
    )
    for t in (hook_t, fwd_t, multi_t):
        with pytest.raises(ValueError, match="overwrite_main_grad needs a lazy"):
            pb.set_overwrite(t, True)
        pb.set_overwrite(t, False)
    with pytest.raises(ValueError, match="padded rows included"):
        pb.bind_grad(hook_t, torch.zeros(3), zero=True)
    with pytest.raises(ValueError, match="unknown grad capture"):
        _take(nn.Linear(2, 2), name="d.weight", grad_capture="eager")


def test_multi_use_tensor_accumulates_into_a_zeroed_target(dev):
    mod = FakeTELinear(IN_F, OUT_F, fuse=True, device=dev)
    full = mod.weight.detach().clone()
    t = _take(mod, te_linear=True, single_use=False)
    slot = _slot([t], dev)
    pb.tensor_view(slot, t).copy_(full)
    pb.bind_weight(t, pb.tensor_view(slot, t))
    target = pb.bind_grad(t, _scratch(t, dev, fill=9.0), zero=True)
    x1, x2 = torch.randn(4, IN_F, device=dev), torch.randn(4, IN_F, device=dev)
    g = torch.randn(4, OUT_F, device=dev)
    (mod(x1) + mod(x2)).backward(g)
    ref_dw, _ = _fake_te_reference(full, [x1, x2], [g, g])
    assert torch.allclose(target, ref_dw, rtol=1e-6, atol=1e-6)


def _overwrite_unit(*, fill=NAN, track=True, name="u.weight"):
    """A fake TE linear under overwrite: bound to a slot, its target a dirty scratch entry."""
    mod = FakeTELinear(IN_F, OUT_F, fuse=True)
    t = _take(mod, name=name, te_linear=True)
    if track:
        pb.track_uses(t)
    slot = _slot([t], "cpu")
    pb.tensor_view(slot, t).copy_(torch.randn(OUT_F, IN_F))
    pb.bind_weight(t, pb.tensor_view(slot, t))
    pb.set_overwrite(t, True)
    target = pb.bind_grad(t, _scratch(t, "cpu", fill=fill))
    pb.reset_uses([t])
    return mod, t, target


def test_use_counter_raises_for_a_single_use_tensor_consumed_twice():
    mod, t, _ = _overwrite_unit()
    x1, x2 = torch.randn(2, IN_F), torch.randn(2, IN_F)
    mod(x1).sum().backward()
    assert pb.check_segment_grads([t], "sub 3") == [] and t.uses == 1
    pb.reset_uses([t])
    with torch.no_grad():  # a no-grad call (evaluation, a checkpoint's first pass) is not a use
        mod(x1)
    assert t.uses == 0
    (mod(x1) + mod(x2)).sum().backward()
    with pytest.raises(RuntimeError, match=r"u.weight consumed 2 times in segment sub 3"):
        pb.check_segment_grads([t], "sub 3")
    pb.set_overwrite(t, False)
    pb.bind_grad(t, _scratch(t, "cpu"))  # a zeroed target may be used any number of times
    assert pb.check_segment_grads([t], "sub 3") == []


def test_use_counter_counts_a_recomputed_use_once():
    """A reentrant checkpoint runs the forward under no_grad and recomputes it in the backward."""
    mod, t, target = _overwrite_unit()
    x = torch.randn(3, IN_F, requires_grad=True)
    g = torch.randn(3, OUT_F)
    y = checkpoint(mod, x, use_reentrant=True)
    assert t.uses == 0  # the first pass built no graph
    y.backward(g)
    assert t.uses == 1 and pb.check_segment_grads([t], "coda") == []
    ref_dw, (ref_dx,) = _fake_te_reference(t.bound.clone(), [x], [g])
    assert torch.equal(target, ref_dw) and torch.equal(x.grad, ref_dx)


def test_overwrite_target_no_gradient_reached_is_zeroed():
    """The forward ran, but no gradient reaches the weight: the dirty entry must not be shipped."""
    mod, t, target = _overwrite_unit(fill=7.0)
    other = nn.Linear(IN_F, OUT_F)
    x = torch.randn(3, IN_F)
    (mod(x).detach() + other(x)).sum().backward()  # the stored linear's branch is detached
    assert t.uses == 1 and not t.grad_written and torch.equal(target, torch.full_like(target, 7.0))
    assert pb.check_segment_grads([t], "sub 0") == ["u.weight"]
    assert torch.equal(target, torch.zeros_like(target)) and t.grad_written
    assert pb.check_segment_grads([t], "sub 0") == []  # idempotent
    # Rebinding clears the flag; a written target is left alone.
    target = pb.bind_grad(t, _scratch(t, "cpu", fill=7.0))
    assert not t.grad_written
    pb.reset_uses([t])
    mod(x).sum().backward()
    assert t.grad_written and pb.check_segment_grads([t], "sub 0") == []
    assert torch.equal(target, torch.ones(3, OUT_F).t() @ x)


def test_overwrite_target_used_only_under_no_grad_is_zeroed():
    """DSA's indexer runs its TE linears under ``no_grad`` when the indexer loss is off: no use
    is counted and no gradient arrives, so the dirty entry is zeroed instead of shipped."""
    mod, t, target = _overwrite_unit(fill=7.0)
    with torch.no_grad():
        mod(torch.randn(3, IN_F))
    assert t.uses == 0 and not t.grad_written
    assert pb.check_segment_grads([t], "sub 1") == ["u.weight"]
    assert torch.equal(target, torch.zeros_like(target))
    # A raw read under grad whose backward runs after the segment's check and reduce is still
    # caught: its gradient arrives with no target bound.
    pb.bind_grad(t, _scratch(t, "cpu", fill=7.0))
    pb.reset_uses([t])
    y = torch.einsum("bd,rd->br", torch.randn(3, IN_F), mod.weight.view(OUT_F, IN_F))
    assert t.uses == 0 and pb.check_segment_grads([t], "sub 1") == ["u.weight"]
    pb.unbind_grad(t)
    with pytest.raises(RuntimeError, match="while no gradient target is bound"):
        y.sum().backward()


def test_overwrite_target_consumed_raw_raises():
    """A TE-owned weight read raw (absorbed MLA's ``weight.view`` einsum) bypasses TE."""
    mod, t, target = _overwrite_unit(fill=7.0)
    x = torch.randn(3, IN_F, requires_grad=True)
    y = torch.einsum("bd,rd->br", x, mod.weight.view(OUT_F, IN_F))
    with pytest.raises(RuntimeError, match="hook-path gradient for an overwrite target: u.weight"):
        y.sum().backward()
    assert torch.equal(target, torch.full_like(target, 7.0))  # nothing was added
    mod.weight.grad = None
    # A raw read under no_grad contributes no gradient: the target is zeroed (Resident's value).
    with torch.no_grad():
        torch.einsum("bd,rd->br", x, mod.weight.view(OUT_F, IN_F))
    assert t.uses == 0 and pb.check_segment_grads([t], "sub 0") == ["u.weight"]
    assert torch.equal(target, torch.zeros_like(target))
    # Untracked tensors are only zeroed (the use checks need the counter).
    _, u, u_target = _overwrite_unit(fill=7.0, track=False, name="v.weight")
    assert pb.check_segment_grads([u], "sub 0") == ["v.weight"]
    assert torch.equal(u_target, torch.zeros_like(u_target))


def test_pad_tail_is_zeroed_on_a_reused_pool_entry(dev):
    """R mod z != 0: after the entry held a larger unit's wgrad, the shard's pad rows stay 0."""
    z = 2
    big = FakeTELinear(IN_F, 6, fuse=True, device=dev)
    small = FakeTELinear(IN_F, 5, fuse=True, device=dev)
    tb = _take(big, name="big.weight", z=z, te_linear=True)
    ts = _take(small, name="small.weight", z=z, te_linear=True)
    assert tb.numel_padded == ts.numel_padded == 6 * IN_F and ts.pad_length == 1
    entry = torch.empty(6 * IN_F, dtype=torch.float32, device=dev)
    for t, mod, out_f in ((tb, big, 6), (ts, small, 5)):
        full = mod.weight.detach().clone()
        slot = _slot([t], dev)
        pb.tensor_view(slot, t).copy_(full)
        pb.bind_weight(t, pb.tensor_view(slot, t))
        pb.set_overwrite(t, True)
        pb.bind_grad(t, entry)
        x = torch.randn(3, IN_F, device=dev)
        mod(x).backward(torch.randn(3, out_f, device=dev))
        pb.unbind_grad(t)
        pb.unbind_weight(t)
    assert torch.isfinite(entry).all()
    padded = entry.view(6, IN_F)
    shards = padded.chunk(z)  # what a reduce-scatter of the region hands each shard rank
    assert torch.equal(shards[-1][-1], torch.zeros(IN_F, device=dev))
    assert not torch.equal(shards[-1][-2], torch.zeros(IN_F, device=dev))


def test_grad_hook_adds_clears_dummies_and_requires_a_target(dev):
    mod = nn.Linear(IN_F, OUT_F, bias=False).to(dev)
    t = _take(mod)
    x = torch.randn(3, IN_F, device=dev)
    with pytest.raises(RuntimeError, match="no gradient target is bound"):
        mod(x).sum().backward()
    mod.weight.grad = None
    target = pb.bind_grad(t, _scratch(t, dev), zero=True)
    mod(x).sum().backward()
    first = target.clone()
    assert torch.equal(first, torch.ones(3, OUT_F, device=dev).t() @ x)
    mod(x).sum().backward()
    assert torch.equal(target, first + first) and mod.weight.grad is None
    # A fused consumer flagged grad_added_to_main_grad: the hook drops the dummy, adds nothing.
    mod.weight.grad_added_to_main_grad = True
    mod(x).sum().backward()
    assert torch.equal(target, first + first)
    assert mod.weight.grad_added_to_main_grad is False and mod.weight.grad is None


def _ddp_main_grad(param):
    """DDP's post-accumulate arithmetic (distributed_data_parallel.py hook) on a plain parameter."""
    if param.grad is not None and (
        not param.grad_added_to_main_grad or getattr(param, "zero_out_wgrad", False)
    ):
        param.main_grad.add_(param.grad.data)
    param.grad = None
    return param.main_grad


VOCAB = 9


class TiedEmbeddingHead(nn.Module):
    """A tied ``[V, H]`` weight: an embedding lookup (the MTP's) and a fused output GEMM."""

    def __init__(self, capture, device):
        super().__init__()
        self.capture = capture
        self.weight = nn.Parameter(torch.randn(VOCAB, IN_F, device=device) * 0.1)

    def forward(self, ids, hidden):
        emb = F.embedding(ids, self.weight)
        if self.capture == "forward":
            return FakeMegatronFusedFn.apply(hidden + emb, self.weight)
        return FakeTELinearFn.apply(hidden + emb, self.weight, True)


@pytest.mark.parametrize("zero_out_wgrad", [True, False])
@pytest.mark.parametrize("capture", ["forward", "lazy"])
def test_tied_weight_fused_and_lookup_in_one_backward_matches_ddp(dev, capture, zero_out_wgrad):
    """The head segment of a tied embedding at PP = 1 (language_module sets ``zero_out_wgrad``)."""
    torch.manual_seed(13)
    mod = TiedEmbeddingHead(capture, dev)
    ref = TiedEmbeddingHead(capture, dev)
    full = mod.weight.detach().clone()
    with torch.no_grad():
        ref.weight.copy_(full)
    ids = torch.randint(0, VOCAB, (2, 5), device=dev)
    hidden = torch.randn(2, 5, IN_F, device=dev)
    g = torch.randn(2, 5, VOCAB, device=dev)
    ref.weight.main_grad = torch.zeros(VOCAB, IN_F, device=dev)
    ref.weight.grad_added_to_main_grad = False
    for module in (mod, ref):
        if zero_out_wgrad:
            module.weight.zero_out_wgrad = True
    ref(ids, hidden).backward(g)
    expected = _ddp_main_grad(ref.weight)
    t = _take(mod, grad_capture=capture, single_use=False)
    slot = _slot([t], dev)
    pb.tensor_view(slot, t).copy_(full)
    pb.bind_weight(t, pb.tensor_view(slot, t))
    target = pb.bind_grad(t, _scratch(t, dev))  # multi-use: zeroed
    mod(ids, hidden).backward(g)
    assert torch.equal(target, expected) and torch.isfinite(target).all()
    assert mod.weight.grad is None and mod.weight.grad_added_to_main_grad is False
    # With zero_out_wgrad the lookup's gradient is kept; without it DDP (and the store) drop it.
    output_gemm_only = _wgrad(g, hidden + F.embedding(ids, full))
    assert torch.equal(target, output_gemm_only) != zero_out_wgrad


def test_classify_and_disable_local_wgrad_fusion():
    lin = nn.Linear(4, 4)
    assert pb.classify_grad_capture(lin, "weight") == "hook"
    assert not pb.is_te_linear(lin)
    cpl = ColumnParallelLinear(
        8, 6, config=_cpl_config(), init_method=nn.init.normal_, bias=True, gather_output=False
    )
    assert pb.classify_grad_capture(cpl, "weight") == "hook"
    cpl.gradient_accumulation_fusion = True
    assert pb.classify_grad_capture(cpl, "weight") == "forward"
    assert pb.classify_grad_capture(cpl, "bias") == "hook"
    assert pb.disable_local_wgrad_fusion(cpl) is True
    assert pb.classify_grad_capture(cpl, "weight") == "hook"
    assert pb.disable_local_wgrad_fusion(cpl) is False


def test_release_store_tensor_restores_registration():
    mod = nn.Linear(IN_F, OUT_F, bias=False)
    full = mod.weight.detach().clone()
    t = _take(mod)
    pb.track_uses(t)
    pb.unbind_weight(t)
    param = pb.release_store_tensor(t, data=full.clone())
    assert param is mod.weight and "weight" in dict(mod.named_parameters())
    assert torch.equal(mod.weight, full) and not hasattr(param, "loop_store_tensor")
    x = torch.randn(2, IN_F)
    mod(x).sum().backward()  # no hook left: an ordinary .grad
    assert mod.weight.grad is not None and not t.handles
    # Unbound and no data: refused (re-registering the NaN placeholder would reach DDP).
    other = nn.Linear(IN_F, OUT_F, bias=False)
    u = _take(other, name="v.weight")
    pb.unbind_weight(u)
    with pytest.raises(ValueError, match="needs the full weight"):
        pb.release_store_tensor(u)
    with pytest.raises(ValueError, match="release data"):
        pb.release_store_tensor(u, data=torch.zeros(2, 2))
    assert "weight" not in dict(other.named_parameters())  # nothing was undone
    # Bound: the bound region is the default.
    slot = _slot([u], "cpu")
    pb.tensor_view(slot, u).copy_(full)
    pb.bind_weight(u, pb.tensor_view(slot, u))
    assert pb.release_store_tensor(u).data_ptr() == slot.data_ptr()
    assert torch.equal(other.weight, full)


# ---------------------------------------------------------------------------------------------
# The DistributedWeight binding against a fake TE 2.20 consumer
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def protocol_available(monkeypatch):
    monkeypatch.setattr(pb, "distributed_weight_unavailable_reason", lambda: None)


def _dw_pair(dev, *, fuse, z=3, dtype=torch.float32, rows=OUT_F):
    """Two identical fake TE linears: one under the data binding, one under the DW binding."""
    torch.manual_seed(3)
    a = FakeTELinear(IN_F, rows, fuse=fuse, dtype=dtype, device=dev)
    b = FakeTELinear(IN_F, rows, fuse=fuse, dtype=dtype, device=dev)
    with torch.no_grad():
        b.weight.copy_(a.weight)
    a.weight.qkv_layout = "sbhd"  # an init-time attribute the stand-in must carry
    b.weight.qkv_layout = "sbhd"
    b.weight.allreduce = True
    full = a.weight.detach().clone()
    rank = z - 1  # the last shard holds the pad rows
    shard = nn.Parameter(pb.shard_slice(full, z, rank))
    ta = _take(a, name="a.weight", z=z, te_linear=True)
    tb = _take(b, name="b.weight", z=z, te_linear=True, binding="distributed_weight", shard=shard)
    return a, b, ta, tb, full


def test_distributed_weight_refused_without_the_protocol(monkeypatch):
    monkeypatch.setattr(pb, "te_version", lambda: "2.16.0+4220403e")
    mod = FakeTELinear(IN_F, OUT_F, fuse=True)
    with pytest.raises(ValueError, match=r"TE 2\.16\.0\+4220403e has no DistributedWeight"):
        _take(mod, z=2, te_linear=True, binding="distributed_weight")
    assert "weight" in dict(mod.named_parameters())  # nothing was hidden
    with pytest.raises(ValueError, match="unknown store binding"):
        _take(mod, binding="dtensor")


def test_distributed_weight_needs_a_te_owner_and_a_matching_shard(protocol_available):
    with pytest.raises(ValueError, match="only TE Linear-family weights"):
        _take(nn.Linear(IN_F, OUT_F), z=2, binding="distributed_weight")
    with pytest.raises(ValueError, match=r"canonical shard must be \(5, 12\), got \(4, 12\)"):
        _take(
            FakeTELinear(IN_F, OUT_F, fuse=True),
            z=2,
            te_linear=True,
            binding="distributed_weight",
            shard=nn.Parameter(torch.zeros(4, IN_F)),
        )
    # The stand-in holds no data, so the shard is optional.
    t = _take(
        FakeTELinear(IN_F, OUT_F, fuse=True), z=2, te_linear=True, binding="distributed_weight"
    )
    assert pb.is_placeholder(t.weight) and tuple(t.weight.shape) == (5, IN_F)


def test_distributed_weight_stand_in(dev, protocol_available):
    a, b, ta, tb, full = _dw_pair(dev, fuse=True, z=3)
    w = b.weight
    assert isinstance(w, pb.RSBFDistributedWeight) and w is tb.weight and w.is_leaf
    assert pb.RSBFDistributedWeight.is_distributed_weight
    assert tuple(w.shape) == tb.shard_shape == (4, IN_F) and w.requires_grad
    assert w._unsharded_shape == torch.Size((OUT_F, IN_F))
    assert w.qkv_layout == "sbhd" and w.allreduce is True and w.loop_store_tensor is tb
    # Backed by a placeholder, not the shard: the DDP wrap moving the shard is not coupled to it.
    assert pb.is_placeholder(w) and w.data_ptr() != tb.shard.data_ptr()
    with pytest.raises(RuntimeError, match="single memory location"):
        with torch.no_grad():
            w.add_(1.0)
    assert "weight" not in dict(b.named_parameters())
    assert dict(pb.named_logical_parameters(b))["weight"] is tb.compute
    assert pb.is_placeholder(tb.compute.data) and tb.compute.shape == full.shape
    det = w.detach()
    assert isinstance(det, pb.RSBFDistributedWeight) and det.loop_store_tensor is tb
    assert type(w * 2) is torch.Tensor  # every other op decays to a plain tensor
    with pytest.raises(RuntimeError, match="used unbound"):
        w.materialize_group_for_forward()
    with pytest.raises(RuntimeError, match="no gradient target"):
        w.grad_buffer()
    del a, ta


@pytest.mark.parametrize(
    "mode,fuse", [("overwrite", True), ("accumulate", True), ("accumulate", False)]
)
def test_distributed_weight_matches_the_data_binding(dev, protocol_available, mode, fuse):
    """LW / RSG (overwrite targets; fused wgrad only) and Resident (accumulator, microbatches)."""
    a, b, ta, tb, full = _dw_pair(dev, fuse=fuse)
    assert ta.grad_capture == tb.grad_capture == ("lazy" if fuse else "hook")
    assert ta.pad_length == 2  # R = 10 rows on z = 3
    targets = []
    for mod, t in ((a, ta), (b, tb)):
        slot = _slot([t], dev)
        view = pb.tensor_view(slot, t)
        view.copy_(full)
        pb.bind_weight(t, view)
        region = _scratch(t, dev, fill=7.0)
        pb.set_overwrite(t, mode == "overwrite")
        steps = 1 if mode == "overwrite" else 3
        torch.manual_seed(5)
        dxs = []
        for step in range(steps):
            x = torch.randn(3, 2, IN_F, device=dev, requires_grad=True)
            g = torch.randn(3, 2, OUT_F, device=dev)
            y = mod(x)
            if mode == "overwrite":
                target = pb.bind_grad(t, region)
            else:
                target = pb.bind_grad(t, region, zero=(step == 0))
            y.backward(g)
            dxs.append(x.grad)
            assert t.weight.grad is None and t.weight.grad_added_to_main_grad is False
        targets.append((region.clone(), dxs, target))
    (ra, dxa, _), (rb, dxb, target_b) = targets
    for da, db in zip(dxa, dxb):
        assert torch.equal(da, db)
    # The fake wgrad GEMM never accumulates in its epilogue, so even the accumulator is bitwise
    # here; TE's fused epilogue makes Resident an rtol comparison on GPU (design 5.4).
    assert torch.equal(ra, rb)
    assert torch.equal(rb[ta.numel :], torch.zeros_like(rb[ta.numel :]))  # pad tail
    assert target_b.shape == full.shape


def test_distributed_weight_grad_buffer_and_finalize_contract(dev, protocol_available):
    """Through TE's dispatcher: a bare wgrad; overwrite targets are written in place."""
    _, b, _, tb, full = _dw_pair(dev, fuse=True)
    w = b.weight
    slot = _slot([tb], dev)
    pb.tensor_view(slot, tb).copy_(full)
    pb.bind_weight(tb, pb.tensor_view(slot, tb))
    wgrad = torch.randn(full.shape, device=dev)
    # Overwrite: TE's GEMM writes the target itself; finalize has nothing to copy.
    pb.set_overwrite(tb, True)
    target = pb.bind_grad(tb, _scratch(tb, dev, fill=7.0))
    assert w.grad_buffer() is target
    w.grad_buffer().copy_(wgrad)
    (dummy,) = _te_finalize_weight_grads(w, [target])
    assert torch.equal(target, wgrad) and tb.grad_written and w.grad_added_to_main_grad
    assert tuple(dummy.shape) == tb.shard_shape
    # A separate wgrad tensor (non-fused DW) is copied into an overwrite target.
    target = pb.bind_grad(tb, _scratch(tb, dev, fill=7.0))
    assert not tb.grad_written
    _te_finalize_weight_grads(w, [wgrad * 2])
    assert torch.equal(target, wgrad * 2)
    # Accumulated target: a fresh buffer, added; the target itself is refused.
    pb.set_overwrite(tb, False)
    target = pb.bind_grad(tb, _scratch(tb, dev), zero=True)
    buf = w.grad_buffer()
    assert buf.data_ptr() != target.data_ptr() and buf.shape == full.shape
    buf.copy_(wgrad)
    _te_finalize_weight_grads(w, [buf])
    _te_finalize_weight_grads(w, [buf])
    assert torch.equal(target, wgrad + wgrad)
    with pytest.raises(RuntimeError, match="written into its accumulated target"):
        w.finalize_group_grads(target)
    # zero_out_wgrad: the dummy is zero (a hook would add it), as TE's get_dummy_wgrad(zero=True).
    w.zero_out_wgrad = True
    (dummy,) = _te_finalize_weight_grads(w, [buf])
    assert torch.equal(dummy, torch.zeros_like(dummy))
    w.grad_added_to_main_grad = False


def test_te_version_survives_a_failing_import(monkeypatch):
    """A TE import raising RuntimeError (CUDA / library mismatch) is "TE unavailable"."""
    import builtins  # pylint: disable=import-outside-toplevel

    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "transformer_engine" or name.startswith("transformer_engine."):
            raise RuntimeError("CUDA driver version is insufficient")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    assert pb.te_version() is None
    assert not pb.te_distributed_weight_available()
    assert "cannot be imported (RuntimeError: CUDA driver version is insufficient)" in (
        pb.distributed_weight_unavailable_reason()
    )
    assert not pb.is_te_linear(nn.Linear(2, 2))
    monkeypatch.undo()
    assert "CUDA driver" not in (pb.distributed_weight_unavailable_reason() or "")


def test_distributed_weight_backward_must_reuse_the_forward_region(dev, protocol_available):
    _, b, _, tb, full = _dw_pair(dev, fuse=True)
    slot_a, slot_b = _slot([tb], dev), _slot([tb], dev)
    pb.tensor_view(slot_a, tb).copy_(full)
    pb.tensor_view(slot_b, tb).copy_(full)
    pb.bind_weight(tb, pb.tensor_view(slot_a, tb))
    y = b(torch.randn(2, IN_F, device=dev))
    pb.bind_weight(tb, pb.tensor_view(slot_b, tb))
    pb.set_overwrite(tb, True)
    pb.bind_grad(tb, _scratch(tb, dev))
    with pytest.raises(RuntimeError, match="re-bound to a different region"):
        y.sum().backward()


def test_distributed_weight_bf16_shard_and_release(dev, protocol_available):
    a, b, ta, tb, full = _dw_pair(dev, fuse=True, dtype=torch.bfloat16)
    for mod, t in ((a, ta), (b, tb)):
        slot = _slot([t], dev, dtype=torch.bfloat16)
        pb.tensor_view(slot, t).copy_(full)
        pb.bind_weight(t, pb.tensor_view(slot, t))
        pb.set_overwrite(t, True)
        pb.bind_grad(t, _scratch(t, dev))
        torch.manual_seed(11)
        x = torch.randn(4, IN_F, device=dev, dtype=torch.bfloat16)
        mod(x).backward(torch.randn(4, OUT_F, device=dev, dtype=torch.bfloat16))
    assert torch.equal(ta.grad_target, tb.grad_target)
    param = pb.release_store_tensor(tb, data=full.clone())
    assert b.weight is param and type(param) is nn.Parameter and param.shape == full.shape


# ---------------------------------------------------------------------------------------------
# TE linears and Megatron's fused local linear on GPU
# ---------------------------------------------------------------------------------------------

TE_VERSION = pb.te_version()
needs_te_gpu = pytest.mark.skipif(
    not (CUDA and TE_VERSION is not None), reason="needs CUDA and Transformer Engine"
)
H, FFN, SEQ, MB = 256, 384, 64, 2
TE_KINDS = ("te_linear", "te_layernorm_column", "te_row")


def _te_config(**overrides):
    kwargs = dict(
        num_layers=1,
        hidden_size=H,
        ffn_hidden_size=FFN,
        num_attention_heads=4,
        use_cpu_initialization=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        disable_parameter_transpose_cache=True,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def _te_module(kind, config):
    # pylint: disable=import-outside-toplevel
    from megatron.core.extensions.transformer_engine import (
        TELayerNormColumnParallelLinear,
        TELinear,
        TERowParallelLinear,
    )
    from megatron.core.utils import init_method_normal  # pylint: disable=import-outside-toplevel

    init = init_method_normal(0.02)
    if kind == "te_linear":
        mod = TELinear(
            H,
            FFN,
            parallel_mode="duplicated",
            config=config,
            init_method=init,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
        )
        in_f = H
    elif kind == "te_layernorm_column":
        mod = TELayerNormColumnParallelLinear(
            H,
            FFN,
            config=config,
            init_method=init,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
        )
        in_f = H
    else:
        mod = TERowParallelLinear(
            FFN,
            H,
            config=config,
            init_method=init,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
        )
        in_f = FFN
    return mod.cuda(), in_f


def _te_pair(kind, *, fuse, **config_overrides):
    """(stored module, reference module, store tensor, slot, full weight, in_features)."""
    config = _te_config(gradient_accumulation_fusion=fuse, **config_overrides)
    torch.manual_seed(7)
    mod, in_f = _te_module(kind, config)
    ref, _ = _te_module(kind, config)
    ref.load_state_dict(mod.state_dict())
    if fuse:
        ref.weight.main_grad = torch.zeros(ref.weight.shape, dtype=torch.float32, device="cuda")
        ref.weight.grad_added_to_main_grad = False
    full = mod.weight.detach().clone()
    t = pb.make_store_tensor(f"{kind}.weight", mod, "weight", shard_size=1)
    assert t.te_linear and t.grad_capture == ("lazy" if fuse else "hook")
    slot = _slot([t], "cuda", dtype=torch.bfloat16)
    pb.unbind_weight(t)
    return mod, ref, t, slot, full, in_f


def _te_io(in_f, out_f, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(SEQ, MB, in_f, device="cuda", dtype=torch.bfloat16, generator=gen)
    g = torch.randn(SEQ, MB, out_f, device="cuda", dtype=torch.bfloat16, generator=gen)
    return x, g


def _run(mod, x, g, before_backward=None):
    x = x.detach().clone().requires_grad_()
    out = mod(x)[0]
    if before_backward is not None:
        before_backward()
    out.backward(g)
    return out.detach(), x.grad


@needs_te_gpu
@pytest.mark.parametrize("fuse", [True, False])
@pytest.mark.parametrize("kind", TE_KINDS)
def test_te_linear_data_binding_matches_reference(kind, fuse):
    mod, ref, t, slot, full, in_f = _te_pair(kind, fuse=fuse)
    out_f = full.shape[0]
    view = pb.tensor_view(slot, t)
    view.copy_(full)
    pb.bind_weight(t, view)
    x, g = _te_io(in_f, out_f, seed=1)
    region = torch.full((t.numel_padded,), NAN, dtype=torch.float32, device="cuda")

    def regather_and_bind_grad():
        # Released, poisoned, re-gathered into the same slot; the lazy target bound only now.
        pb.unbind_weight(t)
        slot.fill_(NAN)
        view.copy_(full)
        pb.bind_weight(t, view)
        pb.bind_grad(t, region, zero=True)

    out, dx = _run(mod, x, g, before_backward=regather_and_bind_grad)
    ref_out, ref_dx = _run(ref, x, g)
    ref_dw = ref.weight.main_grad if fuse else ref.weight.grad.float()
    assert torch.equal(out, ref_out) and torch.equal(dx, ref_dx)
    assert torch.equal(t.grad_target, ref_dw)
    assert mod.weight.grad is None


@needs_te_gpu
@pytest.mark.parametrize("kind", TE_KINDS)
def test_te_overwrite_main_grad_and_resident_accumulation(kind):
    mod, ref, t, slot, full, in_f = _te_pair(kind, fuse=True)
    out_f = full.shape[0]
    pb.tensor_view(slot, t).copy_(full)
    pb.bind_weight(t, pb.tensor_view(slot, t))
    # Overwrite into a dirty entry == the reference overwriting its own dirty main_grad.
    x, g = _te_io(in_f, out_f, seed=2)
    pb.set_overwrite(t, True)
    pb.bind_grad(t, torch.full((t.numel_padded,), 3.0, device="cuda"))
    _run(mod, x, g)
    ref.weight.main_grad.fill_(3.0)
    ref.weight.overwrite_main_grad = True
    _run(ref, x, g)
    assert torch.equal(t.grad_target, ref.weight.main_grad)
    # Resident: one zeroed accumulator over three microbatches == DDP-style main_grad.
    ref.weight.overwrite_main_grad = False
    pb.set_overwrite(t, False)
    ref.weight.main_grad.zero_()
    acc = torch.full((t.numel_padded,), NAN, dtype=torch.float32, device="cuda")
    for step in range(3):
        pb.bind_grad(t, acc, zero=(step == 0))
        x, g = _te_io(in_f, out_f, seed=10 + step)
        _run(mod, x, g)
        _run(ref, x, g)
    assert torch.equal(pb.tensor_view(acc, t), ref.weight.main_grad)


@needs_te_gpu
def test_te_other_slot_rebind_is_recorded():
    """S6.0 (b): whether TE's bf16 backward follows a moved binding (recorded, not asserted)."""
    mod, ref, t, slot, full, in_f = _te_pair("te_linear", fuse=True)
    other = torch.empty_like(slot)
    pb.tensor_view(slot, t).copy_(full)
    pb.bind_weight(t, pb.tensor_view(slot, t))
    x, g = _te_io(in_f, full.shape[0], seed=3)

    def move():
        pb.unbind_weight(t)
        slot.fill_(NAN)
        pb.tensor_view(other, t).copy_(full)
        pb.bind_weight(t, pb.tensor_view(other, t))
        pb.bind_grad(t, torch.zeros(t.numel_padded, device="cuda"), zero=True)

    _, dx = _run(mod, x, g, before_backward=move)
    _, ref_dx = _run(ref, x, g)
    follows = bool(torch.equal(dx, ref_dx))
    print(f"TE {TE_VERSION} TELinear bf16 backward follows a moved .data binding: {follows}")
    assert follows or torch.isnan(dx).any()


@needs_te_gpu
def test_megatron_fused_linear_captures_main_grad_at_forward():
    from megatron.core.tensor_parallel import layers  # pylint: disable=import-outside-toplevel

    if not layers._grad_accum_fusion_available:  # pylint: disable=protected-access
        pytest.skip("fused_weight_gradient_mlp_cuda is not installed")
    config = _te_config(gradient_accumulation_fusion=True, use_cpu_initialization=True)

    def build():
        torch.manual_seed(9)
        return ColumnParallelLinear(
            H, FFN, config=config, init_method=nn.init.normal_, bias=False, gather_output=False
        ).cuda()

    mod, ref = build(), build()
    ref.weight.main_grad = torch.zeros(ref.weight.shape, dtype=torch.float32, device="cuda")
    ref.weight.grad_added_to_main_grad = False
    full = mod.weight.detach().clone()
    t = pb.make_store_tensor("cpl.weight", mod, "weight", shard_size=1)
    assert t.grad_capture == "forward"
    slot = _slot([t], "cuda", dtype=torch.bfloat16)
    pb.tensor_view(slot, t).copy_(full)
    pb.bind_weight(t, pb.tensor_view(slot, t))
    x, g = _te_io(H, FFN, seed=4)
    early = pb.bind_grad(t, torch.full((t.numel_padded,), 3.0, device="cuda"), zero=True)
    late_region = torch.zeros(t.numel_padded, device="cuda")
    _run(mod, x, g, before_backward=lambda: pb.bind_grad(t, late_region, zero=True))
    _run(ref, x, g)
    assert torch.equal(early, ref.weight.main_grad)  # the forward-time target got the wgrad
    assert torch.equal(late_region, torch.zeros_like(late_region))
    assert mod.weight.grad is None


def _quant_recipe(name):
    from transformer_engine.common import recipe  # pylint: disable=import-outside-toplevel

    env = pb.probe_store_environment()
    if name == "fp8_block":
        if not env.fp8_block_scaling:
            pytest.skip(f"FP8 block scaling unavailable: {env.fp8_block_scaling_reason}")
        return recipe.Float8BlockScaling()
    if not env.mxfp8:
        pytest.skip(f"MXFP8 unavailable: {env.mxfp8_reason}")
    return recipe.MXFP8BlockScaling()


def _quant_autocast(quant_recipe):
    import transformer_engine.pytorch as te_pytorch  # pylint: disable=import-outside-toplevel

    if hasattr(te_pytorch, "autocast"):
        return te_pytorch.autocast(enabled=True, recipe=quant_recipe)
    return te_pytorch.fp8_autocast(enabled=True, fp8_recipe=quant_recipe)


@needs_te_gpu
@pytest.mark.parametrize("recipe_name", ["fp8_block", "mxfp8"])
@pytest.mark.parametrize("kind", TE_KINDS)
def test_te_quantized_cast_after_gather(kind, recipe_name):
    """The bound bf16 view is quantized on every call: a re-gather with new weights is used.

    ``disable_parameter_transpose_cache=True`` makes Megatron pass ``is_first_microbatch=None``,
    so TE keeps no quantized workspace across calls (E5). Each step's reference is a module built
    fresh from that step's weights, so a quantized weight cached across calls by the stored module
    (or shared through the config) cannot make both sides agree; the slot is unbound and NaN-filled
    before each re-gather, and the same input must give a different output on the new weights.
    """
    quant_recipe = _quant_recipe(recipe_name)
    mod, _, t, slot, full, in_f = _te_pair(kind, fuse=True)
    out_f = full.shape[0]
    view = pb.tensor_view(slot, t)
    region = torch.zeros(t.numel_padded, dtype=torch.float32, device="cuda")
    x, g = _te_io(in_f, out_f, seed=20)
    outs = []
    for step, weights in enumerate((full, full * 1.5 + 0.01)):
        slot.fill_(NAN)  # released and poisoned ...
        view.copy_(weights)  # ... then the gather of this step's weights into the slot
        pb.bind_weight(t, view)
        pb.set_overwrite(t, True)
        pb.bind_grad(t, region)
        with _quant_autocast(quant_recipe):
            out, dx = _run(mod, x, g)
        ref = _fresh_te_reference(kind, mod, weights, seed=40 + step)
        with _quant_autocast(quant_recipe):
            ref_out, ref_dx = _run(ref, x, g)
        assert torch.equal(out, ref_out) and torch.equal(dx, ref_dx)
        assert torch.equal(t.grad_target, ref.weight.main_grad)
        assert torch.isfinite(out).all()
        outs.append(out)
        pb.unbind_weight(t)
    assert not torch.equal(outs[0], outs[1])


def _fresh_te_reference(kind, stored, weights, seed):
    """A new TE module (its own config and quantizers) holding ``weights`` and ``stored``'s rest."""
    torch.manual_seed(seed)
    ref, _ = _te_module(kind, _te_config(gradient_accumulation_fusion=True))
    missing, unexpected = ref.load_state_dict(stored.state_dict(), strict=False)
    assert "weight" in missing and not unexpected  # the stored weight is hidden
    with torch.no_grad():
        ref.weight.copy_(weights)
    ref.weight.main_grad = torch.full(weights.shape, 3.0, dtype=torch.float32, device="cuda")
    ref.weight.grad_added_to_main_grad = False
    ref.weight.overwrite_main_grad = True
    return ref


@needs_te_gpu
@pytest.mark.skipif(
    not pb.te_distributed_weight_available(),
    reason=f"DistributedWeight binding: {pb.distributed_weight_unavailable_reason()}",
)
@pytest.mark.parametrize("kind", TE_KINDS)
def test_te_distributed_weight_matches_data_binding(kind):
    """TE >= 2.19 only (skipped on the TE 2.16 CI image with the E2 reason)."""
    mod, ref, t, slot, full, in_f = _te_pair(kind, fuse=True)
    out_f = full.shape[0]
    torch.manual_seed(7)
    dw_mod, _ = _te_module(kind, _te_config(gradient_accumulation_fusion=True))
    dw_mod.load_state_dict(ref.state_dict())
    shard = nn.Parameter(pb.shard_slice(full, 2, 1))
    td = pb.make_store_tensor(
        f"{kind}.dw", dw_mod, "weight", shard_size=2, binding="distributed_weight", shard=shard
    )
    x, g = _te_io(in_f, out_f, seed=30)
    results = []
    for module, tensor in ((mod, t), (dw_mod, td)):
        buf = _slot([tensor], "cuda", dtype=torch.bfloat16)
        pb.tensor_view(buf, tensor).copy_(full)
        pb.bind_weight(tensor, pb.tensor_view(buf, tensor))
        region = torch.full((tensor.numel_padded,), 3.0, dtype=torch.float32, device="cuda")
        pb.set_overwrite(tensor, True)
        pb.bind_grad(tensor, region)
        out, dx = _run(module, x, g)
        results.append((out, dx, tensor.grad_target.clone()))
        assert tensor.weight.grad is None
    (out_a, dx_a, dw_a), (out_b, dx_b, dw_b) = results
    assert torch.equal(out_a, out_b) and torch.equal(dx_a, dx_b) and torch.equal(dw_a, dw_b)
    assert math.prod(td.weight.shape) * 2 == td.numel_padded
