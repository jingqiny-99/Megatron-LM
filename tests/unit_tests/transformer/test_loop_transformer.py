# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Loop decoder parity against independent eager PyTorch attention/MLP math.

Architectural sources (not imported by this reference):
Huginn: seal-rg/recurrent-pretraining@1ea7220ec7eb42d13e89db0663df254d0bcdc28e
MoR: raymin0223/mixture_of_recursions@82c97f29cd140efa01f5cc4fe0de73a4c3b6ffd6
The documented proxy uses zero initial state and standard MCore RMSNorm layers.
"""

import copy
from contextlib import nullcontext
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from megatron.core import dist_checkpointing
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.tensor_observation import capture_tensor_observations
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.loop.reporting import loop_metric_report
from megatron.core.transformer.loop_transformer_block import LoopTransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.launch_on_gb200


@pytest.fixture(autouse=True)
def parallel_environment(monkeypatch):
    # FP32 parity must compare full-precision GEMMs, including TE's cuBLAS
    # calls, rather than mix TF32 and IEEE paths on Blackwell.
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    Utils.initialize_model_parallel(1, 1)
    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()
    torch.backends.cuda.matmul.allow_tf32 = old_tf32


def make_config(
    architecture="huginn",
    depth=2,
    bptt=0,
    recompute=False,
    execution="compact",
    dtype=torch.float32,
    untie=False,
    production_shape=False,
    **overrides,
):
    return TransformerConfig(
        num_layers=2 + 2 * depth if untie else 4,
        hidden_size=2048 if production_shape else 128,
        num_attention_heads=16 if production_shape else 2,
        num_query_groups=4 if production_shape else 1,
        kv_channels=128 if production_shape else 64,
        ffn_hidden_size=5632 if production_shape else 256,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=dtype == torch.bfloat16,
        params_dtype=dtype,
        use_cpu_initialization=True,
        gradient_accumulation_fusion=False,
        apply_rope_fusion=False,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        attention_backend=AttnBackend.auto if dtype == torch.bfloat16 else AttnBackend.unfused,
        loop_architecture=architecture,
        loop_prelude_layers=1,
        loop_core_layers=2,
        loop_iterations=depth,
        loop_untie_core=untie,
        loop_bptt_iterations=bptt,
        loop_recompute=recompute,
        loop_token_execution=execution,
        **overrides,
    )


def make_block(*args, **kwargs):
    return LoopTransformerBlock(
        config=make_config(*args, **kwargs),
        spec=get_gpt_layer_with_transformer_engine_spec(),
        pre_process=True,
        post_process=True,
    ).cuda()


class NativeDecoder(nn.Module):
    """No MCore/TE attention, norms, MLPs, routing helpers, or checkpoint calls."""

    def __init__(self, block):
        super().__init__()
        self.c = copy.copy(block.config)
        # Explicit mapping: TE's fused norm weights live on QKV/FC1 modules.
        self.weights = nn.ParameterDict(
            {
                name.replace(".", "__"): nn.Parameter(value.detach().clone())
                for name, value in block.named_parameters()
            }
        )

    def w(self, name):
        return self.weights[name.replace(".", "__")]

    def norm(self, x, name):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() / (variance + self.c.layernorm_epsilon).sqrt()).to(x.dtype) * self.w(name)

    def layer(self, x, number, positions):
        prefix = f"layers.{number}."
        y = self.norm(x, prefix + "self_attention.linear_qkv.layer_norm_weight")
        qkv = F.linear(y, self.w(prefix + "self_attention.linear_qkv.weight"))
        c = self.c
        groups, dim = c.num_query_groups, c.kv_channels
        heads_per_group = c.num_attention_heads // groups
        qkv = qkv.view(x.shape[0], x.shape[1], groups, heads_per_group + 2, dim)
        q = qkv[..., :heads_per_group, :].flatten(2, 3)
        k = qkv[..., heads_per_group, :].repeat_interleave(heads_per_group, dim=2)
        v = qkv[..., heads_per_group + 1, :].repeat_interleave(heads_per_group, dim=2)
        frequencies = 10000.0 ** (-torch.arange(0, dim, 2, device=x.device).float() / dim)
        phase = positions[:, None].float() * frequencies[None]
        phase = torch.cat((phase, phase), dim=-1)[:, None, None]

        def rotate(t):
            a, b = t.chunk(2, -1)
            return (t.float() * phase.cos() + torch.cat((-b, a), -1).float() * phase.sin()).to(
                t.dtype
            )

        q, k = rotate(q), rotate(k)
        scores = torch.einsum("sbhc,tbhc->bhst", q.float(), k.float()) / dim**0.5
        if (
            c.loop_architecture == "mor"
            and c.loop_token_execution == "masked"
            and c.loop_prelude_layers <= number < c.loop_prelude_layers + c.loop_core_layers
        ):
            # Arbitrary-mask attention uses an unfused score GEMM whose output
            # is materialized in the activation dtype before FP32 softmax.
            # Compact/fused attention keeps its score accumulator in FP32.
            q_bh = q.permute(1, 2, 0, 3).reshape(-1, x.shape[0], dim)
            k_bh = k.permute(1, 2, 3, 0).reshape(-1, dim, x.shape[0])
            scores = (
                torch.baddbmm(
                    torch.empty(
                        q_bh.shape[0], x.shape[0], x.shape[0], device=x.device, dtype=x.dtype
                    ),
                    q_bh,
                    k_bh,
                    beta=0,
                    alpha=dim**-0.5,
                )
                .view(x.shape[1], c.num_attention_heads, x.shape[0], x.shape[0])
                .float()
            )
        causal = positions[None, :] > positions[:, None]
        scores = scores.masked_fill(causal, -float("inf"))
        probability = scores.softmax(-1).to(x.dtype)
        attended = torch.einsum("bhst,tbhc->sbhc", probability, v).flatten(2)
        x = x + F.linear(attended, self.w(prefix + "self_attention.linear_proj.weight"))
        if (prefix + "mlp.router.weight").replace(".", "__") in self.weights:
            y = self.norm(x, prefix + "pre_mlp_layernorm.weight")
            logits = F.linear(y, self.w(prefix + "mlp.router.weight"))
            values, indices = logits.topk(c.moe_router_topk, dim=-1)
            probabilities = values.float().softmax(-1).to(y.dtype)
            scores = torch.zeros_like(logits).scatter(-1, indices, probabilities)
            update = torch.zeros_like(x)
            for expert in range(c.num_moe_experts):
                expert_prefix = prefix + "mlp.experts."
                gate, value = F.linear(
                    y, self.w(expert_prefix + f"linear_fc1.weight{expert}")
                ).chunk(2, -1)
                weighted = F.silu(gate) * value * scores[..., expert, None]
                update = update + F.linear(
                    weighted, self.w(expert_prefix + f"linear_fc2.weight{expert}")
                )
            return x + update
        y = self.norm(x, prefix + "mlp.linear_fc1.layer_norm_weight")
        gate, value = F.linear(y, self.w(prefix + "mlp.linear_fc1.weight")).chunk(2, -1)
        return x + F.linear(F.silu(gate) * value, self.w(prefix + "mlp.linear_fc2.weight"))

    def forward(self, x):
        c = self.c
        positions = torch.arange(x.shape[0], device=x.device)
        for index in range(c.loop_prelude_layers):
            x = self.layer(x, index, positions)
        embedding = x
        start = c.loop_prelude_layers
        stop = start + c.loop_core_layers
        self.auxiliary = x.new_zeros((), dtype=torch.float32)
        if c.loop_architecture == "huginn":
            x = torch.zeros_like(embedding)
            first_grad = max(0, c.loop_iterations - (c.loop_bptt_iterations or c.loop_iterations))
            for iteration in range(c.loop_iterations):
                with torch.no_grad() if iteration < first_grad else nullcontext():
                    x = F.linear(torch.cat((x, embedding), -1), self.w("loop_adapter.weight"))
                    offset = iteration * c.loop_core_layers if c.loop_untie_core else 0
                    for index in range(start + offset, stop + offset):
                        x = self.layer(x, index, positions)
                    x = self.norm(x, "loop_norm.weight")
        else:
            logits = F.linear(
                F.silu(F.linear(x, self.w("depth_router.0.weight"))),
                self.w("depth_router.2.weight"),
            ).float()
            probabilities = logits.softmax(-1)
            choice = probabilities.argmax(-1)
            self.chosen_depth = choice.detach() + 1
            selected = probabilities.gather(-1, choice[..., None]).squeeze(-1)
            frequencies = torch.stack(
                [(choice == i).float().mean() for i in range(c.loop_iterations)]
            )
            balance = (
                c.loop_router_balance_coeff
                * c.loop_iterations
                * (frequencies * probabilities.mean((0, 1))).sum()
            )
            self.auxiliary = (
                balance + c.loop_router_z_loss_coeff * logits.logsumexp(-1).square().mean()
            )
            for iteration in range(c.loop_iterations):
                batches = []
                for batch in range(x.shape[1]):
                    indices = (choice[:, batch] >= iteration).nonzero(as_tuple=True)[0]
                    current = x[:, batch : batch + 1]
                    if indices.numel():
                        active = current.index_select(0, indices)
                        for index in range(start, stop):
                            active = self.layer(active, index, indices)
                        current = current.index_copy(0, indices, active)
                    batches.append(current)
                x = torch.cat(batches, 1)
            x = (embedding.float() + selected[..., None] * x.float()).to(x.dtype)
        coda_start = start + c.loop_core_layers * (c.loop_iterations if c.loop_untie_core else 1)
        for index in range(coda_start, c.num_layers):
            x = self.layer(x, index, positions)
        return self.norm(x, "final_layernorm.weight")


def compare(actual, expected, dtype):
    if dtype == torch.float32:
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)
    else:
        a, b = actual.float().flatten(), expected.float().flatten()
        if torch.count_nonzero(a) == 0 and torch.count_nonzero(b) == 0:
            return
        cosine = F.cosine_similarity(a, b, dim=0)
        similarity = 2 * (a * b).sum() / (a.square().sum() + b.square().sum()).clamp_min(1e-30)
        assert cosine > 1 - 1e-3, float(cosine)
        assert similarity > 1 - 1e-3, float(similarity)


@pytest.mark.parametrize("depth", [1, 2, 4])
@pytest.mark.parametrize("bptt,recompute", [(0, False), (2, False), (0, True)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_huginn_native_parity(depth, bptt, recompute, dtype):
    block = make_block(depth=depth, bptt=bptt, recompute=recompute, dtype=dtype)
    check_native(block, dtype)


@pytest.mark.parametrize(
    "execution,dtype",
    [("masked", torch.float32), ("masked", torch.bfloat16), ("compact", torch.bfloat16)],
)
@pytest.mark.parametrize("recompute", [False, True])
def test_mor_native_parity(execution, dtype, recompute):
    block = make_block("mor", depth=4, recompute=recompute, execution=execution, dtype=dtype)
    check_native(block, dtype)


def test_untied_unroll_native_parity():
    check_native(make_block(depth=4, untie=True), torch.float32)


@pytest.mark.parametrize("architecture", ["huginn", "mor"])
def test_production_width_native_parity(architecture):
    block = make_block(architecture, depth=2, dtype=torch.bfloat16, production_shape=True)
    check_native(block, torch.bfloat16)


@pytest.mark.parametrize("architecture", ["huginn", "mor"])
@pytest.mark.parametrize("recompute", [False, True])
def test_moe_native_parity(architecture, recompute):
    # Independent expert equations; auxiliary-loss collective coverage is in
    # test_moe_collective_order_and_empty_local_active_sets with coeff=0.01.
    config = replace(
        make_block(architecture, execution="masked", recompute=recompute).config,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_ffn_hidden_size=256,
        moe_layer_freq=[0, 1, 1, 0],
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        moe_aux_loss_coeff=0.0,
    )
    block = LoopTransformerBlock(
        config=config,
        spec=get_gpt_decoder_block_spec(config, use_transformer_engine=True),
        pre_process=True,
        post_process=True,
    ).cuda()
    check_native(block, torch.float32)


HASH_MOE = dict(num_moe_experts=4, moe_ffn_hidden_size=256, hash_moe_vocab_size=128)


@pytest.mark.parametrize(
    "overrides,feature",
    [
        (dict(HASH_MOE, moe_num_hash_layers=1), "hash-routed MoE"),
        # The legacy alias is folded into moe_num_hash_layers by the config.
        (dict(HASH_MOE, moe_n_hash_layers=1), "hash-routed MoE"),
        (dict(tensor_parallel_num_weight_shards=2), "GTP weight remat"),
        (
            dict(
                num_moe_experts=4,
                moe_ffn_hidden_size=256,
                expert_tensor_parallel_num_weight_shards=2,
            ),
            "GTP weight remat",
        ),
    ],
)
@pytest.mark.parametrize("architecture", ["huginn", "mor"])
def test_rejects_unsupported_dev_features(overrides, feature, architecture):
    # Validation precedes layer construction, so no layer is built or moved to GPU.
    config = make_config(architecture, **overrides)
    with pytest.raises(ValueError, match=feature):
        LoopTransformerBlock(
            config=config,
            spec=get_gpt_layer_with_transformer_engine_spec(),
            pre_process=True,
            post_process=True,
        )


@pytest.mark.parametrize("architecture", ["huginn", "mor"])
@pytest.mark.parametrize("recompute", [False, True])
def test_layer_residuals_observed_once_per_layer_call(architecture, recompute):
    block = make_block(architecture, depth=2, recompute=recompute, execution="masked")
    x = torch.randn(16, 2, 128, device="cuda", requires_grad=True)
    rope = RotaryEmbedding(64, rotary_percent=1.0)(16).cuda()
    seen = []

    def observer(owner, name, kind, tensor, *dims):
        seen.append((owner.layer_number, kind))

    kinds = frozenset({"residual_accumulator", "residual_contribution"})
    with capture_tensor_observations(observer, kinds):
        block(x, attention_mask=None, rotary_pos_emb=rope).sum().backward()
    # Prelude layer 1 and coda layer 4 once; shared core layers 2-3 per
    # iteration. Recompute replays during backward must not observe again.
    calls = [1, 2, 3, 2, 3, 4]
    assert seen == [(n, k) for n in calls for k in sorted(kinds)]


@pytest.mark.parametrize("architecture", ["huginn", "mor"])
def test_evaluation_work_excludes_backward_and_recompute(architecture):
    block = make_block(architecture, recompute=True, dtype=torch.bfloat16).eval()
    x = torch.randn(16, 2, 128, device="cuda", dtype=torch.bfloat16)
    rope = RotaryEmbedding(64, rotary_percent=1.0)(16).cuda()
    with torch.no_grad():
        block(x, attention_mask=None, rotary_pos_emb=rope)
    assert block.loop_metrics["loop forward flops per token"] > 0
    assert block.loop_metrics["loop backward flops per token"] == 0
    assert block.loop_metrics["loop recompute flops per token"] == 0
    assert block.loop_metrics["loop backward token layers"] == 0


@pytest.mark.parametrize("architecture", ["huginn", "mor"])
def test_loop_metric_populations(architecture):
    """Huginn's metrics are per issued token (no populations); MoR's depth metrics are means
    over the valid tokens, whose count the block exposes for the loss function's report
    (megatron/core/transformer/loop/reporting.py)."""
    block = make_block(architecture, depth=3)
    x = torch.randn(16, 2, 128, device="cuda")
    rope = RotaryEmbedding(64, rotary_percent=1.0)(16).cuda()
    kwargs = {}
    if architecture == "mor":
        padding = torch.zeros(2, 16, dtype=torch.bool, device="cuda")
        padding[1, -4:] = True
        kwargs["padding_mask"] = padding
    block(x, attention_mask=None, rotary_pos_emb=rope, **kwargs)
    populations = block.loop_metric_populations
    if architecture == "huginn":
        assert populations == {}
        return
    fractions = {f"depth fraction {index}" for index in (1, 2, 3)}
    assert set(populations) == {"loop depth", *fractions}
    assert all(int(count) == 28 for count in populations.values())
    report = loop_metric_report(block.loop_metrics, issued_tokens=32, populations=populations)
    rounds = block.loop_metrics["loop active token layers"] / block.config.loop_core_layers
    torch.testing.assert_close(report["loop depth"][0], rounds.float(), rtol=1e-6, atol=0)
    assert report["loop depth"][1].item() == 28
    assert report["loop active token layers"][1].item() == 32


def check_native(block, dtype):
    native = NativeDecoder(block).cuda()
    logits = []
    if block.config.loop_architecture == "mor":
        block.depth_router.register_forward_hook(
            lambda module, inputs, output: logits.append(output.detach())
        )
    # BF16 decoder inputs use GPT's embedding initialization scale.
    # Unit-variance inputs make this small initialized residual stack almost
    # exactly identity; the depth gate then cancels through the final RMSNorm,
    # so relative BF16 gradient similarity measures roundoff instead of signal.
    # FP32 also tests the cancellation-prone unit-variance regime.
    input_scale = block.config.init_method_std if dtype == torch.bfloat16 else 1.0
    x = (
        torch.randn(16, 2, block.config.hidden_size, device="cuda", dtype=dtype) * input_scale
    ).requires_grad_(True)
    reference_x = x.detach().clone().requires_grad_(True)
    rope = RotaryEmbedding(block.config.kv_channels, rotary_percent=1.0)(16).cuda()
    y = block(x, attention_mask=None, rotary_pos_emb=rope)
    reference = native(reference_x)
    if logits:
        torch.testing.assert_close(logits[0].argmax(-1) + 1, native.chosen_depth, atol=0, rtol=0)
    compare(y, reference, dtype)
    upstream = torch.randn_like(y)
    ((y * upstream).float().sum() + sum(block.loop_losses.values())).backward()
    ((reference * upstream).float().sum() + native.auxiliary).backward()
    compare(x.grad, reference_x.grad, dtype)
    for name, parameter in block.named_parameters():
        assert parameter.grad is not None, name
        try:
            compare(parameter.grad, native.w(name).grad, dtype)
        except AssertionError as error:
            raise AssertionError(f"Gradient mismatch: {name}") from error
    # A real optimizer update checks tied parameters are updated once.
    optimizer = torch.optim.SGD(block.parameters(), lr=0.03)
    reference_optimizer = torch.optim.SGD(native.parameters(), lr=0.03)
    optimizer.step()
    reference_optimizer.step()
    for name, parameter in block.named_parameters():
        compare(parameter, native.w(name), dtype)


@pytest.mark.parametrize("execution", ["masked", "compact"])
def test_mor_causal_prefix_and_empty_depth(execution):
    block = make_block("mor", depth=4, execution=execution, dtype=torch.bfloat16)
    # All-zero logits choose depth one. Later local active sets are empty.
    with torch.no_grad():
        block.depth_router[2].weight.zero_()
    x = torch.randn(16, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    rope = RotaryEmbedding(64, rotary_percent=1.0)(16).cuda()
    output = block(x, attention_mask=None, rotary_pos_emb=rope)
    changed = x.detach().clone()
    changed[8:] += torch.randn_like(changed[8:])
    other = block(changed, attention_mask=None, rotary_pos_emb=rope)
    torch.testing.assert_close(output[:8], other[:8], atol=0, rtol=0)
    output.float().square().mean().backward()
    assert torch.isfinite(x.grad).all()


def test_repeated_parameter_accumulation_and_resume():
    block = make_block(depth=1, recompute=True)
    reference = NativeDecoder(block)
    rope = RotaryEmbedding(64, rotary_percent=1.0)(16).cuda()
    for depth in [1, 4, 2]:
        block.config.loop_iterations = reference.c.loop_iterations = depth
        x = torch.randn(16, 2, 128, device="cuda", requires_grad=True)
        y = block(x, attention_mask=None, rotary_pos_emb=rope)
        expected = reference(x.detach().clone().requires_grad_(True))
        y.float().sum().backward()
        expected.float().sum().backward()
    for name, parameter in block.named_parameters():
        compare(parameter.grad, reference.w(name).grad, torch.float32)
    restored = make_block(depth=2, recompute=True)
    restored.load_state_dict(block.state_dict())
    torch.testing.assert_close(
        restored(x, attention_mask=None, rotary_pos_emb=rope),
        block(x, attention_mask=None, rotary_pos_emb=rope),
    )


@pytest.mark.parametrize("recompute,bptt", [(False, 0), (True, 0), (False, 2)])
def test_ddp_shared_gradient_readiness_with_changing_depth(recompute, bptt):
    block = make_block(depth=1, recompute=recompute, bptt=bptt)
    ddp = DistributedDataParallel(
        block.config,
        ddp_config=DistributedDataParallelConfig(
            overlap_grad_reduce=True, grad_reduce_in_fp32=True, bucket_size=10000
        ),
        module=block,
        pg_collection=block.pg_collection,
    )
    ddp.broadcast_params()
    native = NativeDecoder(block)
    rope = RotaryEmbedding(64, rotary_percent=1.0)(16).cuda()
    for step, depth in enumerate([1, 4, 2]):
        block.config.loop_iterations = native.c.loop_iterations = depth
        ddp.zero_grad_buffer()
        native.zero_grad(set_to_none=True)
        for microbatch in range(2):
            torch.cuda.manual_seed(1000 + 10 * step + microbatch + torch.distributed.get_rank())
            x = torch.randn(16, 2, 128, device="cuda", requires_grad=True)
            reference_x = x.detach().clone().requires_grad_(True)
            with ddp.no_sync() if microbatch == 0 else nullcontext():
                output = ddp(x, attention_mask=None, rotary_pos_emb=rope)
                (output.float().square().sum() / 2).backward()
            (native(reference_x).float().square().sum() / 2).backward()
        ddp.finish_grad_sync()
        for name, parameter in block.named_parameters():
            expected = native.w(name).grad
            torch.distributed.all_reduce(expected, group=block.pg_collection.dp_cp)
            expected /= torch.distributed.get_world_size(block.pg_collection.dp_cp)
            compare(parameter.main_grad, expected, torch.float32)


def test_sampled_depth_rng_resume():
    block = make_block()
    block.config.loop_sample_depth = True
    rng_state = torch.get_rng_state()
    expected = [block._depth(torch.device("cuda")) for _ in range(8)]
    torch.set_rng_state(rng_state)
    actual = [block._depth(torch.device("cuda")) for _ in range(8)]
    assert expected == actual
    assert set(actual) <= {1, 2, 4, 8}


@pytest.mark.parametrize("architecture", ["huginn", "mor"])
def test_distributed_checkpoint_roundtrip(architecture, tmp_path_dist_ckpt):
    block = make_block(architecture, depth=4, dtype=torch.bfloat16)
    saved = {name: value.detach().clone() for name, value in block.named_parameters()}
    checkpoint_dir = tmp_path_dist_ckpt / architecture
    if torch.distributed.get_rank() == 0:
        checkpoint_dir.mkdir()
    torch.distributed.barrier()
    dist_checkpointing.save(block.sharded_state_dict(), checkpoint_dir)
    restored = make_block(architecture, depth=4, dtype=torch.bfloat16)
    with torch.no_grad():
        for parameter in restored.parameters():
            parameter.zero_()
    restored.load_state_dict(dist_checkpointing.load(restored.sharded_state_dict(), checkpoint_dir))
    for name, value in restored.named_parameters():
        torch.testing.assert_close(value, saved[name], rtol=0, atol=0)


def gather_tp_tensor(tensor, name, group):
    """Independent mapping from physical TP shards to the native full weights."""
    shards = [torch.empty_like(tensor) for _ in range(torch.distributed.get_world_size(group))]
    torch.distributed.all_gather(shards, tensor.contiguous(), group=group)
    if name.endswith("linear_fc1.weight"):
        # SwiGLU owns [gate_rank, value_rank], not a contiguous full-FFN slice.
        return torch.cat([piece.chunk(2, 0)[part] for part in range(2) for piece in shards], 0)
    if name.endswith(("linear_qkv.weight",)):
        return torch.cat(shards, 0)
    if name.endswith(("linear_fc2.weight", "linear_proj.weight")):
        return torch.cat(shards, 1)
    for shard in shards[1:]:
        torch.testing.assert_close(shards[0], shard, rtol=0, atol=0)
    return shards[0]


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("architecture", ["huginn", "mor"])
def test_tensor_parallel_native_parity(tp, architecture):
    Utils.initialize_model_parallel(tp, 1)
    model_parallel_cuda_manual_seed(123)
    torch.manual_seed(123)
    config = TransformerConfig(
        num_layers=4,
        hidden_size=256,
        num_attention_heads=4,
        num_query_groups=4,
        kv_channels=64,
        ffn_hidden_size=512,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        gradient_accumulation_fusion=False,
        apply_rope_fusion=False,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        tensor_model_parallel_size=tp,
        loop_architecture=architecture,
        loop_prelude_layers=1,
        loop_core_layers=2,
        loop_iterations=4,
    )
    block = LoopTransformerBlock(
        config=config,
        spec=get_gpt_layer_with_transformer_engine_spec(),
        pre_process=True,
        post_process=True,
    ).cuda()
    native = NativeDecoder(block)
    group = block.pg_collection.tp
    for name, parameter in block.named_parameters():
        native.weights[name.replace(".", "__")] = nn.Parameter(
            gather_tp_tensor(parameter.detach(), name, group)
        )
    torch.cuda.manual_seed(402)
    x = (torch.randn(16, 2, 256, device="cuda", dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    reference_x = x.detach().clone().requires_grad_(True)
    rope = RotaryEmbedding(64, rotary_percent=1.0)(16).cuda()
    output = block(x, attention_mask=None, rotary_pos_emb=rope)
    expected = native(reference_x)
    compare(output, expected, torch.bfloat16)
    upstream = torch.randn_like(output)
    ((output * upstream).float().sum() + sum(block.loop_losses.values())).backward()
    ((expected * upstream).float().sum() + native.auxiliary).backward()
    compare(x.grad, reference_x.grad, torch.bfloat16)
    for name, parameter in block.named_parameters():
        actual = gather_tp_tensor(parameter.grad, name, group)
        try:
            compare(actual, native.w(name).grad, torch.bfloat16)
        except AssertionError as error:
            raise AssertionError(f"TP gradient mismatch: {name}") from error


class PrescribedDepthRouter(nn.Module):
    """A controlled route replay probe, used only to guarantee EP empty ranks."""

    def __init__(self, router, depth):
        super().__init__()
        self.router = router
        self.depth = depth

    def forward(self, x):
        logits = self.router(x)
        offset = torch.zeros_like(logits)
        offset[..., self.depth - 1] = 20
        return logits + offset


@pytest.mark.parametrize("ep", [1, 4])
@pytest.mark.parametrize(
    "architecture,recompute", [("huginn", False), ("huginn", True), ("mor", False), ("mor", True)]
)
def test_moe_collective_order_and_empty_local_active_sets(ep, architecture, recompute):
    Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=ep)
    model_parallel_cuda_manual_seed(123)
    config = replace(
        make_block(architecture, depth=2, recompute=recompute, dtype=torch.bfloat16).config,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_ffn_hidden_size=256,
        moe_layer_freq=[0, 1, 1, 0],
        expert_model_parallel_size=ep,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        moe_aux_loss_coeff=0.01,
    )
    block = LoopTransformerBlock(
        config=config,
        spec=get_gpt_decoder_block_spec(config, use_transformer_engine=True),
        pre_process=True,
        post_process=True,
    ).cuda()
    if architecture == "mor":
        # Rank zero exits after one recurrence; its EP peers still require a
        # second dispatch/combine. EP=1 also exercises an entirely empty domain.
        depth = 1 if torch.distributed.get_rank(block.pg_collection.ep) == 0 else 2
        block.depth_router = PrescribedDepthRouter(block.depth_router, depth)
    x = torch.randn(16, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    rope = RotaryEmbedding(64, rotary_percent=1.0)(16).cuda()
    output = block(x, attention_mask=None, rotary_pos_emb=rope)
    assert torch.isfinite(output).all()
    (output.float().square().mean() + sum(block.loop_losses.values())).backward()
    assert torch.isfinite(x.grad).all()
    for name, parameter in block.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
