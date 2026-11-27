# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Model shape, FLOP, time and memory terms of the RSBF loop planner (planner v1).

A port of the toolkit cost model of the DSv4-class loop (``cost-model/dsv4``: ``dsv4.py``
``params`` / ``key_means`` / ``layer_flops`` / ``layer_time_ms`` / ``costs`` / ``memory`` /
``step_level_ms`` / ``model_flops_per_sample``, ``proxy_dsv4.derived`` / ``act_scale``,
``proxy_v2._step_level_z`` / ``core_resident`` and ``pp_core_baseline.rsbf_resident_sync_ms``),
with the module-level constants of the reference turned into a ``ModelShape`` and a
``PlannerHardware`` argument. Same formulas, same Python float operation order: on the cost
model's shapes the port reproduces its numbers exactly (``test_loop_planner_costs.py``,
``test_loop_planner_cost_model.py``). The toolkit code itself is not imported.

Units: bytes, FLOPs, ms; a "layer" is one attention sublayer and the MLP / MoE sublayer after it
(the executor's block stride counts sublayers, two per layer). Every number is an estimate.

Layer kinds: ``"CSA"`` (DSv4 compressed sparse attention, ratio 4, with the top-k indexer),
``"HCA"`` (ratio 128), ``"W"`` (window only) and ``"ATT"`` (plain causal self-attention, the
hybrid ``*`` symbol; not in the cost model, defined here). FFN: ``"moe"`` or ``"mlp"``.
"""

import functools
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

__all__ = [
    "DSv4Dims",
    "ModelShape",
    "ParamGroups",
    "PlannerUnsupported",
    "PRO_DIMS",
    "Placement",
    "STORE_GROUP_OF",
    "SeqPhase",
    "UnitCosts",
    "c2c_traffic",
    "key_means",
    "layer_flops",
    "layer_time_ms",
    "memory_components",
    "model_flops_per_sample",
    "place_round_state",
    "pro_ratios",
    "proxy7_ratios",
    "resident_sync_ms",
    "step_level_ms",
    "unit_costs",
]

GB = 1e9


class PlannerUnsupported(ValueError):
    """A model, phase or plan planner v1 cannot model or apply."""


CHAIN_UNITS = 3  # chain record dL/ds bf16 + g_u fp32, in round-state units
KIND_OF_RATIO = {4: "CSA", 128: "HCA", 0: "W"}
DSV4_KINDS = frozenset(("CSA", "HCA", "W"))
# The memory component names (dsv4.memory keys), in the reference's order.
MEM_EXPERT_W = "routed experts: bf16 weights"
MEM_EXPERT_G = "routed experts: grad accumulator"
MEM_DENSE_RES = "dense: Resident groups (bf16 + fp32 grad)"
MEM_DENSE_SH = "dense: shards (bf16 + fp32) /z"
MEM_DENSE_BUF = "dense: weight double buffer + fp32 scratch (one set)"
MEM_OPT = "optimizer (fp32 master + Adam) /N"
MEM_ACT_BLOCK = "activations: max(core item = s layers + block input, coda item)"
MEM_ACT_NESTED = "activations: max(core item = L_core layer inputs + 1 layer, coda item)"
MEM_WINDOW = "dense: block weight window, (s - 1) extra bf16 layer buffers"
MEM_INDEXER = "indexer buffers (dense fallback, 1 CSA layer live)"
MEM_MXFP8 = "MXFP8 per-layer weight casts (transient)"
MEM_GRAPH_IO = "graph I/O + chain records in flight"
MEM_STORE = "state store (peak, from sim)"
MEM_TE_DUMMY = "TE fused-wgrad dummy gradients (one per dense weight shape)"
MEM_COPY_ENGINE = "store copy engines: symmetric buffers (gather mirror + reduce slots)"
MEM_RESERVE = "NCCL / HybridEP buffers"
MEM_MISC = "misc (context, workspaces, fragmentation)"


# ----------------------------------------------------------------------------- dims and shape
@dataclass(frozen=True)
class DSv4Dims:
    """Model dimensions (the cost model's key names, ``proxy_dsv4.PRO``).

    ``DH`` is the attention value head dim (DSv4: ``v_head_dim``, one shared K = V head; plain
    attention: ``kv_channels``); ``OG`` / ``OL`` the DSv4 output groups / low rank; ``NS`` the
    residual streams (1 without mHC); ``KVH`` the key / value heads of plain attention;
    ``FFN_D`` the hidden size of a dense MLP sublayer, ``MLP_MATS`` its weight matrices (3 gated,
    2 plain)."""

    H: int
    V: int
    NEXP: int
    TOPK: int
    FFN_E: int
    FFN_S: int
    HEADS: int
    DH: int
    QL: int
    OG: int
    OL: int
    IDX_H: int
    IDX_D: int
    IDX_TOPK: int
    WIN: int
    NS: int
    KVH: int = 1
    FFN_D: int = 0
    MLP_MATS: int = 3

    @staticmethod
    def from_config(config: Any, vocab_size: int) -> "DSv4Dims":
        """The dims of a built ``TransformerConfig`` (``vocab_size``: the padded vocabulary)."""
        dsv4 = getattr(config, "experimental_attention_variant", None) == "dsv4_hybrid"
        heads = config.num_attention_heads
        kv_channels = config.kv_channels or config.hidden_size // heads
        ffn = config.ffn_hidden_size or 4 * config.hidden_size
        return DSv4Dims(
            H=config.hidden_size,
            V=int(vocab_size),
            NEXP=config.num_moe_experts or 0,
            TOPK=config.moe_router_topk if config.num_moe_experts else 0,
            FFN_E=config.moe_ffn_hidden_size or ffn,
            FFN_S=config.moe_shared_expert_intermediate_size or 0,
            HEADS=heads,
            DH=(config.v_head_dim if dsv4 else kv_channels),
            QL=(config.q_lora_rank or 0) if dsv4 else 0,
            OG=(getattr(config, "o_groups", None) or 0) if dsv4 else 0,
            OL=(getattr(config, "o_lora_rank", None) or 0) if dsv4 else 0,
            IDX_H=config.dsa_indexer_n_heads or 0,
            IDX_D=config.dsa_indexer_head_dim or 0,
            IDX_TOPK=config.dsa_indexer_topk or 0,
            WIN=getattr(config, "csa_window_size", 128) or 0,
            NS=(config.num_residual_streams if config.enable_hyper_connections else 1),
            KVH=config.num_query_groups or heads,
            FFN_D=ffn,
            MLP_MATS=3 if config.gated_linear_unit else 2,
        )


# DSv4 Pro (proxy_dsv4.PRO).
PRO_DIMS = DSv4Dims(
    H=7168,
    V=129280,
    NEXP=384,
    TOPK=6,
    FFN_E=3072,
    FFN_S=2048,
    HEADS=128,
    DH=512,
    QL=1536,
    OG=16,
    OL=1024,
    IDX_H=64,
    IDX_D=128,
    IDX_TOPK=1024,
    WIN=128,
    NS=4,
)


def pro_ratios(lp: int, lc: int, ld: int) -> list:
    """The DSv4 Pro compress-ratio pattern of a ``lp / lc / ld`` split plus the MTP layer
    (``proxy_dsv4.pro_ratios``): ``[128, 128, 4] + [128, 4] * k + [0]``."""
    if lp != 3 or (lc + ld) % 2:
        raise ValueError(
            f"the Pro pattern needs a 3-layer prelude and an even rest, got {lp, lc, ld}"
        )
    return [128, 128, 4] + [128, 4] * ((lc + ld) // 2) + [0]


def proxy7_ratios(lp: int = 1, lc: int = 4, ld: int = 2) -> list:
    """DSV4-PRO-PROXY (``proxy_dsv4.proxy7_ratios``): 7 layers ``1 / 4 / 2`` plus the MTP layer."""
    if (lp, lc, ld) != (1, 4, 2):
        raise ValueError(f"the PRO-PROXY pattern is split 1 / 4 / 2, got {lp, lc, ld}")
    return [128] + [128, 4] * 3 + [0]


def _derived(c: DSv4Dims) -> Dict[str, int]:
    """``proxy_dsv4.derived``: the parameter / FLOP constants of a DSv4 shape."""
    H, DH, NS = c.H, c.DH, c.NS
    d = {}
    d["P_ATT_BASE"] = (
        H * c.QL
        + c.QL
        + c.QL * c.HEADS * DH
        + H * DH
        + DH
        + c.HEADS * DH * c.OL
        + c.OG * c.OL * H
        + c.HEADS
    )
    d["P_COMP4"] = 2 * H * 2 * DH + 4 * 2 * DH + DH
    d["P_COMP128"] = 2 * H * DH + 128 * DH + DH
    d["P_INDEXER"] = (
        c.QL * c.IDX_H * c.IDX_D + H * c.IDX_H + 2 * H * 2 * c.IDX_D + 4 * 2 * c.IDX_D + c.IDX_D
    )
    nm = NS * NS + 2 * NS
    d["P_MHC_LAYER"] = 2 * (NS * H * nm + nm + 3)
    d["P_NORMS"] = 2 * H
    d["P_ROUTER"] = c.NEXP * H
    d["P_SHARED"] = 3 * H * c.FFN_S
    d["P_EXPERT"] = 3 * H * c.FFN_E
    d["P_ROUTED_LAYER"] = c.NEXP * d["P_EXPERT"]
    d["P_ADAPTER"] = 2 * H * H
    return d


# Saved activation bytes per token per layer at Pro width (dsv4.ACT_B_LAYER / ACT_B_MTP), scaled by
# component (proxy_dsv4.ACT_MIX: residual / mHC with H, the q / o path with HEADS x DH, the routed
# path with TOPK x FFN_E; est.).
ACT_B_LAYER_PRO = 0.97e6
ACT_B_MTP_PRO = 1.05e6
ACT_MIX = (0.25, 0.42, 0.33)


def act_scale(c: DSv4Dims) -> float:
    """``proxy_dsv4.act_scale``: saved activation bytes relative to DSv4 Pro (est.)."""
    a, b, e = ACT_MIX
    return a * c.H / 7168 + b * c.HEADS * c.DH / (128 * 512) + e * c.TOPK * c.FFN_E / (6 * 3072)


@dataclass(frozen=True)
class ParamGroups:
    """Parameter counts by placement group (``dsv4.params``; elements, the whole model)."""

    emb_head: float
    prelude_dense: float
    core_dense: float
    coda_dense: float
    mtp_dense: float
    routed_experts: float  # every routed expert of every MoE layer (P_e); per GPU P_e / EP
    largest_dense_layer: float  # dsv4 p_dense_layer("CSA"): one layer's dense weights
    expert_params: float = 0.0  # one routed expert (P_EXPERT)
    dense_by_kind: Mapping[str, float] = field(default_factory=dict)  # one layer, by kind

    def groups(self) -> Dict[str, float]:
        """The dense groups in the reference's order (``dsv4.params`` keys)."""
        return {
            "emb+head": self.emb_head,
            "prelude dense": self.prelude_dense,
            "core dense": self.core_dense,
            "coda dense": self.coda_dense,
            "MTP dense": self.mtp_dense,
        }

    @property
    def dense(self) -> float:
        """``P_d``: every dense parameter."""
        return sum(self.groups().values())


Layer = Tuple[str, str]  # (attention kind, ffn)


@dataclass(frozen=True)
class ModelShape:
    """What the planner needs of a model: dims, the (kind, ffn) layers of each role, parameter
    groups and per-token byte constants.

    ``prelude`` / ``core`` / ``coda``: one ``(kind, ffn)`` per layer; ``mtp``: one per MTP
    depth. ``act_bytes_per_token_layer`` / ``_mtp``: saved activation bytes per token of one
    layer (before ``act_frag``); ``mla_up_bytes_per_token``: what the D42 up-projection recompute
    drops per DSv4 layer; ``elem_bytes_per_token`` / ``mhc_bytes_per_token``: bandwidth-bound
    bytes per token per layer pass (q-path elementwise; mHC); ``ce_bytes_per_token``: the coda
    arena's logits term per token (None: the cost model's ``2 V 2 / 8``, 8 CE chunks)."""

    dims: DSv4Dims
    prelude: Tuple[Layer, ...]
    core: Tuple[Layer, ...]
    coda: Tuple[Layer, ...]
    mtp: Tuple[Layer, ...]
    params: ParamGroups
    act_bytes_per_token_layer: float
    act_bytes_per_token_mtp: float
    mla_up_bytes_per_token: float
    elem_bytes_per_token: float
    mhc_bytes_per_token: float
    expert_grad_bytes: int = 4
    optimizer_bytes_per_param: int = 12
    grace_optimizer_bytes_per_param: int = 12
    ce_bytes_per_token: Optional[float] = None
    distributed_optimizer: bool = True
    # The loop parameter store's runtime terms per (modes, item form) variant
    # (``LoopParamStore.planner_terms``: ``store_variant_key -> StoreMemoryTerms`` or the reason
    # the store cannot run it); None: the cost model's dense terms (``from_dims``, no store).
    store_terms: Optional[Mapping[str, Any]] = None

    def __post_init__(self):
        # The Layer-window gathers are charged per layer kind (``unit_costs``): every kind of
        # every role, the MTP layers included, needs its one-layer dense count.
        layers = self.prelude + self.core + self.coda + self.mtp
        missing = sorted({kind for kind, _ in layers} - set(self.params.dense_by_kind))
        if missing:
            raise PlannerUnsupported(
                f"ModelShape: params.dense_by_kind has no entry for layer kind(s) {missing}"
            )

    @property
    def core_layers(self) -> int:
        """``L_core``."""
        return len(self.core)

    @property
    def state_bytes_per_token(self) -> int:
        """One round state per token: ``NS x H`` bf16 (``dsv4.STATE_TOK``)."""
        return self.dims.NS * self.dims.H * 2

    @property
    def has_csa(self) -> bool:
        """A CSA layer (indexer buffers) exists."""
        return any(kind == "CSA" for kind, _ in self.prelude + self.core + self.coda + self.mtp)

    @property
    def has_coda_dsv4_attention(self) -> bool:
        """A DSv4 attention in the coda or the MTP layers (D42 has something to recompute)."""
        return any(kind in DSV4_KINDS for kind, _ in self.coda + self.mtp)

    @property
    def coda_dsv4_attentions(self) -> int:
        """DSv4 attentions the D42 recompute reaches (coda layers + MTP depths)."""
        return sum(1 for kind, _ in self.coda + self.mtp if kind in DSV4_KINDS)

    @staticmethod
    def from_dims(
        dims: DSv4Dims, split: Tuple[int, int, int], ratios: Sequence[int], **overrides
    ) -> "ModelShape":
        """A cost-model shape: DSv4 layers from compress ``ratios`` (one per layer of the
        ``prelude / core / coda`` split, plus the MTP layer), every FFN a MoE
        (``proxy_dsv4.shape`` + ``dsv4.params``). The cross-check models: DSV4-PRO-192E =
        ``(replace(PRO_DIMS, NEXP=192), (3, 54, 4), pro_ratios(3, 54, 4))``, DSV4-PRO-PROXY =
        ``(PRO_DIMS, (1, 4, 2), proxy7_ratios())``."""
        lp, lc, ld = split
        if len(ratios) != lp + lc + ld + 1:
            raise ValueError(f"{len(ratios)} ratios for a {split} split plus MTP")
        kinds = [KIND_OF_RATIO[r] for r in ratios]
        c = _derived(dims)

        def p_dense_layer(kind):
            extra = {"CSA": c["P_COMP4"] + c["P_INDEXER"], "HCA": c["P_COMP128"], "W": 0}[kind]
            return (
                c["P_ATT_BASE"]
                + extra
                + c["P_MHC_LAYER"]
                + c["P_NORMS"]
                + c["P_ROUTER"]
                + c["P_SHARED"]
            )

        H, NS = dims.H, dims.NS
        pre, core, coda, mtp = kinds[:lp], kinds[lp : lp + lc], kinds[lp + lc : -1], kinds[-1]
        params = ParamGroups(
            emb_head=2 * dims.V * H + H + 114_693,
            prelude_dense=sum(p_dense_layer(k) for k in pre),
            core_dense=sum(p_dense_layer(k) for k in core) + c["P_ADAPTER"] + 2 * NS * H,
            coda_dense=sum(p_dense_layer(k) for k in coda),
            mtp_dense=p_dense_layer(mtp) + 2 * H + 2 * H * H + H + 114_693,
            routed_experts=(lp + lc + ld + 1) * c["P_ROUTED_LAYER"],
            largest_dense_layer=p_dense_layer("CSA"),
            expert_params=c["P_EXPERT"],
            dense_by_kind={k: p_dense_layer(k) for k in ("CSA", "HCA", "W")},
        )
        s = act_scale(dims)
        fields_ = dict(
            dims=dims,
            prelude=tuple((k, "moe") for k in pre),
            core=tuple((k, "moe") for k in core),
            coda=tuple((k, "moe") for k in coda),
            mtp=((mtp, "moe"),),
            params=params,
            act_bytes_per_token_layer=ACT_B_LAYER_PRO * s,
            act_bytes_per_token_mtp=ACT_B_MTP_PRO * s,
            mla_up_bytes_per_token=2 * (2 * dims.HEADS * dims.DH) + 2048,
            elem_bytes_per_token=0.40e6 * dims.HEADS * dims.DH / (128 * 512),
            mhc_bytes_per_token=2 * 14 * dims.H * 2,
        )
        fields_.update(overrides)
        return ModelShape(**fields_)

    @staticmethod
    def from_model(
        model: Any,
        *,
        vocab_size: int,
        expert_parallel_size: int,
        act_bytes_per_token: Optional[float] = None,
        **overrides,
    ) -> "ModelShape":
        """The shape of a built hybrid loop model (``HybridModel`` with a ``LoopHybridStack``,
        possibly wrapped): layer kinds from the stack's sublayer symbols and DSv4 compress
        ratios, dims from its config, parameter counts by role from this rank's parameters
        (expert-parallel ones, ``allreduce == False``, times ``expert_parallel_size``).

        Groups: the embedding, output layer, final norm and mHC output contraction are
        ``emb_head``; ``decoder.layers`` by role; the other loop-stack parameters (adapter, loop
        norm) are ``core_dense``; the MTP block is ``mtp_dense``. Every role must be made of
        (attention, MLP / MoE) sublayer pairs, else
        ``PlannerUnsupported``. Activation bytes:
        ``act_bytes_per_token`` per layer (the MTP layer scaled like the cost model's 1.05 / 0.97),
        or the ported Pro-width estimate x ``act_scale`` (est.).

        Runtime byte defaults (the cost model's ``from_dims`` keeps its own; ``overrides`` win):
        ``grace_optimizer_bytes_per_param = 16`` (fp32 master, both Adam moments and the fp32
        gradient of the CPU-offloaded optimizer) and ``ce_bytes_per_token = 10 V (1 + MTP
        depths)`` (est.: the unfused cross entropy's bf16 logits and fp32 softmax per head, all
        heads alive at the coda backward; the cost model's ``2 V 2 / 8`` assumes 8 fused CE
        chunks). ``expert_parallel_size`` is required: this rank holds ``1 / EP`` of the routed
        experts. With a loop parameter store installed the counts are logical
        (``named_logical_parameters``: the hidden compute parameters at their logical size, the
        store's mirror of canonical shards skipped), so they equal the model's before the
        install and stay the same while the compute parameters are unbound; the store's
        runtime terms come in through ``overrides`` (``store_terms``)."""
        from megatron.core.models.hybrid.layers.utils import Symbols
        from megatron.core.transformer.loop.param_store_binding import named_logical_parameters

        def logical(owner):
            # The registered parameters plus the loop parameter store's hidden compute
            # parameters (the store's mirror of canonical shards is skipped), so the counts are
            # the same before and after a store is installed, bound or not.
            return [p for _, p in named_logical_parameters(owner)]

        if expert_parallel_size < 1:
            raise ValueError(f"expert_parallel_size must be >= 1, got {expert_parallel_size}")
        module = model
        while not hasattr(module, "decoder") and hasattr(module, "module"):
            module = module.module
        stack = getattr(module, "decoder", None)
        roles = getattr(stack, "loop_roles", None)
        if roles is None:
            raise PlannerUnsupported(
                "ModelShape.from_model needs a HybridModel with a LoopHybridStack"
            )
        config = stack.config
        dims = DSv4Dims.from_config(config, vocab_size)
        symbols = list(stack.layer_type_list)
        ratios = [ratio for _, _, ratio in stack._loop_flop_rows]

        def kind_of(symbol, ratio, owner):
            # C / H / W carry their ratio in the symbol (the stack's FLOP rows hold a ratio
            # only for 'D', read from its built attention; loop/flops.py maps the same way).
            if symbol == Symbols.ATTENTION:
                return "ATT"
            if symbol in Symbols.DSV4_COMPRESS_RATIO_MAP:
                ratio = Symbols.DSV4_COMPRESS_RATIO_MAP[symbol]
            elif symbol != Symbols.DS_ATTENTION:
                raise PlannerUnsupported(f"{owner}: sublayer {symbol!r} has no planner cost row")
            if ratio not in KIND_OF_RATIO:
                raise PlannerUnsupported(
                    f"{owner}: DSv4 compress ratio {ratio} has no planner cost row"
                )
            return KIND_OF_RATIO[ratio]

        def pairs(start, end, owner):
            out = []
            for i in range(start, end, 2):
                if i + 1 >= end or symbols[i + 1] not in (Symbols.MLP, Symbols.MOE):
                    raise PlannerUnsupported(
                        f"{owner}: sublayers {start}..{end - 1} are not "
                        "(attention, MLP / MoE) pairs"
                    )
                kind = kind_of(symbols[i], ratios[i], owner)
                ffn = "moe" if symbols[i + 1] == Symbols.MOE else "mlp"
                out.append((i, kind, ffn))
            return out

        def split_count(params):
            dense = expert = 0
            for p in params:
                # A store tensor's logical size (its compute parameter may be unbound, or a
                # shard-shaped DistributedWeight stand-in).
                tensor = getattr(p, "loop_store_tensor", None)
                numel = tensor.numel if tensor is not None else p.numel()
                if getattr(p, "allreduce", True):
                    dense += numel
                else:
                    expert += numel * expert_parallel_size
            return dense, expert

        role_pairs = {
            "prelude": pairs(0, roles.core_start, "prelude"),
            "core": pairs(roles.core_start, roles.core_end, "core"),
            "coda": pairs(roles.core_end, len(symbols), "coda"),
        }
        dense = {"emb+head": 0, "prelude": 0, "core": 0, "coda": 0, "mtp": 0}
        routed = 0
        by_kind: Dict[str, float] = {}
        largest = 0
        layer_of = {}
        for role, items in role_pairs.items():
            for i, kind, _ in items:
                params = logical(stack.layers[i]) + logical(stack.layers[i + 1])
                d_, e_ = split_count(params)
                dense[role] += d_
                routed += e_
                by_kind[kind] = max(by_kind.get(kind, 0), d_)
                largest = max(largest, d_)
                layer_of.update((id(p), role) for p in params)
        seen = set(layer_of)
        for name, p in named_logical_parameters(stack):
            if id(p) in seen:
                continue
            seen.add(id(p))
            role = "emb+head" if name.startswith(("final_norm", "hc_head_")) else "core"
            d_, e_ = split_count([p])
            dense[role] += d_
            routed += e_
        mtp = ()
        mtp_block = getattr(module, "mtp", None)
        mtp_pattern = getattr(module, "mtp_pattern", None)
        if mtp_block is not None and mtp_pattern:
            if len(mtp_pattern) != 2 or mtp_pattern[1] not in (Symbols.MLP, Symbols.MOE):
                raise PlannerUnsupported(
                    f"MTP pattern {mtp_pattern!r} is not one (attention, MLP / MoE) pair"
                )
            ratio = Symbols.DSV4_COMPRESS_RATIO_MAP.get(mtp_pattern[0])
            kind = kind_of(mtp_pattern[0], ratio, "MTP")
            ffn = "moe" if mtp_pattern[1] == Symbols.MOE else "mlp"
            depths = getattr(config, "mtp_num_layers", None) or 0
            mtp = ((kind, ffn),) * depths
            # The MTP layer's dense weights register its kind (the Layer window charges its
            # gathers per kind): each depth's inner layer without the projection and norms
            # (``mtp_model_layer``), or the whole block when it has no such structure.
            for depth_layer in getattr(mtp_block, "layers", None) or [mtp_block]:
                inner = getattr(depth_layer, "mtp_model_layer", depth_layer)
                d_, _ = split_count(logical(inner))
                by_kind[kind] = max(by_kind.get(kind, 0), d_)
            for p in logical(mtp_block):
                if id(p) not in seen:
                    seen.add(id(p))
                    d_, e_ = split_count([p])
                    dense["mtp"] += d_
                    routed += e_
        for p in logical(module):
            if id(p) not in seen:
                seen.add(id(p))
                d_, e_ = split_count([p])
                dense["emb+head"] += d_
                routed += e_
        layers = role_pairs["prelude"] + role_pairs["core"] + role_pairs["coda"]
        moe_layers = sum(1 for _, _, ffn in layers if ffn == "moe") + sum(
            1 for _, ffn in mtp if ffn == "moe"
        )
        params = ParamGroups(
            emb_head=dense["emb+head"],
            prelude_dense=dense["prelude"],
            core_dense=dense["core"],
            coda_dense=dense["coda"],
            mtp_dense=dense["mtp"],
            routed_experts=routed,
            largest_dense_layer=largest,
            expert_params=(routed / (moe_layers * dims.NEXP)) if moe_layers and dims.NEXP else 0.0,
            dense_by_kind=by_kind,
        )
        s = act_scale(dims)
        if act_bytes_per_token is None:
            a_layer, a_mtp = ACT_B_LAYER_PRO * s, ACT_B_MTP_PRO * s
        else:
            a_layer = act_bytes_per_token
            a_mtp = act_bytes_per_token * ACT_B_MTP_PRO / ACT_B_LAYER_PRO
        fields_ = dict(
            dims=dims,
            prelude=tuple((k, f) for _, k, f in role_pairs["prelude"]),
            core=tuple((k, f) for _, k, f in role_pairs["core"]),
            coda=tuple((k, f) for _, k, f in role_pairs["coda"]),
            mtp=mtp,
            params=params,
            act_bytes_per_token_layer=a_layer,
            act_bytes_per_token_mtp=a_mtp,
            mla_up_bytes_per_token=2 * (2 * dims.HEADS * dims.DH) + 2048,
            elem_bytes_per_token=0.40e6 * dims.HEADS * dims.DH / (128 * 512),
            mhc_bytes_per_token=(2 * 14 * dims.H * 2) if dims.NS > 1 else 0.0,
            grace_optimizer_bytes_per_param=16,
            ce_bytes_per_token=10.0 * dims.V * (1 + len(mtp)),
        )
        fields_.update(overrides)
        return ModelShape(**fields_)


# ----------------------------------------------------------------------------- phase and FLOPs
@functools.lru_cache(maxsize=None)
def key_means(S: int, lo: int, hi: int, win: int, idx_topk: int) -> Dict[str, float]:
    """Mean keys per query over the query positions ``[lo, hi)`` of a causal sequence of length
    ``S`` (``dsv4.key_means``): window keys, CSA candidates (ratio 4), indexer top-k, HCA keys
    (ratio 128), and (planner v1) ``full``: every causal key, for plain attention. Cached: the
    caller must not mutate the result."""
    n = hi - lo
    w = c4 = c128 = 0
    for i in range(lo, hi):
        w += min(i + 1, win)
        c4 += (i + 1) // 4
        c128 += (i + 1) // 128
    w, c4, c128 = w / n, c4 / n, c128 / n
    topk = 0
    for i in range(lo, hi):
        topk += min(idx_topk, (i + 1) // 4)
    full = (lo + 1 + hi) * n // 2  # the sum of i + 1 over [lo, hi)
    return dict(win=w, cand=c4, topk=topk / n, hca=c128, full=full / n)


@dataclass(frozen=True)
class SeqPhase:
    """A sequence phase (``dsv4.Phase``): ``S`` tokens per sample over ``cp`` ranks, ``T = S /
    cp`` tokens per GPU, the key means of the whole sequence and of every CP rank's slice, and
    one round state per GPU (``state_gpu`` bytes)."""

    S: int
    cp: int
    T: int
    km_all: Mapping[str, float]
    km_rank: Tuple[Mapping[str, float], ...]
    state_gpu: int

    @staticmethod
    def of(shape: ModelShape, S: int, cp: int = 1) -> "SeqPhase":
        """The phase of ``shape`` at sequence length ``S`` and context parallelism ``cp``."""
        T = S // cp
        win, topk = shape.dims.WIN, shape.dims.IDX_TOPK
        return SeqPhase(
            S=S,
            cp=cp,
            T=T,
            km_all=key_means(S, 0, S, win, topk),
            km_rank=tuple(key_means(S, r * T, (r + 1) * T, win, topk) for r in range(cp)),
            state_gpu=shape.state_bytes_per_token * T,
        )


def layer_flops(kind: str, km: Mapping[str, float], dims: DSv4Dims, ffn: str = "moe") -> dict:
    """Forward FLOPs per token of one layer by component (``dsv4.layer_flops``; ``mx`` = the
    MXFP8-eligible GEMMs; ``teacher`` / ``idx_bwd`` run in the training forward only).

    ``ATT`` (planner v1, est.): QKV and output projections ``2 H (HEADS DH + 2 KVH DH) + 2
    HEADS DH H`` and ``4 HEADS DH`` per causal key. ``mlp``: the dense MLP in ``shared_mx``
    (``2 MLP_MATS H FFN_D``), no router, no routed GEMM. Without mHC (``NS = 1``) no ``mhc``."""
    c = dims
    moe = ffn == "moe"
    f = {}
    if kind == "ATT":
        f["dense_mx"] = 2 * (c.H * c.HEADS * c.DH + 2 * c.H * c.KVH * c.DH + c.HEADS * c.DH * c.H)
        f["small_bf16"] = 2 * c.NEXP * c.H if moe else 0
    else:
        f["dense_mx"] = 2 * (c.H * c.QL + c.QL * c.HEADS * c.DH + c.H * c.DH + c.OG * c.OL * c.H)
        if moe:
            f["small_bf16"] = 2 * (c.HEADS * c.DH) * c.OL + 2 * c.NEXP * c.H
        else:
            f["small_bf16"] = 2 * (c.HEADS * c.DH) * c.OL
        if kind == "CSA":
            f["dense_mx"] += 2 * c.QL * c.IDX_H * c.IDX_D
            # Compressors and the indexer weight projection; the cost model writes the
            # compressed KV width as 512 (Pro's DH).
            f["small_bf16"] += (
                2 * 2 * c.H * 2 * c.DH + 2 * c.H * c.IDX_H + 2 * 2 * c.H * 2 * c.IDX_D
            )
        elif kind == "HCA":
            f["small_bf16"] += 2 * 2 * c.H * c.DH
    if moe:
        f["shared_mx"] = 6 * c.H * c.FFN_S
        f["routed_mx"] = c.TOPK * 6 * c.H * c.FFN_E
    else:
        f["shared_mx"] = 2 * c.MLP_MATS * c.H * c.FFN_D
        f["routed_mx"] = 0
    if kind == "ATT":
        keys = km["full"]
    else:
        keys = km["win"] + (km["topk"] if kind == "CSA" else km["hca"] if kind == "HCA" else 0)
    f["attn"] = keys * (4 * c.HEADS * c.DH)
    idx_pair = 2 * c.IDX_H * c.IDX_D
    f["idx"] = km["cand"] * idx_pair if kind == "CSA" else 0.0
    f["teacher"] = km["topk"] * (2 * c.HEADS * c.DH) if kind == "CSA" else 0.0
    f["idx_bwd"] = km["topk"] * (2 * idx_pair) if kind == "CSA" else 0.0
    NS = c.NS
    f["mhc"] = 2 * 2 * NS * c.H * (NS * NS + 2 * NS) if NS > 1 else 0
    return f


def _mx(hw: Any, prec: str) -> float:
    return hw.mxfp8_speedup / (1 + hw.mxfp8_quant) if prec == "mxfp8" else 1.0


def layer_time_ms(
    kind: str,
    km: Mapping[str, float],
    T: int,
    prec: str,
    mode: str,
    shape: ModelShape,
    ffn: str,
    hw: Any,
) -> dict:
    """Time (ms) of one layer pass for ``T`` tokens on one GPU by component
    (``dsv4.layer_time_ms``); ``mode``: ``sweep`` | ``fwdtrain`` | ``bwd``."""
    f = layer_flops(kind, km, shape.dims, ffn)
    P = hw.peak_flops
    mx = _mx(hw, prec)
    t = {}
    t["gemm"] = (
        T
        * (
            f["dense_mx"] / (P * hw.mfu_gemm * mx)
            + f["shared_mx"] / (P * hw.mfu_gemm * mx)
            + f["small_bf16"] / (P * hw.mfu_small)
        )
        * 1e3
    )
    t["moe_gemm"] = T * f["routed_mx"] / (P * hw.mfu_moe * mx) * 1e3
    t["attn"] = T * f["attn"] / (P * hw.mfu_sattn) * 1e3
    t["indexer"] = T * f["idx"] / (P * hw.mfu_idx) * 1e3
    t["bw"] = T * (shape.elem_bytes_per_token + shape.mhc_bytes_per_token) / hw.hbm_kernel_bw * 1e3
    t["teacher"] = 0.0
    if mode == "fwdtrain":
        t["teacher"] = (
            T * (f["teacher"] / (P * hw.mfu_sattn) + f["idx_bwd"] / (P * hw.mfu_idx)) * 1e3
        )
    if mode == "bwd":
        t = dict(
            gemm=2 * t["gemm"],
            moe_gemm=2 * t["moe_gemm"],
            attn=hw.attn_bwd_factor * t["attn"],
            indexer=0.0,
            bw=2 * t["bw"],
            teacher=0.0,
        )
    return t


def _role_time_ms(layers, km, T, prec, mode, shape, hw):
    tot = {}
    for kind, ffn in layers:
        for k, v in layer_time_ms(kind, km, T, prec, mode, shape, ffn, hw).items():
            tot[k] = tot.get(k, 0.0) + v
    return tot


def _train_gf_tok(layers, km, dims, fwd_only_once):
    tot = 0.0
    for kind, ffn in layers:
        for k, v in layer_flops(kind, km, dims, ffn).items():
            once = fwd_only_once and k in ("idx", "teacher", "idx_bwd")
            tot += (1 if once else 3) * v
    return tot / 1e9


def model_flops_per_sample(shape: ModelShape, ph: SeqPhase, depth: int, hw: Any = None) -> float:
    """Useful training FLOPs of one sample of ``depth`` rounds (``dsv4.model_flops_per_sample``;
    no recompute, no sweep or rebuild): 3 x the training-path forward, the forward-only indexer
    work once (``hw.mfu_fwd_only_x1``, the doc_s3 profile), plus the vocab heads (main + one per
    MTP depth) and the MTP projections."""
    once = True if hw is None else hw.mfu_fwd_only_x1
    km = ph.km_all
    d = shape.dims
    core = _train_gf_tok(shape.core, km, d, once)
    heads = 1 + len(shape.mtp)
    head_flop = 2 * d.H * d.V
    mtp_proj_flop = 2 * d.H * d.H * (1 + d.NS)
    edge = (
        _train_gf_tok(shape.prelude, km, d, once)
        + _train_gf_tok(shape.coda, km, d, once)
        + _train_gf_tok(shape.mtp, km, d, once)
        + 3 * (heads * head_flop + len(shape.mtp) * mtp_proj_flop) / 1e9
    )
    return ph.S * (depth * core + edge) * 1e9


# ----------------------------------------------------------------------------- unit costs
@dataclass(frozen=True)
class UnitCosts:
    """Per-EU task durations of one sample (ms): ``f`` (sweep round), ``b`` (core item), ``coda``,
    ``pre``, ``sweep`` (prelude forward); ``split``: per task ``gpu`` / ``ep`` / ``cp`` / ``lw`` /
    ``host`` / ``moe_imb`` (and ``c2c`` on ``f`` / ``b`` when charged); ``moe_share``: the
    routed-GEMM share of each task (the simulator's ``active_scale``)."""

    d: Mapping[str, float]
    split: Mapping[str, Mapping[str, float]]
    moe_share: Mapping[str, float]


def _ep_bytes_per_pass(T: int, EP: int, dims: DSv4Dims) -> float:
    """NVLink bytes per GPU per MoE layer per pass (dispatch bf16 + combine bf16), EP inside one
    NVLink domain (``dsv4.ep_bytes_per_pass``, ``domains = 1``)."""
    tok = dims.H * 2 + dims.H * 2
    return T * dims.TOPK * tok * (1 - 1 / (EP / 1)) if EP > 1 else 0.0


def unit_costs(
    shape: ModelShape,
    ph: SeqPhase,
    hw: Any,
    *,
    prec: str,
    EP: int,
    ep_overlap: str,
    graphs: str,
    nested: bool,
    lw_core: bool,
    lw_edges: bool = False,
    z_lw: int = 64,
    whole_round: bool = False,
    coda_rc: bool = False,
    emb_rs: bool = False,
    c2c_bytes: Optional[Mapping[str, float]] = None,
    nominal: bool = False,
    rebuild_fraction: float = 1.0,
    moe_imb: float = 1.0,
) -> UnitCosts:
    """Task durations per EU for one sample (``dsv4.costs``, EP inside one NVLink domain).

    ``ep_overlap``: ``raw`` or ``two_chunk`` (``max(0.5 raw, raw - 0.9 g) + kappa_EP x rest``);
    ``graphs``: ``full`` / ``partial`` / ``eager`` (host time per layer pass); ``nested``: the core
    item rebuilds the round first (``rebuild_fraction`` x a sweep; 1.0 = the cost model's 5
    units); ``whole_round``: the core item gathers each Layer-window layer twice (the regather
    block variant; no rebuild); ``lw_core`` / ``lw_edges``: Layer-window gathers of the core /
    prelude-coda-MTP dense layers over ``z_lw`` GPUs (move E); ``coda_rc``: D42 (the coda item
    re-runs the up projection); ``emb_rs``: move M (fp32 head / embedding grad reduce-scatter per
    coda / prelude item); ``c2c_bytes`` (``f`` / ``b``): Grace C2C bytes per sample-round of
    round-state data on Grace, charged as a hidden transfer; ``nominal``: GPU compute only."""
    kap = hw.kappa
    kap_ep = hw.kappa_ep
    if ep_overlap not in ("raw", "two_chunk"):
        raise ValueError(f"unknown EP overlap {ep_overlap!r}")
    floor = hw.two_chunk_exposed if ep_overlap == "two_chunk" else None
    T = ph.T
    dims = shape.dims
    role_layers = {"pre": shape.prelude, "core": shape.core, "coda": shape.coda, "mtp": shape.mtp}

    def roles(mode):
        return [
            {
                role: _role_time_ms(layers, km, T, prec, mode, shape, hw)
                for role, layers in role_layers.items()
            }
            for km in ph.km_rank
        ]

    R = {m: roles(m) for m in ("sweep", "fwdtrain", "bwd")}
    ncp = len(ph.km_rank)

    def tsum(d):
        return sum(d.values())

    def role_ms(mode, role, worst=True):
        vals = [tsum(R[mode][r][role]) for r in range(ncp)]
        return max(vals) if worst else sum(vals) / ncp

    def comp(mode, role):
        out = {}
        for r in range(ncp):
            for k, v in R[mode][r][role].items():
                out[k] = out.get(k, 0.0) + v / ncp
        return out

    def moe(role, mode):
        # An empty role (no prelude, no MTP layer) has no components.
        return comp(mode, role).get("moe_gemm", 0.0)

    g = {}
    imb = {}
    for role in ("pre", "core", "coda", "mtp"):
        for mode in ("sweep", "fwdtrain", "bwd"):
            g[(role, mode)] = role_ms(mode, role, worst=False)
            imb[(role, mode)] = role_ms(mode, role, worst=True) - g[(role, mode)]
    head_f = T * (2 * dims.H * dims.V) / (hw.peak_flops * hw.mfu_gemm) * 1e3
    mtp_proj_f = T * (2 * dims.H * dims.H * (1 + dims.NS)) / (hw.peak_flops * hw.mfu_gemm) * 1e3

    nvl_b = _ep_bytes_per_pass(T, EP, dims)
    ep_raw = max(nvl_b / hw.nvl_bytes_per_s, 0.0 / hw.ib_bytes_per_s) * 1e3
    cpb = {"CSA": 0.0, "HCA": 0.0, "W": 0.0, "ATT": 0.0}
    if ph.cp > 1:
        frac = (ph.cp - 1) / ph.cp
        # Compressed KV + indexer K all-gathers and the window halo (512 / 128 in the cost
        # model: Pro's DH / WIN).
        halo = dims.WIN * dims.NS * dims.H * 2
        cpb["CSA"] = (ph.S / 4 * dims.DH * 2 + ph.S / 4 * dims.IDX_D * 2) * frac + halo
        cpb["HCA"] = (ph.S / 128 * dims.DH * 2) * frac + halo
        cpb["W"] = halo
        cpb["ATT"] = ph.S * 2 * dims.KVH * dims.DH * 2 * frac  # K and V all-gather (est.)
    pl = shape.params.dense_by_kind
    zf = (z_lw - 1) / z_lw

    def lw_ag(k):
        return 2 * pl[k] * zf / hw.nvl_bytes_per_s * 1e3

    def lw_rs(k):
        return 4 * pl[k] * zf / hw.nvl_bytes_per_s * 1e3

    def pass_overhead(kinds, n_ep_pairs, lw_ag_n, lw_rs_n, gpu_ms, lw_on, lw_extra=0.0):
        L = len(kinds)
        gl = gpu_ms / L / max(1, n_ep_pairs) if L else 0.0
        if floor is not None:
            exp1 = max(floor * ep_raw, ep_raw - 0.9 * gl)
            ep_exp = exp1 + kap_ep * (ep_raw - exp1)
        else:
            ep_exp = ep_raw
        moe_layers = sum(1 for _, ffn in kinds if ffn == "moe")
        ep_ms = moe_layers * n_ep_pairs * ep_exp
        cp_ms = sum(cpb[k] for k, _ in kinds) / hw.nvl_bytes_per_s * 1e3 * n_ep_pairs
        lw_ms = 0.0
        if lw_on:
            c = sum(lw_ag_n * lw_ag(k) + lw_rs_n * lw_rs(k) for k, _ in kinds)
            if lw_extra:
                c += lw_extra
            nvl_ep = moe_layers * n_ep_pairs * nvl_b / hw.nvl_bytes_per_s * 1e3
            exposed = max(0.0, c + nvl_ep - gpu_ms)
            lw_ms = exposed + kap * (c - min(c, exposed))
        return ep_ms, cp_ms, lw_ms

    def host_ms(nlayers, passes_f, passes_b, gpu_ms):
        if graphs == "full":
            hf = hb = hw.host_full_ms
        elif graphs == "partial":
            hf, hb = hw.host_partial_fwd_ms, hw.host_partial_bwd_ms
        elif graphs == "eager":
            hf, hb = hw.host_eager_fwd_ms, hw.host_eager_bwd_ms
        else:
            raise ValueError(f"unknown graph mode {graphs!r}")
        host = nlayers * (hf * passes_f + hb * passes_b)
        return max(0.0, host - gpu_ms) + hw.host_item_ms

    def task(
        role_modes, extra_gpu=0.0, lw_on=False, ag_n=0, rs_n=0, n_pairs=1, pf=1, pb=0, lw_extra=0.0
    ):
        gpu = extra_gpu + sum(g[(r, m)] for r, m in role_modes)
        im = sum(imb[(r, m)] for r, m in role_modes)
        kinds = []
        roles_seen = []
        for r, _ in role_modes:
            if r not in roles_seen:
                roles_seen.append(r)
        for r in roles_seen:
            kinds += list(role_layers[r])
        if nominal:
            return gpu, dict(gpu=gpu, ep=0, cp=0, lw=0, host=0, moe_imb=0)
        ep_ms, cp_ms, lw_ms = pass_overhead(kinds, n_pairs, ag_n, rs_n, gpu, lw_on, lw_extra)
        h = host_ms(len(kinds), pf, pb, gpu + ep_ms)
        mi = (moe_imb - 1.0) * sum(moe(r, m) for r, m in role_modes)
        return (
            gpu + im + ep_ms + cp_ms + lw_ms + h + mi,
            dict(gpu=gpu, ep=ep_ms, cp=cp_ms + im, lw=lw_ms, host=h, moe_imb=mi),
        )

    f, sf = task([("core", "sweep")], lw_on=lw_core, ag_n=1, rs_n=0, n_pairs=1, pf=1)
    rebuild = rebuild_fraction if (nested and not whole_round) else 0.0
    b0, sb0 = task(
        [("core", "fwdtrain"), ("core", "bwd")],
        lw_on=lw_core,
        ag_n=2 if whole_round else 1,
        rs_n=1,
        n_pairs=2,
        pf=1,
        pb=1,
    )
    if rebuild > 0:
        br, sbr = task([("core", "sweep")], lw_on=lw_core, ag_n=1, rs_n=0, n_pairs=1, pf=1)
        b = b0 + rebuild * br
        sb = {k: sb0[k] + rebuild * sbr[k] for k in sb0}
    else:
        b, sb = b0, sb0
    coda_extra = 3 * ((1 + len(shape.mtp)) * head_f + len(shape.mtp) * mtp_proj_f)
    if coda_rc:
        mx_ = _mx(hw, prec)
        coda_extra += (
            shape.coda_dsv4_attentions
            * T
            * (
                2 * dims.QL * dims.HEADS * dims.DH / (hw.peak_flops * hw.mfu_gemm * mx_)
                + shape.elem_bytes_per_token / hw.hbm_kernel_bw
            )
            * 1e3
        )
    if emb_rs:
        e_rs = 4 * dims.V * dims.H * zf / hw.nvl_bytes_per_s * 1e3
        ekw = dict(lw_on=True, ag_n=1 if lw_edges else 0, rs_n=1 if lw_edges else 0, lw_extra=e_rs)
    else:
        ekw = dict(lw_on=lw_edges, ag_n=1, rs_n=1)
    coda, sc = task(
        [("coda", "fwdtrain"), ("coda", "bwd"), ("mtp", "fwdtrain"), ("mtp", "bwd")],
        extra_gpu=coda_extra,
        n_pairs=2,
        pf=1,
        pb=1,
        **ekw,
    )
    pre, sp = task([("pre", "fwdtrain"), ("pre", "bwd")], n_pairs=2, pf=1, pb=1, **ekw)
    sweep, ss = task([("pre", "sweep")], lw_on=lw_edges, ag_n=1, rs_n=0, n_pairs=1, pf=1)
    if c2c_bytes:
        for key in ("f", "b"):
            sp_ = sf if key == "f" else sb
            t = c2c_bytes[key] / hw.c2c_bytes_per_s * 1e3
            ex = max(0.0, t - sp_["gpu"])
            ch = ex + kap * (t - ex)
            sp_["c2c"] = ch
            if key == "f":
                f += ch
            else:
                b += ch
    d = dict(f=f, b=b, coda=coda, pre=pre, sweep=sweep)
    split = dict(f=sf, b=sb, coda=sc, pre=sp, sweep=ss)

    def share(part, whole):
        return part / whole if whole else 0.0

    moe_b = moe("core", "fwdtrain") + moe("core", "bwd") + rebuild * moe("core", "sweep")
    moe_share = dict(
        f=share(moe("core", "sweep"), f),
        b=share(moe_b, b),
        coda=share(sum(moe(r, m) for r in ("coda", "mtp") for m in ("fwdtrain", "bwd")), coda),
        pre=share(moe("pre", "fwdtrain") + moe("pre", "bwd"), pre),
    )
    return UnitCosts(d=d, split=split, moe_share=moe_share)


# ----------------------------------------------------------------------------- memory
def memory_components(
    shape: ModelShape,
    ph: SeqPhase,
    hw: Any,
    peak_units: float,
    *,
    EP: int,
    N: int,
    prec: str,
    layout: str,
    nested: bool,
    block_stride: Optional[int] = None,
    block_window: bool = False,
    coda_rc: bool = False,
    adam_on_grace: bool = False,
    edges_mode: str = "Resident",
    emb_mode: str = "Resident",
    z_lw: int = 64,
    store_terms: Any = None,
) -> Tuple[Dict[str, float], float]:
    """Per-GPU HBM components (bytes, ``dsv4.memory`` names and order) and the Grace bytes of
    the Adam state when ``adam_on_grace``.

    ``layout``: ``layer_window`` (the cost model: core dense in the Layer window over ``z_lw``
    GPUs; ``edges_mode`` / ``emb_mode`` per the moves E / M) or ``resident`` (every dense group
    Resident, no shard buffers: ``proxy_v2.core_resident``). ``nested``: the nested arena
    (``L_core`` layer inputs + one layer); ``block_stride`` ``s``: ``s`` layers + the block input
    (``s = L_core`` is the whole round), the stash is placed by the caller; ``block_window``: the
    block's gathered weights stay resident (``layer_window`` only).

    ``store_terms``: the loop parameter store's runtime terms of this option
    (``StoreMemoryTerms``; None: the cost model's terms above). Then the dense groups follow the
    store exactly: per group the stored elements by the store's mode (Resident: compute copy +
    fp32 accumulator; Resident + sharded grads: compute copy; Layer window: nothing) plus the
    canonical shards and their fp32 gradients of **every** stored group, the rest of the group
    replicated (weight + fp32 gradient); the window slots and fp32 scratch pool as the store
    sized them (``MEM_DENSE_BUF``, no separate block window); the distributed optimizer over the
    replicate group (``N / z`` ranks: stored shards and replicated parameters), experts over
    ``N``; the coda item arena adds the head and sharded-grad transient targets and the
    hook-class embedding ``.grad``, the prelude item arena (not modelled otherwise) the same of
    the embed segment; the FP8 cast term covers the window units an item keeps; TE's fused-wgrad
    dummies are a persistent term; a store with copy engines adds their symmetric buffers
    (``MEM_COPY_ENGINE``, only when above 0: NCCL stores keep the same components)."""
    if layout not in ("layer_window", "resident"):
        raise ValueError(f"unknown layout {layout!r}")
    if layout == "resident":
        edges_mode = emb_mode = "Resident"
    params = shape.params
    groups = params.groups()
    P_e = params.routed_experts
    T = ph.T
    act_frag = hw.act_frag
    r = {}
    r[MEM_EXPERT_W] = 2 * P_e / EP
    r[MEM_EXPERT_G] = shape.expert_grad_bytes * P_e / EP
    pl_max = params.largest_dense_layer
    P_d = sum(groups.values())
    if store_terms is not None:
        r[MEM_DENSE_RES], r[MEM_DENSE_SH] = _store_dense_bytes(groups, store_terms)
        r[MEM_DENSE_BUF] = float(store_terms.buffer_bytes)
        z = store_terms.shard_axis
        stored = sum(store_terms.group_numel.values())
        shards = sum(store_terms.group_shard_numel.values())
        if shape.distributed_optimizer:
            per_gpu = ((shards + (P_d - stored)) * z + P_e) / N
        else:
            per_gpu = shards + (P_d - stored) + P_e / EP
        opt = shape.optimizer_bytes_per_param * per_gpu
        opt_grace = shape.grace_optimizer_bytes_per_param * per_gpu
    else:
        dense_res = 0.0
        dense_sh = 0.0
        modes = {
            "emb+head": emb_mode,
            "prelude dense": edges_mode,
            "coda dense": edges_mode,
            "MTP dense": edges_mode,
            "core dense": "Layer window",
        }
        any_sharded = False
        for k, P in groups.items():
            m = modes[k]
            if m == "Resident":
                dense_res += 6 * P
            elif m == "Resident + sharded grads":
                dense_res += 2 * P
                dense_sh += 6 * P / z_lw
                any_sharded = True
            else:
                dense_sh += 6 * P / z_lw
                any_sharded = True
        r[MEM_DENSE_RES] = dense_res
        r[MEM_DENSE_SH] = dense_sh
        r[MEM_DENSE_BUF] = 12 * pl_max if any_sharded else 0.0
        if layout == "resident":
            r[MEM_DENSE_RES] += 6 * groups["core dense"]
            r[MEM_DENSE_SH] = 0.0
            r[MEM_DENSE_BUF] = 0.0
        if shape.distributed_optimizer:
            opt = shape.optimizer_bytes_per_param * (P_d + P_e) / N
            opt_grace = shape.grace_optimizer_bytes_per_param * (P_d + P_e) / N
        else:
            opt = shape.optimizer_bytes_per_param * (P_d + P_e / EP)
            opt_grace = shape.grace_optimizer_bytes_per_param * (P_d + P_e / EP)
    r[MEM_OPT] = 0.0 if adam_on_grace else opt
    act_layer = shape.act_bytes_per_token_layer * T * act_frag
    state_tok = shape.state_bytes_per_token
    stash = shape.core_layers * T * state_tok if nested else 0.0
    core_item = stash + act_layer
    ce = (
        (2 * T * shape.dims.V * 2 / 8)
        if shape.ce_bytes_per_token is None
        else T * shape.ce_bytes_per_token
    )
    a_mtp = shape.act_bytes_per_token_mtp if shape.mtp else 0.0
    coda_item = (
        len(shape.coda) * shape.act_bytes_per_token_layer + len(shape.mtp) * a_mtp
    ) * T * act_frag + ce
    if coda_rc:
        coda_item -= shape.coda_dsv4_attentions * shape.mla_up_bytes_per_token * T * act_frag
    prelude_item = 0.0
    if store_terms is not None:
        coda_item += (
            store_terms.head_scratch_coda
            + store_terms.rsg_transient_coda
            + store_terms.hook_grad_coda
        )
        prelude_item = store_terms.rsg_transient_prelude + store_terms.hook_grad_prelude
    if block_stride is not None:
        if nested:
            raise ValueError("block checkpoint: pass nested=False")
        core_item = block_stride * act_layer + T * state_tok
        r[MEM_ACT_BLOCK] = max(core_item, coda_item, prelude_item)
    else:
        r[MEM_ACT_NESTED] = max(core_item, coda_item, prelude_item)
    if (
        store_terms is None
        and block_stride is not None
        and block_window
        and layout == "layer_window"
    ):
        r[MEM_WINDOW] = (block_stride - 1) * 2 * pl_max
    r[MEM_INDEXER] = T * ph.S * 4 if shape.has_csa else 0.0
    if prec == "mxfp8":
        cast = pl_max
        if store_terms is not None:
            cast = max(1, store_terms.cast_units_kept) * (
                store_terms.unit_bytes_max / store_terms.elem_bytes
            )
        r[MEM_MXFP8] = 2.06 * (cast + (shape.dims.NEXP // EP if EP else 0) * params.expert_params)
    if store_terms is not None and store_terms.te_dummy_wgrad:
        r[MEM_TE_DUMMY] = float(store_terms.te_dummy_wgrad)
    copy_engine = 0 if store_terms is None else getattr(store_terms, "copy_engine_bytes", 0)
    if copy_engine > 0:
        r[MEM_COPY_ENGINE] = float(copy_engine)
    st = peak_units * ph.state_gpu
    r[MEM_GRAPH_IO] = 6 * ph.state_gpu
    r[MEM_STORE] = st
    r[MEM_RESERVE] = hw.ep_reserve_bytes
    r[MEM_MISC] = hw.misc_bytes
    return r, (opt_grace if adam_on_grace else 0.0)


# The planner's dense groups (``ParamGroups.groups``) -> the loop parameter store's groups.
STORE_GROUP_OF = {
    "emb+head": "emb_head",
    "prelude dense": "prelude",
    "core dense": "core",
    "coda dense": "coda",
    "MTP dense": "mtp",
}


def _store_dense_bytes(groups: Mapping[str, float], terms: Any) -> Tuple[float, float]:
    """``(MEM_DENSE_RES, MEM_DENSE_SH)`` of the dense groups under a parameter store: the
    replicated part of each group (weight + fp32 gradient), the stored part by its mode (copy +
    accumulator / copy / nothing), and the canonical shards + fp32 shard gradients."""
    e = terms.elem_bytes
    resident = shards = 0.0
    for name, P in groups.items():
        group = STORE_GROUP_OF[name]
        resident += (e + 4) * (P - terms.group_numel.get(group, 0))
        shards += (e + 4) * terms.group_shard_numel.get(group, 0)
        copy = terms.group_copy_numel.get(group, 0)
        mode = terms.group_modes.get(group, "resident")
        if mode == "resident":
            resident += (e + 4) * copy
        elif mode == "resident_sharded_grads":
            resident += e * copy
    return resident, shards


@dataclass(frozen=True)
class Placement:
    """Where round-state data goes (``dsv4.evaluate_moves.place``, ``unified``): the state store
    plus the block-input stash (``spill``) take the free HBM under the cap first, then Grace."""

    store: float
    need: float
    spill: float
    in_hbm: float
    on_grace: float
    share: float
    hbm: float
    grace: float
    adam_grace: float
    room: float
    mem_ok: bool
    deficit: float


def place_round_state(
    components: Mapping[str, float], adam_grace: float, *, stash_need: float, hw: Any
) -> Placement:
    """Place the state store (``components[MEM_STORE]``) and ``stash_need`` bytes of block
    inputs: HBM first up to ``hw.hbm_cap_bytes``, the rest on Grace after the Adam state.
    ``mem_ok``: HBM within the cap and Grace within ``hw.grace_bytes``."""
    st = components[MEM_STORE]
    spill = stash_need + st
    base_h = sum(components.values()) - st
    room = hw.hbm_bytes - hw.planner_reserve_bytes - base_h
    in_h = min(spill, max(0.0, room))
    on_g = spill - in_h
    share = on_g / spill if spill else 0.0
    grace = adam_grace + on_g
    return Placement(
        store=st,
        need=stash_need,
        spill=spill,
        in_hbm=in_h,
        on_grace=on_g,
        share=share,
        hbm=base_h + in_h,
        grace=grace,
        adam_grace=adam_grace,
        room=room,
        mem_ok=room >= 0 and grace <= hw.grace_bytes,
        deficit=max(0.0, -room, grace - hw.grace_bytes),
    )


def c2c_traffic(
    share_stash: float, share_state: float, stash_units: int, state: float
) -> Tuple[float, float, float]:
    """Grace C2C bytes per sample-round of the Grace share of round-state data
    (``evaluate_moves.traffic``): the sweep writes the stash units and the round input; the core
    item reads them and the chain record, and writes the chain record. ``(write, read,
    item write)``."""
    fw = (share_stash * stash_units + share_state) * state
    br = (share_stash * stash_units + share_state * (1 + CHAIN_UNITS)) * state
    bw = share_state * CHAIN_UNITS * state
    return fw, br, bw


# ----------------------------------------------------------------------------- step level
def step_level_ms(
    shape: ModelShape, hw: Any, *, EP: int, N: int, adam_on_grace: bool, dense_axis: int = 64
) -> Dict[str, float]:
    """Step-level costs (ms; ``proxy_v2._step_level_z``, ``dsv4.step_level_ms`` at
    ``dense_axis = 64``): expert grad RS (fp32) + param AG (bf16) over EDP on IB when ``N >
    EP``, the dense shard sync over the replicate group when ``N > dense_axis``, and the
    optimizer step on Grace (65% of the LPDDR5X peak + the C2C traffic) or in HBM."""
    groups = shape.params.groups()
    P_e = shape.params.routed_experts
    P_d = sum(groups.values())
    out = {}
    edp = N // EP
    if edp > 1:
        f = (edp - 1) / edp
        rs = 4 * P_e / EP * f / hw.ib_bytes_per_s * 1e3
        ag = 2 * P_e / EP * f / hw.ib_bytes_per_s * 1e3
        out["expert grad RS (fp32) + param AG (bf16) over EDP, IB"] = rs + hw.edp_ag_factor * ag
    rep = N // dense_axis
    if rep > 1:
        f = (rep - 1) / rep
        out["dense shard RS + AG over replicate group, IB"] = (
            (4 + 2) * P_d / dense_axis * f / hw.ib_bytes_per_s * 1e3 * hw.dense_rep_factor
        )
    Pg = (P_d + P_e) / N
    if adam_on_grace:
        grace_bw = hw.grace_opt_eff * hw.lpddr_peak_per_gpu
        out["optimizer step on Grace"] = (28 * Pg / grace_bw + 6 * Pg / hw.c2c_bytes_per_s) * 1e3
    else:
        out["optimizer step in HBM"] = 30 * Pg / hw.hbm_opt_bw * 1e3
    return out


def resident_sync_ms(shape: ModelShape, hw: Any, moves: str, group: int, layout: str) -> float:
    """In-domain step-level sync of the Resident dense groups over ``group`` GPUs on NVLink
    (``pp_core_baseline.rsbf_resident_sync_ms``): fp32 RS + bf16 AG (6 bytes per parameter) of
    the embedding / head (bf16 AG only, 2 bytes, with move M) and of the prelude / coda / MTP
    dense groups (unless move E); in the ``resident`` layout the core dense group too (est.)."""
    g = shape.params
    edges = g.prelude_dense + g.coda_dense + g.mtp_dense
    gb = [(2 if "M" in moves else 6, g.emb_head)]
    if "E" not in moves:
        gb.append((6, edges))
    if layout == "resident":
        gb.append((6, g.core_dense))
    return sum(b * p for b, p in gb) * (group - 1) / group / hw.nvl_bytes_per_s * 1e3
