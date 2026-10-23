# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The loop-metric reporting rule (``megatron/core/transformer/loop/reporting.py``).

Every loop metric is reported as ``[value summed over its population, population size]``:
token-layer counts are already microbatch totals, every other loop metric is a mean over its
population (issued tokens, or the valid tokens a MoR decoder names in
``loop_metric_populations``). Both entry scripts' loss functions report by it, so
``train_step``'s ``sum / weight`` gives the per-token mean in the eager and the RSBF schedule
(the stage-2 finding: eager logged ``loop active token layers`` per microbatch, RSBF per
token). Pure host tests: no process group, no model.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.loop.reporting import (
    LOOP_TOKEN_COUNT_METRICS,
    loop_metric_entry,
    loop_metric_report,
)

SEQ, BATCH, CORE = 8, 2, 4


def test_token_count_metrics_are_the_token_layer_counts():
    assert LOOP_TOKEN_COUNT_METRICS == {"loop active token layers", "loop backward token layers"}


@pytest.mark.parametrize("population", [16, torch.tensor(16)], ids=["int", "tensor"])
def test_mean_metric_entry_is_value_times_population(population):
    metric = torch.tensor(2.5, requires_grad=True)
    entry = loop_metric_entry("loop depth", metric * 1, population=population)
    assert entry.dtype == torch.float32 and not entry.requires_grad
    assert entry.tolist() == [40.0, 16.0]


@pytest.mark.parametrize("name", sorted(LOOP_TOKEN_COUNT_METRICS))
def test_token_count_entry_is_the_total(name):
    total = torch.tensor(float(BATCH * SEQ * 3 * CORE))
    entry = loop_metric_entry(name, total, population=BATCH * SEQ)
    assert entry.tolist() == [BATCH * SEQ * 3 * CORE, BATCH * SEQ]
    # train_step logs sum / weight: rounds x core sublayers per token.
    assert (entry[0] / entry[1]).item() == 3 * CORE


def test_report_uses_the_named_populations_and_issued_tokens_otherwise():
    metrics = {
        "loop depth": torch.tensor(2.0),
        "depth fraction 1": torch.tensor(0.25),
        "loop active token layers": torch.tensor(96.0),
        "loop forward flops per token": torch.tensor(1.5e6),
    }
    valid = torch.tensor(12)
    report = loop_metric_report(
        metrics,
        issued_tokens=BATCH * SEQ,
        populations={"loop depth": valid, "depth fraction 1": valid},
    )
    assert list(report) == list(metrics)
    assert report["loop depth"].tolist() == [24.0, 12.0]
    assert report["depth fraction 1"].tolist() == [3.0, 12.0]
    assert report["loop active token layers"].tolist() == [96.0, 16.0]
    assert report["loop forward flops per token"].tolist() == [1.5e6 * 16, 16.0]
    assert loop_metric_report(None, issued_tokens=4) == {}


def _reduce(reports):
    """``train_step``'s per-key reduction (one rank): Σ value / Σ weight."""
    reduced = {}
    for key in reports[0]:
        total = torch.vstack([report[key].view(-1) for report in reports]).sum(0)
        reduced[key] = (total[0] / total[1]).item()
    return reduced


def test_eager_microbatch_and_rsbf_sample_entries_reduce_alike():
    """The stage-2 open finding's numbers (S 4096, K 8): an eager microbatch of two rows at
    depths 2 and 3 and the two RSBF per-sample entries reduce to E[d] K token layers per token
    and E[d] rounds."""
    seq, core, depths = 4096, 8, (2, 3)
    eager = loop_metric_report(
        {
            "loop depth": torch.tensor(sum(depths) / len(depths)),
            "loop active token layers": torch.tensor(float(sum(depths) * seq * core)),
        },
        issued_tokens=len(depths) * seq,
    )
    rsbf = [
        {
            "loop depth": torch.tensor([float(d * seq), float(seq)]),
            "loop active token layers": torch.tensor([float(seq * d * core), float(seq)]),
        }
        for d in depths
    ]
    assert (
        _reduce([eager])
        == _reduce(rsbf)
        == {"loop depth": 2.5, "loop active token layers": 2.5 * core}
    )


@pytest.fixture(params=["pretrain_gpt", "pretrain_hybrid"])
def entry(request, monkeypatch):
    """An entry script's module with stub args (default loss branch, no checks)."""
    module = __import__(request.param)
    monkeypatch.setattr(
        module,
        "get_args",
        lambda: SimpleNamespace(
            logits_load_dir=None,
            check_for_nan_in_loss_and_grad=False,
            check_for_spiky_loss=False,
            modelopt_enabled=False,
        ),
    )
    monkeypatch.setattr(module, "has_nvidia_modelopt", False, raising=False)
    return module


def test_loss_funcs_report_loop_metrics_by_the_rule(entry):
    """Loop losses stay weighted by the loss-token count; loop metrics follow the rule with
    the issued tokens (the per-token loss tensor's numel) or the named population."""
    loss_mask = torch.ones(BATCH, SEQ)
    loss_mask[1, -4:] = 0  # 12 loss tokens of 16 issued
    output = torch.full((BATCH, SEQ), 0.5)
    balance = torch.tensor(0.25, requires_grad=True)
    valid = torch.tensor(14)
    loss, num_tokens, report = entry.loss_func(
        loss_mask,
        output,
        loop_losses={"depth balance loss": balance},
        loop_metrics={
            "loop depth": torch.tensor(3.0),
            "loop active token layers": torch.tensor(float(BATCH * SEQ * 3 * CORE)),
            "loop forward flops per token": torch.tensor(100.0),
        },
        loop_metric_populations={"loop depth": valid},
    )
    assert num_tokens == 12
    assert loss.item() == pytest.approx(12 * 0.5 + 12 * 0.25)
    assert report["depth balance loss"].tolist() == [3.0, 12]
    assert report["loop depth"].tolist() == [42.0, 14.0]
    assert report["loop active token layers"].tolist() == [BATCH * SEQ * 3 * CORE, 16.0]
    assert report["loop forward flops per token"].tolist() == [1600.0, 16.0]
    reduced = _reduce([report])
    assert reduced["loop active token layers"] == 3 * CORE
    assert reduced["loop forward flops per token"] == 100.0


def test_forward_steps_pass_the_decoder_populations():
    """Both ``forward_step`` closures hand the decoder's ``loop_metric_populations`` to the
    loss function, next to its ``loop_metrics`` (source check: the forward steps need a
    model and the training globals)."""
    import inspect

    import pretrain_gpt
    import pretrain_hybrid

    for module in (pretrain_gpt, pretrain_hybrid):
        source = inspect.getsource(module.forward_step)
        assert 'loop_metrics=getattr(decoder, "loop_metrics", None)' in source
        assert (
            'loop_metric_populations=getattr(decoder, "loop_metric_populations", None)' in source
        ), module.__name__
