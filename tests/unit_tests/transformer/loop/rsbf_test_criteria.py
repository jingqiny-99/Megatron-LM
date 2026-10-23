# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Test-only exit criteria for the hybrid loop program (not collected by pytest).

``OnlineHaltingCriterion`` is shaped like the plan's learned halting head: a fp32 contraction
of the mHC streams over the non-padding tokens, a carried criterion state, a per-round
auxiliary loss and a differentiable ``init_state``. M1 ships no production online criterion;
this one exercises the online path, criterion-state checkpointing and its gradients.
``PaddingDepthCriterion`` is an online criterion with depths prescribed through the batch.

``retire_late`` / ``retire_early`` build buggy host plans for the executor's exit-enforcement
tests.
"""

import torch
from torch import nn

from megatron.core.transformer.loop.program import CriterionStep, ExitCriterion


class HaltingHead(nn.Module):
    """The halting criterion's parameters: per-stream weights ``w`` (rounds) and ``v``
    (``init_state``), ``[n, h]`` fp32, seeded on CPU."""

    def __init__(self, num_streams, hidden, *, device, seed=0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.w = nn.Parameter((torch.randn(num_streams, hidden, generator=generator)).to(device))
        self.v = nn.Parameter((torch.randn(num_streams, hidden, generator=generator)).to(device))


def stream_contraction(x, weight, valid):
    """``sum_j w_j . mean_valid(x[:, :, j])`` per row in fp32: ``x`` ``[S, c, n * h]``,
    ``weight`` ``[n, h]``, ``valid`` ``[c, S]`` bool or None; returns ``[c]``."""
    sequence, rows = x.shape[:2]
    streams = x.float().reshape(sequence, rows, *weight.shape)
    if valid is None:
        mean = streams.mean(0)
    else:
        mask = valid.t().float()[:, :, None, None]
        mean = (streams * mask).sum(0) / mask.sum(0).clamp_min(1.0)
    return (mean * weight.float()).sum((-2, -1))


class OnlineHaltingCriterion(ExitCriterion):
    """Halts when the accumulated ``sigmoid(h_r)`` reaches ``tau`` (online: depths unknown).

    ``crit_r = crit_{r-1} + sigmoid(h_r)``, ``h_r`` the stream contraction of ``s_r``;
    ``aux_loss = coeff * sum_rows sigmoid(h_r)`` (plus ``carry * sum(crit_{r-1})`` when
    ``carry`` is set, so ``dL/dcrit`` crosses rounds and reaches ``init_state``);
    ``crit_0 = scale * contraction(e, v)``.
    """

    state_width = 1
    differentiable = True

    def __init__(self, head, *, tau=1.5, coeff=0.01, scale=0.1, carry=0.0):
        self.head = head
        self.tau = tau
        self.coeff = coeff
        self.scale = scale
        self.carry = carry

    def init_state(self, injection, valid):
        """``scale * contraction(e, v)``, ``[c, 1]``."""
        return (self.scale * stream_contraction(injection, self.head.v, valid))[:, None]

    def step(self, state, crit_state, r, valid, depth):
        """Accumulate the halting probability; exit at ``tau``."""
        probability = torch.sigmoid(stream_contraction(state, self.head.w, valid))
        crit = crit_state + probability[:, None]
        aux_loss = self.coeff * probability.sum()
        if self.carry:
            aux_loss = aux_loss + self.carry * crit_state.sum()
        return CriterionStep(exit=crit[:, 0] >= self.tau, crit_state=crit, aux_loss=aux_loss)


class ConstantExitCriterion(ExitCriterion):
    """Exit bits always ``value`` (forcing tests: ``R_max`` and ``r_min`` are the program's)."""

    def __init__(self, value):
        self.value = value

    def step(self, state, crit_state, r, valid, depth):
        """A constant exit bit per row."""
        exit_bits = torch.full((state.shape[1],), self.value, dtype=torch.bool, device=state.device)
        return CriterionStep(exit=exit_bits, crit_state=crit_state)


class PaddingDepthCriterion(ExitCriterion):
    """An online criterion (no ``known_depths``) whose exit round is written into the batch: a
    row with ``p`` padding tokens exits after round ``p + 1``. ``crit_0`` holds that target and
    travels with the sample's checkpoints, so tests can prescribe online depths per sample and
    still have the host learn the exits only from the per-round exit bitmap."""

    state_width = 1

    def init_state(self, injection, valid):
        """``[c, 1]`` fp32: one plus the row's padding tokens (one without a padding mask)."""
        rows = injection.shape[1]
        target = torch.ones(rows, 1, dtype=torch.float32, device=injection.device)
        if valid is not None:
            target = target + (~valid).sum(1, keepdim=True).to(torch.float32)
        return target

    def step(self, state, crit_state, r, valid, depth):
        """Exit once ``r`` reaches the target; the state is carried unchanged."""
        return CriterionStep(exit=torch.ge(r, crit_state[:, 0]), crit_state=crit_state)


# ---------------------------------------------------------------------------
# Buggy host plans for the executor's exit-enforcement tests (monkeypatch
# ``LoopRSBFExecutor._newly_exited`` with one of these).
# ---------------------------------------------------------------------------


def retire_late(original, sample):
    """A host plan that keeps wave-local ``sample`` active one round past its exit (it is
    swept after its device exit bit is set)."""
    held = {}

    def newly_exited(self, active, r, *, depths, bits):
        newly = original(self, active, r, depths=depths, bits=bits)
        if sample in newly and sample not in held:
            held[sample] = r
            newly = [i for i in newly if i != sample]
        elif sample in active and held.get(sample) == r - 1:
            newly = sorted(newly + [sample])
        return newly

    return newly_exited


def retire_early(original, sample, at_round):
    """A host plan that retires wave-local ``sample`` after ``at_round``, before its exit."""

    def newly_exited(self, active, r, *, depths, bits):
        newly = original(self, active, r, depths=depths, bits=bits)
        if r == at_round and sample in active and sample not in newly:
            newly = sorted(newly + [sample])
        elif r > at_round:
            newly = [i for i in newly if i != sample]
        return newly

    return newly_exited
