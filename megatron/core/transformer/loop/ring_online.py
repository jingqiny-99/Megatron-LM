# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Causal stop decisions on the fixed-shape, separate-edge GLM pipeline."""

import torch

from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator


class RingOnlineCommunicator(P2PCommunicator):
    """Each forward tensor and its adjoint carry one extra sequence row on the wire."""

    def _communicate(self, *, tensor_shape, **kwargs):
        if tensor_shape is not None:
            sequence, batch, hidden = tensor_shape
            tensor_shape = (sequence + 1, batch, hidden)
        return super()._communicate(tensor_shape=tensor_shape, **kwargs)


class RingOnlineExit:
    """One optimizer step. Only an executed gate on the last stage originates a decision."""

    def __init__(self, config, cp_group):
        self.config = config
        self.cp_group = cp_group
        self.stopped = {}
        self.events = {}

    def receive(self, sample, lap, wire):
        """Strip metadata before the model. Unknown depth is encoded as zero."""
        if lap > 1 and sample not in self.stopped:
            depth = int(wire[-1, 0, 0].item())
            if depth < 0 or depth >= lap:
                raise RuntimeError(f"Online ring received depth {depth} entering lap {lap}")
            if depth:
                self.stopped[sample] = depth
            elif lap == self.config.loop_iterations + 1:
                raise RuntimeError(f"Online ring coda received an undecided sample {sample}")
        return wire[:-1]

    def send(self, sample, state):
        # The metadata row has no gradient path into the model. Native backward transfers
        # an identically shaped tensor, with zero gradient for this row.
        header = state.new_full((1, *state.shape[1:]), self.stopped.get(sample, 0))
        return torch.cat((state, header), dim=0)

    def gate(self, sample, lap, state, *, padding_mask=None):
        from megatron.core.transformer.loop.online_exit import online_exit_decision

        if sample in self.stopped or (sample, lap) in self.events:
            raise RuntimeError(
                f"Re-entered a completed online ring gate for sample {sample}, lap {lap}"
            )
        if lap == self.config.loop_iterations:
            stop, score = True, None
        else:
            decision, value = online_exit_decision(
                self.config,
                state,
                lap,
                valid=None if padding_mask is None else ~padding_mask.bool(),
                cp_group=self.cp_group,
                return_score=True,
            )
            stop, score = bool(decision.item()), float(value.item())
        self.events[sample, lap] = stop, score
        if stop:
            self.stopped[sample] = lap
        return stop

    def finish(self, samples, *, owner):
        if self.stopped.keys() != set(range(samples)):
            raise RuntimeError("Online ring did not receive a terminal depth for every sample")
        depths = tuple(self.stopped[i] for i in range(samples))
        if owner:
            expected = {(i, r): r == d for i, d in enumerate(depths) for r in range(1, d + 1)}
            if {k: v[0] for k, v in self.events.items()} != expected:
                raise RuntimeError(
                    "Online ring terminal depths disagree with the executed gate decisions"
                )
        elif self.events:
            raise RuntimeError("Only the final pipeline stage can originate online ring decisions")
        return depths

    def records(self):
        return tuple((i, r, stop, score) for (i, r), (stop, score) in sorted(self.events.items()))
