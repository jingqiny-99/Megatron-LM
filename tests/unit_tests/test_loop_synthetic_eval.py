# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""The pure helpers of ``examples/loop/synthetic_eval.py`` (CPU; no process group).

The evaluation batch of a packed row equals the ``GPTDataset`` sample of the same row with SBHD
document packing field by field (tokens, labels, loss mask, positions, padding mask, causal
mask), so the evaluation forward sees exactly what training saw; rows split over data-parallel
ranks in equal forward counts with padding slots excluded from the scores; checkpoint and
evaluation-set discovery; the record format.
"""

import importlib.util
import json
import os
import types

import numpy
import pytest
import torch

from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.datasets.indexed_dataset import DType, IndexedDataset, IndexedDatasetBuilder
from megatron.core.datasets.utils import Split
from megatron.core.tokenizers import MegatronTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VOCAB = 512
EOD = VOCAB - 1
BOS, ANS = 1, 5


def _load_module():
    path = os.path.join(ROOT, "examples", "loop", "synthetic_eval.py")
    spec = importlib.util.spec_from_file_location("synthetic_eval_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


se = _load_module()


class _Args:
    reset_position_ids = False
    reset_attention_mask = False
    eod_mask_loss = False
    create_attention_mask_in_dataloader = False


def _problem(rng, body):
    """A synthetic-task-shaped document: BOS, task, body, ANS, answer, EOD."""
    tokens = [BOS, 6] + rng.integers(64, 128, size=body).tolist()
    answer = int(rng.integers(64, 128))
    return tokens + [ANS, answer, EOD]


def _eval_arrays(rows):
    """Evaluation arrays (the toolkit's eval_s*.npz fields) from rows of problems."""
    seq_length = max(sum(len(p) for p in row) for row in rows)
    tokens = numpy.zeros((len(rows), seq_length), dtype=numpy.int32)
    lengths, answer_row, answer_pos, answer = [], [], [], []
    for r, row in enumerate(rows):
        offset = 0
        for problem in row:
            tokens[r, offset : offset + len(problem)] = problem
            answer_row.append(r)
            answer_pos.append(offset + len(problem) - 2)
            answer.append(problem[-2])
            offset += len(problem)
        lengths.append(offset)
    return {
        "tokens": tokens,
        "row_length": numpy.asarray(lengths, dtype=numpy.int32),
        "answer_row": numpy.asarray(answer_row, dtype=numpy.int32),
        "answer_pos": numpy.asarray(answer_pos, dtype=numpy.int32),
        "answer": numpy.asarray(answer, dtype=numpy.int32),
        "steps": numpy.ones(len(answer), dtype=numpy.int32),
    }


@pytest.mark.parametrize("create_attention_mask", [False, True])
def test_microbatch_equals_the_packed_gpt_dataset_row(tmp_path, monkeypatch, create_attention_mask):
    """A packed training row (content <= S) rebuilt from its tokens and content length gives
    the same fields as ``GPTDataset.__getitem__`` (NullTokenizer, SBHD document packing)."""
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    seq_length = 64
    rng = numpy.random.default_rng(0)
    prefix = str(tmp_path / "train_text_document")
    builder = IndexedDatasetBuilder(prefix + ".bin", dtype=DType.optimal_dtype(VOCAB))
    for _ in range(60):
        document = _problem(rng, int(rng.integers(3, 20)))
        builder.add_document(document, [len(document)])
    builder.finalize(prefix + ".idx")
    tokenizer = MegatronTokenizer.from_pretrained(
        metadata_path={"library": "null-text"}, vocab_size=VOCAB
    )
    assert tokenizer.eod == EOD
    config = GPTDatasetConfig(
        random_seed=1234,
        sequence_length=seq_length,
        tokenizer=tokenizer,
        sbhd_document_packing=True,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        create_attention_mask=create_attention_mask,
        split="990,9,1",
        blend=([prefix], None),
        path_to_cache=str(tmp_path / "cache"),
    )
    dataset = GPTDataset(
        IndexedDataset(prefix, mmap=True),
        prefix,
        numpy.arange(60, dtype=numpy.int32),
        20,
        Split.train,
        config,
    )
    args = _Args()
    args.create_attention_mask_in_dataloader = create_attention_mask
    compared = 0
    for idx in range(len(dataset)):
        sample = dataset[idx]
        content = int((~sample["padding_mask"]).sum())
        if content >= seq_length:  # a full row's last label is the next row's token
            continue
        arrays = {
            "tokens": sample["tokens"].numpy()[None].astype(numpy.int32),
            "row_length": numpy.asarray([content], dtype=numpy.int32),
        }
        batch = se.build_microbatch(arrays, [0], args, EOD)
        for name, value in sample.items():
            assert torch.equal(batch[name][0], value.to(batch[name].dtype)), (idx, name)
            assert batch[name].dtype == value.dtype, name
        assert set(batch) == set(sample)
        # whole problems per row: the row's content starts with BOS and ends with EOD
        assert sample["tokens"][0] == BOS and sample["tokens"][content - 1] == EOD
        compared += 1
    assert compared >= 5


def test_padding_slots_and_scores():
    rng = numpy.random.default_rng(1)
    rows = [[_problem(rng, 4), _problem(rng, 2)], [_problem(rng, 5)], [_problem(rng, 1)]]
    arrays = _eval_arrays(rows)
    seq_length = arrays["tokens"].shape[1]
    # Three rows over two DP ranks at MBS 1: two forwards each, one padding slot.
    assert se.forwards_per_rank(3, 2, 1) == 2
    assert se.rank_rows(3, 0, 2, 1) == [0, 1] and se.rank_rows(3, 1, 2, 1) == [2, -1]
    assert se.rank_rows(3, 0, 2, 2) == [0, 1] and se.rank_rows(3, 1, 2, 2) == [2, -1]
    assert se.rank_rows(16, 3, 8, 1) == [6, 7] and se.rank_rows(1, 7, 8, 1) == [-1]
    batch = se.build_microbatch(arrays, [2, -1], _Args(), EOD)
    row0 = torch.from_numpy(arrays["tokens"][0].astype(numpy.int64))
    assert torch.equal(batch["tokens"][1], row0)  # a padding slot is row 0 ...
    assert batch["loss_mask"][1].sum() == 0 and batch["loss_mask"][0].sum() > 0  # ... unscored
    # Perfect predictions everywhere: every real problem counts once.
    predictions = numpy.zeros((2, seq_length), dtype=numpy.int64)
    for b, row in enumerate([0, 1]):
        for k in numpy.flatnonzero(arrays["answer_row"] == row):
            predictions[b, arrays["answer_pos"][k] - 1] = arrays["answer"][k]
    counts = [0, 0]
    se.score(arrays, [0, 1], predictions, counts)
    assert counts == [3, 3]
    # A prediction read at the answer position itself (off by one) scores nothing.
    shifted = numpy.roll(predictions, 1, axis=1)
    counts = [0, 0]
    se.score(arrays, [0, 1], shifted, counts)
    assert counts == [0, 3]
    # Padding slots are not scored even if they duplicate a real row.
    counts = [0, 0]
    se.score(arrays, [0, -1], predictions, counts)
    assert counts == [2, 2]


def test_load_eval_arrays_checks_and_truncates(tmp_path):
    rng = numpy.random.default_rng(2)
    arrays = _eval_arrays([[_problem(rng, 3)], [_problem(rng, 3), _problem(rng, 1)]])
    path = str(tmp_path / "eval_s1.npz")
    numpy.savez(path, **arrays)
    width = arrays["tokens"].shape[1]
    loaded = se.load_eval_arrays(path, width)
    assert all(numpy.array_equal(loaded[k], arrays[k]) for k in se.EVAL_FIELDS)
    one = se.load_eval_arrays(path, width, max_rows=1)
    assert one["tokens"].shape[0] == 1 and one["answer"].tolist() == arrays["answer"][:1].tolist()
    with pytest.raises(ValueError, match="--seq-length"):
        se.load_eval_arrays(path, width + 1)
    numpy.savez(path, tokens=arrays["tokens"])
    with pytest.raises(ValueError, match="missing fields"):
        se.load_eval_arrays(path, width)


def test_checkpoint_iterations(tmp_path):
    for name in ("iter_0000250", "iter_0000500", "iter_0000010", "other", "iter_12"):
        (tmp_path / name).mkdir()
    (tmp_path / "iter_0000750").write_text("not a directory")
    load = str(tmp_path)
    assert se.checkpoint_iterations(load) == [10, 250, 500]
    assert se.select_iterations("all", load) == [10, 250, 500]
    assert se.select_iterations("latest", load) == [500]
    assert se.select_iterations("500,250", load) == [500, 250]
    with pytest.raises(ValueError, match="no checkpoint for \\[300\\]"):
        se.select_iterations("250,300", load)
    with pytest.raises(ValueError, match="integers"):
        se.select_iterations("first", load)
    with pytest.raises(ValueError, match="no iter_"):
        se.select_iterations("all", str(tmp_path / "other"))


def _task_dir(root, task, steps_eval, steps_train=2):
    task_dir = root / task
    task_dir.mkdir()
    manifest = {
        "task": task,
        "steps_eval": [1, steps_eval],
        "steps_train": [1, steps_train],
        "chance_accuracy": 0.25,
    }
    if task == "phop":  # the manifest's per-step baseline and overlap (absent: None)
        manifest["splits"] = {
            "eval": {"per_steps": {"1": {"baseline_accuracy": 0.3, "train_overlap": 2}}}
        }
    (task_dir / "manifest.json").write_text(json.dumps(manifest))
    for s in range(1, steps_eval + 1):
        (task_dir / f"eval_s{s}.npz").write_bytes(b"")
    return task_dir


def test_discover_eval_sets(tmp_path):
    _task_dir(tmp_path, "vartrack", 2)
    phop = _task_dir(tmp_path, "phop", 3)
    (tmp_path / "phop" / "eval_s9.npz").write_bytes(b"")  # not listed by the manifest
    sets = se.discover_eval_sets(str(tmp_path))
    assert [(s.task, s.steps) for s in sets] == [
        ("phop", 1),
        ("phop", 2),
        ("phop", 3),
        ("vartrack", 1),
        ("vartrack", 2),
    ]
    assert sets[0].train_steps == (1, 2) and sets[0].chance == 0.25
    assert (sets[0].baseline, sets[0].train_overlap) == (0.3, 2)
    assert (sets[1].baseline, sets[1].train_overlap) == (None, None)
    assert (sets[3].baseline, sets[3].train_overlap) == (None, None)  # no splits in manifest
    assert [s.task for s in se.discover_eval_sets(str(tmp_path), ["vartrack"])] == ["vartrack"] * 2
    assert len(se.discover_eval_sets(str(phop))) == 3  # one task directory
    with pytest.raises(ValueError, match="no task directory for \\['arith'\\]"):
        se.discover_eval_sets(str(tmp_path), ["arith"])
    os.remove(phop / "eval_s2.npz")
    with pytest.raises(ValueError, match="eval_s2.npz is missing"):
        se.discover_eval_sets(str(tmp_path))


def test_record_and_depths():
    sets = [
        se.EvalSet("phop", 1, "a", (1, 2), 0.25, baseline=0.3),
        se.EvalSet("phop", 3, "b", (1, 2), 0.25),
        se.EvalSet("arith", 1, "c", (1, 8), None, baseline=0.015, train_overlap=1000),
    ]
    record = se.make_record(250, None, sets, [[3, 4], [1, 4], [0, 2]], load="L", data="D")
    assert record == {
        "format": "synth-eval-v1",
        "iteration": 250,
        "depth": None,
        "load": "L",
        "data": "D",
        "tasks": {
            "phop": {
                "train_steps": [1, 2],
                "chance": 0.25,
                "steps": {"1": [3, 4], "3": [1, 4]},
                "baseline": {"1": 0.3, "3": None},
                "train_overlap": {"1": None, "3": None},
            },
            "arith": {
                "train_steps": [1, 8],
                "chance": None,
                "steps": {"1": [0, 2]},
                "baseline": {"1": 0.015},
                "train_overlap": {"1": 1000},
            },
        },
    }
    json.dumps(record)

    class _Loop:
        def __init__(self, **overrides):
            values = dict(loop_untie_core=False, loop_architecture="huginn")
            values["loop_bptt_iterations"] = 0
            values.update(overrides)
            self.config = types.SimpleNamespace(**values)

        def set_loop_depths(self, depths):
            pass

    assert se.check_depths(None, object(), 4) == [None]
    assert se.check_depths([3, 1, 3], _Loop(), 4) == [None, 1, 3]
    assert se.check_depths([4, 1], _Loop(), 4) == [None, 1, 4]  # R itself is accepted
    with pytest.raises(ValueError, match="1..4"):
        se.check_depths([5], _Loop(), 4)
    with pytest.raises(ValueError, match="1..4"):
        se.check_depths([0], _Loop(), 4)
    with pytest.raises(ValueError, match="loop decoder"):
        se.check_depths([1], object(), 4)
    # refused before any checkpoint load (LoopHybridStack would raise at the first forward)
    with pytest.raises(ValueError, match="--loop-untie-core"):
        se.check_depths([1], _Loop(loop_untie_core=True), 4)
    with pytest.raises(ValueError, match="--loop-architecture huginn"):
        se.check_depths([1], _Loop(loop_architecture="ouro"), 4)
    with pytest.raises(ValueError, match="--loop-bptt-iterations 0"):
        se.check_depths([1], _Loop(loop_bptt_iterations=2), 4)
    assert se.check_depths(None, _Loop(loop_untie_core=True), 4) == [None]


def test_arguments():
    import argparse

    parser = se.add_synthetic_eval_args(argparse.ArgumentParser())
    args = parser.parse_args(["--synth-eval-data", "D", "--synth-eval-depths", "1", "4"])
    assert args.synth_eval_data == "D" and args.synth_eval_depths == [1, 4]
    assert args.synth_eval_iterations == "all" and args.synth_eval_output is None
