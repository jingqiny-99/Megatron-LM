# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The whole-document SBHD row packer (``megatron/core/datasets/sbhd_packing.py``).

Invariants on synthetic document lengths: (P1) a document that fits in a row is in exactly one
row, whole; (P2) a longer document is its ``C``-token windows (one-token overlap with the extra
token) plus at most one remainder piece; (P3) every row holds 1..C tokens and at least one
piece; (P4) tokens are conserved; (P5) the packing is deterministic; (P6) a document of exactly
``C`` tokens fills a row alone; (P7) one open row is next fit. Hand-checked rows, ``split_at``,
the window bound of open rows. The C++ ``build_packed_rows`` equals the Python reference when
the dataset helpers are built (GPU nodes compile them; skipped otherwise).
"""

import importlib

import numpy
import pytest
import torch

from megatron.core.datasets.sbhd_packing import (
    PackedRows,
    document_pieces,
    pack_rows,
    pack_rows_fast,
    row_capacity,
)
from megatron.core.datasets.utils import compile_helpers
from tests.unit_tests.test_utilities import Utils


def _rows(packed):
    return [
        [tuple(int(v) for v in piece) for piece in packed.row(k)] for k in range(packed.num_rows)
    ]


def _expected_pieces(length, sequence_length, extra):
    """The pieces a document of ``length`` tokens must become (P1 / P2)."""
    capacity = row_capacity(sequence_length, extra)
    if length <= 0:
        return []
    if length <= capacity:
        return [(0, length)]
    pieces, offset = [], 0
    while offset + capacity <= length:
        pieces.append((offset, capacity))
        offset += sequence_length
    if length - offset >= (2 if extra else 1):
        pieces.append((offset, length - offset))
    return pieces


def _check_invariants(packed, document_index, lengths, sequence_length, extra):
    capacity = row_capacity(sequence_length, extra)
    rows = _rows(packed)
    assert packed.row_offsets[0] == 0 and packed.row_offsets[-1] == packed.pieces.shape[0]
    assert numpy.all(numpy.diff(packed.row_offsets) >= 1)  # P3: no empty row
    for row in rows:
        content = sum(length for _, _, length in row)
        assert 1 <= content <= capacity  # P3
    numpy.testing.assert_array_equal(packed.row_tokens(), [sum(p[2] for p in row) for row in rows])
    # P1 / P2: every occurrence of a document becomes exactly its expected pieces, and a
    # document that fits is one piece in one row.
    seen = {}
    for row in rows:
        for document, offset, length in row:
            seen.setdefault(document, []).append((offset, length))
    occurrences = {}
    for document in document_index:
        occurrences[int(document)] = occurrences.get(int(document), 0) + 1
    for document, count in occurrences.items():
        expected = _expected_pieces(int(lengths[document]), sequence_length, extra) * count
        assert sorted(seen.get(document, [])) == sorted(expected), document
    assert set(seen) <= set(occurrences)
    # P4: tokens conserved.
    total = sum(
        sum(length for _, length in _expected_pieces(int(lengths[d]), sequence_length, extra))
        for d in document_index
    )
    assert int(packed.pieces[:, 2].sum()) == total
    # Full windows are rows of their own.
    for row in rows:
        if any(length == capacity for _, _, length in row):
            assert len(row) == 1


def _next_fit(document_index, lengths, sequence_length, extra):
    """An independent next fit (the open-rows-1 oracle, P7)."""
    capacity = row_capacity(sequence_length, extra)
    rows, current, used = [], [], 0
    for document, offset, length, window in document_pieces(
        document_index, lengths, sequence_length=sequence_length, add_extra_token=extra
    ):
        if window or length == capacity:
            rows.append([(document, offset, length)])
            continue
        if current and used + length > capacity:
            rows.append(current)
            current, used = [], 0
        current.append((document, offset, length))
        used += length
        if used == capacity:
            rows.append(current)
            current, used = [], 0
    if current:
        rows.append(current)
    return rows


def _lengths(seed, count, high):
    return numpy.random.default_rng(seed).integers(0, high, size=count, dtype=numpy.int32)


# ---------------------------------------------------------------------------
# Pieces and hand-checked rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("extra", [True, False])
def test_document_pieces_cut_long_documents_as_the_unpacked_windows(extra):
    """S = 8: lengths C - 1, C, C + 1, S + 1, 2S + 1, a 1-token remainder, an empty
    document."""
    sequence_length = 8
    capacity = row_capacity(sequence_length, extra)
    lengths = numpy.array(
        [capacity - 1, capacity, capacity + 1, sequence_length + 1, 2 * sequence_length + 1, 0],
        dtype=numpy.int32,
    )
    pieces = list(
        document_pieces(
            numpy.arange(len(lengths)),
            lengths,
            sequence_length=sequence_length,
            add_extra_token=extra,
        )
    )
    by_doc = {}
    for document, offset, length, window in pieces:
        by_doc.setdefault(document, []).append((offset, length, window))
    assert by_doc[0] == [(0, capacity - 1, False)]
    assert by_doc[1] == [(0, capacity, False)]  # exactly C: a normal piece that fills a row
    assert 5 not in by_doc  # empty documents yield nothing
    if extra:
        # C + 1 = S + 2: one window and a 2-token remainder starting at S (the window's label
        # token, as the unpacked sample index's next window).
        assert by_doc[2] == [(0, capacity, True), (sequence_length, 2, False)]
        # S + 1 = C fits whole.
        assert by_doc[3] == [(0, capacity, False)]
        # 2S + 1: two windows sharing token S; the 1-token remainder (token 2S, already the
        # second window's label) is dropped.
        assert by_doc[4] == [(0, capacity, True), (sequence_length, capacity, True)]
    else:
        assert by_doc[2] == [(0, capacity, True), (sequence_length, 1, False)]
        assert by_doc[3] == [(0, capacity, True), (sequence_length, 1, False)]
        assert by_doc[4] == [(0, capacity, True), (8, capacity, True), (16, 1, False)]


def test_hand_checked_rows_with_the_extra_token():
    """S = 4, C = 5, lengths [3, 2, 4, 1, 6, 10] in document order."""
    lengths = numpy.array([3, 2, 4, 1, 6, 10], dtype=numpy.int32)
    packed = pack_rows(numpy.arange(6), lengths, sequence_length=4, add_extra_token=True)
    assert _rows(packed) == [
        [(0, 0, 3), (1, 0, 2)],
        [(2, 0, 4), (3, 0, 1)],
        [(4, 0, 5)],
        [(5, 0, 5)],
        [(5, 4, 5)],
        [(4, 4, 2), (5, 8, 2)],
    ]
    assert packed.num_rows_first == packed.num_rows == 6
    numpy.testing.assert_array_equal(packed.row_tokens(), [5, 5, 5, 5, 5, 4])


def test_hand_checked_rows_without_the_extra_token():
    """S = C = 4: first fit places the 1-token document in the first open row, a document of
    exactly C tokens is emitted alone at once."""
    lengths = numpy.array([3, 2, 4, 1, 6, 10], dtype=numpy.int32)
    packed = pack_rows(numpy.arange(6), lengths, sequence_length=4, add_extra_token=False)
    assert _rows(packed) == [
        [(2, 0, 4)],
        [(0, 0, 3), (3, 0, 1)],
        [(4, 0, 4)],
        [(1, 0, 2), (4, 4, 2)],
        [(5, 0, 4)],
        [(5, 4, 4)],
        [(5, 8, 2)],
    ]


def test_the_oldest_open_row_is_closed_when_the_window_is_full():
    """open_rows = 2, C = 5: three 3-token documents; the third closes the oldest row."""
    lengths = numpy.array([3, 3, 3, 2], dtype=numpy.int32)
    packed = pack_rows(
        numpy.arange(4), lengths, sequence_length=4, add_extra_token=True, open_rows=2
    )
    assert _rows(packed) == [[(0, 0, 3)], [(1, 0, 3), (3, 0, 2)], [(2, 0, 3)]]


# ---------------------------------------------------------------------------
# Invariants on random lengths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("extra", [True, False])
@pytest.mark.parametrize("open_rows", [1, 4, 64])
@pytest.mark.parametrize("high", [64, 300])
def test_invariants_on_random_lengths(extra, open_rows, high):
    """P1-P4 with short documents (``high`` 64 at S 128) and with documents longer than a row
    (``high`` 300)."""
    sequence_length = 128
    lengths = _lengths(7, 400, high)
    document_index = numpy.random.default_rng(1).permutation(len(lengths)).astype(numpy.int32)
    packed = pack_rows(
        document_index,
        lengths,
        sequence_length=sequence_length,
        add_extra_token=extra,
        open_rows=open_rows,
    )
    _check_invariants(packed, document_index, lengths, sequence_length, extra)


@pytest.mark.parametrize("extra", [True, False])
def test_open_rows_one_is_next_fit(extra):
    """P7."""
    lengths = _lengths(3, 300, 300)
    document_index = numpy.arange(len(lengths))
    packed = pack_rows(
        document_index, lengths, sequence_length=128, add_extra_token=extra, open_rows=1
    )
    assert _rows(packed) == _next_fit(document_index, lengths, 128, extra)


def test_an_exact_capacity_document_closes_no_open_row():
    """open_rows = 1, C = 5: the 5-token document is emitted alone at once and leaves the open
    row (document 0) open, so document 2 still fills it."""
    lengths = numpy.array([3, 5, 2], dtype=numpy.int32)
    packed = pack_rows(
        numpy.arange(3), lengths, sequence_length=4, add_extra_token=True, open_rows=1
    )
    assert _rows(packed) == [[(1, 0, 5)], [(0, 0, 3), (2, 0, 2)]]


def test_exact_capacity_documents_fill_a_row_alone():
    """P6: a document of C tokens has no padding; one of S tokens (with the extra token)
    leaves one padded position."""
    lengths = numpy.array([9, 8], dtype=numpy.int32)
    packed = pack_rows(numpy.arange(2), lengths, sequence_length=8, add_extra_token=True)
    assert _rows(packed) == [[(0, 0, 9)], [(1, 0, 8)]]
    numpy.testing.assert_array_equal(packed.row_tokens(), [9, 8])


def test_packing_is_deterministic_and_order_dependent():
    """P5: the same inputs give identical arrays; another document order other rows."""
    lengths = _lengths(5, 500, 200)
    order = numpy.random.default_rng(2).permutation(len(lengths))
    first = pack_rows(order, lengths, sequence_length=128, add_extra_token=True)
    second = pack_rows(order.copy(), lengths.copy(), sequence_length=128, add_extra_token=True)
    numpy.testing.assert_array_equal(first.row_offsets, second.row_offsets)
    numpy.testing.assert_array_equal(first.pieces, second.pieces)
    other = pack_rows(order[::-1], lengths, sequence_length=128, add_extra_token=True)
    assert _rows(other) != _rows(first)


def test_split_at_packs_the_segments_independently():
    lengths = _lengths(9, 200, 100)
    order = numpy.random.default_rng(4).permutation(len(lengths))
    split = 120
    packed = pack_rows(order, lengths, sequence_length=128, add_extra_token=True, split_at=split)
    head = pack_rows(order[:split], lengths, sequence_length=128, add_extra_token=True)
    tail = pack_rows(order[split:], lengths, sequence_length=128, add_extra_token=True)
    assert packed.num_rows_first == head.num_rows
    assert _rows(packed) == _rows(head) + _rows(tail)
    joined = PackedRows.concatenate([head, tail], num_rows_first=head.num_rows)
    numpy.testing.assert_array_equal(joined.row_offsets, packed.row_offsets)
    numpy.testing.assert_array_equal(joined.pieces, packed.pieces)
    with pytest.raises(ValueError, match="split_at"):
        pack_rows(order, lengths, sequence_length=128, add_extra_token=True, split_at=201)
    with pytest.raises(ValueError, match="open_rows"):
        pack_rows(order, lengths, sequence_length=128, add_extra_token=True, open_rows=0)


def test_empty_input_gives_no_rows():
    packed = pack_rows(
        numpy.zeros(0, dtype=numpy.int32),
        numpy.zeros(0, dtype=numpy.int32),
        sequence_length=8,
        add_extra_token=True,
    )
    assert packed.num_rows == 0 and packed.pieces.shape == (0, 3)


# ---------------------------------------------------------------------------
# The C++ packer (dataset helpers built: GPU nodes)
# ---------------------------------------------------------------------------


def require_cpp_helpers():
    """The C++ dataset helpers module: imported when built, compiled (rank 0, as
    test_gpt_dataset.py) on GPU nodes, skipped elsewhere (the Mac CPU venv has no pybind11)."""
    try:
        from megatron.core.datasets import helpers_cpp
    except ImportError:
        if not torch.cuda.is_available():
            pytest.skip("the C++ dataset helpers are not built here (GPU nodes compile them)")
        Utils.initialize_distributed()
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()
        helpers_cpp = importlib.import_module("megatron.core.datasets.helpers_cpp")
    if not hasattr(helpers_cpp, "build_packed_rows"):
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            compile_helpers()  # for the next run: this process already imported the old build
        pytest.fail("stale helpers_cpp build without build_packed_rows (recompiled; rerun)")
    return helpers_cpp


@pytest.fixture(scope="module")
def cpp_packer():
    return require_cpp_helpers().build_packed_rows


@pytest.mark.parametrize("seed", range(20))
def test_cpp_packer_equals_the_reference(cpp_packer, seed):
    """20 random length sets (short and long documents, both label conventions, open rows,
    split_at) and the mock dataset's lengths: identical arrays."""
    rng = numpy.random.default_rng(seed)
    extra = bool(seed % 2)
    sequence_length = int(rng.choice([16, 64, 128]))
    lengths = rng.integers(0, int(rng.choice([40, 200, 500])), size=300, dtype=numpy.int32)
    order = rng.permutation(len(lengths)).astype(numpy.int32)
    open_rows = int(rng.choice([1, 3, 64]))
    split_at = None if seed % 3 == 0 else int(rng.integers(0, len(order) + 1))
    kwargs = dict(
        sequence_length=sequence_length,
        add_extra_token=extra,
        open_rows=open_rows,
        split_at=split_at,
    )
    reference = pack_rows(order, lengths, **kwargs)
    fast = pack_rows_fast(order, lengths, **kwargs)
    numpy.testing.assert_array_equal(fast.row_offsets, reference.row_offsets)
    numpy.testing.assert_array_equal(fast.pieces, reference.pieces)
    assert fast.num_rows_first == reference.num_rows_first


def test_cpp_packer_equals_the_reference_on_mock_lengths(cpp_packer):
    from megatron.core.datasets.gpt_dataset import MockGPTLowLevelDataset

    lengths = numpy.random.default_rng(seed=MockGPTLowLevelDataset.seed).integers(
        low=1, high=MockGPTLowLevelDataset.max_sequence_length, size=20000, dtype=numpy.int32
    )
    order = numpy.random.RandomState(1234).permutation(len(lengths)).astype(numpy.int32)
    for extra in (True, False):
        reference = pack_rows(order, lengths, sequence_length=4096, add_extra_token=extra)
        fast = pack_rows_fast(order, lengths, sequence_length=4096, add_extra_token=extra)
        numpy.testing.assert_array_equal(fast.row_offsets, reference.row_offsets)
        numpy.testing.assert_array_equal(fast.pieces, reference.pieces)
