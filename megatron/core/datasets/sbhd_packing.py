# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Whole-document SBHD row packing (``GPTDatasetConfig.sbhd_document_packing``).

A packed dataset row holds whole documents, in document-index order, up to the row capacity
``C = sequence_length + add_extra_token_to_sequence`` tokens, followed by row-tail padding. The
packer never splits a document that fits in a row; a longer document is cut into ``C``-token
windows exactly as the default (unpacked) sample index cuts it: windows start at offsets ``0,
S, 2S, ...`` (with the extra token, consecutive windows share one token, the previous window's
label), each full window is a row of its own without padding, and the remainder is packed like
a document. Attention over a row stays plain causal (documents of a row see the earlier
documents of the row); the dataset emits a row-tail padding mask that every consumer excludes.

The row index is built once per dataset (and cached with the other indices) by a first fit
over a bounded window of open rows, so rows stay close to document order; it is a pure function
of its inputs. ``pack_rows`` is the Python reference; ``pack_rows_fast`` runs the same loop in
the C++ dataset helpers (``build_packed_rows``) when they are built and falls back to the
reference otherwise. Both return identical arrays.
"""

import dataclasses
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy


@dataclasses.dataclass(frozen=True)
class PackedRows:
    """The packed row index.

    ``row_offsets`` is ``int64 [num_rows + 1]``: row ``k`` holds the pieces
    ``pieces[row_offsets[k]:row_offsets[k + 1]]``. ``pieces`` is ``int64 [num_pieces, 3]``:
    ``(document id, token offset, token count)``, the document id being the value of the
    document index (the low-level dataset's document). ``num_rows_first`` is the number of rows
    packed from the first document-index segment (before ``split_at``; all rows without it).
    """

    row_offsets: numpy.ndarray
    pieces: numpy.ndarray
    num_rows_first: int

    @property
    def num_rows(self) -> int:
        """The number of rows."""
        return int(self.row_offsets.shape[0]) - 1

    def row(self, index: int) -> numpy.ndarray:
        """The ``[n, 3]`` pieces of row ``index`` (in index order, before any shuffle)."""
        return self.pieces[self.row_offsets[index] : self.row_offsets[index + 1]]

    def row_tokens(self) -> numpy.ndarray:
        """``int64 [num_rows]``: the content tokens (before padding) of every row."""
        counts = numpy.concatenate(([0], numpy.cumsum(self.pieces[:, 2], dtype=numpy.int64)))
        return counts[self.row_offsets[1:]] - counts[self.row_offsets[:-1]]

    @staticmethod
    def concatenate(parts: Sequence["PackedRows"], num_rows_first: int) -> "PackedRows":
        """The rows of ``parts`` in order (e.g. one part per epoch); ``num_rows_first`` is
        set by the caller."""
        offsets = [numpy.zeros(1, dtype=numpy.int64)]
        pieces = []
        base = 0
        for part in parts:
            offsets.append(part.row_offsets[1:].astype(numpy.int64) + base)
            pieces.append(part.pieces.astype(numpy.int64).reshape(-1, 3))
            base += int(part.pieces.shape[0])
        return PackedRows(
            row_offsets=numpy.concatenate(offsets),
            pieces=(
                numpy.concatenate(pieces) if pieces else numpy.zeros((0, 3), dtype=numpy.int64)
            ),
            num_rows_first=num_rows_first,
        )


def row_capacity(sequence_length: int, add_extra_token: bool) -> int:
    """``C``: the tokens a row holds (the sequence plus the label token of its last
    position when ``add_extra_token``)."""
    return sequence_length + int(bool(add_extra_token))


def document_pieces(
    document_index: numpy.ndarray,
    sequence_lengths: numpy.ndarray,
    *,
    sequence_length: int,
    add_extra_token: bool,
    min_piece: Optional[int] = None,
) -> Iterator[Tuple[int, int, int, bool]]:
    """Yield ``(document, offset, length, full_window)`` in document-index order.

    A document of at most ``C`` tokens is one piece (an empty document yields nothing). A
    longer document is cut as the default sample index cuts it: ``C``-token windows at offsets
    ``0, S, 2S, ...`` while ``offset + C <= length`` (each a ``full_window`` piece; with the
    extra token consecutive windows share one token), then the remainder ``(offset, length -
    offset)`` as a normal piece when it has at least ``min_piece`` tokens. ``min_piece``
    defaults to 2 with the extra token (a 1-token remainder is the previous window's label
    token: it carries no new (token, label) pair) and to 1 without it. ``min_piece`` applies
    to remainders only: a whole document is kept at any length, so with the extra token a row
    holding only a 1-token document (possible at an epoch end) has no loss label; the loss sums
    over the loss mask and the step normalises by the global token count, so such a row adds
    nothing.
    """
    capacity = row_capacity(sequence_length, add_extra_token)
    if min_piece is None:
        min_piece = 2 if add_extra_token else 1
    for document in document_index:
        document = int(document)
        length = int(sequence_lengths[document])
        if length <= 0:
            continue
        if length <= capacity:
            yield document, 0, length, False
            continue
        offset = 0
        while offset + capacity <= length:
            yield document, offset, capacity, True
            offset += sequence_length
        remainder = length - offset
        if remainder >= min_piece:
            yield document, offset, remainder, False


def _pack_segment(
    document_index: numpy.ndarray,
    sequence_lengths: numpy.ndarray,
    sequence_length: int,
    add_extra_token: bool,
    open_rows: int,
) -> List[List[Tuple[int, int, int]]]:
    """First fit over ``open_rows`` open rows of one document-index segment; the rows in
    emission order."""
    capacity = row_capacity(sequence_length, add_extra_token)
    emitted: List[List[Tuple[int, int, int]]] = []
    open_used: List[int] = []
    open_pieces: List[List[Tuple[int, int, int]]] = []
    for document, offset, length, full_window in document_pieces(
        document_index,
        sequence_lengths,
        sequence_length=sequence_length,
        add_extra_token=add_extra_token,
    ):
        piece = (document, offset, length)
        if full_window or length == capacity:
            # A full row on its own (no open row has room: every open row holds a token).
            emitted.append([piece])
            continue
        for slot, used in enumerate(open_used):
            if used + length <= capacity:
                open_used[slot] = used + length
                open_pieces[slot].append(piece)
                if open_used[slot] == capacity:
                    emitted.append(open_pieces.pop(slot))
                    open_used.pop(slot)
                break
        else:
            if len(open_used) == open_rows:
                emitted.append(open_pieces.pop(0))
                open_used.pop(0)
            open_used.append(length)
            open_pieces.append([piece])
    emitted.extend(open_pieces)
    return emitted


def _validate(sequence_length: int, open_rows: int, split_at: Optional[int], size: int) -> None:
    if sequence_length < 1:
        raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")
    if open_rows < 1:
        raise ValueError(f"open_rows must be >= 1, got {open_rows}")
    if split_at is not None and not 0 <= split_at <= size:
        raise ValueError(f"split_at must be in [0, {size}], got {split_at}")


def pack_rows(
    document_index: numpy.ndarray,
    sequence_lengths: numpy.ndarray,
    *,
    sequence_length: int,
    add_extra_token: bool,
    open_rows: int = 64,
    split_at: Optional[int] = None,
) -> PackedRows:
    """Pack the pieces of ``document_index`` (``document_pieces``) into rows (the reference).

    First fit over a bounded window of open rows, in document order: a full-window piece and a
    piece of exactly ``C`` tokens are each a closed row of their own (emitted at once, no open
    row closed); a normal piece of ``p`` tokens goes to the first
    open row (in creation order) with ``used + p <= C``; without one, the oldest open row is
    emitted first when ``open_rows`` rows are open, then a new row is opened with the piece; a
    row that reaches ``C`` tokens is emitted at once; after the last piece the open rows are
    emitted in creation order. ``open_rows = 1`` is next fit. ``split_at`` (an index into
    ``document_index``) packs ``[0, split_at)`` and ``[split_at, end)`` independently
    (``num_rows_first`` rows from the first segment). Deterministic: a pure function of the
    arguments.
    """
    document_index = numpy.asarray(document_index)
    _validate(sequence_length, open_rows, split_at, document_index.shape[0])
    segments = (
        [document_index]
        if split_at is None
        else [document_index[:split_at], document_index[split_at:]]
    )
    rows: List[List[Tuple[int, int, int]]] = []
    num_rows_first = 0
    for segment_index, segment in enumerate(segments):
        rows.extend(
            _pack_segment(segment, sequence_lengths, sequence_length, add_extra_token, open_rows)
        )
        if segment_index == 0:
            num_rows_first = len(rows)
    row_offsets = numpy.zeros(len(rows) + 1, dtype=numpy.int64)
    numpy.cumsum([len(row) for row in rows], out=row_offsets[1:])
    pieces = numpy.array([piece for row in rows for piece in row], dtype=numpy.int64).reshape(-1, 3)
    return PackedRows(row_offsets=row_offsets, pieces=pieces, num_rows_first=num_rows_first)


def _cpp_packer():
    """The C++ ``build_packed_rows``, or None when the dataset helpers are not built (or
    predate it)."""
    try:
        from megatron.core.datasets import helpers_cpp
    except ImportError:
        return None
    return getattr(helpers_cpp, "build_packed_rows", None)


def fast_packer_available() -> bool:
    """Whether ``pack_rows_fast`` runs the C++ packer."""
    return _cpp_packer() is not None


def pack_rows_fast(
    document_index: numpy.ndarray,
    sequence_lengths: numpy.ndarray,
    *,
    sequence_length: int,
    add_extra_token: bool,
    open_rows: int = 64,
    split_at: Optional[int] = None,
) -> PackedRows:
    """``pack_rows`` through the C++ dataset helpers when they are built (same result), else
    the Python reference."""
    packer = _cpp_packer()
    if packer is None:
        return pack_rows(
            document_index,
            sequence_lengths,
            sequence_length=sequence_length,
            add_extra_token=add_extra_token,
            open_rows=open_rows,
            split_at=split_at,
        )
    # int32 as build_sample_idx: the document index and the low-level dataset's lengths
    # are int32 already (no copy of the lengths).
    document_index = numpy.ascontiguousarray(document_index, dtype=numpy.int32)
    _validate(sequence_length, open_rows, split_at, document_index.shape[0])
    row_offsets, pieces, num_rows_first = packer(
        document_index,
        numpy.ascontiguousarray(sequence_lengths, dtype=numpy.int32),
        int(sequence_length),
        int(bool(add_extra_token)),
        int(open_rows),
        -1 if split_at is None else int(split_at),
    )
    return PackedRows(
        row_offsets=numpy.asarray(row_offsets, dtype=numpy.int64),
        pieces=numpy.asarray(pieces, dtype=numpy.int64).reshape(-1, 3),
        num_rows_first=int(num_rows_first),
    )
