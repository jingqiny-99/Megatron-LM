# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""SBHD document packing in ``GPTDataset`` (``GPTDatasetConfig.sbhd_document_packing``).

CPU (the packed path needs no C++ helpers): ``MockGPTDataset`` and an on-disk indexed dataset
packed at S 512 and 4096 with and without the extra token: the length is the row count, a
row's tokens are its pieces' tokens, documents that fit are never split, the padding mask is
the row tail, the loss mask is 0 from ``content - 1``, plain positions and causal masks, the
batch-padding row, the loss-mask cache (a short row fetched before a full row must not leak its
zeros), ``reset_position_ids``, row-based epochs (several epochs, separate and joint final
epoch), a fast-cache-load run reading the packed files of a first run, the cache description
(packing keys only when on; the default description equal to its value at the base commit) and
the config rejections. With the C++ helpers (GPU nodes): a two-component blend and the default
(unpacked) path's samples equal to goldens recorded at the base commit (vendor 0acc168de).
"""

import hashlib
import os

import numpy
import pytest
import torch

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import (
    GPTDataset,
    GPTDatasetConfig,
    MockGPTDataset,
    MockGPTLowLevelDataset,
    _build_document_index,
)
from megatron.core.datasets.indexed_dataset import DType, IndexedDataset, IndexedDatasetBuilder
from megatron.core.datasets.sbhd_packing import PackedRows, pack_rows, row_capacity
from megatron.core.datasets.utils import Split
from megatron.core.tokenizers import MegatronTokenizer
from tests.unit_tests.data.test_sbhd_packing import require_cpp_helpers

VOCAB = 8192
SEED = 1234


@pytest.fixture(scope="module")
def tokenizer():
    return MegatronTokenizer.from_pretrained(
        metadata_path={"library": "null-text"}, vocab_size=VOCAB
    )


@pytest.fixture(scope="module")
def cpp_helpers():
    """The C++ dataset helpers (module scope: set up before ``single_process`` patches
    ``torch.distributed``)."""
    return require_cpp_helpers()


@pytest.fixture
def single_process(monkeypatch):
    """Every rank builds its own indices (per-rank tmp caches in multi-rank runs)."""
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)


def _config(tokenizer, sequence_length=512, *, packing=True, **overrides):
    overrides.setdefault("reset_position_ids", False)
    overrides.setdefault("reset_attention_mask", False)
    overrides.setdefault("eod_mask_loss", False)
    overrides.setdefault("create_attention_mask", False)
    overrides.setdefault("split", "990,9,1")
    return GPTDatasetConfig(
        random_seed=SEED,
        sequence_length=sequence_length,
        tokenizer=tokenizer,
        sbhd_document_packing=packing,
        **overrides,
    )


def _mock(tokenizer, config, num_samples=64, documents=2000):
    return MockGPTDataset(
        MockGPTLowLevelDataset(tokenizer),
        None,
        numpy.arange(documents, dtype=numpy.int32),
        num_samples,
        Split.train,
        config,
    )


def _content(dataset, idx):
    return sum(length for _, _, length in dataset.packed_row_pieces(idx))


def _row_text(dataset, idx):
    """The row's pieces' tokens (unpadded), from the low-level dataset."""
    return numpy.concatenate(
        [
            dataset.dataset.get(document, offset=offset, length=length)
            for document, offset, length in dataset.packed_row_pieces(idx)
        ]
    )


# ---------------------------------------------------------------------------
# Rows, fields and masks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("extra", [True, False])
@pytest.mark.parametrize("sequence_length", [512, 4096])
def test_mock_rows_hold_whole_documents_and_their_masks(tokenizer, sequence_length, extra):
    config = _config(tokenizer, sequence_length, add_extra_token_to_sequence=extra)
    dataset = _mock(tokenizer, config, num_samples=32, documents=300)
    packed = dataset.packed_rows
    assert isinstance(packed, PackedRows) and len(dataset) == packed.num_rows >= 32
    capacity = row_capacity(sequence_length, extra)
    lengths = dataset.dataset.sequence_lengths
    # P1 over the whole index: every exposed document that fits is one whole piece in one row.
    for document in range(300):
        rows = packed.pieces[packed.pieces[:, 0] == document]
        if lengths[document] <= capacity:
            assert rows.tolist() == [[document, 0, lengths[document]]]
    for idx in range(min(len(dataset), 24)):
        sample = dataset[idx]
        content = _content(dataset, idx)
        text = _row_text(dataset, idx)
        assert len(text) == content <= capacity
        padded = numpy.concatenate([text, numpy.full(capacity - content, -1)])
        expected_tokens = torch.from_numpy(padded[:sequence_length]).clamp_min(0)
        assert torch.equal(sample["tokens"], expected_tokens)
        if extra:
            expected_labels = torch.from_numpy(padded[1:]).clamp_min(0)
        else:
            expected_labels = torch.from_numpy(numpy.append(padded[1:], -1)).clamp_min(0)
        assert torch.equal(sample["labels"], expected_labels)
        positions = torch.arange(sequence_length)
        assert sample["padding_mask"].dtype == torch.bool
        assert torch.equal(sample["padding_mask"], positions >= content)
        assert torch.equal(sample["loss_mask"], (positions < content - 1).float())
        assert torch.equal(sample["position_ids"], positions)
        assert "attention_mask" not in sample


def test_causal_attention_mask_and_the_batch_padding_row(tokenizer):
    config = _config(tokenizer, 512, create_attention_mask=True)
    dataset = _mock(tokenizer, config, num_samples=8, documents=100)
    sample = dataset[3]
    causal = ~torch.tril(torch.ones(512, 512, dtype=torch.bool)).unsqueeze(0)
    assert torch.equal(sample["attention_mask"], causal)
    pad = dataset[None]
    first = dataset[0]
    assert torch.equal(pad["tokens"], first["tokens"])
    assert torch.equal(pad["loss_mask"], torch.zeros(512))
    assert torch.equal(pad["padding_mask"], first["padding_mask"])


def test_cached_loss_mask_is_not_corrupted_by_a_short_row(tokenizer):
    """F15: the first fetch of a worker caches the masks; a short row's padding must not
    reach the cache (a later full row would inherit its zeros)."""
    config = _config(tokenizer, 512)
    dataset = _mock(tokenizer, config, num_samples=64, documents=600)
    contents = [_content(dataset, idx) for idx in range(len(dataset))]
    short = next(idx for idx, content in enumerate(contents) if content < 300)
    full = next(idx for idx, content in enumerate(contents) if content == 513)
    assert dataset.masks_and_position_ids_are_cacheable
    dataset[short]
    sample = dataset[full]
    assert torch.equal(sample["loss_mask"], torch.ones(512))
    assert not sample["padding_mask"].any()
    assert torch.equal(
        dataset[short]["loss_mask"], (torch.arange(512) < contents[short] - 1).float()
    )


def test_reset_position_ids_restart_at_each_document_end(tokenizer):
    config = _config(tokenizer, 512, reset_position_ids=True, eod_mask_loss=True)
    dataset = _mock(tokenizer, config, num_samples=8, documents=200)
    eod = tokenizer.eod
    for idx in range(6):
        sample = dataset[idx]
        tokens = sample["tokens"]
        expected, position = [], 0
        for token in tokens.tolist():
            expected.append(position)
            position = 0 if token == eod else position + 1
        assert sample["position_ids"].tolist() == expected
        content = _content(dataset, idx)
        mask = (torch.arange(512) < content - 1).float()
        mask[tokens == eod] = 0.0
        assert torch.equal(sample["loss_mask"], mask)


# ---------------------------------------------------------------------------
# Epochs in rows
# ---------------------------------------------------------------------------


def _epoch_rows(indices, lengths, sequence_length, epochs):
    """The per-epoch packed rows the dataset must build (one RandomState, one permutation per
    epoch)."""
    random_state = numpy.random.RandomState(SEED)
    parts = []
    for _ in range(epochs):
        order = _build_document_index(indices, 1, random_state, separate_final_epoch=False)
        parts.append(
            (
                order,
                pack_rows(order, lengths, sequence_length=sequence_length, add_extra_token=True),
            )
        )
    return parts


@pytest.mark.parametrize("case", ["separate", "joint", "many"])
def test_epochs_are_counted_in_rows(tokenizer, case):
    """Rows of whole epochs until they reach num_samples; the final epoch stays out of the
    global shuffle below 80 % use. High fill (rows < token windows) never trips the token
    rule's asserts: they do not run on the packing path."""
    documents, sequence_length = 40, 4096
    indices = numpy.arange(documents, dtype=numpy.int32)
    lengths = MockGPTLowLevelDataset(tokenizer).sequence_lengths
    rows_1 = _epoch_rows(indices, lengths, sequence_length, 1)[0][1].num_rows
    num_samples = {"separate": rows_1 + 1, "joint": 2 * rows_1 - 1, "many": 3 * rows_1 + 2}[case]
    config = _config(tokenizer, sequence_length)
    dataset = _mock(tokenizer, config, num_samples=num_samples, documents=documents)
    epochs = {"separate": 2, "joint": 2, "many": 4}[case]
    parts = _epoch_rows(indices, lengths, sequence_length, epochs)
    total = sum(part.num_rows for _, part in parts)
    assert len(dataset) == total >= num_samples
    assert total - parts[-1][1].num_rows < num_samples
    numpy.testing.assert_array_equal(
        dataset.document_index, numpy.concatenate([order for order, _ in parts])
    )
    expected = PackedRows.concatenate([part for _, part in parts], num_rows_first=0)
    numpy.testing.assert_array_equal(dataset.packed_rows.pieces, expected.pieces)
    numpy.testing.assert_array_equal(dataset.packed_rows.row_offsets, expected.row_offsets)
    rows_before_final = total - parts[-1][1].num_rows
    shuffle = numpy.asarray(dataset.shuffle_index)
    assert sorted(shuffle.tolist()) == list(range(total))
    separate = num_samples - rows_before_final < int(0.8 * parts[-1][1].num_rows)
    assert separate == (case != "joint")
    if separate:
        assert sorted(shuffle[:rows_before_final].tolist()) == list(range(rows_before_final))
    else:
        assert shuffle[:rows_before_final].max() >= rows_before_final


def test_one_epoch_without_num_samples(tokenizer):
    config = _config(tokenizer, 4096)
    dataset = _mock(tokenizer, config, num_samples=None, documents=50)
    parts = _epoch_rows(
        numpy.arange(50, dtype=numpy.int32), dataset.dataset.sequence_lengths, 4096, 1
    )
    assert len(dataset) == parts[0][1].num_rows


# ---------------------------------------------------------------------------
# Cache, description, indexed data, config
# ---------------------------------------------------------------------------

# MockGPTDataset(seed 1234, S 512, split "990,9,1" (mock: "1,1,1"), 64 samples, train, null
# tokenizer of 8192) at vendor 0acc168de: its unique description and hash.
BASE_DESCRIPTION_HASH = "00d0c9bbd8434637d45094ff7a3d8339"


def test_cache_description_has_the_packing_keys_only_when_on(tokenizer, monkeypatch):
    packed = _mock(tokenizer, _config(tokenizer, 512, sbhd_packing_open_rows=8), documents=100)
    assert '"sbhd_document_packing": true' in packed.unique_description
    assert '"sbhd_packing_open_rows": 8' in packed.unique_description
    # The default path (its index build needs the C++ helpers; not under test here).
    monkeypatch.setattr(
        GPTDataset, "_build_document_sample_shuffle_indices", lambda self: (None,) * 3
    )
    default = _mock(tokenizer, _config(tokenizer, 512, packing=False), documents=1000)
    assert "sbhd" not in default.unique_description
    assert default.unique_description_hash == BASE_DESCRIPTION_HASH
    assert packed.unique_description_hash != default.unique_description_hash


def test_fast_cache_load_reads_the_packed_files_of_a_first_run(tokenizer, tmp_path, single_process):
    cache = str(tmp_path / "cache")
    first = _mock(tokenizer, _config(tokenizer, 512, path_to_cache=cache), documents=300)
    base = f"{first.unique_description_hash}-MockGPTDataset-train"
    for name in (
        "description.txt",
        "document_index.npy",
        "packed_row_offsets.npy",
        "packed_pieces.npy",
        "shuffle_index.npy",
    ):
        assert os.path.isfile(os.path.join(cache, f"{base}-{name}")), name
    second = _mock(
        tokenizer, _config(tokenizer, 512, path_to_cache=cache, fast_cache_load=True), documents=300
    )
    assert len(second) == len(first)
    for idx in (0, 5, len(first) - 1):
        for key, value in first[idx].items():
            assert torch.equal(second[idx][key], value), (idx, key)


def _indexed_dataset(tmp_path, name, lengths, vocab=VOCAB, eod=VOCAB - 1):
    prefix = str(tmp_path / name)
    builder = IndexedDatasetBuilder(prefix + ".bin", dtype=DType.optimal_dtype(vocab))
    rng = numpy.random.default_rng(len(name))
    for length in lengths:
        tokens = rng.integers(0, vocab - 1, size=int(length) - 1).tolist() + [eod]
        builder.add_document(tokens, [len(tokens)])
    builder.finalize(prefix + ".idx")
    return prefix


def test_indexed_dataset_rows(tokenizer, tmp_path, single_process):
    """The real (indexed) dataset path, packed at S 64."""
    lengths = numpy.random.default_rng(0).integers(2, 150, size=80)
    prefix = _indexed_dataset(tmp_path, "docs", lengths)
    config = _config(tokenizer, 64, blend=([prefix], None), path_to_cache=str(tmp_path / "cache"))
    indexed = IndexedDataset(prefix, mmap=True)
    dataset = GPTDataset(
        indexed, prefix, numpy.arange(80, dtype=numpy.int32), 40, Split.train, config
    )
    assert len(dataset) >= 40
    for document in range(80):
        rows = dataset.packed_rows.pieces[dataset.packed_rows.pieces[:, 0] == document]
        if lengths[document] <= 65:
            assert rows.tolist() == [[document, 0, lengths[document]]]
    for idx in range(10):
        sample = dataset[idx]
        content = _content(dataset, idx)
        text = _row_text(dataset, idx)
        assert torch.equal(sample["tokens"][: min(content, 64)], torch.from_numpy(text[:64]).long())
        assert torch.equal(sample["padding_mask"], torch.arange(64) >= content)
        assert torch.equal(sample["loss_mask"], (torch.arange(64) < content - 1).float())


@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(inter_document_masking=True), "inter_document_masking"),
        (dict(defer_npy_index_mmap=True, path_to_cache="unused"), "defer_npy_index_mmap"),
        (dict(context_parallel_size=2), "context_parallel_size"),
        (dict(dynamic_context_parallel=True), "dynamic_context_parallel"),
        (dict(sbhd_packing_open_rows=0), "sbhd_packing_open_rows"),
    ],
)
def test_config_rejections(tokenizer, overrides, match):
    with pytest.raises(ValueError, match=match):
        _config(tokenizer, 512, **overrides)
    # Without packing the same options are accepted (except the packing-only field).
    if "sbhd_packing_open_rows" not in overrides:
        _config(tokenizer, 512, packing=False, **overrides)


# ---------------------------------------------------------------------------
# With the C++ dataset helpers (GPU nodes)
# ---------------------------------------------------------------------------


def test_two_component_blend(tokenizer, cpp_helpers, tmp_path, single_process):
    prefixes = [
        _indexed_dataset(tmp_path, f"part{i}", numpy.random.default_rng(i).integers(2, 120, 60))
        for i in range(2)
    ]
    config = _config(
        tokenizer,
        64,
        blend=(prefixes, [0.5, 0.5]),
        split="80,10,10",
        path_to_cache=str(tmp_path / "cache"),
    )
    train, _, _ = BlendedMegatronDatasetBuilder(
        GPTDataset, [40, 4, 4], lambda: True, config
    ).build()
    assert len(train) >= 40
    for component in train.datasets:
        assert component.config.sbhd_document_packing and component.packed_rows is not None
    for idx in range(len(train)):
        sample = train[idx]
        content = int((~sample["padding_mask"]).sum())
        assert torch.equal(sample["padding_mask"], torch.arange(64) >= content)


def _digest(sample):
    digest = hashlib.sha256()
    for key in ("tokens", "labels", "loss_mask", "position_ids"):
        digest.update(key.encode())
        digest.update(numpy.ascontiguousarray(sample[key].numpy()).tobytes())
    return digest.hexdigest()[:16]


# The first 16 train samples of BlendedMegatronDatasetBuilder(MockGPTDataset, [100, 100, 100])
# (seed 1234, S 1024, mid-level surplus 0.005, masks off), recorded at vendor 0acc168de.
BASE_DIGESTS = {
    True: [
        "cd1762c7df321506",
        "8f805fca3695c5a1",
        "daf3d408a79dd43f",
        "24522176867eb89c",
        "e2bb8c84a8ac54ec",
        "fedc3c551977b1ff",
        "e638ea18a8464f41",
        "c640a6691d013858",
        "4c93a9147dbe5db5",
        "7ba387aadac7145f",
        "9b6e28e0823684e8",
        "ba750ac4653b8d3b",
        "e7740d3c5b0380f8",
        "b7deccf680061676",
        "bd20a3cffe7434e9",
        "509149ee819c1dfb",
    ],
    False: [
        "9b50e0d888d585a9",
        "42eb698f72bd0d2b",
        "a94a21e7fd8f6d5d",
        "e4c58a932e45d4dc",
        "b88ca93d6d47c895",
        "9a38ccc7bacf95de",
        "401bd231bd950ee3",
        "fb3f7ff2164190ea",
        "976b1aff32e1744e",
        "cea77e761825a232",
        "41dd864b9ca876bf",
        "4f48bac64bd07633",
        "11a74833322eb51a",
        "58af0d50f7663268",
        "b91e176ee1f45405",
        "ab04e4c2f2d5daf5",
    ],
}


@pytest.mark.parametrize("extra", [True, False])
def test_default_path_equals_the_base_commit(tokenizer, cpp_helpers, extra):
    """Packing off: the unpacked samples are those of vendor 0acc168de."""
    config = GPTDatasetConfig(
        random_seed=SEED,
        sequence_length=1024,
        split="990,9,1",
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        create_attention_mask=False,
        tokenizer=tokenizer,
        mid_level_dataset_surplus=0.005,
        add_extra_token_to_sequence=extra,
    )
    train, _, _ = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [100, 100, 100], lambda: True, config
    ).build()
    assert len(train) == 66592
    assert "padding_mask" not in train[0]
    assert [_digest(train[idx]) for idx in range(16)] == BASE_DIGESTS[extra]
