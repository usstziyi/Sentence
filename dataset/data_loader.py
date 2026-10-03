"""
Reconstructed COFETT data_loader.py.

This module is designed to match the public COFETT train.py call:

    data_loader = EEGDataLoader(config.data_path)

    data_splits = data_loader.get_all_splits(
        paradigm=config.paradigm,
        subject=config.subject,
        categories=config.categories,
        sentence_ratio=sentence_ratio,
        train_repeats=n_repeats,
        lang=config.lang,
        emb_type=config.emb_type,
        mode=mode,
    )

Expected reconstructed HDF5 layout:

    /para01/sub01/<category>/<sentence_key>/
        attrs["text"]
        eeg/rep_01
        eeg/rep_02
        ...
        embeddings/zh/labse
        embeddings/zh/jina

    /para02/sub01/<category>/<sentence_key>/
        eeg/rep_01 ... rep_16
        embeddings/...

Public COFETT code confirms the outer hierarchy
(paradigm -> subject -> category -> sentence -> embeddings), but the missing
original data_loader.py means the exact EEG subgroup/dataset names cannot be
recovered. The "eeg/rep_XX" convention is therefore a reconstruction.

Split reconstruction:
    para01:
        repetitions 1..3 -> train
        repetition 4     -> test

    para02:
        repetitions 1..14 -> candidate train repetitions
        repetitions 15,16 -> fixed test

For para02, this matches the paper's description of the fixed 15th/16th
repetitions as test data. sentence_ratio is applied only to the training
sentence identities; the test sentence set remains fixed.

An optional LengthBucketBatchSampler is included as a project extension.
It is NOT part of the public COFETT train.py. It groups equal-length EEG trials
into the same mini-batch without padding or masks.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def normalize_paradigm(
    paradigm: str,
) -> str:
    text = paradigm.strip().lower()
    if text in {"para1", "para01"}:
        return "para01"
    if text in {"para2", "para02"}:
        return "para02"
    raise ValueError(
        f"Unsupported paradigm: {paradigm!r}"
    )


def normalize_subject(
    subject: str,
) -> str:
    text = subject.strip().lower()

    if text.startswith("sub-"):
        number = int(text[4:])
    elif text.startswith("sub"):
        number = int(text[3:])
    else:
        number = int(text)

    return f"sub{number:02d}"


def _decode_attr(
    value,
):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _parse_repeat_number(
    name: str,
) -> int:
    if not name.startswith("rep_"):
        raise ValueError(
            f"Unexpected EEG repetition dataset name: {name}"
        )
    return int(name.split("_", 1)[1])


@dataclass(frozen=True)
class SampleSpec:
    h5_path: str
    eeg_dataset_path: str
    embedding_dataset_path: str | None
    category: int
    text: str
    sentence_path: str
    repeat_index: int
    n_channels: int
    n_times: int
    sfreq: float


class EEGTextDataset(Dataset):
    """
    Lazy HDF5 EEG/Text dataset.

    h5py file handles are opened lazily per process so this remains safe when a
    PyTorch DataLoader later uses num_workers > 0.
    """

    def __init__(
        self,
        samples: Sequence[SampleSpec],
        *,
        require_embeddings: bool = True,
    ):
        self.samples = list(samples)
        self.require_embeddings = require_embeddings
        self._h5: h5py.File | None = None

    def __len__(self) -> int:
        return len(self.samples)

    def _file(self) -> h5py.File:
        if self._h5 is None:
            if not self.samples:
                raise RuntimeError(
                    "Cannot open HDF5 for an empty dataset."
                )
            self._h5 = h5py.File(
                self.samples[0].h5_path,
                "r",
            )
        return self._h5

    def __getitem__(
        self,
        index: int,
    ) -> dict:
        spec = self.samples[index]
        h5 = self._file()

        eeg = np.asarray(
            h5[spec.eeg_dataset_path][()],
            dtype=np.float32,
        )

        if eeg.ndim != 2:
            raise ValueError(
                f"EEG dataset {spec.eeg_dataset_path} "
                f"must have shape (C, T), got {eeg.shape}"
            )

        embeddings: dict[str, dict[str, torch.Tensor]] = {}

        if spec.embedding_dataset_path is not None:
            parts = spec.embedding_dataset_path.strip("/").split("/")
            # ... / embeddings / <lang> / <emb_type>
            lang = parts[-2]
            emb_type = parts[-1]

            embedding = np.asarray(
                h5[spec.embedding_dataset_path][()],
                dtype=np.float32,
            ).squeeze()

            embeddings = {
                lang: {
                    emb_type: torch.from_numpy(
                        embedding.copy()
                    ).float()
                }
            }
        elif self.require_embeddings:
            raise KeyError(
                "This sample has no requested text embedding."
            )

        return {
            "eeg_data": torch.from_numpy(
                eeg.copy()
            ).float(),
            "embeddings": embeddings,
            "category": torch.tensor(
                spec.category,
                dtype=torch.long,
            ),
            "text": spec.text,
            "repeat_index": spec.repeat_index,
            "n_times": spec.n_times,
            "sfreq": spec.sfreq,
            "sentence_path": spec.sentence_path,
        }

    def eeg_length(
        self,
        index: int,
    ) -> int:
        return self.samples[index].n_times

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state


class EEGDataLoader:
    """
    Loader matching the public COFETT train.py interface.

    Note:
        Despite its name, this is a dataset/split manager rather than
        torch.utils.data.DataLoader itself.
    """

    def __init__(
        self,
        data_path: str | Path,
        *,
        seed: int = 42,
        require_embeddings: bool = True,
    ):
        self.data_path = str(
            Path(data_path).expanduser().resolve()
        )
        self.seed = int(seed)
        self.require_embeddings = require_embeddings

        if not Path(self.data_path).exists():
            raise FileNotFoundError(
                self.data_path
            )

    def _collect_sentence_records(
        self,
        *,
        paradigm: str,
        subject: str,
        categories: Sequence[int] | None,
        lang: str,
        emb_type: str,
    ) -> list[dict]:
        paradigm = normalize_paradigm(
            paradigm
        )
        subject = normalize_subject(
            subject
        )

        category_filter = (
            None
            if categories is None
            else {int(value) for value in categories}
        )

        records: list[dict] = []

        with h5py.File(
            self.data_path,
            "r",
        ) as h5:
            if paradigm not in h5:
                raise KeyError(
                    f"Paradigm {paradigm!r} not found. "
                    f"Available: {list(h5.keys())}"
                )

            paradigm_group = h5[paradigm]

            if subject not in paradigm_group:
                raise KeyError(
                    f"Subject {subject!r} not found under {paradigm}. "
                    f"Available: {list(paradigm_group.keys())}"
                )

            subject_group = paradigm_group[subject]

            for category_name in subject_group.keys():
                try:
                    category = int(
                        category_name
                    )
                except ValueError:
                    continue

                if (
                    category_filter is not None
                    and category not in category_filter
                ):
                    continue

                category_group = subject_group[
                    category_name
                ]

                for sentence_key in category_group.keys():
                    sentence_group = category_group[
                        sentence_key
                    ]

                    if not isinstance(
                        sentence_group,
                        h5py.Group,
                    ):
                        continue

                    if "eeg" not in sentence_group:
                        continue

                    text = str(
                        _decode_attr(
                            sentence_group.attrs.get(
                                "text",
                                sentence_key,
                            )
                        )
                    )

                    embedding_path = (
                        f"{sentence_group.name}/embeddings/"
                        f"{lang}/{emb_type}"
                    )

                    if embedding_path not in h5:
                        if self.require_embeddings:
                            raise KeyError(
                                "Requested embedding is missing:\n"
                                f"  {embedding_path}\n"
                                "Attach/generate text embeddings before training, "
                                "or construct EEGDataLoader(..., "
                                "require_embeddings=False) for EEG-only inspection."
                            )
                        embedding_path = None

                    repetitions: list[dict] = []

                    eeg_group = sentence_group["eeg"]
                    for repetition_name in eeg_group.keys():
                        ds = eeg_group[
                            repetition_name
                        ]
                        if not isinstance(
                            ds,
                            h5py.Dataset,
                        ):
                            continue

                        repeat_index = _parse_repeat_number(
                            repetition_name
                        )

                        if ds.ndim != 2:
                            raise ValueError(
                                f"{ds.name} must be (C,T), got {ds.shape}"
                            )

                        repetitions.append(
                            {
                                "repeat_index": repeat_index,
                                "eeg_dataset_path": ds.name,
                                "n_channels": int(ds.shape[0]),
                                "n_times": int(ds.shape[1]),
                                "sfreq": float(
                                    ds.attrs.get(
                                        "sfreq",
                                        h5.attrs.get(
                                            "target_sfreq",
                                            np.nan,
                                        ),
                                    )
                                ),
                            }
                        )

                    repetitions.sort(
                        key=lambda item: item[
                            "repeat_index"
                        ]
                    )

                    if not repetitions:
                        continue

                    records.append(
                        {
                            "category": category,
                            "text": text,
                            "sentence_path": sentence_group.name,
                            "embedding_path": embedding_path,
                            "repetitions": repetitions,
                        }
                    )

        if not records:
            raise ValueError(
                "No sentence records matched the requested "
                f"paradigm={paradigm}, subject={subject}, "
                f"categories={categories}."
            )

        records.sort(
            key=lambda item: (
                item["category"],
                item["sentence_path"],
            )
        )
        return records

    def _split_repeat_sets(
        self,
        paradigm: str,
        available_repeats: set[int],
        train_repeats: int,
    ) -> tuple[set[int], set[int]]:
        paradigm = normalize_paradigm(
            paradigm
        )

        if paradigm == "para01":
            canonical_test = {4}
            max_train_repeat = min(
                int(train_repeats),
                3,
            )
            canonical_train = set(
                range(
                    1,
                    max_train_repeat + 1,
                )
            )
        else:
            canonical_test = {15, 16}
            max_train_repeat = min(
                int(train_repeats),
                14,
            )
            canonical_train = set(
                range(
                    1,
                    max_train_repeat + 1,
                )
            )

        train = (
            canonical_train
            & available_repeats
        )
        test = (
            canonical_test
            & available_repeats
        )

        if not train:
            raise ValueError(
                "No requested training repetitions are present. "
                f"Available repeats: {sorted(available_repeats)}"
            )

        if not test:
            # Useful for partial/local preprocessing while making the fallback
            # explicit rather than silently returning an empty test set.
            sorted_repeats = sorted(
                available_repeats
            )
            fallback_count = (
                1
                if paradigm == "para01"
                else min(2, len(sorted_repeats))
            )

            test = set(
                sorted_repeats[
                    -fallback_count:
                ]
            )

            train -= test

            if not train:
                raise ValueError(
                    "Partial dataset fallback consumed all repetitions; "
                    "cannot form both train and test sets."
                )

            import warnings

            warnings.warn(
                "Canonical COFETT test repetitions were not found. "
                f"Using final available repetition(s) {sorted(test)} as "
                "a clearly marked reconstruction fallback.",
                RuntimeWarning,
                stacklevel=2,
            )

        return train, test

    def _choose_training_sentences(
        self,
        records: list[dict],
        sentence_ratio: float,
    ) -> set[str]:
        if not 0 < sentence_ratio <= 1:
            raise ValueError(
                "sentence_ratio must satisfy 0 < ratio <= 1"
            )

        identities = [
            record["sentence_path"]
            for record in records
        ]

        if sentence_ratio >= 1.0:
            return set(identities)

        rng = random.Random(
            self.seed
        )

        shuffled = identities.copy()
        rng.shuffle(shuffled)

        n_keep = max(
            1,
            math.ceil(
                len(shuffled)
                * sentence_ratio
            ),
        )

        return set(
            shuffled[:n_keep]
        )

    def _build_sample(
        self,
        record: dict,
        repetition: dict,
    ) -> SampleSpec:
        return SampleSpec(
            h5_path=self.data_path,
            eeg_dataset_path=repetition[
                "eeg_dataset_path"
            ],
            embedding_dataset_path=record[
                "embedding_path"
            ],
            category=int(
                record["category"]
            ),
            text=str(
                record["text"]
            ),
            sentence_path=str(
                record["sentence_path"]
            ),
            repeat_index=int(
                repetition["repeat_index"]
            ),
            n_channels=int(
                repetition["n_channels"]
            ),
            n_times=int(
                repetition["n_times"]
            ),
            sfreq=float(
                repetition["sfreq"]
            ),
        )

    def get_all_splits(
        self,
        paradigm: str,
        subject: str,
        categories: Sequence[int] | None,
        sentence_ratio: float,
        train_repeats: int,
        lang: str,
        emb_type: str,
        mode: str = "grid",
    ) -> dict[str, EEGTextDataset]:
        """
        Return {"train": ..., "test": ...} in the form expected by train.py.

        `mode` is accepted for compatibility with the public train.py.
        The reconstructed split is governed by paradigm, repetition index and
        sentence_ratio.
        """
        if mode not in {
            "grid",
            "category",
        }:
            raise ValueError(
                f"Unsupported mode: {mode!r}"
            )

        records = self._collect_sentence_records(
            paradigm=paradigm,
            subject=subject,
            categories=categories,
            lang=lang,
            emb_type=emb_type,
        )

        available_repeats = {
            int(rep["repeat_index"])
            for record in records
            for rep in record["repetitions"]
        }

        train_repeat_set, test_repeat_set = (
            self._split_repeat_sets(
                paradigm,
                available_repeats,
                train_repeats,
            )
        )

        training_sentence_paths = (
            self._choose_training_sentences(
                records,
                sentence_ratio,
            )
        )

        train_samples: list[SampleSpec] = []
        test_samples: list[SampleSpec] = []

        for record in records:
            for repetition in record[
                "repetitions"
            ]:
                repeat_index = int(
                    repetition["repeat_index"]
                )

                if (
                    record["sentence_path"]
                    in training_sentence_paths
                    and repeat_index
                    in train_repeat_set
                ):
                    train_samples.append(
                        self._build_sample(
                            record,
                            repetition,
                        )
                    )

                if (
                    repeat_index
                    in test_repeat_set
                ):
                    test_samples.append(
                        self._build_sample(
                            record,
                            repetition,
                        )
                    )

        if not train_samples:
            raise ValueError(
                "Training split is empty."
            )
        if not test_samples:
            raise ValueError(
                "Test split is empty."
            )

        train_samples.sort(
            key=lambda item: (
                item.category,
                item.sentence_path,
                item.repeat_index,
            )
        )
        test_samples.sort(
            key=lambda item: (
                item.category,
                item.sentence_path,
                item.repeat_index,
            )
        )

        return {
            "train": EEGTextDataset(
                train_samples,
                require_embeddings=self.require_embeddings,
            ),
            "test": EEGTextDataset(
                test_samples,
                require_embeddings=self.require_embeddings,
            ),
        }

    def summary(
        self,
        *,
        paradigm: str,
        subject: str,
        categories: Sequence[int] | None = None,
        lang: str = "zh",
        emb_type: str = "labse",
    ) -> dict:
        records = self._collect_sentence_records(
            paradigm=paradigm,
            subject=subject,
            categories=categories,
            lang=lang,
            emb_type=emb_type,
        )

        lengths = [
            rep["n_times"]
            for record in records
            for rep in record["repetitions"]
        ]
        repeats = sorted(
            {
                rep["repeat_index"]
                for record in records
                for rep in record["repetitions"]
            }
        )

        return {
            "n_sentences": len(records),
            "n_eeg_trials": len(lengths),
            "categories": sorted(
                {
                    record["category"]
                    for record in records
                }
            ),
            "repeats": repeats,
            "min_n_times": min(lengths),
            "max_n_times": max(lengths),
            "unique_lengths": len(
                set(lengths)
            ),
        }


class LengthBucketBatchSampler(
    Sampler[list[int]]
):
    """
    Project extension: equal-length mini-batches without padding.

    Every yielded batch contains only samples with identical `n_times`.
    Batch order is shuffled across length buckets each epoch.

    Example
    -------
    sampler = LengthBucketBatchSampler(
        dataset,
        batch_size=32,
        shuffle=True,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
    )
    """

    def __init__(
        self,
        dataset: EEGTextDataset,
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ):
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive"
            )

        self.dataset = dataset
        self.batch_size = int(
            batch_size
        )
        self.shuffle = bool(
            shuffle
        )
        self.drop_last = bool(
            drop_last
        )
        self.seed = int(
            seed
        )
        self.epoch = 0

        self._buckets: dict[
            int,
            list[int],
        ] = {}

        for index, spec in enumerate(
            dataset.samples
        ):
            self._buckets.setdefault(
                spec.n_times,
                [],
            ).append(index)

    def set_epoch(
        self,
        epoch: int,
    ) -> None:
        self.epoch = int(
            epoch
        )

    def __iter__(
        self,
    ) -> Iterator[list[int]]:
        rng = random.Random(
            self.seed + self.epoch
        )

        batches: list[list[int]] = []

        for n_times in sorted(
            self._buckets
        ):
            indices = self._buckets[
                n_times
            ].copy()

            if self.shuffle:
                rng.shuffle(indices)

            for start in range(
                0,
                len(indices),
                self.batch_size,
            ):
                batch = indices[
                    start : start + self.batch_size
                ]

                if (
                    self.drop_last
                    and len(batch)
                    < self.batch_size
                ):
                    continue

                batches.append(
                    batch
                )

        if self.shuffle:
            rng.shuffle(batches)

        yield from batches

    def __len__(self) -> int:
        total = 0

        for indices in self._buckets.values():
            if self.drop_last:
                total += (
                    len(indices)
                    // self.batch_size
                )
            else:
                total += math.ceil(
                    len(indices)
                    / self.batch_size
                )

        return total


def attach_sentence_embeddings(
    h5_path: str | Path,
    encoder,
    *,
    lang: str,
    emb_type: str,
    batch_size: int = 64,
    overwrite: bool = False,
) -> None:
    """
    Attach one embedding per sentence to the reconstructed HDF5.

    `encoder` can be:
        - an object exposing .get_embedding(list[str]), or
        - any callable accepting list[str] and returning (N,D).

    This helper is intentionally model-agnostic so it can be used with
    LaBSE, Jina, GTE, E5, BGE, Qwen, etc.
    """
    h5_path = Path(
        h5_path
    )

    with h5py.File(
        h5_path,
        "a",
    ) as h5:
        sentence_groups: list[
            tuple[str, str]
        ] = []

        for paradigm in h5.keys():
            if not isinstance(
                h5[paradigm],
                h5py.Group,
            ):
                continue

            for subject in h5[
                paradigm
            ].keys():
                subject_group = h5[
                    paradigm
                ][subject]

                for category in subject_group.keys():
                    category_group = subject_group[
                        category
                    ]

                    for sentence_key in category_group.keys():
                        group = category_group[
                            sentence_key
                        ]

                        if not isinstance(
                            group,
                            h5py.Group,
                        ):
                            continue
                        if "eeg" not in group:
                            continue

                        text = str(
                            _decode_attr(
                                group.attrs.get(
                                    "text",
                                    sentence_key,
                                )
                            )
                        )

                        target_path = (
                            f"{group.name}/embeddings/"
                            f"{lang}/{emb_type}"
                        )

                        if (
                            target_path in h5
                            and not overwrite
                        ):
                            continue

                        sentence_groups.append(
                            (
                                group.name,
                                text,
                            )
                        )

        for start in range(
            0,
            len(sentence_groups),
            batch_size,
        ):
            batch = sentence_groups[
                start : start + batch_size
            ]
            texts = [
                text
                for _, text in batch
            ]

            if hasattr(
                encoder,
                "get_embedding",
            ):
                vectors = encoder.get_embedding(
                    texts
                )
            else:
                vectors = encoder(
                    texts
                )

            if isinstance(
                vectors,
                torch.Tensor,
            ):
                vectors = (
                    vectors.detach()
                    .cpu()
                    .numpy()
                )

            vectors = np.asarray(
                vectors,
                dtype=np.float32,
            )

            if vectors.ndim != 2:
                raise ValueError(
                    "Embedding encoder must return shape (N, D)."
                )

            if vectors.shape[0] != len(
                batch
            ):
                raise ValueError(
                    "Embedding batch size mismatch."
                )

            for (
                group_path,
                _,
            ), vector in zip(
                batch,
                vectors,
                strict=True,
            ):
                group = h5[
                    group_path
                ]

                emb_group = (
                    group.require_group(
                        "embeddings"
                    )
                    .require_group(
                        lang
                    )
                )

                if emb_type in emb_group:
                    del emb_group[
                        emb_type
                    ]

                emb_group.create_dataset(
                    emb_type,
                    data=vector,
                )


if __name__ == "__main__":
    # Minimal structure inspection example:
    #
    # loader = EEGDataLoader(
    #     "data/target.h5",
    #     require_embeddings=False,
    # )
    #
    # print(
    #     loader.summary(
    #         paradigm="para02",
    #         subject="sub01",
    #         categories=None,
    #         lang="zh",
    #         emb_type="labse",
    #     )
    # )
    pass
