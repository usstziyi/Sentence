"""
Reconstructed COFETT / Chisco-2.0 preprocessing pipeline.

IMPORTANT
---------
The original COFETT repository currently references preprocessing/data-loader
modules that are not publicly present. This file is therefore a reconstruction
based on:

1. The COFETT paper's Appendix E / preprocessing figure.
2. The public COFETT train.py/config.py/similarity.py interfaces.
3. The public BIDS structure of OpenNeuro ds006317 / NEMAR on006317.
4. The public PsychoPy stimulus files in baoyudu/COFETT.

Source-supported preprocessing steps:
    - resample 1000 Hz -> 500 Hz
    - PREP-style robust referencing / bad-channel handling / line-noise cleanup
    - 50 Hz notch
    - 1 Hz zero-phase FIR high-pass; no explicit low-pass
    - EOG regression
    - extended Infomax ICA, 30 components
    - ICLabel-assisted artifact identification
    - segmentation into reading / recall(inner speech) / rest
    - AutoReject

Reconstruction choices made here because the original preprocessing.py is absent:
    - Continuous-data order used here:
          resample -> PREP -> filtering -> EOG regression -> ICA -> segmentation
          -> optional AutoReject per equal-length bucket.
      The paper's prose and flow figure do not expose enough implementation
      detail to recover the exact original ordering.
    - Event-code mapping is inferred from the public BIDS event timing:
          65329 -> reading onset
          65379 -> recall / inner-speech onset
          65381 -> rest onset
    - para1 repetition index is mapped from BIDS session (1..4).
    - para2 repetition index is mapped as (session - 1) * 4 + run (1..16).
    - para1 run k is mapped to psychopy/text1-k.xlsx.
    - para2 is mapped to psychopy/text2.xlsx.

The code intentionally preserves variable-length recall EEG. It does NOT pad
or truncate trials during preprocessing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import mne
import numpy as np
import pandas as pd


LOGGER = logging.getLogger("cofett.preprocessing")

READING_EVENT = 65329
RECALL_EVENT = 65379
REST_EVENT = 65381

EVENT_NAMES = {
    READING_EVENT: "reading",
    RECALL_EVENT: "recall",
    REST_EVENT: "rest",
}

SENTENCE_COLUMN_CANDIDATES = (
    "句子",
    "sentence",
    "text",
    "sentence_zh",
    "中文",
)

CATEGORY_COLUMN_CANDIDATES = (
    "类别",
    "分类",
    "category",
    "category_id",
    "class",
    "label",
    "标签",
)


@dataclass(frozen=True)
class RecordingSpec:
    subject_bids: str
    subject_h5: str
    session_bids: str
    session_index: int
    task_bids: str
    paradigm_h5: str
    run_index: int
    eeg_path: Path
    events_path: Path
    channels_path: Path


@dataclass
class TrialSegment:
    trial_index: int
    text: str
    category: int
    phase: str
    onset: float
    stop: float
    eeg: np.ndarray
    sfreq: float
    repeat_index: int
    source_edf: str
    session: str
    run: int
    stimulus_file: str

    @property
    def duration(self) -> float:
        return self.stop - self.onset

    @property
    def n_times(self) -> int:
        return int(self.eeg.shape[-1])


@dataclass
class PreprocessConfig:
    bids_root: Path
    stimuli_dir: Path
    output_h5: Path

    target_sfreq: float = 500.0
    highpass_hz: float = 1.0
    line_freq_hz: float = 50.0

    phase: str = "recall"

    use_prep: bool = True
    prep_ransac: bool = True
    allow_prep_fallback: bool = True

    use_eog_regression: bool = True
    use_ica: bool = True
    auto_apply_ica: bool = True
    ica_n_components: int = 30
    ica_probability_threshold: float = 0.80
    ica_artifact_labels: tuple[str, ...] = (
        "eye blink",
        "muscle artifact",
    )

    use_autoreject: bool = True
    autoreject_min_bucket_size: int = 6

    montage_name: str = "standard_1005"
    montage_file: Path | None = None
    strict_montage: bool = False

    random_state: int = 42
    n_jobs: int = 1
    overwrite: bool = False


def _normalize_subject_for_h5(subject_bids: str) -> str:
    match = re.fullmatch(r"sub-(\d+)", subject_bids)
    if not match:
        raise ValueError(f"Unexpected subject name: {subject_bids}")
    return f"sub{int(match.group(1)):02d}"


def _normalize_paradigm(task_bids: str) -> str:
    match = re.fullmatch(r"para0?(\d+)", task_bids)
    if not match:
        raise ValueError(f"Unexpected task/paradigm: {task_bids}")
    return f"para{int(match.group(1)):02d}"


def _extract_index(name: str, prefix: str) -> int:
    match = re.fullmatch(rf"{re.escape(prefix)}-(\d+)", name)
    if not match:
        raise ValueError(f"Could not parse {prefix} index from {name!r}")
    return int(match.group(1))


def discover_recordings(
    bids_root: str | Path,
    subjects: Sequence[str] | None = None,
) -> list[RecordingSpec]:
    """Discover BIDS EDF recordings and their TSV sidecars."""
    bids_root = Path(bids_root).expanduser().resolve()

    requested = None
    if subjects:
        requested = set()
        for subject in subjects:
            if subject.startswith("sub-"):
                requested.add(subject)
            elif subject.startswith("sub"):
                requested.add(f"sub-{int(subject[3:]):02d}")
            else:
                requested.add(f"sub-{int(subject):02d}")

    pattern = re.compile(
        r"(?P<subject>sub-\d+)_(?P<session>ses-\d+)_"
        r"task-(?P<task>para\d+)_run-(?P<run>\d+)_eeg\.edf$"
    )

    specs: list[RecordingSpec] = []

    for eeg_path in sorted(bids_root.glob("sub-*/ses-*/eeg/*_eeg.edf")):
        match = pattern.fullmatch(eeg_path.name)
        if not match:
            continue

        subject_bids = match.group("subject")
        if requested is not None and subject_bids not in requested:
            continue

        session_bids = match.group("session")
        task_bids = match.group("task")
        run_index = int(match.group("run"))

        stem = eeg_path.name.removesuffix("_eeg.edf")
        events_path = eeg_path.with_name(f"{stem}_events.tsv")
        channels_path = eeg_path.with_name(f"{stem}_channels.tsv")

        missing = [
            str(path)
            for path in (events_path, channels_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing BIDS sidecars for recording:\n"
                + "\n".join(missing)
            )

        specs.append(
            RecordingSpec(
                subject_bids=subject_bids,
                subject_h5=_normalize_subject_for_h5(subject_bids),
                session_bids=session_bids,
                session_index=_extract_index(session_bids, "ses"),
                task_bids=task_bids,
                paradigm_h5=_normalize_paradigm(task_bids),
                run_index=run_index,
                eeg_path=eeg_path,
                events_path=events_path,
                channels_path=channels_path,
            )
        )

    if not specs:
        raise FileNotFoundError(
            f"No BIDS EDF recordings found under {bids_root}"
        )

    return specs


def _column_lookup(
    df: pd.DataFrame,
    candidates: Iterable[str],
    *,
    required: bool = True,
) -> str | None:
    original = list(df.columns)
    normalized = {
        str(column).strip().lower(): column
        for column in original
    }

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]

    if required:
        raise KeyError(
            "Could not identify required spreadsheet column. "
            f"Candidates={list(candidates)}; columns={original}"
        )
    return None


def stimulus_file_for_recording(
    stimuli_dir: str | Path,
    spec: RecordingSpec,
) -> Path:
    """Map a BIDS recording to the public COFETT stimulus spreadsheet."""
    stimuli_dir = Path(stimuli_dir)

    if spec.paradigm_h5 == "para01":
        path = stimuli_dir / f"text1-{spec.run_index}.xlsx"
    elif spec.paradigm_h5 == "para02":
        path = stimuli_dir / "text2.xlsx"
    else:
        raise ValueError(f"Unsupported paradigm: {spec.paradigm_h5}")

    if not path.exists():
        raise FileNotFoundError(
            f"Stimulus spreadsheet not found: {path}"
        )
    return path


def load_stimulus_table(
    path: str | Path,
) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_excel(path)

    sentence_col = _column_lookup(
        df,
        SENTENCE_COLUMN_CANDIDATES,
    )
    category_col = _column_lookup(
        df,
        CATEGORY_COLUMN_CANDIDATES,
    )

    out = pd.DataFrame(
        {
            "text": df[sentence_col].astype(str),
            "category": pd.to_numeric(
                df[category_col],
                errors="raise",
            ).astype(int),
        }
    )
    return out.reset_index(drop=True)


def load_events(
    events_path: str | Path,
) -> pd.DataFrame:
    events = pd.read_csv(events_path, sep="\t")
    required = {"onset", "value"}
    missing = required - set(events.columns)
    if missing:
        raise KeyError(
            f"{events_path} is missing event columns: {sorted(missing)}"
        )

    events = events.copy()
    events["value"] = pd.to_numeric(
        events["value"],
        errors="coerce",
    ).astype("Int64")

    events = events[
        events["value"].isin(
            [READING_EVENT, RECALL_EVENT, REST_EVENT]
        )
    ].reset_index(drop=True)

    return events


def parse_trial_boundaries(
    events: pd.DataFrame,
) -> list[dict[str, float | int]]:
    """
    Parse public ds006317 event triples:

        65329 -> reading onset
        65379 -> recall onset
        65381 -> rest onset
    """
    rows = events[["onset", "value"]].to_dict("records")

    trials: list[dict[str, float | int]] = []
    i = 0

    while i < len(rows):
        if int(rows[i]["value"]) != READING_EVENT:
            i += 1
            continue

        if i + 2 >= len(rows):
            raise ValueError(
                "Incomplete trial at end of events table."
            )

        triple = rows[i : i + 3]
        values = [int(row["value"]) for row in triple]

        expected = [
            READING_EVENT,
            RECALL_EVENT,
            REST_EVENT,
        ]
        if values != expected:
            raise ValueError(
                "Unexpected event sequence while reconstructing trials: "
                f"got {values}, expected {expected}"
            )

        trials.append(
            {
                "trial_index": len(trials),
                "reading_onset": float(triple[0]["onset"]),
                "recall_onset": float(triple[1]["onset"]),
                "rest_onset": float(triple[2]["onset"]),
            }
        )
        i += 3

    if not trials:
        raise ValueError("No complete EEG trials were found.")

    return trials


def _repeat_index(spec: RecordingSpec) -> int:
    """
    Reconstructed repetition mapping.

    para01: four BIDS sessions -> repetitions 1..4
    para02: four sessions x four runs -> repetitions 1..16
    """
    if spec.paradigm_h5 == "para01":
        return spec.session_index

    if spec.paradigm_h5 == "para02":
        return (
            (spec.session_index - 1) * 4
            + spec.run_index
        )

    raise ValueError(
        f"Unsupported paradigm: {spec.paradigm_h5}"
    )


def apply_bids_channel_types(
    raw: mne.io.BaseRaw,
    channels_path: str | Path,
) -> None:
    """Apply BIDS channel types using *_channels.tsv."""
    channels = pd.read_csv(channels_path, sep="\t")

    if not {"name", "type"}.issubset(channels.columns):
        raise KeyError(
            f"{channels_path} must contain name/type columns."
        )

    type_map = {
        "EEG": "eeg",
        "EOG": "eog",
        "TRIG": "stim",
        "STIM": "stim",
        "ECG": "ecg",
        "EMG": "emg",
    }

    mapping: dict[str, str] = {}
    raw_names = set(raw.ch_names)

    for _, row in channels.iterrows():
        name = str(row["name"])
        bids_type = str(row["type"]).upper()
        if name in raw_names and bids_type in type_map:
            mapping[name] = type_map[bids_type]

    if mapping:
        raw.set_channel_types(mapping, verbose=False)


def attach_montage(
    raw: mne.io.BaseRaw,
    config: PreprocessConfig,
) -> None:
    """
    Attach electrode positions.

    A custom cap montage is preferable when available. standard_1005 is only a
    fallback because the public BIDS mirror does not provide a per-recording
    electrodes.tsv in the paths inspected during reconstruction.
    """
    if config.montage_file is not None:
        montage = mne.channels.read_custom_montage(
            config.montage_file
        )
    else:
        montage = mne.channels.make_standard_montage(
            config.montage_name
        )

    on_missing = "raise" if config.strict_montage else "warn"

    raw.set_montage(
        montage,
        match_case=False,
        on_missing=on_missing,
        verbose=False,
    )


def run_prep(
    raw: mne.io.BaseRaw,
    config: PreprocessConfig,
) -> mne.io.BaseRaw:
    """Run PyPREP when available."""
    if not config.use_prep:
        return raw

    try:
        from pyprep.prep_pipeline import PrepPipeline
    except ImportError as exc:
        if not config.allow_prep_fallback:
            raise ImportError(
                "PyPREP is required for PREP processing. "
                "Install it with: pip install pyprep"
            ) from exc

        LOGGER.warning(
            "PyPREP is unavailable; falling back to average EEG reference. "
            "This is NOT equivalent to the paper's PREP pipeline."
        )
        raw = raw.copy()
        raw.set_eeg_reference(
            "average",
            projection=False,
            verbose=False,
        )
        return raw

    montage = raw.get_montage()
    if montage is None:
        raise RuntimeError(
            "PREP requires a montage. Supply --montage-file or "
            "use a compatible standard montage."
        )

    line_freqs = np.arange(
        config.line_freq_hz,
        raw.info["sfreq"] / 2.0,
        config.line_freq_hz,
    )

    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": line_freqs,
    }

    try:
        prep = PrepPipeline(
            raw.copy(),
            prep_params,
            montage,
            ransac=config.prep_ransac,
            random_state=config.random_state,
        )
        prep.fit()
        return prep.raw
    except Exception:
        if not config.allow_prep_fallback:
            raise

        LOGGER.exception(
            "PyPREP failed; falling back to average EEG reference. "
            "This changes the preprocessing relative to the paper."
        )
        fallback = raw.copy()
        fallback.set_eeg_reference(
            "average",
            projection=False,
            verbose=False,
        )
        return fallback


def apply_filters(
    raw: mne.io.BaseRaw,
    config: PreprocessConfig,
) -> mne.io.BaseRaw:
    """Apply the paper-described 50-Hz notch + 1-Hz high-pass."""
    raw = raw.copy()

    picks = mne.pick_types(
        raw.info,
        eeg=True,
        eog=True,
        exclude=[],
    )

    raw.notch_filter(
        freqs=[config.line_freq_hz],
        picks=picks,
        method="fir",
        phase="zero",
        n_jobs=config.n_jobs,
        verbose=False,
    )

    raw.filter(
        l_freq=config.highpass_hz,
        h_freq=None,
        picks=picks,
        method="fir",
        phase="zero",
        n_jobs=config.n_jobs,
        verbose=False,
    )

    return raw


def apply_eog_regression(
    raw: mne.io.BaseRaw,
    config: PreprocessConfig,
) -> mne.io.BaseRaw:
    """Regress EOG channels from EEG channels."""
    if not config.use_eog_regression:
        return raw

    eog_picks = mne.pick_types(
        raw.info,
        eog=True,
        exclude=[],
    )
    if len(eog_picks) == 0:
        LOGGER.warning(
            "No EOG channels found; skipping EOG regression."
        )
        return raw

    from mne.preprocessing import EOGRegression

    regression = EOGRegression(
        picks="eeg",
        picks_artifact="eog",
        proj=False,
    )
    regression.fit(raw)

    return regression.apply(
        raw,
        copy=True,
    )


def apply_ica(
    raw: mne.io.BaseRaw,
    config: PreprocessConfig,
    report_path: Path,
) -> mne.io.BaseRaw:
    """
    Fit extended Infomax ICA and label components with ICLabel.

    The paper reports manual review after ICLabel. Because the original manual
    decisions are unavailable, this reconstruction can optionally auto-remove
    high-confidence artifact components.
    """
    if not config.use_ica:
        return raw

    try:
        from mne_icalabel import label_components
    except ImportError as exc:
        raise ImportError(
            "mne-icalabel is required for ICA labeling. "
            "Install it with: pip install mne-icalabel"
        ) from exc

    eeg_picks = mne.pick_types(
        raw.info,
        eeg=True,
        exclude="bads",
    )

    max_components = max(1, len(eeg_picks) - 1)
    n_components = min(
        config.ica_n_components,
        max_components,
    )

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method="infomax",
        fit_params={"extended": True},
        random_state=config.random_state,
        max_iter="auto",
    )

    ica.fit(
        raw,
        picks=eeg_picks,
        reject_by_annotation=True,
        verbose=False,
    )

    labels = label_components(
        raw,
        ica,
        method="iclabel",
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = pd.DataFrame(
        {
            "component": np.arange(
                len(labels["labels"])
            ),
            "label": labels["labels"],
            "probability": labels["y_pred_proba"],
        }
    )

    report["auto_exclude"] = (
        report["label"].isin(
            config.ica_artifact_labels
        )
        & (
            report["probability"]
            >= config.ica_probability_threshold
        )
    )

    report.to_csv(
        report_path,
        index=False,
    )

    if not config.auto_apply_ica:
        LOGGER.warning(
            "ICA report written to %s, but components were not removed. "
            "This mode is intended for manual review.",
            report_path,
        )
        return raw

    exclude = (
        report.loc[
            report["auto_exclude"],
            "component",
        ]
        .astype(int)
        .tolist()
    )

    ica.exclude = exclude

    LOGGER.info(
        "ICA: excluding %d/%d components: %s",
        len(exclude),
        n_components,
        exclude,
    )

    return ica.apply(
        raw.copy(),
        exclude=exclude,
        verbose=False,
    )


def preprocess_continuous_recording(
    spec: RecordingSpec,
    config: PreprocessConfig,
) -> mne.io.BaseRaw:
    """Load and clean one continuous EDF recording."""
    LOGGER.info(
        "Preprocessing %s",
        spec.eeg_path.name,
    )

    raw = mne.io.read_raw_edf(
        spec.eeg_path,
        preload=True,
        verbose=False,
    )

    apply_bids_channel_types(
        raw,
        spec.channels_path,
    )

    if raw.info["sfreq"] != config.target_sfreq:
        raw.resample(
            config.target_sfreq,
            npad="auto",
            n_jobs=config.n_jobs,
            verbose=False,
        )

    attach_montage(
        raw,
        config,
    )

    raw = run_prep(
        raw,
        config,
    )

    raw = apply_filters(
        raw,
        config,
    )

    raw = apply_eog_regression(
        raw,
        config,
    )

    report_name = (
        f"{spec.subject_bids}_{spec.session_bids}_"
        f"task-{spec.task_bids}_run-{spec.run_index:02d}_ica.csv"
    )

    raw = apply_ica(
        raw,
        config,
        config.output_h5.parent
        / "ica_reports"
        / report_name,
    )

    return raw


def _phase_interval(
    trial: dict[str, float | int],
    phase: str,
    next_reading_onset: float | None,
) -> tuple[float, float]:
    if phase == "reading":
        return (
            float(trial["reading_onset"]),
            float(trial["recall_onset"]),
        )

    if phase == "recall":
        return (
            float(trial["recall_onset"]),
            float(trial["rest_onset"]),
        )

    if phase == "rest":
        if next_reading_onset is None:
            raise ValueError(
                "Cannot infer final rest stop without a following reading onset."
            )
        return (
            float(trial["rest_onset"]),
            float(next_reading_onset),
        )

    raise ValueError(
        f"Unsupported phase: {phase}"
    )


def extract_segments(
    raw: mne.io.BaseRaw,
    spec: RecordingSpec,
    config: PreprocessConfig,
) -> list[TrialSegment]:
    """Extract variable-length EEG segments and align them to stimulus rows."""
    events = load_events(
        spec.events_path
    )
    trials = parse_trial_boundaries(
        events
    )

    stimulus_path = stimulus_file_for_recording(
        config.stimuli_dir,
        spec,
    )
    stimuli = load_stimulus_table(
        stimulus_path
    )

    if len(stimuli) != len(trials):
        raise ValueError(
            "Stimulus/event trial count mismatch for "
            f"{spec.eeg_path.name}: "
            f"{len(stimuli)} stimulus rows vs {len(trials)} trials. "
            "The reconstruction intentionally refuses to guess the alignment."
        )

    eeg_picks = mne.pick_types(
        raw.info,
        eeg=True,
        exclude=[],
    )

    if len(eeg_picks) == 0:
        raise RuntimeError(
            "No EEG channels remain after preprocessing."
        )

    segments: list[TrialSegment] = []
    sfreq = float(raw.info["sfreq"])
    repeat_index = _repeat_index(spec)

    for i, trial in enumerate(trials):
        next_reading = None
        if i + 1 < len(trials):
            next_reading = float(
                trials[i + 1]["reading_onset"]
            )

        onset, stop = _phase_interval(
            trial,
            config.phase,
            next_reading,
        )

        start_idx, stop_idx = raw.time_as_index(
            [onset, stop],
            use_rounding=True,
        )

        if stop_idx <= start_idx:
            raise ValueError(
                f"Invalid segment interval: {onset} -> {stop}"
            )

        data = raw.get_data(
            picks=eeg_picks,
            start=int(start_idx),
            stop=int(stop_idx),
        ).astype(np.float32, copy=False)

        row = stimuli.iloc[i]

        segments.append(
            TrialSegment(
                trial_index=i,
                text=str(row["text"]),
                category=int(row["category"]),
                phase=config.phase,
                onset=onset,
                stop=stop,
                eeg=data,
                sfreq=sfreq,
                repeat_index=repeat_index,
                source_edf=str(spec.eeg_path),
                session=spec.session_bids,
                run=spec.run_index,
                stimulus_file=stimulus_path.name,
            )
        )

    return segments


def autoreject_equal_length_buckets(
    segments: list[TrialSegment],
    raw: mne.io.BaseRaw,
    config: PreprocessConfig,
) -> list[TrialSegment]:
    """
    Apply AutoReject separately to exact-length buckets.

    This preserves the variable-length dataset while satisfying MNE Epochs'
    equal-shape requirement inside each AutoReject fit.
    """
    if not config.use_autoreject:
        return segments

    try:
        from autoreject import AutoReject
    except ImportError as exc:
        raise ImportError(
            "autoreject is required. Install it with: pip install autoreject"
        ) from exc

    by_length: dict[int, list[TrialSegment]] = {}
    for segment in segments:
        by_length.setdefault(
            segment.n_times,
            [],
        ).append(segment)

    eeg_picks = mne.pick_types(
        raw.info,
        eeg=True,
        exclude=[],
    )
    eeg_info = mne.pick_info(
        raw.info,
        eeg_picks,
        copy=True,
    )

    cleaned: list[TrialSegment] = []

    for n_times, bucket in sorted(by_length.items()):
        if len(bucket) < config.autoreject_min_bucket_size:
            LOGGER.warning(
                "AutoReject skipped for n_times=%d because bucket size=%d "
                "< minimum=%d.",
                n_times,
                len(bucket),
                config.autoreject_min_bucket_size,
            )
            cleaned.extend(bucket)
            continue

        data = np.stack(
            [segment.eeg for segment in bucket],
            axis=0,
        )

        epochs = mne.EpochsArray(
            data,
            eeg_info,
            tmin=0.0,
            verbose=False,
        )

        ar = AutoReject(
            random_state=config.random_state,
            n_jobs=config.n_jobs,
            verbose=False,
        )

        epochs_clean, reject_log = ar.fit_transform(
            epochs,
            return_log=True,
        )

        keep_mask = ~np.asarray(
            reject_log.bad_epochs,
            dtype=bool,
        )

        clean_array = epochs_clean.get_data(
            copy=True
        ).astype(np.float32)

        clean_cursor = 0
        for keep, segment in zip(
            keep_mask,
            bucket,
            strict=True,
        ):
            if not keep:
                LOGGER.info(
                    "AutoReject dropped trial %s / rep %s / n_times=%s",
                    segment.trial_index,
                    segment.repeat_index,
                    n_times,
                )
                continue

            segment.eeg = clean_array[
                clean_cursor
            ]
            clean_cursor += 1
            cleaned.append(segment)

    cleaned.sort(
        key=lambda item: item.trial_index
    )
    return cleaned


def _sentence_key(
    text: str,
) -> str:
    digest = hashlib.sha1(
        text.encode("utf-8")
    ).hexdigest()[:12]
    return f"sentence_{digest}"


def write_segments_to_h5(
    h5: h5py.File,
    spec: RecordingSpec,
    segments: list[TrialSegment],
    config: PreprocessConfig,
) -> None:
    """Write segments in a hierarchy compatible with COFETT similarity.py."""
    subject_group = (
        h5.require_group(spec.paradigm_h5)
        .require_group(spec.subject_h5)
    )

    for segment in segments:
        category_group = subject_group.require_group(
            str(segment.category)
        )

        sentence_group = category_group.require_group(
            _sentence_key(segment.text)
        )

        sentence_group.attrs["text"] = segment.text
        sentence_group.attrs["category"] = segment.category
        sentence_group.attrs["paradigm"] = spec.paradigm_h5

        eeg_group = sentence_group.require_group(
            "eeg"
        )

        dataset_name = (
            f"rep_{segment.repeat_index:02d}"
        )

        if dataset_name in eeg_group:
            if not config.overwrite:
                LOGGER.info(
                    "Skipping existing dataset %s/%s",
                    sentence_group.name,
                    dataset_name,
                )
                continue
            del eeg_group[dataset_name]

        ds = eeg_group.create_dataset(
            dataset_name,
            data=segment.eeg,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        ds.attrs["sfreq"] = segment.sfreq
        ds.attrs["n_channels"] = segment.eeg.shape[0]
        ds.attrs["n_times"] = segment.n_times
        ds.attrs["duration"] = segment.duration
        ds.attrs["phase"] = segment.phase
        ds.attrs["trial_index"] = segment.trial_index
        ds.attrs["repeat_index"] = segment.repeat_index
        ds.attrs["session"] = segment.session
        ds.attrs["run"] = segment.run
        ds.attrs["onset"] = segment.onset
        ds.attrs["stop"] = segment.stop
        ds.attrs["source_edf"] = segment.source_edf
        ds.attrs["stimulus_file"] = segment.stimulus_file

        # Create the expected location used by public similarity.py.
        # Embedding datasets can be attached later:
        #
        # embeddings/<lang>/<emb_type>
        sentence_group.require_group("embeddings")


def process_recording(
    spec: RecordingSpec,
    config: PreprocessConfig,
    h5: h5py.File,
) -> None:
    raw = preprocess_continuous_recording(
        spec,
        config,
    )

    segments = extract_segments(
        raw,
        spec,
        config,
    )

    segments = autoreject_equal_length_buckets(
        segments,
        raw,
        config,
    )

    write_segments_to_h5(
        h5,
        spec,
        segments,
        config,
    )

    LOGGER.info(
        "Saved %d %s segments from %s",
        len(segments),
        config.phase,
        spec.eeg_path.name,
    )


def preprocess_dataset(
    config: PreprocessConfig,
    subjects: Sequence[str] | None = None,
) -> None:
    config.output_h5.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    specs = discover_recordings(
        config.bids_root,
        subjects=subjects,
    )

    mode = "a"

    with h5py.File(
        config.output_h5,
        mode,
    ) as h5:
        h5.attrs["dataset"] = "Chisco-2.0 / COFETT reconstruction"
        h5.attrs["target_sfreq"] = config.target_sfreq
        h5.attrs["phase"] = config.phase
        h5.attrs["event_reading"] = READING_EVENT
        h5.attrs["event_recall"] = RECALL_EVENT
        h5.attrs["event_rest"] = REST_EVENT
        h5.attrs["reconstructed_pipeline"] = True
        h5.attrs[
            "reconstruction_note"
        ] = (
            "Original COFETT preprocessing.py is not publicly available; "
            "this HDF5 was generated by a source-grounded reconstruction."
        )

        for index, spec in enumerate(
            specs,
            start=1,
        ):
            LOGGER.info(
                "[%d/%d] %s",
                index,
                len(specs),
                spec.eeg_path,
            )
            process_recording(
                spec,
                config,
                h5,
            )
            h5.flush()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstructed Chisco-2.0 / COFETT EEG preprocessing."
        )
    )

    parser.add_argument(
        "--bids-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--stimuli-dir",
        type=Path,
        required=True,
        help="Directory containing text1-1.xlsx ... text1-4.xlsx and text2.xlsx",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/target.h5"),
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
    )
    parser.add_argument(
        "--phase",
        choices=["reading", "recall", "rest"],
        default="recall",
    )
    parser.add_argument(
        "--montage",
        default="standard_1005",
    )
    parser.add_argument(
        "--montage-file",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--strict-montage",
        action="store_true",
    )
    parser.add_argument(
        "--no-prep",
        action="store_true",
    )
    parser.add_argument(
        "--no-eog-regression",
        action="store_true",
    )
    parser.add_argument(
        "--no-ica",
        action="store_true",
    )
    parser.add_argument(
        "--ica-report-only",
        action="store_true",
        help="Run ICA+ICLabel but do not automatically remove components.",
    )
    parser.add_argument(
        "--no-autoreject",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
    )

    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    args = build_arg_parser().parse_args()

    config = PreprocessConfig(
        bids_root=args.bids_root,
        stimuli_dir=args.stimuli_dir,
        output_h5=args.output,
        phase=args.phase,
        montage_name=args.montage,
        montage_file=args.montage_file,
        strict_montage=args.strict_montage,
        use_prep=not args.no_prep,
        use_eog_regression=not args.no_eog_regression,
        use_ica=not args.no_ica,
        auto_apply_ica=not args.ica_report_only,
        use_autoreject=not args.no_autoreject,
        n_jobs=args.n_jobs,
        overwrite=args.overwrite,
    )

    preprocess_dataset(
        config,
        subjects=args.subjects,
    )


if __name__ == "__main__":
    main()
