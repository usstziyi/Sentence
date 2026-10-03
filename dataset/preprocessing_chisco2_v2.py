# MIT License
#
# Copyright (c) 2024 Yu Bao, Harbin Institute of Technology
#
# This file is a Chisco-2.0 adaptation of the official Chisco 1.0
# preprocessing.py released under the MIT License:
# https://github.com/zhangzihan-is-good/Chisco
#
# Modifications:
# - BIDS/OpenNeuro ds006317 directory support
# - Chisco-2.0 event parsing
# - variable-length Reading / Recall / Rest segmentation
# - optional EOG regression described in the COFETT paper
# - AutoReject grouped by equal temporal length
# - ICA adapted to variable-length segments
# - optional HDF5 output compatible with the reconstructed COFETT data_loader.py
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import re
import string
import warnings
from collections import defaultdict
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd

from pyprep.prep_pipeline import PrepPipeline
from mne_icalabel import label_components


# =============================================================================
# 1. Chisco-2.0 constants
# =============================================================================

SAMPLE_RATE = 500
IC_NUM = 30

# Same channel-removal list as the official Chisco 1.0 preprocessing.py.
USELESS_CHANNELS = [
    "11", "110", "EKG", "EMG",
    "84", "85", "10", "111",
]

EOG_CHANNELS = ["VEO", "HEO"]

# -------------------------------------------------------------------------
# Chisco-2.0 event codes in public ds006317 *_events.tsv
#
# The public PsychoPy code sends:
#   50  -> word-by-word Reading routine
#   100 -> Recall / inner-speech routine
#   101 -> blank/rest routine
#
# In the public BIDS export these per-trial markers appear as:
#   65329 -> Reading onset
#   65379 -> Recall onset
#   65381 -> Rest onset
#
# 65480 / 65481 also occur, but they are not the three per-trial phase markers.
# -------------------------------------------------------------------------
READING_EVENT = 65329
RECALL_EVENT = 65379
REST_EVENT = 65381

# PsychoPy skips these characters when advancing the 0.4 s highlighting timer.
PUNCTUATION = set(string.punctuation + "，。？！：；、“”‘’（）()《》〈〉【】[]…—· ")

# The experiment uses:
#   Reading = 0.4 * x seconds
#   Recall  = 0.4 * (x + 1) seconds
#   Rest    = 1.8 seconds
CHAR_DURATION = 0.4
REST_DURATION = 1.8

SENTENCE_COLUMN_CANDIDATES = [
    "句子", "sentence", "text", "中文",
]

CATEGORY_COLUMN_CANDIDATES = [
    "类别", "分类", "category", "category_id",
    "class", "label", "标签",
]


# =============================================================================
# 2. Arguments
# =============================================================================

parser = argparse.ArgumentParser(
    description="Chisco-2.0 preprocessing based on the official Chisco 1.0 pipeline",
    formatter_class=argparse.RawTextHelpFormatter,
)

parser.add_argument(
    "--bids_root",
    type=str,
    required=True,
    help="Root directory of OpenNeuro ds006317, e.g. D:/AI/ds006317",
)
parser.add_argument(
    "--stimulus_dir",
    type=str,
    required=True,
    help=(
        "COFETT psychopy folder containing text1-1.xlsx ... text1-4.xlsx "
        "and text2.xlsx"
    ),
)
parser.add_argument(
    "--montage_file",
    type=str,
    default="montage.csv",
    help=(
        "Custom montage CSV. The official Chisco 1.0 montage.csv is the "
        "recommended starting point because Chisco-2.0 uses the same acquisition setup."
    ),
)
parser.add_argument(
    "--subject",
    "-i",
    type=str,
    default="sub-01",
    help="Subject, e.g. sub-01 or sub-02",
)
parser.add_argument(
    "--task",
    choices=["all", "para1", "para2"],
    default="all",
    help="Only preprocess one paradigm or both",
)
parser.add_argument(
    "--method_str",
    "-m",
    default="chisco2",
    help="Identifier used in output folder names",
)
parser.add_argument(
    "--count_limit",
    "-c",
    type=int,
    default=10086,
)
parser.add_argument(
    "--test",
    "-t",
    action="store_true",
    help="Only process the first recording",
)

# Keep the official Chisco 1.0 switch semantics:
# --not_prep / --not_reject disable steps that are ON by default.
parser.add_argument(
    "--not_prep",
    "-p",
    dest="prep",
    action="store_false",
    help="Do not run PREP",
)
parser.set_defaults(prep=True)

parser.add_argument(
    "--not_reject",
    "-a",
    dest="autoreject",
    action="store_false",
    help="Do not run AutoReject",
)
parser.set_defaults(autoreject=True)

parser.add_argument(
    "--ica",
    "-I",
    action="store_true",
    help=(
        "Run extended-Infomax ICA + ICLabel. "
        "Like the original Chisco script, ICA is OFF unless this flag is supplied."
    ),
)
parser.add_argument(
    "--ransac",
    "-R",
    action="store_true",
    help="Enable RANSAC inside PREP",
)
parser.add_argument(
    "--step",
    "-s",
    action="store_true",
    help="Save intermediate continuous FIF files",
)

parser.add_argument(
    "--no_eog_regression",
    dest="eog_regression",
    action="store_false",
    help=(
        "Disable EOG regression. COFETT Appendix E describes multivariate "
        "H-EOG/V-EOG regression; it was not present in the Chisco 1.0 script."
    ),
)
parser.set_defaults(eog_regression=True)

parser.add_argument(
    "--ica_report_only",
    action="store_true",
    help=(
        "Fit ICA and save ICLabel reports but do not automatically remove components. "
        "Useful for reproducing the COFETT paper's manual IC review."
    ),
)

parser.add_argument(
    "--min_autoreject_bucket",
    type=int,
    default=8,
    help=(
        "Minimum number of equal-length trials required to fit AutoReject. "
        "Smaller buckets are kept unchanged."
    ),
)
parser.add_argument(
    "--n_jobs",
    type=int,
    default=1,
)
parser.add_argument(
    "--output_root",
    type=str,
    default="preprocess_output",
)
parser.add_argument(
    "--h5_output",
    type=str,
    default=None,
    help=(
        "Optional HDF5 file. If given, Recall segments are appended using the "
        "layout expected by the reconstructed data_loader.py."
    ),
)
parser.add_argument(
    "--overwrite_h5",
    action="store_true",
)

args = parser.parse_args()


# =============================================================================
# 3. Paths / runtime settings
# =============================================================================

BIDS_ROOT = Path(args.bids_root).expanduser().resolve()
STIMULUS_DIR = Path(args.stimulus_dir).expanduser().resolve()
MONTAGE_FILE = Path(args.montage_file).expanduser().resolve()
SUBJECT = normalize_subject_arg = args.subject
PREP = args.prep
AUTO_REJECT = args.autoreject
ICA = args.ica
RANSAC = args.ransac
TEST = args.test
STEP = args.step

OUTPUT_FOLDER = (
    Path(args.output_root)
    / f"{args.method_str}_{SUBJECT.replace('-', '')}"
)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

for subdir in [
    "pkl",
    "fif",
    "ica_reports",
    "logs",
]:
    (OUTPUT_FOLDER / subdir).mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# 4. Helpers: BIDS / stimulus metadata
# =============================================================================

def normalize_subject(subject: str) -> tuple[str, str]:
    """
    Return:
        BIDS name: sub-01
        HDF5 name: sub01
    """
    subject = subject.strip()

    if subject.startswith("sub-"):
        number = int(subject[4:])
    elif subject.startswith("sub"):
        number = int(subject[3:])
    else:
        number = int(subject)

    return f"sub-{number:02d}", f"sub{number:02d}"


def parse_bids_recording(edf_path: Path) -> dict:
    pattern = re.compile(
        r"(?P<subject>sub-\d+)_"
        r"(?P<session>ses-\d+)_"
        r"task-(?P<task>para\d+)_"
        r"run-(?P<run>\d+)_eeg\.edf$"
    )

    match = pattern.fullmatch(edf_path.name)
    if not match:
        raise ValueError(
            f"Unexpected Chisco-2.0 filename: {edf_path.name}"
        )

    session = match.group("session")
    session_index = int(session.split("-")[1])
    run_index = int(match.group("run"))
    task = match.group("task")

    return {
        "subject_bids": match.group("subject"),
        "session": session,
        "session_index": session_index,
        "task": task,
        "run_index": run_index,
    }


def get_sidecar_paths(edf_path: Path) -> tuple[Path, Path]:
    stem = edf_path.name.removesuffix("_eeg.edf")
    events_path = edf_path.with_name(
        f"{stem}_events.tsv"
    )
    channels_path = edf_path.with_name(
        f"{stem}_channels.tsv"
    )

    if not events_path.exists():
        raise FileNotFoundError(
            f"Missing events TSV: {events_path}"
        )
    if not channels_path.exists():
        raise FileNotFoundError(
            f"Missing channels TSV: {channels_path}"
        )

    return events_path, channels_path


def stimulus_file_for_recording(
    task: str,
    run_index: int,
) -> Path:
    """
    Chisco-2.0 design:

    para1:
        Text 1 consists of four 281-sentence session lists.
        Run k -> text1-k.xlsx.

    para2:
        The same 252-sentence Text 2 list is repeated across the
        4 sessions/day x 4 days.
        -> text2.xlsx.
    """
    if task in {"para1", "para01"}:
        path = STIMULUS_DIR / f"text1-{run_index}.xlsx"
    elif task in {"para2", "para02"}:
        path = STIMULUS_DIR / "text2.xlsx"
    else:
        raise ValueError(
            f"Unsupported task: {task}"
        )

    if not path.exists():
        raise FileNotFoundError(
            f"Stimulus spreadsheet not found: {path}"
        )

    return path


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
) -> str | None:
    normalized = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        key = candidate.lower()
        if key in normalized:
            return normalized[key]

    if required:
        raise KeyError(
            f"Could not find a column among {candidates}. "
            f"Spreadsheet columns: {list(df.columns)}"
        )

    return None


def load_stimulus_metadata(
    task: str,
    run_index: int,
) -> tuple[pd.DataFrame, Path]:
    stimulus_path = stimulus_file_for_recording(
        task,
        run_index,
    )

    df = pd.read_excel(
        stimulus_path
    )

    sentence_col = find_column(
        df,
        SENTENCE_COLUMN_CANDIDATES,
        required=True,
    )

    category_col = find_column(
        df,
        CATEGORY_COLUMN_CANDIDATES,
        required=False,
    )

    metadata = pd.DataFrame(
        {
            "text": df[sentence_col]
            .fillna("")
            .astype(str),
        }
    )

    if category_col is not None:
        metadata["category"] = pd.to_numeric(
            df[category_col],
            errors="coerce",
        ).fillna(-1).astype(int)
    else:
        warnings.warn(
            f"No category column found in {stimulus_path.name}; "
            "category will be set to -1.",
            RuntimeWarning,
        )
        metadata["category"] = -1

    metadata["sentence_index"] = np.arange(
        len(metadata),
        dtype=int,
    )

    return metadata, stimulus_path


# =============================================================================
# 5. Helpers: channels / montage / PREP / filters
# =============================================================================

def apply_channel_types(
    raw: mne.io.BaseRaw,
    channels_path: Path,
) -> None:
    """
    Use BIDS channels.tsv when possible, while preserving the original
    Chisco VEO/HEO handling.
    """
    channels_df = pd.read_csv(
        channels_path,
        sep="\t",
    )

    mapping = {}
    raw_names = set(raw.ch_names)

    if {"name", "type"}.issubset(
        channels_df.columns
    ):
        bids_to_mne = {
            "EEG": "eeg",
            "EOG": "eog",
            "ECG": "ecg",
            "EMG": "emg",
            "TRIG": "stim",
            "STIM": "stim",
        }

        for _, row in channels_df.iterrows():
            name = str(row["name"])
            channel_type = str(
                row["type"]
            ).upper()

            if (
                name in raw_names
                and channel_type in bids_to_mne
            ):
                mapping[name] = bids_to_mne[
                    channel_type
                ]

    # Keep explicit VEO / HEO behavior from Chisco 1.0.
    for eog_name in EOG_CHANNELS:
        if eog_name in raw_names:
            mapping[eog_name] = "eog"

    if mapping:
        raw.set_channel_types(
            mapping,
            verbose=False,
        )


def run_prep(
    raw: mne.io.BaseRaw,
    custom_montage,
) -> mne.io.BaseRaw:
    if not PREP:
        print("Not running PyPREP")
        return raw.copy()

    print("Running PyPREP")

    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": np.arange(
            50,
            SAMPLE_RATE / 2,
            50,
        ),
    }

    prep = PrepPipeline(
        raw,
        prep_params,
        custom_montage,
        ransac=RANSAC,
    )
    prep.fit()

    return prep.raw


def apply_cofett_filters(
    raw: mne.io.BaseRaw,
) -> mne.io.BaseRaw:
    """
    COFETT Appendix E:
      - 50-Hz power-line notch filtering
      - 1-Hz zero-phase FIR high-pass
      - no low-pass

    Figure 8 also depicts the line harmonics (100, 150, 200 Hz), so this
    implementation removes all 50-Hz harmonics below Nyquist.
    """
    raw = raw.copy()

    picks = mne.pick_types(
        raw.info,
        eeg=True,
        eog=True,
        exclude=[],
    )

    notch_freqs = np.arange(
        50,
        SAMPLE_RATE / 2,
        50,
    )

    raw.notch_filter(
        freqs=notch_freqs,
        picks=picks,
        method="fir",
        phase="zero",
        n_jobs=args.n_jobs,
        verbose=False,
    )

    raw.filter(
        l_freq=1.0,
        h_freq=None,
        picks=picks,
        method="fir",
        phase="zero",
        n_jobs=args.n_jobs,
        verbose=False,
    )

    return raw


def apply_eog_regression(
    raw: mne.io.BaseRaw,
) -> mne.io.BaseRaw:
    """
    COFETT Appendix E describes multivariate EOG regression using H-EOG/V-EOG.

    Exact code/order was not released. This implementation applies MNE's
    EOGRegression to the continuous filtered signal before segmentation.
    """
    if not args.eog_regression:
        print("Not running EOG regression")
        return raw

    existing_eog = [
        name
        for name in EOG_CHANNELS
        if name in raw.ch_names
    ]

    if not existing_eog:
        warnings.warn(
            "VEO/HEO were not found; skipping EOG regression.",
            RuntimeWarning,
        )
        return raw

    from mne.preprocessing import EOGRegression

    print(
        "Running EOG regression with:",
        existing_eog,
    )

    regression = EOGRegression(
        picks="eeg",
        picks_artifact=existing_eog,
        proj=False,
    )

    regression.fit(
        raw,
    )

    return regression.apply(
        raw,
        copy=True,
    )


# =============================================================================
# 6. Chisco-2.0 event parsing
# =============================================================================

def load_trial_table(
    events_path: Path,
    stimulus_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one row per trial.

    We use the three BIDS phase markers as onset anchors and the documented
    experimental timing to make exact-length segments:

        reading duration = 0.4 * x
        recall duration  = 0.4 * (x + 1)
        rest duration    = 1.8

    x is inferred from the observed gap:
        recall_onset - reading_onset ~= 0.4 * x

    This is preferable to using raw marker-to-marker sample counts directly,
    because monitor-refresh jitter would otherwise create many nearly-identical
    sequence lengths and make AutoReject / later length-bucket training harder.
    """
    events = pd.read_csv(
        events_path,
        sep="\t",
    )

    required = {
        "onset",
        "value",
    }
    missing = required - set(
        events.columns
    )
    if missing:
        raise KeyError(
            f"{events_path} missing columns: {sorted(missing)}"
        )

    events = events.copy()
    events["value"] = pd.to_numeric(
        events["value"],
        errors="coerce",
    )

    events = events[
        events["value"].isin(
            [
                READING_EVENT,
                RECALL_EVENT,
                REST_EVENT,
            ]
        )
    ].reset_index(drop=True)

    rows = []
    index = 0

    while index + 2 < len(events):
        triple = events.iloc[
            index : index + 3
        ]

        codes = triple[
            "value"
        ].astype(int).tolist()

        expected = [
            READING_EVENT,
            RECALL_EVENT,
            REST_EVENT,
        ]

        if codes != expected:
            raise ValueError(
                "Unexpected Chisco-2.0 phase-marker sequence near "
                f"row {index}: {codes}; expected {expected}"
            )

        reading_onset = float(
            triple.iloc[0]["onset"]
        )
        recall_onset = float(
            triple.iloc[1]["onset"]
        )
        rest_onset = float(
            triple.iloc[2]["onset"]
        )

        observed_reading_duration = (
            recall_onset
            - reading_onset
        )

        # x = number of actually highlighted non-punctuation characters.
        x = int(
            round(
                observed_reading_duration
                / CHAR_DURATION
            )
        )

        if x <= 0:
            raise ValueError(
                f"Invalid inferred sentence length x={x}"
            )

        expected_reading_duration = (
            CHAR_DURATION * x
        )
        expected_recall_duration = (
            CHAR_DURATION * (x + 1)
        )

        observed_recall_duration = (
            rest_onset
            - recall_onset
        )

        # Monitor refresh / marker delivery produces small timing deviations.
        if abs(
            observed_reading_duration
            - expected_reading_duration
        ) > 0.15:
            warnings.warn(
                "Reading timing deviates from the documented 0.4*x rule: "
                f"observed={observed_reading_duration:.3f}s, "
                f"expected={expected_reading_duration:.3f}s",
                RuntimeWarning,
            )

        if abs(
            observed_recall_duration
            - expected_recall_duration
        ) > 0.20:
            warnings.warn(
                "Recall timing deviates from the documented 0.4*(x+1) rule: "
                f"observed={observed_recall_duration:.3f}s, "
                f"expected={expected_recall_duration:.3f}s",
                RuntimeWarning,
            )

        rows.append(
            {
                "trial_index": len(rows),
                "reading_onset": reading_onset,
                "recall_onset": recall_onset,
                "rest_onset": rest_onset,
                "x": x,
                "reading_duration": expected_reading_duration,
                "recall_duration": expected_recall_duration,
                "rest_duration": REST_DURATION,
                "observed_reading_duration": observed_reading_duration,
                "observed_recall_duration": observed_recall_duration,
            }
        )

        index += 3

    trial_table = pd.DataFrame(
        rows
    )

    if len(trial_table) != len(
        stimulus_metadata
    ):
        raise ValueError(
            "EEG trial count != stimulus row count.\n"
            f"events: {len(trial_table)} trials\n"
            f"stimuli: {len(stimulus_metadata)} rows\n"
            "Refusing to silently truncate because that can misalign EEG and text."
        )

    trial_table = pd.concat(
        [
            trial_table.reset_index(
                drop=True
            ),
            stimulus_metadata.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    return trial_table


# =============================================================================
# 7. Variable-length segmentation
# =============================================================================

def extract_segment(
    raw: mne.io.BaseRaw,
    eeg_picks,
    onset: float,
    duration: float,
) -> np.ndarray:
    """
    Return EEG with shape:
        (C, T)

    Stop is exclusive so a nominal 2.4 s segment at 500 Hz has exactly:
        2.4 * 500 = 1200 samples
    """
    sfreq = float(
        raw.info["sfreq"]
    )

    start_sample = int(
        round(onset * sfreq)
    )
    n_samples = int(
        round(duration * sfreq)
    )
    stop_sample = (
        start_sample + n_samples
    )

    if stop_sample > raw.n_times:
        raise ValueError(
            "Segment exceeds the available EEG samples: "
            f"onset={onset:.3f}, duration={duration:.3f}"
        )

    return raw.get_data(
        picks=eeg_picks,
        start=start_sample,
        stop=stop_sample,
    ).astype(
        np.float32,
        copy=False,
    )


def segment_recording(
    raw: mne.io.BaseRaw,
    trial_table: pd.DataFrame,
    source_edf: Path,
    recording_info: dict,
    stimulus_file: Path,
) -> dict[str, list[dict]]:
    eeg_picks = mne.pick_types(
        raw.info,
        eeg=True,
        exclude=[],
    )

    phase_segments = {
        "reading": [],
        "recall": [],
        "rest": [],
    }

    for _, row in trial_table.iterrows():
        common = {
            "text": str(
                row["text"]
            ),
            "category": int(
                row["category"]
            ),
            "sentence_index": int(
                row["sentence_index"]
            ),
            "trial_index": int(
                row["trial_index"]
            ),
            "x": int(
                row["x"]
            ),
            "sfreq": float(
                raw.info["sfreq"]
            ),
            "source_edf": str(
                source_edf
            ),
            "session": recording_info[
                "session"
            ],
            "run": int(
                recording_info[
                    "run_index"
                ]
            ),
            "task": recording_info[
                "task"
            ],
            "stimulus_file": stimulus_file.name,
        }

        phase_specs = {
            "reading": (
                float(
                    row["reading_onset"]
                ),
                float(
                    row["reading_duration"]
                ),
            ),
            "recall": (
                float(
                    row["recall_onset"]
                ),
                float(
                    row["recall_duration"]
                ),
            ),
            "rest": (
                float(
                    row["rest_onset"]
                ),
                float(
                    row["rest_duration"]
                ),
            ),
        }

        for phase, (
            onset,
            duration,
        ) in phase_specs.items():
            eeg = extract_segment(
                raw,
                eeg_picks,
                onset,
                duration,
            )

            item = {
                **common,
                "phase": phase,
                "onset": onset,
                "duration": duration,
                "n_times": int(
                    eeg.shape[-1]
                ),
                "input_features": eeg,
            }

            phase_segments[
                phase
            ].append(item)

    return phase_segments


# =============================================================================
# 8. AutoReject for variable-length Chisco-2.0
# =============================================================================

def autoreject_segments(
    segments: list[dict],
    raw: mne.io.BaseRaw,
    phase: str,
) -> list[dict]:
    """
    Chisco 1.0 can put all trials into one MNE Epochs object because its
    Reading/Imagine windows are fixed-length.

    Chisco-2.0 cannot: sentence length x changes T.

    Solution:
        group by n_times -> AutoReject within each equal-length group.

    This keeps the true variable length and introduces no zero-padding.
    """
    if not AUTO_REJECT:
        print(
            f"Not running AutoReject for {phase}"
        )
        return segments

    from autoreject import AutoReject

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

    by_length = defaultdict(
        list
    )

    for item in segments:
        by_length[
            item["n_times"]
        ].append(item)

    cleaned = []

    print(
        f"AutoReject [{phase}] "
        f"{len(segments)} trials / "
        f"{len(by_length)} length buckets"
    )

    for n_times in sorted(
        by_length
    ):
        bucket = by_length[
            n_times
        ]

        if (
            len(bucket)
            < args.min_autoreject_bucket
        ):
            print(
                f"  skip T={n_times}: "
                f"only {len(bucket)} trials"
            )
            cleaned.extend(
                bucket
            )
            continue

        data = np.stack(
            [
                item["input_features"]
                for item in bucket
            ],
            axis=0,
        )

        epochs = mne.EpochsArray(
            data,
            eeg_info,
            tmin=0.0,
            verbose=False,
        )

        # Keep CV valid for smaller buckets.
        cv = min(
            10,
            len(bucket),
        )

        ar = AutoReject(
            cv=cv,
            n_jobs=args.n_jobs,
            random_state=97,
            verbose=False,
        )

        epochs_clean, reject_log = (
            ar.fit_transform(
                epochs,
                return_log=True,
            )
        )

        bad_mask = np.asarray(
            reject_log.bad_epochs,
            dtype=bool,
        )

        clean_data = (
            epochs_clean
            .get_data(copy=True)
            .astype(np.float32)
        )

        clean_cursor = 0

        for is_bad, item in zip(
            bad_mask,
            bucket,
            strict=True,
        ):
            if is_bad:
                print(
                    "  dropped:",
                    f"trial={item['trial_index']}",
                    f"T={n_times}",
                )
                continue

            item = item.copy()
            item[
                "input_features"
            ] = clean_data[
                clean_cursor
            ]
            clean_cursor += 1

            cleaned.append(
                item
            )

    cleaned.sort(
        key=lambda item: item[
            "trial_index"
        ]
    )

    return cleaned


# =============================================================================
# 9. ICA adapted to variable-length segments
# =============================================================================

def ica_clean_segments(
    segments: list[dict],
    raw: mne.io.BaseRaw,
    phase: str,
    recording_tag: str,
) -> list[dict]:
    """
    Official Chisco 1.0:
        fit ICA separately on fixed-length Reading and Imagine Epochs.

    Chisco-2.0:
        concatenate all surviving equal-channel segments for one phase along
        time -> fit one ICA -> clean -> split back to original variable lengths.

    This preserves variable T while staying close to the official logic.
    """
    if not ICA:
        return segments

    if not segments:
        return segments

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

    lengths = [
        item["input_features"].shape[-1]
        for item in segments
    ]

    concat_data = np.concatenate(
        [
            item["input_features"]
            for item in segments
        ],
        axis=-1,
    )

    concat_raw = mne.io.RawArray(
        concat_data,
        eeg_info,
        verbose=False,
    )

    n_components = min(
        IC_NUM,
        len(eeg_picks) - 1,
    )

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        random_state=97,
        max_iter="auto",
        method="infomax",
        fit_params=dict(
            extended=True,
        ),
    )

    ica.fit(
        concat_raw,
        picks="eeg",
        verbose=False,
    )

    ic_labels = label_components(
        concat_raw,
        ica,
        method="iclabel",
    )

    labels = list(
        ic_labels["labels"]
    )
    probabilities = np.asarray(
        ic_labels["y_pred_proba"]
    )

    # Same automatic decision rule as the official Chisco 1.0 preprocessing.py.
    auto_exclude = [
        idx
        for idx, label in enumerate(
            labels
        )
        if label not in [
            "brain",
            "other",
        ]
    ]

    report = pd.DataFrame(
        {
            "component": np.arange(
                len(labels)
            ),
            "label": labels,
            "probability": probabilities,
            "auto_exclude": [
                idx in auto_exclude
                for idx in range(
                    len(labels)
                )
            ],
        }
    )

    report_path = (
        OUTPUT_FOLDER
        / "ica_reports"
        / f"{recording_tag}_{phase}_iclabel.csv"
    )

    report.to_csv(
        report_path,
        index=False,
    )

    print(
        f"{phase} ICLabel report:",
        report_path,
    )
    print(
        f"{phase} auto-exclude:",
        auto_exclude,
    )

    if args.ica_report_only:
        print(
            "ICA report-only mode: "
            "no components were removed."
        )
        return segments

    ica.exclude = auto_exclude

    clean_raw = concat_raw.copy()

    ica.apply(
        clean_raw,
        exclude=auto_exclude,
        verbose=False,
    )

    clean_data = clean_raw.get_data()

    output = []
    cursor = 0

    for item, length in zip(
        segments,
        lengths,
        strict=True,
    ):
        updated = item.copy()
        updated[
            "input_features"
        ] = clean_data[
            :,
            cursor : cursor + length,
        ].astype(
            np.float32,
            copy=False,
        )

        output.append(
            updated
        )
        cursor += length

    return output


# =============================================================================
# 10. Save functions
# =============================================================================

def save_segments_to_pickle(
    segments: list[dict],
    edf_path: Path,
    phase: str,
    extra_str: str = "",
) -> Path:
    output_path = (
        OUTPUT_FOLDER
        / "pkl"
        / (
            f"{edf_path.stem}_"
            f"{phase}_{args.method_str}{extra_str}.pkl"
        )
    )

    data_to_save = []

    for item in segments:
        # Keep Chisco 1.0-compatible name "input_features".
        # Chisco 1.0 saved one epoch as (1,C,T); preserve that convention.
        data_to_save.append(
            {
                "text": item["text"],
                "category": item["category"],
                "input_features": item[
                    "input_features"
                ][None, ...],
                "phase": item["phase"],
                "trial_index": item[
                    "trial_index"
                ],
                "sentence_index": item[
                    "sentence_index"
                ],
                "x": item["x"],
                "n_times": item[
                    "n_times"
                ],
                "sfreq": item["sfreq"],
                "session": item[
                    "session"
                ],
                "run": item["run"],
                "task": item["task"],
            }
        )

    with open(
        output_path,
        "wb",
    ) as file:
        pickle.dump(
            data_to_save,
            file,
        )

    return output_path


def repeat_index(
    task: str,
    session_index: int,
    run_index: int,
) -> int:
    """
    Reconstructed repetition numbering from the public experimental design.

    para1:
        the same Text-1 run is repeated once/day across 4 days
        -> repetitions 1..4 = session index

    para2:
        the same Text-2 set is repeated across 4 sessions/day x 4 days
        -> repetitions 1..16
    """
    if task in {
        "para1",
        "para01",
    }:
        return session_index

    if task in {
        "para2",
        "para02",
    }:
        return (
            (session_index - 1) * 4
            + run_index
        )

    raise ValueError(
        f"Unsupported task: {task}"
    )


def sentence_key(
    sentence_index: int,
    text: str,
) -> str:
    digest = hashlib.sha1(
        text.encode("utf-8")
    ).hexdigest()[:8]

    return (
        f"sentence_{sentence_index:03d}_"
        f"{digest}"
    )


def append_recall_to_h5(
    recall_segments: list[dict],
    recording_info: dict,
) -> None:
    """
    Optional bridge to the reconstructed data_loader.py.

    Layout:
        /para01/sub01/<category>/<sentence>/eeg/rep_XX
        /para02/sub01/<category>/<sentence>/eeg/rep_XX
    """
    if args.h5_output is None:
        return

    h5_path = Path(
        args.h5_output
    ).expanduser().resolve()

    h5_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    _, subject_h5 = normalize_subject(
        recording_info[
            "subject_bids"
        ]
    )

    paradigm_h5 = (
        "para01"
        if recording_info[
            "task"
        ] in {"para1", "para01"}
        else "para02"
    )

    rep = repeat_index(
        recording_info["task"],
        recording_info[
            "session_index"
        ],
        recording_info[
            "run_index"
        ],
    )

    with h5py.File(
        h5_path,
        "a",
    ) as h5:
        h5.attrs[
            "target_sfreq"
        ] = SAMPLE_RATE

        h5.attrs[
            "preprocessing_origin"
        ] = (
            "Chisco-2.0 adaptation of official Chisco 1.0 preprocessing.py"
        )

        subject_group = (
            h5.require_group(
                paradigm_h5
            )
            .require_group(
                subject_h5
            )
        )

        for item in recall_segments:
            category_group = (
                subject_group
                .require_group(
                    str(
                        item[
                            "category"
                        ]
                    )
                )
            )

            key = sentence_key(
                item[
                    "sentence_index"
                ],
                item["text"],
            )

            sentence_group = (
                category_group
                .require_group(
                    key
                )
            )

            sentence_group.attrs[
                "text"
            ] = item["text"]
            sentence_group.attrs[
                "category"
            ] = item[
                "category"
            ]
            sentence_group.attrs[
                "sentence_index"
            ] = item[
                "sentence_index"
            ]

            eeg_group = (
                sentence_group
                .require_group(
                    "eeg"
                )
            )

            dataset_name = (
                f"rep_{rep:02d}"
            )

            if dataset_name in eeg_group:
                if not args.overwrite_h5:
                    raise FileExistsError(
                        f"HDF5 dataset already exists: "
                        f"{eeg_group.name}/{dataset_name}"
                    )

                del eeg_group[
                    dataset_name
                ]

            ds = eeg_group.create_dataset(
                dataset_name,
                data=item[
                    "input_features"
                ],
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

            ds.attrs[
                "sfreq"
            ] = item["sfreq"]
            ds.attrs[
                "n_times"
            ] = item[
                "input_features"
            ].shape[-1]
            ds.attrs[
                "phase"
            ] = "recall"
            ds.attrs[
                "session"
            ] = item[
                "session"
            ]
            ds.attrs[
                "run"
            ] = item["run"]

            # Reserved group for LaBSE/Jina/etc.
            sentence_group.require_group(
                "embeddings"
            )


# =============================================================================
# 11. One EDF recording
# =============================================================================

def process_edf_file(
    edf_path: Path,
) -> None:
    print(
        "\n"
        + "=" * 80
    )
    print(
        "Processing:",
        edf_path,
    )
    print(
        "=" * 80
    )

    recording_info = parse_bids_recording(
        edf_path
    )

    events_path, channels_path = (
        get_sidecar_paths(
            edf_path
        )
    )

    stimulus_metadata, stimulus_file = (
        load_stimulus_metadata(
            recording_info[
                "task"
            ],
            recording_info[
                "run_index"
            ],
        )
    )

    raw = mne.io.read_raw_edf(
        edf_path,
        preload=True,
        verbose=False,
    )

    print(
        "Original:",
        f"{raw.info['sfreq']} Hz,",
        f"{len(raw.ch_names)} channels,",
        f"{raw.n_times} samples",
    )

    # ---------------------------------------------------------------------
    # Same first operation as official Chisco 1.0.
    # ---------------------------------------------------------------------
    raw.resample(
        SAMPLE_RATE,
        npad="auto",
        n_jobs=args.n_jobs,
        verbose=False,
    )

    apply_channel_types(
        raw,
        channels_path,
    )

    if not MONTAGE_FILE.exists():
        raise FileNotFoundError(
            f"Montage file not found: {MONTAGE_FILE}\n"
            "Use the official Chisco montage.csv or another verified montage."
        )

    custom_montage = (
        mne.channels
        .read_custom_montage(
            MONTAGE_FILE
        )
    )

    raw.set_montage(
        custom_montage,
        match_case=False,
        on_missing="warn",
        verbose=False,
    )

    existing_useless = [
        name
        for name in USELESS_CHANNELS
        if name in raw.ch_names
    ]

    if existing_useless:
        print(
            "Dropping Chisco useless channels:",
            existing_useless,
        )
        raw.drop_channels(
            existing_useless
        )

    # Re-assert EOG types after channel removal.
    eog_mapping = {
        name: "eog"
        for name in EOG_CHANNELS
        if name in raw.ch_names
    }

    if eog_mapping:
        raw.set_channel_types(
            eog_mapping,
            verbose=False,
        )

    # PREP
    raw_new = run_prep(
        raw,
        custom_montage,
    )

    print(
        "Still bad channels:",
        raw_new.info["bads"],
    )

    # COFETT-specific explicit filtering.
    raw_new = apply_cofett_filters(
        raw_new
    )

    # COFETT Appendix E-specific ocular regression.
    raw_new = apply_eog_regression(
        raw_new
    )

    if STEP:
        fif_path = (
            OUTPUT_FOLDER
            / "fif"
            / (
                f"{edf_path.stem}_"
                f"{args.method_str}-raw.fif"
            )
        )

        raw_new.save(
            fif_path,
            overwrite=True,
        )

    # ---------------------------------------------------------------------
    # Chisco-2.0 replacement for Chisco 1.0's fixed Epochs:
    #   65380 + [0,5] / [5,8.3]
    #
    # becomes:
    #   phase markers + variable 0.4*x / 0.4*(x+1) / 1.8 windows.
    # ---------------------------------------------------------------------
    trial_table = load_trial_table(
        events_path,
        stimulus_metadata,
    )

    print(
        "Trials:",
        len(trial_table),
    )
    print(
        "x range:",
        int(trial_table["x"].min()),
        "->",
        int(trial_table["x"].max()),
    )

    phase_segments = segment_recording(
        raw_new,
        trial_table,
        edf_path,
        recording_info,
        stimulus_file,
    )

    for phase in [
        "reading",
        "recall",
        "rest",
    ]:
        unique_lengths = sorted(
            {
                item[
                    "input_features"
                ].shape[-1]
                for item in phase_segments[
                    phase
                ]
            }
        )

        print(
            f"{phase}:",
            len(
                phase_segments[
                    phase
                ]
            ),
            "segments; T values =",
            unique_lengths,
        )

    # AutoReject separately for each variable-length phase.
    for phase in [
        "reading",
        "recall",
        "rest",
    ]:
        phase_segments[
            phase
        ] = autoreject_segments(
            phase_segments[
                phase
            ],
            raw_new,
            phase,
        )

    recording_tag = (
        f"{recording_info['subject_bids']}_"
        f"{recording_info['session']}_"
        f"task-{recording_info['task']}_"
        f"run-{recording_info['run_index']:02d}"
    )

    # ICA separately for Reading / Recall / Rest, analogous to the original
    # Chisco code fitting separate ICAs for Reading and Imagine.
    if ICA:
        for phase in [
            "reading",
            "recall",
            "rest",
        ]:
            phase_segments[
                phase
            ] = ica_clean_segments(
                phase_segments[
                    phase
                ],
                raw_new,
                phase,
                recording_tag,
            )

    suffix = (
        "_rej"
        if AUTO_REJECT
        else ""
    )

    if ICA:
        suffix += (
            "_ica_report"
            if args.ica_report_only
            else "_ica"
        )

    for phase in [
        "reading",
        "recall",
        "rest",
    ]:
        path = save_segments_to_pickle(
            phase_segments[
                phase
            ],
            edf_path,
            phase,
            extra_str=suffix,
        )
        print(
            f"Saved {phase}:",
            path,
        )

    append_recall_to_h5(
        phase_segments[
            "recall"
        ],
        recording_info,
    )


# =============================================================================
# 12. Traverse ds006317 BIDS recordings
# =============================================================================

def main() -> None:
    subject_bids, _ = normalize_subject(
        SUBJECT
    )

    subject_root = (
        BIDS_ROOT
        / subject_bids
    )

    if not subject_root.exists():
        raise FileNotFoundError(
            f"Subject directory not found: {subject_root}"
        )

    edf_files = sorted(
        subject_root.glob(
            "ses-*/eeg/*_eeg.edf"
        )
    )

    if args.task != "all":
        edf_files = [
            path
            for path in edf_files
            if (
                f"task-{args.task}_"
                in path.name
            )
        ]

    if not edf_files:
        raise FileNotFoundError(
            f"No matching EDF files found under {subject_root}"
        )

    if TEST:
        edf_files = edf_files[
            :1
        ]

    edf_files = edf_files[
        : args.count_limit
    ]

    print(
        "Found",
        len(edf_files),
        "recordings",
    )

    for edf_path in edf_files:
        process_edf_file(
            edf_path
        )


if __name__ == "__main__":
    main()
