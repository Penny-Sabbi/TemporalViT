"""
TemporalViT robust experimental pipeline for sky-image-based PV power forecasting.

Designed for Google Colab and the SKIPP'D dataset from TorchGeo.

Main improvements over the earlier script
-----------------------------------------
1. Leakage-resistant chronological day-based train/validation/test split.
2. Sequences never cross day boundaries or large timestamp gaps.
3. Persistence, power-only, image-only, CNN-LSTM, direct-patch ViT, and
   proposed convolutional-stem TemporalViT comparisons.
4. Key ablation studies.
5. Repeated-seed experiments with mean ± standard deviation.
6. Bootstrap confidence intervals and paired statistical testing.
7. Parameter count, model size, profiler FLOPs, inference latency,
   throughput, peak GPU memory, and epoch training time.
8. Day-wise, scatter, residual, metric, efficiency, ablation, data-size,
   and multi-horizon visualizations.
9. Checkpoint resumption and permanent Google Drive saving.
10. The corrected proposed architecture is:
      - 64 x 64 grayscale input
      - three stride-2 convolutional stem layers
      - 8 x 8 feature grid = 64 tokens per frame
      - 8 frames = 512 visual tokens + one CLS token
      - embedding dimension 128
      - four transformer blocks
      - four attention heads
      - 128-dimensional historical-power encoder

IMPORTANT COMPUTE NOTE
----------------------
The full paper experiment is expensive. The code is intentionally staged.
This completion version loads or resumes the previously trained main benchmark,
then runs the three-seed ablation, multi-horizon, and data-size studies. It also
generates persistence skill scores, day-wise and rapid-ramp analyses, day-block
bootstrap confidence intervals, statistical tests, figures, and writing-ready
Section 4 tables. Completed runs are skipped unless force_retrain is True.
"""

# ============================================================
# 0. INSTALL DEPENDENCIES IN COLAB
# ============================================================

import os
import sys
import subprocess

REQUIRED_PACKAGES = [
    "torchgeo",
    "timm",
    "scipy",
    "pandas",
    "matplotlib",
    "psutil",
]


def install_packages() -> None:
    """Install only the packages missing from the current environment."""
    import importlib.util

    missing = []
    import_names = {
        "torchgeo": "torchgeo",
        "timm": "timm",
        "scipy": "scipy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "psutil": "psutil",
    }
    for package in REQUIRED_PACKAGES:
        if importlib.util.find_spec(import_names[package]) is None:
            missing.append(package)

    if missing:
        print("Installing missing packages:", missing)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", *missing]
        )


install_packages()

# ============================================================
# 1. IMPORTS
# ============================================================

import io
import gc
import glob
import json
import math
import time
import copy
import random
import warnings
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import psutil

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 2. GOOGLE DRIVE AND CONFIGURATION
# ============================================================

try:
    from google.colab import drive

    drive.mount("/content/drive")
except Exception as exc:
    print("Google Drive mount was skipped:", exc)


@dataclass
class ExperimentConfig:
    # Dataset and cache
    dataset_root: str = "/content/skippd_data"
    cache_dir: str = "/content/skippd_robust_cache"
    image_size: int = 64
    sequence_length: int = 8
    forecast_horizon_minutes: int = 1

    # Use more than the earlier 8,000-sample experiment.
    # Set to None to process all available records when resources permit.
    max_raw_samples: Optional[int] = 30000

    # Sequence validity
    max_allowed_gap_minutes: float = 2.5

    # Leakage-resistant chronological day split
    train_fraction_days: float = 0.70
    val_fraction_days: float = 0.15
    purge_days_between_splits: int = 1

    # Training requested by the user
    batch_size: int = 16
    epochs: int = 100
    early_stop_patience: int = 70
    gradient_clip_norm: float = 1.0
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    num_workers: int = 2
    use_amp: bool = False

    # Learning rates
    lr_power_only: float = 1e-3
    lr_cnn_models: float = 1e-4
    lr_transformers: float = 3e-4

    # Model dimensions
    embed_dim: int = 128
    transformer_depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.10
    drop_path_max: float = 0.10
    lstm_hidden: int = 128

    # Repeated runs. Three seeds are recommended for the final paper.
    main_seeds: Tuple[int, ...] = (42, 52, 62)
    ablation_seeds: Tuple[int, ...] = (42, 52, 62)
    multi_horizon_seeds: Tuple[int, ...] = (42, 52, 62)
    data_size_seeds: Tuple[int, ...] = (42, 52, 62)

    # Staged experiment switches
    run_main_benchmarks: bool = True
    run_ablations: bool = True
    run_multi_horizon: bool = True
    run_data_size_study: bool = True

    # Additional analyses required for a strong Results and Discussion chapter.
    run_forecast_skill_analysis: bool = True
    run_daywise_ramp_analysis: bool = True
    run_day_block_bootstrap: bool = True
    ramp_event_quantile: float = 0.90

    # Continue with later experiments if an individual run fails. All failures
    # are written to Drive so they can be rerun after the unattended session.
    continue_on_error: bool = True

    # Main benchmark models. Keep all for a robust comparison.
    main_models: Tuple[str, ...] = (
        "persistence",
        "power_only_lstm",
        "image_only_cnn_lstm",
        "cnn_lstm_baseline",
        "direct_patch_temporal_vit",
        "temporal_vit_proposed",
    )

    # Multi-horizon study
    horizon_minutes: Tuple[int, ...] = (1, 5, 10, 15, 20, 25, 30)

    # Data-size study uses training-sequence counts. None means all training data.
    data_size_counts: Tuple[Optional[int], ...] = (2000, 5000, 10000, None)

    # Statistical analysis
    bootstrap_repetitions: int = 1000
    permutation_repetitions: int = 5000

    # Runtime profiling
    latency_warmup_batches: int = 20
    latency_repetitions: int = 100
    profile_batch_size: int = 16

    # Output
    project_folder_name: str = "TemporalViT_Robust_Paper_Study"
    # Fixed run folder so interrupted Colab sessions can resume the same experiment.
    # Change this name only when starting a genuinely new experiment.
    run_folder_name: str = "TemporalViT_Robust_Chronological_Study_v1"
    # Renamed image-storage folder requested by the user.
    figure_folder_name: str = "TemporalViT_Robust_Paper_Figures"
    force_retrain: bool = False

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CONFIG = ExperimentConfig()
RUN_NAME = CONFIG.run_folder_name

DRIVE_ROOT = Path("/content/drive/MyDrive")
PROJECT_ROOT = DRIVE_ROOT / CONFIG.project_folder_name
RUN_ROOT = PROJECT_ROOT / RUN_NAME
FIGURE_DIR = RUN_ROOT / CONFIG.figure_folder_name
CHECKPOINT_DIR = RUN_ROOT / "checkpoints"
TABLE_DIR = RUN_ROOT / "tables"
PREDICTION_DIR = RUN_ROOT / "predictions"
LOG_DIR = RUN_ROOT / "logs"
CACHE_DIR = Path(CONFIG.cache_dir)

for directory in [
    RUN_ROOT,
    FIGURE_DIR,
    CHECKPOINT_DIR,
    TABLE_DIR,
    PREDICTION_DIR,
    LOG_DIR,
    CACHE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

with open(RUN_ROOT / "experiment_config.json", "w", encoding="utf-8") as handle:
    json.dump(asdict(CONFIG), handle, indent=2, default=str)

print("Device:", CONFIG.device)
print("Run folder:", RUN_ROOT)
print("All paper figures will be saved in:", FIGURE_DIR)

# ============================================================
# 3. REPRODUCIBILITY AND GENERAL UTILITIES
# ============================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(data), handle, indent=2)


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save publication figures permanently to Google Drive as PNG and PDF."""
    png_path = FIGURE_DIR / f"{stem}.png"
    pdf_path = FIGURE_DIR / f"{stem}.pdf"
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print("Saved figure:", png_path)
    plt.show()
    plt.close(fig)


# ============================================================
# 4. LOAD SKIPP'D AND TIMESTAMPS
# ============================================================

from torchgeo.datasets import SKIPPD

print("Loading SKIPP'D dataset...")
skippd = SKIPPD(
    root=CONFIG.dataset_root,
    download=True,
    checksum=False,
)
print(skippd)
print("Total raw records:", len(skippd))


def locate_times_file(dataset: Any, root: str) -> Path:
    candidates: List[Path] = []

    if hasattr(dataset, "split") and hasattr(dataset, "task"):
        candidates.append(
            Path(root) / f"times_{dataset.split}_{dataset.task}.npy"
        )

    candidates.extend(Path(p) for p in glob.glob(os.path.join(root, "**", "times_*.npy"), recursive=True))
    candidates.extend(Path(p) for p in glob.glob(os.path.join(root, "**", "*time*.npy"), recursive=True))

    unique_candidates = []
    seen = set()
    for path in candidates:
        if path.exists() and str(path) not in seen:
            unique_candidates.append(path)
            seen.add(str(path))

    if not unique_candidates:
        raise FileNotFoundError(
            "Could not locate the SKIPP'D timestamp .npy file under " + root
        )

    # Prefer a filename that includes both split and task.
    selected = unique_candidates[0]
    print("Timestamp file:", selected)
    return selected


TIMES_PATH = locate_times_file(skippd, CONFIG.dataset_root)
all_timestamps_raw = np.load(TIMES_PATH, allow_pickle=True)
all_timestamps = pd.to_datetime(all_timestamps_raw).to_numpy(dtype="datetime64[ns]")
print("Timestamp records:", len(all_timestamps))

# ============================================================
# 5. ROBUST IMAGE/POWER EXTRACTION AND UINT8 CACHE
# ============================================================

pil_resize_grayscale = T.Compose(
    [
        T.Resize((CONFIG.image_size, CONFIG.image_size)),
        T.Grayscale(num_output_channels=1),
        T.PILToTensor(),
    ]
)


def find_image(sample: Dict[str, Any]) -> Any:
    preferred_keys = [
        "image",
        "sky_image",
        "sky",
        "img",
        "rgb",
        "input",
    ]

    for key in preferred_keys:
        if key in sample:
            value = sample[key]
            if isinstance(value, Image.Image):
                return value
            if torch.is_tensor(value) and value.ndim in (2, 3):
                return value

    for value in sample.values():
        if isinstance(value, Image.Image):
            return value
        if torch.is_tensor(value) and value.ndim in (2, 3):
            return value

    raise KeyError(f"No image-like field found. Available keys: {list(sample.keys())}")


def find_power(sample: Dict[str, Any]) -> float:
    preferred_keys = [
        "power",
        "pv_power",
        "ac_power",
        "dc_power",
        "Power",
        "power_ac",
        "power_dc",
        "output_power",
        "target",
        "label",
        "y",
    ]

    for key in preferred_keys:
        if key in sample:
            value = sample[key]
            if torch.is_tensor(value):
                return float(value.detach().cpu().reshape(-1)[0])
            if isinstance(value, (int, float, np.number)):
                return float(value)

    excluded_fragments = ("time", "date", "index", "id", "lat", "lon")
    scalar_candidates = []
    for key, value in sample.items():
        if any(fragment in key.lower() for fragment in excluded_fragments):
            continue
        if isinstance(value, (int, float, np.number)):
            scalar_candidates.append((key, float(value)))
        elif torch.is_tensor(value) and value.numel() == 1:
            scalar_candidates.append((key, float(value.item())))

    if scalar_candidates:
        key, value = scalar_candidates[0]
        print(f"Warning: using fallback scalar field '{key}' as PV power.")
        return value

    raise KeyError(f"No power field found. Available keys: {list(sample.keys())}")


def tensor_or_pil_to_uint8(image: Any) -> torch.Tensor:
    if torch.is_tensor(image):
        tensor = image.detach().cpu()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3:
            raise ValueError(f"Unsupported tensor image shape: {tuple(tensor.shape)}")
        if tensor.shape[0] not in (1, 3):
            tensor = tensor.permute(2, 0, 1)
        image = T.ToPILImage()(tensor)

    if not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image type: {type(image)}")

    output = pil_resize_grayscale(image)
    if output.dtype != torch.uint8:
        output = output.clamp(0, 255).to(torch.uint8)
    return output


cache_tag = f"n{CONFIG.max_raw_samples or 'all'}_s{CONFIG.image_size}"
IMAGE_CACHE_PATH = CACHE_DIR / f"skippd_images_uint8_{cache_tag}.pt"
POWER_CACHE_PATH = CACHE_DIR / f"skippd_powers_{cache_tag}.pt"
TIME_CACHE_PATH = CACHE_DIR / f"skippd_times_{cache_tag}.npy"


def build_cache() -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    total_available = min(len(skippd), len(all_timestamps))
    if CONFIG.max_raw_samples is not None:
        total = min(total_available, CONFIG.max_raw_samples)
    else:
        total = total_available

    images: List[torch.Tensor] = []
    powers: List[float] = []

    print("Building uint8 cache for", total, "records...")
    start = time.time()

    for index in range(total):
        sample = skippd[index]
        images.append(tensor_or_pil_to_uint8(find_image(sample)))
        powers.append(find_power(sample))

        if (index + 1) % 500 == 0 or (index + 1) == total:
            elapsed = time.time() - start
            print(f"Cached {index + 1}/{total} records in {elapsed:.1f}s")

    image_tensor = torch.stack(images, dim=0)
    power_tensor = torch.tensor(powers, dtype=torch.float32)
    timestamps = all_timestamps[:total]

    torch.save(image_tensor, IMAGE_CACHE_PATH)
    torch.save(power_tensor, POWER_CACHE_PATH)
    np.save(TIME_CACHE_PATH, timestamps)

    print("Cache complete:")
    print(" images:", tuple(image_tensor.shape), image_tensor.dtype)
    print(" powers:", tuple(power_tensor.shape), power_tensor.dtype)
    print(" times:", timestamps.shape)
    return image_tensor, power_tensor, timestamps


if IMAGE_CACHE_PATH.exists() and POWER_CACHE_PATH.exists() and TIME_CACHE_PATH.exists():
    print("Loading existing robust cache...")
    cached_images = torch.load(IMAGE_CACHE_PATH, map_location="cpu")
    cached_powers = torch.load(POWER_CACHE_PATH, map_location="cpu")
    cached_timestamps = np.load(TIME_CACHE_PATH, allow_pickle=False)
else:
    cached_images, cached_powers, cached_timestamps = build_cache()

assert len(cached_images) == len(cached_powers) == len(cached_timestamps)
print("Cached dataset length:", len(cached_images))

# ============================================================
# 6. TIMESTAMP INTERVAL, VALID SEQUENCES, AND DAY SPLITS
# ============================================================


def estimate_sampling_interval_minutes(timestamps: np.ndarray) -> float:
    if len(timestamps) < 2:
        return 1.0
    deltas = np.diff(timestamps).astype("timedelta64[s]").astype(np.float64) / 60.0
    deltas = deltas[(deltas > 0) & np.isfinite(deltas)]
    if len(deltas) == 0:
        return 1.0
    # Ignore overnight gaps when estimating the normal cadence.
    q90 = np.quantile(deltas, 0.90)
    short_deltas = deltas[deltas <= q90]
    return float(np.median(short_deltas)) if len(short_deltas) else float(np.median(deltas))


SAMPLING_INTERVAL_MINUTES = estimate_sampling_interval_minutes(cached_timestamps)
print(f"Estimated sampling interval: {SAMPLING_INTERVAL_MINUTES:.3f} minute(s)")


def minutes_to_steps(minutes: int) -> int:
    return max(1, int(round(minutes / max(SAMPLING_INTERVAL_MINUTES, 1e-6))))


def timestamp_date(ts: np.datetime64) -> np.datetime64:
    return ts.astype("datetime64[D]")


def build_valid_start_indices(
    timestamps: np.ndarray,
    sequence_length: int,
    horizon_steps: int,
    max_gap_minutes: float,
) -> List[int]:
    """Build sequences that stay within one day and contain no large time gaps."""
    valid: List[int] = []
    last_start = len(timestamps) - sequence_length - horizon_steps

    for start in range(max(0, last_start + 1)):
        end_input = start + sequence_length
        target_index = end_input + horizon_steps - 1
        window = timestamps[start : target_index + 1]

        if timestamp_date(window[0]) != timestamp_date(window[-1]):
            continue

        deltas = np.diff(window).astype("timedelta64[s]").astype(np.float64) / 60.0
        if len(deltas) and (
            np.any(deltas <= 0) or np.any(deltas > max_gap_minutes)
        ):
            continue

        valid.append(start)

    return valid


def target_index_from_start(start: int, sequence_length: int, horizon_steps: int) -> int:
    return start + sequence_length + horizon_steps - 1


def target_day_from_start(start: int, horizon_steps: int) -> np.datetime64:
    target_index = target_index_from_start(
        start, CONFIG.sequence_length, horizon_steps
    )
    return timestamp_date(cached_timestamps[target_index])


def create_chronological_day_split(
    valid_starts: Sequence[int],
    horizon_steps: int,
) -> Dict[str, List[np.datetime64]]:
    unique_days = sorted(
        {
            target_day_from_start(start, horizon_steps)
            for start in valid_starts
        }
    )

    if len(unique_days) < 6:
        raise RuntimeError(
            f"Only {len(unique_days)} valid days were found; more data are needed for a robust split."
        )

    n_days = len(unique_days)
    raw_train_end = max(1, int(round(CONFIG.train_fraction_days * n_days)))
    raw_val_end = max(raw_train_end + 1, int(round((CONFIG.train_fraction_days + CONFIG.val_fraction_days) * n_days)))
    raw_val_end = min(raw_val_end, n_days - 1)

    purge = CONFIG.purge_days_between_splits
    train_end = max(1, raw_train_end - purge)
    val_start = min(n_days - 2, raw_train_end + purge)
    val_end = max(val_start + 1, raw_val_end - purge)
    test_start = min(n_days - 1, raw_val_end + purge)

    train_days = unique_days[:train_end]
    val_days = unique_days[val_start:val_end]
    test_days = unique_days[test_start:]

    if not train_days or not val_days or not test_days:
        print("Not enough days for purge gaps; repeating split without purge days.")
        train_end = max(1, int(0.70 * n_days))
        val_end = max(train_end + 1, int(0.85 * n_days))
        train_days = unique_days[:train_end]
        val_days = unique_days[train_end:val_end]
        test_days = unique_days[val_end:]

    return {
        "train": train_days,
        "val": val_days,
        "test": test_days,
        "all": unique_days,
    }


def split_starts_by_days(
    valid_starts: Sequence[int],
    horizon_steps: int,
    split_days: Dict[str, Sequence[np.datetime64]],
) -> Dict[str, List[int]]:
    day_sets = {
        key: set(values)
        for key, values in split_days.items()
        if key in ("train", "val", "test")
    }
    output = {"train": [], "val": [], "test": []}

    for start in valid_starts:
        day = target_day_from_start(start, horizon_steps)
        for split_name in ("train", "val", "test"):
            if day in day_sets[split_name]:
                output[split_name].append(start)
                break

    return output


MAIN_HORIZON_STEPS = minutes_to_steps(CONFIG.forecast_horizon_minutes)
main_valid_starts = build_valid_start_indices(
    cached_timestamps,
    CONFIG.sequence_length,
    MAIN_HORIZON_STEPS,
    CONFIG.max_allowed_gap_minutes,
)
MAIN_SPLIT_DAYS = create_chronological_day_split(main_valid_starts, MAIN_HORIZON_STEPS)
main_split_starts = split_starts_by_days(
    main_valid_starts,
    MAIN_HORIZON_STEPS,
    MAIN_SPLIT_DAYS,
)

split_summary = {
    "sampling_interval_minutes": SAMPLING_INTERVAL_MINUTES,
    "forecast_horizon_minutes": CONFIG.forecast_horizon_minutes,
    "forecast_horizon_steps": MAIN_HORIZON_STEPS,
    "sequence_length": CONFIG.sequence_length,
    "valid_sequences": len(main_valid_starts),
    "train_sequences": len(main_split_starts["train"]),
    "validation_sequences": len(main_split_starts["val"]),
    "test_sequences": len(main_split_starts["test"]),
    "train_day_count": len(MAIN_SPLIT_DAYS["train"]),
    "validation_day_count": len(MAIN_SPLIT_DAYS["val"]),
    "test_day_count": len(MAIN_SPLIT_DAYS["test"]),
    "train_first_day": str(MAIN_SPLIT_DAYS["train"][0]),
    "train_last_day": str(MAIN_SPLIT_DAYS["train"][-1]),
    "validation_first_day": str(MAIN_SPLIT_DAYS["val"][0]),
    "validation_last_day": str(MAIN_SPLIT_DAYS["val"][-1]),
    "test_first_day": str(MAIN_SPLIT_DAYS["test"][0]),
    "test_last_day": str(MAIN_SPLIT_DAYS["test"][-1]),
    "purge_days_between_splits": CONFIG.purge_days_between_splits,
}
save_json(split_summary, TABLE_DIR / "chronological_split_summary.json")
print(json.dumps(split_summary, indent=2))

# Save exact split membership for reproducibility.
split_rows = []
for split_name, starts in main_split_starts.items():
    for start in starts:
        target_idx = target_index_from_start(start, CONFIG.sequence_length, MAIN_HORIZON_STEPS)
        split_rows.append(
            {
                "split": split_name,
                "start_index": start,
                "target_index": target_idx,
                "start_timestamp": str(cached_timestamps[start]),
                "target_timestamp": str(cached_timestamps[target_idx]),
            }
        )
pd.DataFrame(split_rows).to_csv(TABLE_DIR / "chronological_split_membership.csv", index=False)

# ============================================================
# 7. DATASET AND DATALOADER BUILDERS
# ============================================================


class CachedSkySequenceDataset(Dataset):
    def __init__(
        self,
        images_uint8: torch.Tensor,
        powers: torch.Tensor,
        starts: Sequence[int],
        sequence_length: int,
        horizon_steps: int,
        power_mean: float,
        power_std: float,
    ) -> None:
        self.images_uint8 = images_uint8
        self.powers = powers
        self.starts = list(starts)
        self.sequence_length = sequence_length
        self.horizon_steps = horizon_steps
        self.power_mean = float(power_mean)
        self.power_std = float(max(power_std, 1e-6))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        start = self.starts[index]
        end_input = start + self.sequence_length
        target_index = end_input + self.horizon_steps - 1

        images = self.images_uint8[start:end_input].float().div(255.0)
        images = (images - 0.5) / 0.5

        power_seq_raw = self.powers[start:end_input]
        target_raw = self.powers[target_index].view(1)

        power_seq = (power_seq_raw - self.power_mean) / self.power_std
        target = (target_raw - self.power_mean) / self.power_std

        return {
            "images": images,
            "power_seq": power_seq,
            "target": target,
            "target_raw": target_raw,
            "target_index": torch.tensor(target_index, dtype=torch.long),
        }


@dataclass
class LoaderBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_starts: List[int]
    val_starts: List[int]
    test_starts: List[int]
    power_mean: float
    power_std: float
    horizon_steps: int


def compute_training_power_stats(
    train_starts: Sequence[int], horizon_steps: int
) -> Tuple[float, float]:
    target_indices = [
        target_index_from_start(start, CONFIG.sequence_length, horizon_steps)
        for start in train_starts
    ]
    targets = cached_powers[target_indices]
    return float(targets.mean().item()), float(targets.std().item() + 1e-6)


def make_loader(
    dataset: Dataset,
    shuffle: bool,
    seed: int,
    batch_size: Optional[int] = None,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size or CONFIG.batch_size,
        shuffle=shuffle,
        num_workers=CONFIG.num_workers,
        pin_memory=(CONFIG.device == "cuda"),
        persistent_workers=(CONFIG.num_workers > 0),
        generator=generator,
    )


def build_loader_bundle(
    split_starts: Dict[str, Sequence[int]],
    horizon_steps: int,
    seed: int,
    train_limit: Optional[int] = None,
) -> LoaderBundle:
    train_starts = list(split_starts["train"])
    val_starts = list(split_starts["val"])
    test_starts = list(split_starts["test"])

    if train_limit is not None and train_limit < len(train_starts):
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(train_starts), size=train_limit, replace=False))
        train_starts = [train_starts[i] for i in selected]

    power_mean, power_std = compute_training_power_stats(train_starts, horizon_steps)

    train_ds = CachedSkySequenceDataset(
        cached_images,
        cached_powers,
        train_starts,
        CONFIG.sequence_length,
        horizon_steps,
        power_mean,
        power_std,
    )
    val_ds = CachedSkySequenceDataset(
        cached_images,
        cached_powers,
        val_starts,
        CONFIG.sequence_length,
        horizon_steps,
        power_mean,
        power_std,
    )
    test_ds = CachedSkySequenceDataset(
        cached_images,
        cached_powers,
        test_starts,
        CONFIG.sequence_length,
        horizon_steps,
        power_mean,
        power_std,
    )

    return LoaderBundle(
        train_loader=make_loader(train_ds, True, seed),
        val_loader=make_loader(val_ds, False, seed),
        test_loader=make_loader(test_ds, False, seed),
        train_starts=train_starts,
        val_starts=val_starts,
        test_starts=test_starts,
        power_mean=power_mean,
        power_std=power_std,
        horizon_steps=horizon_steps,
    )

# ============================================================
# 8. MODEL DEFINITIONS
# ============================================================


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-normalized transformer block with explicit DropPath residuals."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        attended, _ = self.attn(normalized, normalized, normalized, need_weights=False)
        x = x + self.drop_path1(attended)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class ConvStemPatchEmbedding(nn.Module):
    """Three stride-2 convolutions: 64x64 -> 8x8 -> 64 tokens per frame."""

    def __init__(self, image_size: int, embed_dim: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, embed_dim // 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.grid_size = image_size // 8
        self.num_patches = self.grid_size * self.grid_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return x.flatten(2).transpose(1, 2)


class DirectPatchEmbedding(nn.Module):
    """Conventional ViT patchification with 8x8 patches: 64 tokens per frame."""

    def __init__(self, image_size: int, embed_dim: int, patch_size: int = 8) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.proj = nn.Conv2d(
            1,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class PersistenceModel(nn.Module):
    """No-training baseline: predict the last observed normalized PV power."""

    def forward(self, images: torch.Tensor, power_seq: torch.Tensor) -> torch.Tensor:
        del images
        return power_seq[:, -1:].contiguous()


class PowerOnlyLSTM(nn.Module):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_dim, batch_first=True)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, images: torch.Tensor, power_seq: torch.Tensor) -> torch.Tensor:
        del images
        sequence = power_seq.unsqueeze(-1)
        _, (hidden, _) = self.lstm(sequence)
        return self.regressor(hidden[-1])


class SharedCNNEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.output_dim = 128 * 4 * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


class ImageOnlyCNNLSTM(nn.Module):
    def __init__(self, lstm_hidden: int = 128) -> None:
        super().__init__()
        self.cnn = SharedCNNEncoder()
        self.lstm = nn.LSTM(self.cnn.output_dim, lstm_hidden, batch_first=True)
        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, images: torch.Tensor, power_seq: torch.Tensor) -> torch.Tensor:
        del power_seq
        batch, frames, channels, height, width = images.shape
        features = self.cnn(images.reshape(batch * frames, channels, height, width))
        features = features.reshape(batch, frames, -1)
        _, (hidden, _) = self.lstm(features)
        return self.regressor(hidden[-1])


class CNNLSTMBaseline(nn.Module):
    def __init__(self, seq_len: int, lstm_hidden: int = 128) -> None:
        super().__init__()
        self.cnn = SharedCNNEncoder()
        self.lstm = nn.LSTM(self.cnn.output_dim, lstm_hidden, batch_first=True)
        self.power_encoder = nn.Sequential(
            nn.Linear(seq_len, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, images: torch.Tensor, power_seq: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = images.shape
        features = self.cnn(images.reshape(batch * frames, channels, height, width))
        features = features.reshape(batch, frames, -1)
        _, (hidden, _) = self.lstm(features)
        image_feature = hidden[-1]
        power_feature = self.power_encoder(power_seq)
        return self.regressor(torch.cat([image_feature, power_feature], dim=1))


class TemporalViTRegressor(nn.Module):
    """Generic direct-patch or convolutional-stem TemporalViT with ablation flags."""

    def __init__(
        self,
        image_size: int,
        seq_len: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        drop_path_max: float,
        use_conv_stem: bool = True,
        use_spatial_pos: bool = True,
        use_temporal_pos: bool = True,
        use_power: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.use_spatial_pos = use_spatial_pos
        self.use_temporal_pos = use_temporal_pos
        self.use_power = use_power

        self.patch_embed = (
            ConvStemPatchEmbedding(image_size, embed_dim)
            if use_conv_stem
            else DirectPatchEmbedding(image_size, embed_dim, patch_size=8)
        )
        self.num_patches = self.patch_embed.num_patches

        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, 1, self.num_patches, embed_dim),
            requires_grad=use_spatial_pos,
        )
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, seq_len, 1, embed_dim),
            requires_grad=use_temporal_pos,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        drop_rates = torch.linspace(0.0, drop_path_max, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=drop_rates[index],
                )
                for index in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)

        if use_power:
            self.power_encoder = nn.Sequential(
                nn.Linear(seq_len, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            fused_dim = embed_dim * 2
        else:
            self.power_encoder = None
            fused_dim = embed_dim

        self.regressor = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.use_spatial_pos:
            nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        else:
            nn.init.zeros_(self.spatial_pos_embed)
        if self.use_temporal_pos:
            nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        else:
            nn.init.zeros_(self.temporal_pos_embed)

    def forward(self, images: torch.Tensor, power_seq: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = images.shape
        if frames != self.seq_len:
            raise ValueError(f"Expected {self.seq_len} frames but received {frames}")

        frame_batch = images.reshape(batch * frames, channels, height, width)
        tokens = self.patch_embed(frame_batch)
        tokens = tokens.reshape(batch, frames, self.num_patches, self.embed_dim)

        if self.use_spatial_pos:
            tokens = tokens + self.spatial_pos_embed
        if self.use_temporal_pos:
            tokens = tokens + self.temporal_pos_embed

        tokens = tokens.reshape(batch, frames * self.num_patches, self.embed_dim)
        cls = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls, tokens], dim=1)

        for block in self.blocks:
            x = block(x)

        image_feature = self.final_norm(x)[:, 0]

        if self.use_power:
            assert self.power_encoder is not None
            power_feature = self.power_encoder(power_seq)
            fused = torch.cat([image_feature, power_feature], dim=1)
        else:
            fused = image_feature

        return self.regressor(fused)


# Correct architecture metadata for the paper figure and methods section.
architecture_metadata = {
    "input_image_size": [CONFIG.image_size, CONFIG.image_size],
    "input_channels": 1,
    "sequence_length": CONFIG.sequence_length,
    "conv_stem_layers": 3,
    "conv_stem_stride_per_layer": 2,
    "conv_stem_output_grid": [CONFIG.image_size // 8, CONFIG.image_size // 8],
    "patches_per_frame": (CONFIG.image_size // 8) ** 2,
    "visual_tokens": CONFIG.sequence_length * (CONFIG.image_size // 8) ** 2,
    "cls_tokens": 1,
    "total_transformer_tokens": 1 + CONFIG.sequence_length * (CONFIG.image_size // 8) ** 2,
    "embedding_dimension": CONFIG.embed_dim,
    "transformer_blocks": CONFIG.transformer_depth,
    "attention_heads": CONFIG.num_heads,
    "mlp_ratio": CONFIG.mlp_ratio,
    "historical_power_encoder_dimension": CONFIG.embed_dim,
}
save_json(architecture_metadata, TABLE_DIR / "corrected_proposed_architecture_metadata.json")

# ============================================================
# 9. MODEL SPECIFICATIONS
# ============================================================


@dataclass
class ModelSpec:
    key: str
    label: str
    factory: Callable[[], nn.Module]
    learning_rate: float
    warmup_epochs: int
    trainable: bool = True


def proposed_factory(**overrides: Any) -> nn.Module:
    kwargs = dict(
        image_size=CONFIG.image_size,
        seq_len=CONFIG.sequence_length,
        embed_dim=CONFIG.embed_dim,
        depth=CONFIG.transformer_depth,
        num_heads=CONFIG.num_heads,
        mlp_ratio=CONFIG.mlp_ratio,
        dropout=CONFIG.dropout,
        drop_path_max=CONFIG.drop_path_max,
        use_conv_stem=True,
        use_spatial_pos=True,
        use_temporal_pos=True,
        use_power=True,
    )
    kwargs.update(overrides)
    return TemporalViTRegressor(**kwargs)


MODEL_SPECS: Dict[str, ModelSpec] = {
    "persistence": ModelSpec(
        key="persistence",
        label="Persistence",
        factory=lambda: PersistenceModel(),
        learning_rate=0.0,
        warmup_epochs=0,
        trainable=False,
    ),
    "power_only_lstm": ModelSpec(
        key="power_only_lstm",
        label="Power-only LSTM",
        factory=lambda: PowerOnlyLSTM(hidden_dim=64),
        learning_rate=CONFIG.lr_power_only,
        warmup_epochs=CONFIG.warmup_epochs,
    ),
    "image_only_cnn_lstm": ModelSpec(
        key="image_only_cnn_lstm",
        label="Image-only CNN-LSTM",
        factory=lambda: ImageOnlyCNNLSTM(CONFIG.lstm_hidden),
        learning_rate=CONFIG.lr_cnn_models,
        warmup_epochs=CONFIG.warmup_epochs,
    ),
    "cnn_lstm_baseline": ModelSpec(
        key="cnn_lstm_baseline",
        label="CNN-LSTM baseline",
        factory=lambda: CNNLSTMBaseline(CONFIG.sequence_length, CONFIG.lstm_hidden),
        learning_rate=CONFIG.lr_cnn_models,
        warmup_epochs=CONFIG.warmup_epochs,
    ),
    "direct_patch_temporal_vit": ModelSpec(
        key="direct_patch_temporal_vit",
        label="Direct-patch TemporalViT",
        factory=lambda: proposed_factory(use_conv_stem=False),
        learning_rate=CONFIG.lr_transformers,
        warmup_epochs=CONFIG.warmup_epochs,
    ),
    "temporal_vit_proposed": ModelSpec(
        key="temporal_vit_proposed",
        label="Proposed TemporalViT",
        factory=lambda: proposed_factory(),
        learning_rate=CONFIG.lr_transformers,
        warmup_epochs=CONFIG.warmup_epochs,
    ),
}

ABLATION_SPECS: Dict[str, ModelSpec] = {
    "ablation_no_conv_stem": ModelSpec(
        key="ablation_no_conv_stem",
        label="Without convolutional stem",
        factory=lambda: proposed_factory(use_conv_stem=False),
        learning_rate=CONFIG.lr_transformers,
        warmup_epochs=CONFIG.warmup_epochs,
    ),
    "ablation_no_temporal_pos": ModelSpec(
        key="ablation_no_temporal_pos",
        label="Without temporal embeddings",
        factory=lambda: proposed_factory(use_temporal_pos=False),
        learning_rate=CONFIG.lr_transformers,
        warmup_epochs=CONFIG.warmup_epochs,
    ),
    "ablation_no_spatial_pos": ModelSpec(
        key="ablation_no_spatial_pos",
        label="Without spatial embeddings",
        factory=lambda: proposed_factory(use_spatial_pos=False),
        learning_rate=CONFIG.lr_transformers,
        warmup_epochs=CONFIG.warmup_epochs,
    ),
    "ablation_no_power": ModelSpec(
        key="ablation_no_power",
        label="Without historical-power fusion",
        factory=lambda: proposed_factory(use_power=False),
        learning_rate=CONFIG.lr_transformers,
        warmup_epochs=CONFIG.warmup_epochs,
    ),
    "ablation_no_droppath": ModelSpec(
        key="ablation_no_droppath",
        label="Without DropPath",
        factory=lambda: proposed_factory(drop_path_max=0.0),
        learning_rate=CONFIG.lr_transformers,
        warmup_epochs=CONFIG.warmup_epochs,
    ),
    "temporal_vit_proposed": MODEL_SPECS["temporal_vit_proposed"],
}

# ============================================================
# 10. METRICS, CONFIDENCE INTERVALS, AND STATISTICAL TESTS
# ============================================================


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    errors = predictions - targets
    absolute_errors = np.abs(errors)
    squared_errors = errors ** 2

    mae = float(np.mean(absolute_errors))
    mse = float(np.mean(squared_errors))
    rmse = float(np.sqrt(mse))
    denominator = float(np.sum((targets - np.mean(targets)) ** 2))
    r2 = float(1.0 - np.sum(squared_errors) / denominator) if denominator > 0 else float("nan")
    mbe = float(np.mean(errors))
    correlation = float(np.corrcoef(targets, predictions)[0, 1]) if len(targets) > 1 else float("nan")
    target_range = float(np.max(targets) - np.min(targets))
    nrmse_range = float(rmse / target_range) if target_range > 0 else float("nan")

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mbe": mbe,
        "pearson_r": correlation,
        "nrmse_range": nrmse_range,
    }


def bootstrap_metric_intervals(
    targets: np.ndarray,
    predictions: np.ndarray,
    repetitions: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    n = len(targets)
    values = {name: [] for name in ("mae", "rmse", "r2")}

    for _ in range(repetitions):
        indices = rng.integers(0, n, size=n)
        metric_values = regression_metrics(targets[indices], predictions[indices])
        for name in values:
            values[name].append(metric_values[name])

    output = {}
    for name, samples in values.items():
        low, high = np.percentile(samples, [2.5, 97.5])
        output[name] = {"ci_low": float(low), "ci_high": float(high)}
    return output


def paired_sign_flip_permutation_test(
    baseline_abs_errors: np.ndarray,
    proposed_abs_errors: np.ndarray,
    repetitions: int,
    seed: int,
) -> Dict[str, float]:
    """One-sided test of whether baseline absolute errors exceed proposed errors."""
    differences = baseline_abs_errors - proposed_abs_errors
    observed = float(np.mean(differences))
    rng = np.random.default_rng(seed)
    exceedances = 0

    for _ in range(repetitions):
        signs = rng.choice((-1.0, 1.0), size=len(differences))
        permuted = float(np.mean(differences * signs))
        if permuted >= observed:
            exceedances += 1

    p_value = (exceedances + 1) / (repetitions + 1)
    return {"mean_mae_difference": observed, "one_sided_p_value": float(p_value)}


# ============================================================
# 11. TRAINING, EVALUATION, RESUME, AND CHECKPOINTING
# ============================================================


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def make_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(0, warmup_epochs * steps_per_epoch)
    total_steps = max(1, total_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    batch_count = 0

    cuda_sync()
    start_time = time.perf_counter()

    for batch in loader:
        images = batch["images"].to(CONFIG.device, non_blocking=True)
        power_seq = batch["power_seq"].to(CONFIG.device, non_blocking=True)
        targets = batch["target"].to(CONFIG.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda" if CONFIG.device == "cuda" else "cpu",
            enabled=CONFIG.use_amp and CONFIG.device == "cuda",
        ):
            predictions = model(images, power_seq)
            loss = criterion(predictions, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += float(loss.item())
        batch_count += 1

    cuda_sync()
    epoch_time = time.perf_counter() - start_time
    return total_loss / max(1, batch_count), epoch_time


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    power_mean: float,
    power_std: float,
) -> Dict[str, Any]:
    model.eval()
    normalized_losses: List[float] = []
    predictions_raw: List[float] = []
    targets_raw: List[float] = []
    target_indices: List[int] = []

    criterion = nn.MSELoss()

    for batch in loader:
        images = batch["images"].to(CONFIG.device, non_blocking=True)
        power_seq = batch["power_seq"].to(CONFIG.device, non_blocking=True)
        targets = batch["target"].to(CONFIG.device, non_blocking=True)

        outputs = model(images, power_seq)
        normalized_losses.append(float(criterion(outputs, targets).item()))

        outputs_raw = outputs * power_std + power_mean
        predictions_raw.extend(outputs_raw.detach().cpu().reshape(-1).tolist())
        targets_raw.extend(batch["target_raw"].reshape(-1).tolist())
        target_indices.extend(batch["target_index"].reshape(-1).tolist())

    predictions_np = np.asarray(predictions_raw, dtype=np.float64)
    targets_np = np.asarray(targets_raw, dtype=np.float64)
    indices_np = np.asarray(target_indices, dtype=np.int64)
    metrics = regression_metrics(targets_np, predictions_np)
    metrics["normalized_mse_loss"] = float(np.mean(normalized_losses))

    return {
        "metrics": metrics,
        "predictions": predictions_np,
        "targets": targets_np,
        "target_indices": indices_np,
    }


@dataclass
class RunResult:
    model_key: str
    model_label: str
    seed: int
    model: nn.Module
    history: pd.DataFrame
    metrics: Dict[str, float]
    predictions: np.ndarray
    targets: np.ndarray
    target_indices: np.ndarray
    best_epoch: int
    total_training_seconds: float
    checkpoint_path: str


def train_or_load_model(
    spec: ModelSpec,
    seed: int,
    loaders: LoaderBundle,
    experiment_group: str,
) -> RunResult:
    set_seed(seed)
    run_dir = CHECKPOINT_DIR / experiment_group / spec.key / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_path = run_dir / "best_model.pt"
    last_path = run_dir / "last_training_state.pt"
    history_path = run_dir / "history.csv"
    metrics_path = run_dir / "metrics.json"
    predictions_path = PREDICTION_DIR / experiment_group / f"{spec.key}_seed_{seed}.csv"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    model = spec.factory().to(CONFIG.device)

    if (
        metrics_path.exists()
        and history_path.exists()
        and predictions_path.exists()
        and best_path.exists()
        and not CONFIG.force_retrain
    ):
        print(f"Loading completed run: {experiment_group}/{spec.key}/seed_{seed}")
        model.load_state_dict(torch.load(best_path, map_location=CONFIG.device))
        history = pd.read_csv(history_path)
        stored = json.loads(metrics_path.read_text(encoding="utf-8"))
        prediction_frame = pd.read_csv(predictions_path)
        return RunResult(
            model_key=spec.key,
            model_label=spec.label,
            seed=seed,
            model=model,
            history=history,
            metrics=stored["metrics"],
            predictions=prediction_frame["prediction"].to_numpy(),
            targets=prediction_frame["target"].to_numpy(),
            target_indices=prediction_frame["target_index"].to_numpy(dtype=np.int64),
            best_epoch=int(stored["best_epoch"]),
            total_training_seconds=float(stored["total_training_seconds"]),
            checkpoint_path=str(best_path),
        )

    if not spec.trainable:
        print(f"Evaluating non-trainable model: {spec.label}")
        evaluation = evaluate_model(
            model,
            loaders.test_loader,
            loaders.power_mean,
            loaders.power_std,
        )
        history = pd.DataFrame()
        best_epoch = 0
        total_training_seconds = 0.0
        torch.save(model.state_dict(), best_path)
    else:
        print("\n" + "=" * 80)
        print(f"Training {spec.label} | seed={seed}")
        print("Trainable parameters:", count_parameters(model))
        print("=" * 80)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=spec.learning_rate,
            weight_decay=CONFIG.weight_decay,
            betas=(0.9, 0.999),
        )
        scheduler = make_warmup_cosine_scheduler(
            optimizer,
            spec.warmup_epochs,
            CONFIG.epochs,
            len(loaders.train_loader),
        )
        criterion = nn.MSELoss()
        scaler = torch.cuda.amp.GradScaler(
            enabled=CONFIG.use_amp and CONFIG.device == "cuda"
        )

        history_records: List[Dict[str, float]] = []
        start_epoch = 0
        best_val_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        total_training_seconds = 0.0

        if last_path.exists() and not CONFIG.force_retrain:
            print("Resuming interrupted run from:", last_path)
            state = torch.load(last_path, map_location=CONFIG.device)
            model.load_state_dict(state["model_state"])
            optimizer.load_state_dict(state["optimizer_state"])
            scheduler.load_state_dict(state["scheduler_state"])
            scaler.load_state_dict(state["scaler_state"])
            start_epoch = int(state["epoch"] + 1)
            best_val_loss = float(state["best_val_loss"])
            best_epoch = int(state["best_epoch"])
            epochs_without_improvement = int(state["epochs_without_improvement"])
            total_training_seconds = float(state["total_training_seconds"])
            history_records = list(state["history_records"])

        for epoch in range(start_epoch, CONFIG.epochs):
            train_loss, epoch_time = train_one_epoch(
                model,
                loaders.train_loader,
                optimizer,
                scheduler,
                criterion,
                scaler,
            )
            total_training_seconds += epoch_time

            validation = evaluate_model(
                model,
                loaders.val_loader,
                loaders.power_mean,
                loaders.power_std,
            )
            val_metrics = validation["metrics"]
            current_lr = float(optimizer.param_groups[0]["lr"])

            record = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_metrics["normalized_mse_loss"],
                "val_mae": val_metrics["mae"],
                "val_mse": val_metrics["mse"],
                "val_rmse": val_metrics["rmse"],
                "val_r2": val_metrics["r2"],
                "learning_rate": current_lr,
                "epoch_time_seconds": epoch_time,
            }
            history_records.append(record)

            print(
                f"Epoch {epoch + 1:03d}/{CONFIG.epochs} | "
                f"train={train_loss:.6f} | val={record['val_loss']:.6f} | "
                f"MAE={record['val_mae']:.6f} | RMSE={record['val_rmse']:.6f} | "
                f"R2={record['val_r2']:.6f} | time={epoch_time:.2f}s"
            )

            if record["val_loss"] < best_val_loss - 1e-7:
                best_val_loss = record["val_loss"]
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                torch.save(
                    {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    best_path,
                )
            else:
                epochs_without_improvement += 1

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "epochs_without_improvement": epochs_without_improvement,
                    "total_training_seconds": total_training_seconds,
                    "history_records": history_records,
                },
                last_path,
            )

            pd.DataFrame(history_records).to_csv(history_path, index=False)

            if epochs_without_improvement >= CONFIG.early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch + 1}; no validation improvement "
                    f"for {CONFIG.early_stop_patience} epochs."
                )
                break

        if not best_path.exists():
            torch.save(
                {key: value.detach().cpu() for key, value in model.state_dict().items()},
                best_path,
            )

        model.load_state_dict(torch.load(best_path, map_location=CONFIG.device))
        history = pd.DataFrame(history_records)
        history.to_csv(history_path, index=False)

        evaluation = evaluate_model(
            model,
            loaders.test_loader,
            loaders.power_mean,
            loaders.power_std,
        )

    prediction_frame = pd.DataFrame(
        {
            "target_index": evaluation["target_indices"],
            "timestamp": [
                str(cached_timestamps[index])
                for index in evaluation["target_indices"]
            ],
            "target": evaluation["targets"],
            "prediction": evaluation["predictions"],
            "error": evaluation["predictions"] - evaluation["targets"],
            "absolute_error": np.abs(evaluation["predictions"] - evaluation["targets"]),
        }
    ).sort_values("target_index")
    prediction_frame.to_csv(predictions_path, index=False)

    intervals = bootstrap_metric_intervals(
        evaluation["targets"],
        evaluation["predictions"],
        CONFIG.bootstrap_repetitions,
        seed,
    )
    stored_metrics = {
        "model_key": spec.key,
        "model_label": spec.label,
        "seed": seed,
        "metrics": evaluation["metrics"],
        "bootstrap_95_ci": intervals,
        "best_epoch": best_epoch,
        "total_training_seconds": total_training_seconds,
        "parameter_count": count_parameters(model),
        "checkpoint": str(best_path),
    }
    save_json(stored_metrics, metrics_path)

    return RunResult(
        model_key=spec.key,
        model_label=spec.label,
        seed=seed,
        model=model,
        history=history,
        metrics=evaluation["metrics"],
        predictions=evaluation["predictions"],
        targets=evaluation["targets"],
        target_indices=evaluation["target_indices"],
        best_epoch=best_epoch,
        total_training_seconds=total_training_seconds,
        checkpoint_path=str(best_path),
    )

# ============================================================
# 12. COMPUTATIONAL EFFICIENCY PROFILING
# ============================================================


def model_state_size_mb(model: nn.Module) -> float:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / (1024 ** 2)


@torch.no_grad()
def profile_forward_flops(
    model: nn.Module,
    images: torch.Tensor,
    power_seq: torch.Tensor,
) -> float:
    """Estimate forward FLOPs using PyTorch profiler-supported operations."""
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if CONFIG.device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        with torch.profiler.profile(
            activities=activities,
            with_flops=True,
            record_shapes=False,
            profile_memory=False,
        ) as profiler:
            _ = model(images, power_seq)
            cuda_sync()

        total_flops = 0.0
        for event in profiler.key_averages():
            if event.flops is not None:
                total_flops += float(event.flops)
        return total_flops
    except Exception as exc:
        print("FLOP profiling failed:", exc)
        return float("nan")


@torch.no_grad()
def profile_model_efficiency(
    result: RunResult,
    loaders: LoaderBundle,
) -> Dict[str, float]:
    model = result.model.to(CONFIG.device).eval()
    first_batch = next(iter(loaders.test_loader))
    images = first_batch["images"][: CONFIG.profile_batch_size].to(CONFIG.device)
    power_seq = first_batch["power_seq"][: CONFIG.profile_batch_size].to(CONFIG.device)

    # Batch-one input for per-forecast FLOPs and latency.
    images_one = images[:1]
    power_one = power_seq[:1]

    for _ in range(CONFIG.latency_warmup_batches):
        _ = model(images, power_seq)
    cuda_sync()

    latencies = []
    if CONFIG.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for _ in range(CONFIG.latency_repetitions):
        cuda_sync()
        start = time.perf_counter()
        _ = model(images, power_seq)
        cuda_sync()
        latencies.append(time.perf_counter() - start)

    batch_size = images.shape[0]
    mean_batch_seconds = float(np.mean(latencies))
    mean_sample_ms = mean_batch_seconds * 1000.0 / batch_size
    throughput = batch_size / mean_batch_seconds

    if CONFIG.device == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_memory_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

    flops = profile_forward_flops(model, images_one, power_one)
    avg_epoch_time = (
        float(result.history["epoch_time_seconds"].mean())
        if not result.history.empty and "epoch_time_seconds" in result.history
        else 0.0
    )

    return {
        "model_key": result.model_key,
        "model_label": result.model_label,
        "seed": result.seed,
        "parameters": count_parameters(model),
        "model_size_mb": model_state_size_mb(model),
        "estimated_forward_flops": flops,
        "estimated_forward_gflops": flops / 1e9 if np.isfinite(flops) else float("nan"),
        "inference_latency_ms_per_sample": mean_sample_ms,
        "throughput_samples_per_second": throughput,
        "peak_inference_memory_mb": peak_memory_mb,
        "average_epoch_time_seconds": avg_epoch_time,
        "total_training_seconds": result.total_training_seconds,
    }

# ============================================================
# 13. EXPERIMENT RUNNERS AND AGGREGATION
# ============================================================


FAILED_RUNS: List[Dict[str, Any]] = []


def save_pipeline_heartbeat(message: str) -> None:
    """Write the latest pipeline status to Google Drive."""
    timestamp = datetime.now().isoformat(timespec="seconds")
    heartbeat_path = RUN_ROOT / "pipeline_heartbeat.txt"
    heartbeat_path.write_text(f"{timestamp}\n{message}\n", encoding="utf-8")


def record_failed_run(
    experiment_group: str,
    model_key: str,
    seed: int,
    exc: BaseException,
) -> None:
    failure = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment_group": experiment_group,
        "model_key": model_key,
        "seed": seed,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
    }
    FAILED_RUNS.append(failure)
    pd.DataFrame(FAILED_RUNS).to_csv(TABLE_DIR / "failed_runs.csv", index=False)
    with open(LOG_DIR / "failed_runs.log", "a", encoding="utf-8") as handle:
        handle.write("\n" + "=" * 100 + "\n")
        handle.write(json.dumps(failure, indent=2))
        handle.write("\n")


def run_model_set(
    specs: Dict[str, ModelSpec],
    model_keys: Sequence[str],
    seeds: Sequence[int],
    loaders_by_seed: Dict[int, LoaderBundle],
    experiment_group: str,
) -> Dict[Tuple[str, int], RunResult]:
    outputs: Dict[Tuple[str, int], RunResult] = {}

    for key in model_keys:
        if key not in specs:
            raise KeyError(f"Unknown model key: {key}")
        spec = specs[key]

        # Persistence has no random initialization; evaluate it once.
        active_seeds = seeds[:1] if not spec.trainable else seeds

        for seed in active_seeds:
            status = f"Running {experiment_group}/{key}/seed_{seed}"
            print("\n" + status)
            save_pipeline_heartbeat(status)
            try:
                outputs[(key, seed)] = train_or_load_model(
                    spec,
                    seed,
                    loaders_by_seed[seed],
                    experiment_group,
                )
                save_pipeline_heartbeat(
                    f"Completed {experiment_group}/{key}/seed_{seed}"
                )
            except Exception as exc:
                print(
                    f"ERROR in {experiment_group}/{key}/seed_{seed}: "
                    f"{type(exc).__name__}: {exc}"
                )
                record_failed_run(experiment_group, key, seed, exc)
                if not CONFIG.continue_on_error:
                    raise
            finally:
                cleanup_memory()

    return outputs

def results_to_long_dataframe(results: Dict[Tuple[str, int], RunResult]) -> pd.DataFrame:
    rows = []
    for (_, _), result in results.items():
        row = {
            "model_key": result.model_key,
            "model": result.model_label,
            "seed": result.seed,
            "best_epoch": result.best_epoch,
            "total_training_seconds": result.total_training_seconds,
        }
        row.update(result.metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_results(long_frame: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "mae",
        "mse",
        "rmse",
        "r2",
        "mbe",
        "pearson_r",
        "nrmse_range",
        "total_training_seconds",
        "best_epoch",
    ]
    rows = []
    for (model_key, model), group in long_frame.groupby(["model_key", "model"], sort=False):
        row = {
            "model_key": model_key,
            "model": model,
            "runs": len(group),
        }
        for metric in metric_columns:
            if metric in group:
                row[f"{metric}_mean"] = float(group[metric].mean())
                row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)

# ============================================================
# 14. MAIN BENCHMARKS
# ============================================================

main_loaders_by_seed: Dict[int, LoaderBundle] = {
    seed: build_loader_bundle(
        main_split_starts,
        MAIN_HORIZON_STEPS,
        seed,
    )
    for seed in CONFIG.main_seeds
}

main_results: Dict[Tuple[str, int], RunResult] = {}
main_long = pd.DataFrame()
main_summary = pd.DataFrame()
efficiency_frame = pd.DataFrame()

if CONFIG.run_main_benchmarks:
    main_results = run_model_set(
        MODEL_SPECS,
        CONFIG.main_models,
        CONFIG.main_seeds,
        main_loaders_by_seed,
        experiment_group="main_benchmarks",
    )

    main_long = results_to_long_dataframe(main_results)
    main_summary = aggregate_results(main_long)
    main_long.to_csv(TABLE_DIR / "main_benchmark_results_all_seeds.csv", index=False)
    main_summary.to_csv(TABLE_DIR / "main_benchmark_results_mean_std.csv", index=False)

    print("\nMain benchmark summary:")
    print(main_summary.to_string(index=False))

    # Efficiency is profiled on the first available seed for each model.
    efficiency_rows = []
    for model_key in CONFIG.main_models:
        matching = [result for (key, _), result in main_results.items() if key == model_key]
        if not matching:
            continue
        result = matching[0]
        efficiency_rows.append(
            profile_model_efficiency(result, main_loaders_by_seed[result.seed])
        )
        cleanup_memory()

    efficiency_frame = pd.DataFrame(efficiency_rows)
    efficiency_frame.to_csv(TABLE_DIR / "computational_efficiency_results.csv", index=False)

# ============================================================
# 15. PAIRED STATISTICAL COMPARISON: BASELINE VS PROPOSED
# ============================================================


def align_predictions(a: RunResult, b: RunResult) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_a = pd.DataFrame(
        {
            "target_index": a.target_indices,
            "target_a": a.targets,
            "pred_a": a.predictions,
        }
    )
    frame_b = pd.DataFrame(
        {
            "target_index": b.target_indices,
            "target_b": b.targets,
            "pred_b": b.predictions,
        }
    )
    merged = frame_a.merge(frame_b, on="target_index", how="inner")
    targets = merged["target_a"].to_numpy()
    return targets, merged["pred_a"].to_numpy(), merged["pred_b"].to_numpy()


if main_results:
    statistical_rows = []
    common_seeds = sorted(
        set(seed for key, seed in main_results if key == "cnn_lstm_baseline")
        & set(seed for key, seed in main_results if key == "temporal_vit_proposed")
    )

    for seed in common_seeds:
        baseline = main_results[("cnn_lstm_baseline", seed)]
        proposed = main_results[("temporal_vit_proposed", seed)]
        targets, baseline_preds, proposed_preds = align_predictions(baseline, proposed)
        baseline_abs = np.abs(baseline_preds - targets)
        proposed_abs = np.abs(proposed_preds - targets)

        try:
            wilcoxon_result = wilcoxon(
                baseline_abs,
                proposed_abs,
                alternative="greater",
                zero_method="wilcox",
            )
            wilcoxon_stat = float(wilcoxon_result.statistic)
            wilcoxon_p = float(wilcoxon_result.pvalue)
        except ValueError:
            wilcoxon_stat = float("nan")
            wilcoxon_p = float("nan")

        permutation = paired_sign_flip_permutation_test(
            baseline_abs,
            proposed_abs,
            CONFIG.permutation_repetitions,
            seed,
        )
        statistical_rows.append(
            {
                "seed": seed,
                "baseline_mae": float(np.mean(baseline_abs)),
                "proposed_mae": float(np.mean(proposed_abs)),
                "mae_reduction_percent": float(
                    100.0 * (np.mean(baseline_abs) - np.mean(proposed_abs)) / np.mean(baseline_abs)
                ),
                "wilcoxon_statistic": wilcoxon_stat,
                "wilcoxon_one_sided_p_value": wilcoxon_p,
                **permutation,
            }
        )

    statistical_frame = pd.DataFrame(statistical_rows)
    statistical_frame.to_csv(TABLE_DIR / "paired_statistical_tests.csv", index=False)

# ============================================================
# 16. MAIN PAPER FIGURES
# ============================================================


def first_seed_result(model_key: str) -> Optional[RunResult]:
    matches = [result for (key, _), result in main_results.items() if key == model_key]
    return sorted(matches, key=lambda item: item.seed)[0] if matches else None


def plot_training_and_validation_loss() -> None:
    baseline = first_seed_result("cnn_lstm_baseline")
    proposed = first_seed_result("temporal_vit_proposed")
    if baseline is None or proposed is None:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(baseline.history["epoch"], baseline.history["train_loss"], label="Baseline train loss")
    ax.plot(baseline.history["epoch"], baseline.history["val_loss"], label="Baseline validation loss")
    ax.plot(proposed.history["epoch"], proposed.history["train_loss"], label="Proposed train loss")
    ax.plot(proposed.history["epoch"], proposed.history["val_loss"], label="Proposed validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized MSE loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, "Fig04_training_validation_loss")


def plot_epoch_training_time() -> None:
    baseline = first_seed_result("cnn_lstm_baseline")
    proposed = first_seed_result("temporal_vit_proposed")
    if baseline is None or proposed is None:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(baseline.history["epoch"], baseline.history["epoch_time_seconds"], label="Baseline epoch time")
    ax.plot(proposed.history["epoch"], proposed.history["epoch_time_seconds"], label="Proposed epoch time")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Seconds")
    ax.set_title("Epoch Training Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, "Fig05_epoch_training_time")


def prediction_frame_from_result(result: RunResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_index": result.target_indices,
            "timestamp": pd.to_datetime(cached_timestamps[result.target_indices]),
            "actual": result.targets,
            "predicted": result.predictions,
        }
    ).sort_values("timestamp")


def select_day_panels(result: RunResult, minimum_points: int = 20) -> List[pd.Timestamp]:
    frame = prediction_frame_from_result(result)
    frame["day"] = frame["timestamp"].dt.date
    roughness = []
    for day, group in frame.groupby("day"):
        ordered = group.sort_values("timestamp")
        if len(ordered) < minimum_points:
            continue
        diffs = np.diff(ordered["actual"].to_numpy())
        roughness.append((day, float(np.var(diffs))))

    roughness.sort(key=lambda item: item[1])
    if len(roughness) < 4:
        return [pd.Timestamp(day) for day, _ in roughness]
    selected = roughness[:2] + roughness[-2:]
    return [pd.Timestamp(day) for day, _ in selected]


def plot_daywise(result: RunResult, stem: str, title: str) -> None:
    frame = prediction_frame_from_result(result)
    frame["day"] = frame["timestamp"].dt.date
    selected_days = select_day_panels(result)
    if not selected_days:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.reshape(-1)

    for axis, day_ts in zip(axes, selected_days):
        day = day_ts.date()
        group = frame[frame["day"] == day].sort_values("timestamp")
        axis.plot(group["timestamp"], group["actual"], label="Actual PV power")
        axis.plot(group["timestamp"], group["predicted"], label="Predicted PV power")
        axis.set_title(f"Actual vs predicted power for {day}")
        axis.set_xlabel("Time")
        axis.set_ylabel("PV power")
        axis.grid(True, alpha=0.3)
        axis.legend()
        axis.tick_params(axis="x", rotation=30)

    for axis in axes[len(selected_days):]:
        axis.axis("off")

    fig.suptitle(title, y=1.01)
    save_figure(fig, stem)


def plot_combined_scatter() -> None:
    baseline = first_seed_result("cnn_lstm_baseline")
    proposed = first_seed_result("temporal_vit_proposed")
    if baseline is None or proposed is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    pairs = [
        (baseline, "(a) CNN-LSTM baseline"),
        (proposed, "(b) Proposed TemporalViT"),
    ]
    all_values = np.concatenate([baseline.targets, baseline.predictions, proposed.targets, proposed.predictions])
    low, high = float(np.min(all_values)), float(np.max(all_values))

    for axis, (result, title) in zip(axes, pairs):
        axis.scatter(result.targets, result.predictions, alpha=0.45)
        axis.plot([low, high], [low, high], linestyle="--", label="Perfect prediction")
        axis.set_title(title)
        axis.set_xlabel("Actual PV power")
        axis.set_ylabel("Predicted PV power")
        axis.grid(True, alpha=0.3)
        axis.legend()

    save_figure(fig, "Fig08_actual_vs_predicted_scatter")


def plot_residual_distributions() -> None:
    baseline = first_seed_result("cnn_lstm_baseline")
    proposed = first_seed_result("temporal_vit_proposed")
    if baseline is None or proposed is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(baseline.predictions - baseline.targets, bins=40, alpha=0.7, label="CNN-LSTM")
    axes[0].hist(proposed.predictions - proposed.targets, bins=40, alpha=0.7, label="TemporalViT")
    axes[0].axvline(0.0, linestyle="--")
    axes[0].set_xlabel("Prediction residual")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Residual Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(baseline.targets, baseline.predictions - baseline.targets, alpha=0.35, label="CNN-LSTM")
    axes[1].scatter(proposed.targets, proposed.predictions - proposed.targets, alpha=0.35, label="TemporalViT")
    axes[1].axhline(0.0, linestyle="--")
    axes[1].set_xlabel("Actual PV power")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuals versus Actual Power")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    save_figure(fig, "Fig10_residual_analysis")


def plot_main_metric_comparison() -> None:
    if main_summary.empty:
        return
    ordered = main_summary.copy()
    metrics = ["mae_mean", "rmse_mean"]
    x = np.arange(len(ordered))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].bar(x - width / 2, ordered[metrics[0]], width, yerr=ordered.get("mae_std", 0), label="MAE")
    axes[0].bar(x + width / 2, ordered[metrics[1]], width, yerr=ordered.get("rmse_std", 0), label="RMSE")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ordered["model"], rotation=30, ha="right")
    axes[0].set_ylabel("Error")
    axes[0].set_title("Forecasting Error Comparison")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, ordered["r2_mean"], yerr=ordered.get("r2_std", 0))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ordered["model"], rotation=30, ha="right")
    axes[1].set_ylabel("R²")
    axes[1].set_title("Coefficient of Determination")
    axes[1].grid(True, axis="y", alpha=0.3)

    save_figure(fig, "Fig11_main_model_metric_comparison")


def plot_efficiency_comparison() -> None:
    if efficiency_frame.empty:
        return

    frame = efficiency_frame.copy()
    x = np.arange(len(frame))
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.reshape(-1)

    columns = [
        ("parameters", "Trainable parameters"),
        ("model_size_mb", "Model size (MB)"),
        ("inference_latency_ms_per_sample", "Inference latency (ms/sample)"),
        ("peak_inference_memory_mb", "Peak inference memory (MB)"),
    ]

    for axis, (column, title) in zip(axes, columns):
        axis.bar(x, frame[column])
        axis.set_xticks(x)
        axis.set_xticklabels(frame["model_label"], rotation=30, ha="right")
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.3)

    save_figure(fig, "Fig12_computational_efficiency")


if main_results:
    plot_training_and_validation_loss()
    plot_epoch_training_time()

    baseline_result = first_seed_result("cnn_lstm_baseline")
    proposed_result = first_seed_result("temporal_vit_proposed")
    if baseline_result is not None:
        plot_daywise(
            baseline_result,
            "Fig06_daywise_cnn_lstm_baseline",
            "CNN-LSTM Baseline: Actual versus Predicted PV Power",
        )
    if proposed_result is not None:
        plot_daywise(
            proposed_result,
            "Fig07_daywise_temporal_vit_proposed",
            "Proposed TemporalViT: Actual versus Predicted PV Power",
        )

    plot_combined_scatter()
    plot_residual_distributions()
    plot_main_metric_comparison()
    plot_efficiency_comparison()

# ============================================================
# 17. ABLATION STUDY
# ============================================================

ablation_results: Dict[Tuple[str, int], RunResult] = {}
ablation_summary = pd.DataFrame()
ablation_tests = pd.DataFrame()


def alias_run_result(
    result: RunResult,
    model_key: str,
    model_label: str,
) -> RunResult:
    """Create a presentation alias without mutating the original main result."""
    return RunResult(
        model_key=model_key,
        model_label=model_label,
        seed=result.seed,
        model=result.model,
        history=result.history.copy(),
        metrics=dict(result.metrics),
        predictions=result.predictions.copy(),
        targets=result.targets.copy(),
        target_indices=result.target_indices.copy(),
        best_epoch=result.best_epoch,
        total_training_seconds=result.total_training_seconds,
        checkpoint_path=result.checkpoint_path,
    )


if CONFIG.run_ablations:
    print("\n" + "=" * 90)
    print("STARTING ABLATION STUDY")
    print("=" * 90)

    ablation_loaders_by_seed = {
        seed: build_loader_bundle(main_split_starts, MAIN_HORIZON_STEPS, seed)
        for seed in CONFIG.ablation_seeds
    }

    # Reuse the already completed full proposed model and direct-patch
    # comparator. Only genuinely new variants are trained below.
    for seed in CONFIG.ablation_seeds:
        loaders = ablation_loaders_by_seed[seed]

        try:
            full_result = main_results.get(("temporal_vit_proposed", seed))
            if full_result is None:
                full_result = train_or_load_model(
                    MODEL_SPECS["temporal_vit_proposed"],
                    seed,
                    loaders,
                    experiment_group="main_benchmarks",
                )
            ablation_results[("temporal_vit_proposed", seed)] = alias_run_result(
                full_result,
                "temporal_vit_proposed",
                "Complete proposed TemporalViT",
            )
        except Exception as exc:
            record_failed_run("ablations_reuse", "temporal_vit_proposed", seed, exc)
            if not CONFIG.continue_on_error:
                raise

        try:
            direct_result = main_results.get(("direct_patch_temporal_vit", seed))
            if direct_result is None:
                direct_result = train_or_load_model(
                    MODEL_SPECS["direct_patch_temporal_vit"],
                    seed,
                    loaders,
                    experiment_group="main_benchmarks",
                )
            ablation_results[("ablation_no_conv_stem", seed)] = alias_run_result(
                direct_result,
                "ablation_no_conv_stem",
                "Without convolutional stem",
            )
        except Exception as exc:
            record_failed_run("ablations_reuse", "ablation_no_conv_stem", seed, exc)
            if not CONFIG.continue_on_error:
                raise
        cleanup_memory()

    new_ablation_keys = (
        "ablation_no_temporal_pos",
        "ablation_no_spatial_pos",
        "ablation_no_power",
        "ablation_no_droppath",
    )
    new_ablation_results = run_model_set(
        ABLATION_SPECS,
        new_ablation_keys,
        CONFIG.ablation_seeds,
        ablation_loaders_by_seed,
        experiment_group="ablations",
    )
    ablation_results.update(new_ablation_results)

    if ablation_results:
        ablation_long = results_to_long_dataframe(ablation_results)
        ablation_summary = aggregate_results(ablation_long)

        ablation_order = [
            "temporal_vit_proposed",
            "ablation_no_conv_stem",
            "ablation_no_temporal_pos",
            "ablation_no_spatial_pos",
            "ablation_no_power",
            "ablation_no_droppath",
        ]
        ablation_summary["order"] = ablation_summary["model_key"].map(
            {key: index for index, key in enumerate(ablation_order)}
        )
        ablation_summary = (
            ablation_summary.sort_values("order")
            .drop(columns="order")
            .reset_index(drop=True)
        )

        full_rows = ablation_summary[
            ablation_summary["model_key"] == "temporal_vit_proposed"
        ]
        if not full_rows.empty:
            full_row = full_rows.iloc[0]
            ablation_summary["mae_increase_percent"] = (
                (ablation_summary["mae_mean"] - full_row["mae_mean"])
                / max(abs(full_row["mae_mean"]), 1e-12)
                * 100.0
            )
            ablation_summary["rmse_increase_percent"] = (
                (ablation_summary["rmse_mean"] - full_row["rmse_mean"])
                / max(abs(full_row["rmse_mean"]), 1e-12)
                * 100.0
            )
            ablation_summary["r2_reduction"] = (
                full_row["r2_mean"] - ablation_summary["r2_mean"]
            )
            ref_mask = ablation_summary["model_key"] == "temporal_vit_proposed"
            ablation_summary.loc[
                ref_mask,
                ["mae_increase_percent", "rmse_increase_percent", "r2_reduction"],
            ] = 0.0

        ablation_long.to_csv(
            TABLE_DIR / "ablation_results_all_seeds.csv", index=False
        )
        ablation_summary.to_csv(
            TABLE_DIR / "ablation_results_mean_std.csv", index=False
        )

        # Paired tests: does removing a component increase absolute error?
        ablation_test_rows: List[Dict[str, Any]] = []
        for variant_key in ablation_order[1:]:
            for seed in CONFIG.ablation_seeds:
                full = ablation_results.get(("temporal_vit_proposed", seed))
                variant = ablation_results.get((variant_key, seed))
                if full is None or variant is None:
                    continue
                targets, full_preds, variant_preds = align_predictions(full, variant)
                full_abs = np.abs(full_preds - targets)
                variant_abs = np.abs(variant_preds - targets)
                try:
                    test = wilcoxon(
                        variant_abs,
                        full_abs,
                        alternative="greater",
                        zero_method="wilcox",
                    )
                    statistic = float(test.statistic)
                    p_value = float(test.pvalue)
                except ValueError:
                    statistic = float("nan")
                    p_value = float("nan")
                permutation = paired_sign_flip_permutation_test(
                    variant_abs,
                    full_abs,
                    CONFIG.permutation_repetitions,
                    seed,
                )
                ablation_test_rows.append(
                    {
                        "variant_key": variant_key,
                        "variant": variant.model_label,
                        "seed": seed,
                        "full_mae": float(np.mean(full_abs)),
                        "variant_mae": float(np.mean(variant_abs)),
                        "mae_increase_percent": float(
                            100.0
                            * (np.mean(variant_abs) - np.mean(full_abs))
                            / max(np.mean(full_abs), 1e-12)
                        ),
                        "wilcoxon_statistic": statistic,
                        "wilcoxon_one_sided_p_value": p_value,
                        "mean_variant_minus_full_absolute_error": permutation[
                            "mean_mae_difference"
                        ],
                        "permutation_one_sided_p_value": permutation[
                            "one_sided_p_value"
                        ],
                    }
                )
        ablation_tests = pd.DataFrame(ablation_test_rows)
        ablation_tests.to_csv(
            TABLE_DIR / "ablation_paired_statistical_tests.csv", index=False
        )

        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        x = np.arange(len(ablation_summary))
        labels = ablation_summary["model"].tolist()
        panels = [
            ("mae_mean", "mae_std", "MAE", "(a) Mean Absolute Error"),
            ("rmse_mean", "rmse_std", "RMSE", "(b) Root Mean Squared Error"),
            ("r2_mean", "r2_std", "R²", "(c) Coefficient of Determination"),
        ]
        for axis, (mean_col, std_col, ylabel, title) in zip(axes, panels):
            axis.bar(
                x,
                ablation_summary[mean_col],
                yerr=ablation_summary.get(std_col, 0.0),
                capsize=4,
            )
            axis.set_xticks(x)
            axis.set_xticklabels(labels, rotation=32, ha="right")
            axis.set_ylabel(ylabel)
            axis.set_title(title)
            axis.grid(True, axis="y", alpha=0.3)
        save_figure(fig, "Fig13_ablation_study")


# ============================================================
# 18. MULTI-HORIZON FORECASTING STUDY
# ============================================================

multi_horizon_results: Dict[Tuple[int, str, int], RunResult] = {}
multi_horizon_summary = pd.DataFrame()
multi_horizon_tests = pd.DataFrame()


def fixed_split_for_horizon(horizon_steps: int) -> Dict[str, List[int]]:
    valid_starts = build_valid_start_indices(
        cached_timestamps,
        CONFIG.sequence_length,
        horizon_steps,
        CONFIG.max_allowed_gap_minutes,
    )
    # Reuse the same calendar-day partitions so horizon comparisons are fair.
    return split_starts_by_days(valid_starts, horizon_steps, MAIN_SPLIT_DAYS)


if CONFIG.run_multi_horizon:
    print("\n" + "=" * 90)
    print("STARTING MULTI-HORIZON FORECASTING STUDY")
    print("=" * 90)

    horizon_rows: List[Dict[str, Any]] = []
    horizon_test_rows: List[Dict[str, Any]] = []

    for horizon_minutes in CONFIG.horizon_minutes:
        horizon_steps = minutes_to_steps(horizon_minutes)
        horizon_split = fixed_split_for_horizon(horizon_steps)
        split_counts = {key: len(value) for key, value in horizon_split.items()}
        save_json(
            {
                "horizon_minutes": horizon_minutes,
                "horizon_steps": horizon_steps,
                **split_counts,
            },
            TABLE_DIR / f"multi_horizon_{horizon_minutes:02d}min_split_summary.json",
        )

        for seed in CONFIG.multi_horizon_seeds:
            loaders = build_loader_bundle(horizon_split, horizon_steps, seed)

            for model_key in ("cnn_lstm_baseline", "temporal_vit_proposed"):
                spec = MODEL_SPECS[model_key]
                try:
                    # The one-minute result is identical to the main benchmark.
                    if horizon_minutes == CONFIG.forecast_horizon_minutes:
                        result = main_results.get((model_key, seed))
                        if result is None:
                            result = train_or_load_model(
                                spec,
                                seed,
                                loaders,
                                experiment_group="main_benchmarks",
                            )
                    else:
                        group_name = f"multi_horizon_{horizon_minutes:02d}min"
                        result = train_or_load_model(spec, seed, loaders, group_name)

                    multi_horizon_results[(horizon_minutes, model_key, seed)] = result
                    row = {
                        "horizon_minutes": horizon_minutes,
                        "horizon_steps": horizon_steps,
                        "train_sequences": len(horizon_split["train"]),
                        "validation_sequences": len(horizon_split["val"]),
                        "test_sequences": len(horizon_split["test"]),
                        "model_key": model_key,
                        "model": spec.label,
                        "seed": seed,
                        "best_epoch": result.best_epoch,
                        "total_training_seconds": result.total_training_seconds,
                    }
                    row.update(result.metrics)
                    horizon_rows.append(row)
                except Exception as exc:
                    record_failed_run(
                        f"multi_horizon_{horizon_minutes:02d}min",
                        model_key,
                        seed,
                        exc,
                    )
                    if not CONFIG.continue_on_error:
                        raise
                finally:
                    cleanup_memory()

            baseline = multi_horizon_results.get(
                (horizon_minutes, "cnn_lstm_baseline", seed)
            )
            proposed = multi_horizon_results.get(
                (horizon_minutes, "temporal_vit_proposed", seed)
            )
            if baseline is not None and proposed is not None:
                targets, baseline_preds, proposed_preds = align_predictions(
                    baseline, proposed
                )
                baseline_abs = np.abs(baseline_preds - targets)
                proposed_abs = np.abs(proposed_preds - targets)
                try:
                    test = wilcoxon(
                        baseline_abs,
                        proposed_abs,
                        alternative="greater",
                        zero_method="wilcox",
                    )
                    statistic = float(test.statistic)
                    p_value = float(test.pvalue)
                except ValueError:
                    statistic = float("nan")
                    p_value = float("nan")
                permutation = paired_sign_flip_permutation_test(
                    baseline_abs,
                    proposed_abs,
                    CONFIG.permutation_repetitions,
                    seed + horizon_minutes,
                )
                horizon_test_rows.append(
                    {
                        "horizon_minutes": horizon_minutes,
                        "seed": seed,
                        "baseline_mae": float(np.mean(baseline_abs)),
                        "proposed_mae": float(np.mean(proposed_abs)),
                        "proposed_mae_reduction_percent": float(
                            100.0
                            * (np.mean(baseline_abs) - np.mean(proposed_abs))
                            / max(np.mean(baseline_abs), 1e-12)
                        ),
                        "wilcoxon_statistic": statistic,
                        "wilcoxon_one_sided_p_value": p_value,
                        **permutation,
                    }
                )

    horizon_long = pd.DataFrame(horizon_rows)
    if not horizon_long.empty:
        horizon_long.to_csv(
            TABLE_DIR / "multi_horizon_results_all_seeds.csv", index=False
        )

        summary_rows: List[Dict[str, Any]] = []
        for (horizon, model_key, model), group in horizon_long.groupby(
            ["horizon_minutes", "model_key", "model"], sort=True
        ):
            row = {
                "horizon_minutes": int(horizon),
                "model_key": model_key,
                "model": model,
                "runs": len(group),
                "test_sequences": int(group["test_sequences"].iloc[0]),
            }
            for metric in ("mae", "mse", "rmse", "r2", "mbe", "pearson_r"):
                row[f"{metric}_mean"] = float(group[metric].mean())
                row[f"{metric}_std"] = (
                    float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
                )
            summary_rows.append(row)
        multi_horizon_summary = pd.DataFrame(summary_rows)

        # Quantify degradation relative to one-minute forecasting.
        for model_key in ("cnn_lstm_baseline", "temporal_vit_proposed"):
            model_mask = multi_horizon_summary["model_key"] == model_key
            one_minute = multi_horizon_summary[
                model_mask
                & (
                    multi_horizon_summary["horizon_minutes"]
                    == CONFIG.forecast_horizon_minutes
                )
            ]
            if one_minute.empty:
                continue
            reference = one_minute.iloc[0]
            multi_horizon_summary.loc[
                model_mask, "mae_increase_vs_1min_percent"
            ] = (
                (
                    multi_horizon_summary.loc[model_mask, "mae_mean"]
                    - reference["mae_mean"]
                )
                / max(reference["mae_mean"], 1e-12)
                * 100.0
            )
            multi_horizon_summary.loc[
                model_mask, "rmse_increase_vs_1min_percent"
            ] = (
                (
                    multi_horizon_summary.loc[model_mask, "rmse_mean"]
                    - reference["rmse_mean"]
                )
                / max(reference["rmse_mean"], 1e-12)
                * 100.0
            )
            multi_horizon_summary.loc[model_mask, "r2_change_vs_1min"] = (
                multi_horizon_summary.loc[model_mask, "r2_mean"]
                - reference["r2_mean"]
            )

        # Proposed-model advantage over the CNN-LSTM at each horizon.
        relative_rows: List[Dict[str, Any]] = []
        for horizon in sorted(multi_horizon_summary["horizon_minutes"].unique()):
            base = multi_horizon_summary[
                (multi_horizon_summary["horizon_minutes"] == horizon)
                & (multi_horizon_summary["model_key"] == "cnn_lstm_baseline")
            ]
            prop = multi_horizon_summary[
                (multi_horizon_summary["horizon_minutes"] == horizon)
                & (multi_horizon_summary["model_key"] == "temporal_vit_proposed")
            ]
            if base.empty or prop.empty:
                continue
            base_row, prop_row = base.iloc[0], prop.iloc[0]
            relative_rows.append(
                {
                    "horizon_minutes": int(horizon),
                    "proposed_mae_reduction_percent": float(
                        100.0
                        * (base_row["mae_mean"] - prop_row["mae_mean"])
                        / max(base_row["mae_mean"], 1e-12)
                    ),
                    "proposed_rmse_reduction_percent": float(
                        100.0
                        * (base_row["rmse_mean"] - prop_row["rmse_mean"])
                        / max(base_row["rmse_mean"], 1e-12)
                    ),
                    "proposed_r2_difference": float(
                        prop_row["r2_mean"] - base_row["r2_mean"]
                    ),
                }
            )
        pd.DataFrame(relative_rows).to_csv(
            TABLE_DIR / "multi_horizon_relative_improvements.csv", index=False
        )
        multi_horizon_summary.to_csv(
            TABLE_DIR / "multi_horizon_results_mean_std.csv", index=False
        )

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        for model_key, label in [
            ("cnn_lstm_baseline", "CNN-LSTM baseline"),
            ("temporal_vit_proposed", "Proposed TemporalViT"),
        ]:
            frame = multi_horizon_summary[
                multi_horizon_summary["model_key"] == model_key
            ].sort_values("horizon_minutes")
            axes[0].errorbar(
                frame["horizon_minutes"],
                frame["mae_mean"],
                yerr=frame["mae_std"],
                marker="o",
                capsize=3,
                label=label,
            )
            axes[1].errorbar(
                frame["horizon_minutes"],
                frame["rmse_mean"],
                yerr=frame["rmse_std"],
                marker="o",
                capsize=3,
                label=label,
            )
            axes[2].errorbar(
                frame["horizon_minutes"],
                frame["r2_mean"],
                yerr=frame["r2_std"],
                marker="o",
                capsize=3,
                label=label,
            )
        for axis, ylabel, title in zip(
            axes,
            ["MAE", "RMSE", "R²"],
            [
                "(a) Mean Absolute Error",
                "(b) Root Mean Squared Error",
                "(c) Coefficient of Determination",
            ],
        ):
            axis.set_xlabel("Forecast horizon (minutes)")
            axis.set_ylabel(ylabel)
            axis.set_title(title)
            axis.grid(True, alpha=0.3)
            axis.legend()
        save_figure(fig, "Fig09_multi_horizon_performance")

    multi_horizon_tests = pd.DataFrame(horizon_test_rows)
    multi_horizon_tests.to_csv(
        TABLE_DIR / "multi_horizon_paired_statistical_tests.csv", index=False
    )


# ============================================================
# 19. DATA-SIZE / SAMPLE-EFFICIENCY STUDY
# ============================================================

data_size_results: Dict[Tuple[int, str, int], RunResult] = {}
data_size_summary = pd.DataFrame()

if CONFIG.run_data_size_study:
    print("\n" + "=" * 90)
    print("STARTING DATA-SIZE / SAMPLE-EFFICIENCY STUDY")
    print("=" * 90)

    data_size_rows: List[Dict[str, Any]] = []
    full_train_count = len(main_split_starts["train"])

    for requested_count in CONFIG.data_size_counts:
        actual_count = (
            full_train_count
            if requested_count is None
            else min(int(requested_count), full_train_count)
        )

        for seed in CONFIG.data_size_seeds:
            loaders = build_loader_bundle(
                main_split_starts,
                MAIN_HORIZON_STEPS,
                seed,
                train_limit=actual_count,
            )

            for model_key in ("cnn_lstm_baseline", "temporal_vit_proposed"):
                spec = MODEL_SPECS[model_key]
                try:
                    # Reuse full-data main results rather than retraining them.
                    if actual_count == full_train_count:
                        result = main_results.get((model_key, seed))
                        if result is None:
                            result = train_or_load_model(
                                spec,
                                seed,
                                loaders,
                                experiment_group="main_benchmarks",
                            )
                    else:
                        group_name = f"data_size_{actual_count}"
                        result = train_or_load_model(spec, seed, loaders, group_name)

                    data_size_results[(actual_count, model_key, seed)] = result
                    row = {
                        "training_sequences": actual_count,
                        "training_fraction_percent": 100.0
                        * actual_count
                        / full_train_count,
                        "model_key": model_key,
                        "model": spec.label,
                        "seed": seed,
                        "best_epoch": result.best_epoch,
                        "total_training_seconds": result.total_training_seconds,
                    }
                    row.update(result.metrics)
                    data_size_rows.append(row)
                except Exception as exc:
                    record_failed_run(
                        f"data_size_{actual_count}", model_key, seed, exc
                    )
                    if not CONFIG.continue_on_error:
                        raise
                finally:
                    cleanup_memory()

    data_size_long = pd.DataFrame(data_size_rows)
    if not data_size_long.empty:
        data_size_long.to_csv(
            TABLE_DIR / "data_size_results_all_seeds.csv", index=False
        )

        summary_rows: List[Dict[str, Any]] = []
        for (count, fraction, model_key, model), group in data_size_long.groupby(
            ["training_sequences", "training_fraction_percent", "model_key", "model"],
            sort=True,
        ):
            row = {
                "training_sequences": int(count),
                "training_fraction_percent": float(fraction),
                "model_key": model_key,
                "model": model,
                "runs": len(group),
            }
            for metric in ("mae", "rmse", "r2"):
                row[f"{metric}_mean"] = float(group[metric].mean())
                row[f"{metric}_std"] = (
                    float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
                )
            summary_rows.append(row)
        data_size_summary = pd.DataFrame(summary_rows)

        # Error reduction obtained by using the complete training set.
        for model_key in ("cnn_lstm_baseline", "temporal_vit_proposed"):
            model_mask = data_size_summary["model_key"] == model_key
            full_row = data_size_summary[
                model_mask
                & (data_size_summary["training_sequences"] == full_train_count)
            ]
            if full_row.empty:
                continue
            full_row = full_row.iloc[0]
            data_size_summary.loc[model_mask, "mae_gap_to_full_percent"] = (
                (
                    data_size_summary.loc[model_mask, "mae_mean"]
                    - full_row["mae_mean"]
                )
                / max(full_row["mae_mean"], 1e-12)
                * 100.0
            )
            data_size_summary.loc[model_mask, "rmse_gap_to_full_percent"] = (
                (
                    data_size_summary.loc[model_mask, "rmse_mean"]
                    - full_row["rmse_mean"]
                )
                / max(full_row["rmse_mean"], 1e-12)
                * 100.0
            )

        data_size_summary.to_csv(
            TABLE_DIR / "data_size_results_mean_std.csv", index=False
        )

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        for model_key, label in [
            ("cnn_lstm_baseline", "CNN-LSTM baseline"),
            ("temporal_vit_proposed", "Proposed TemporalViT"),
        ]:
            frame = data_size_summary[
                data_size_summary["model_key"] == model_key
            ].sort_values("training_sequences")
            for axis, metric, std in [
                (axes[0], "mae_mean", "mae_std"),
                (axes[1], "rmse_mean", "rmse_std"),
                (axes[2], "r2_mean", "r2_std"),
            ]:
                axis.errorbar(
                    frame["training_sequences"],
                    frame[metric],
                    yerr=frame[std],
                    marker="o",
                    capsize=3,
                    label=label,
                )
        for axis, ylabel, title in zip(
            axes,
            ["MAE", "RMSE", "R²"],
            [
                "(a) Mean Absolute Error",
                "(b) Root Mean Squared Error",
                "(c) Coefficient of Determination",
            ],
        ):
            axis.set_xlabel("Number of training sequences")
            axis.set_ylabel(ylabel)
            axis.set_title(title)
            axis.grid(True, alpha=0.3)
            axis.legend()
        save_figure(fig, "Fig14_data_size_study")


# ============================================================
# 20. FORECAST SKILL RELATIVE TO PERSISTENCE
# ============================================================

forecast_skill_frame = pd.DataFrame()

if CONFIG.run_forecast_skill_analysis and not main_summary.empty:
    persistence = main_summary[
        main_summary["model_key"] == "persistence"
    ]
    if not persistence.empty:
        reference = persistence.iloc[0]
        forecast_skill_frame = main_summary.copy()
        forecast_skill_frame["mae_improvement_vs_persistence_percent"] = (
            100.0
            * (reference["mae_mean"] - forecast_skill_frame["mae_mean"])
            / max(reference["mae_mean"], 1e-12)
        )
        forecast_skill_frame["rmse_improvement_vs_persistence_percent"] = (
            100.0
            * (reference["rmse_mean"] - forecast_skill_frame["rmse_mean"])
            / max(reference["rmse_mean"], 1e-12)
        )
        forecast_skill_frame["mse_skill_score"] = (
            1.0
            - forecast_skill_frame["mse_mean"]
            / max(reference["mse_mean"], 1e-12)
        )
        forecast_skill_frame.to_csv(
            TABLE_DIR / "forecast_skill_relative_to_persistence.csv", index=False
        )


# ============================================================
# 21. DAY-WISE AND RAPID-RAMP ERROR ANALYSIS
# ============================================================

daywise_summary = pd.DataFrame()
ramp_summary = pd.DataFrame()


def build_ramp_labels(target_indices: np.ndarray, targets: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "target_index": target_indices,
            "timestamp": pd.to_datetime(cached_timestamps[target_indices]),
            "target": targets,
        }
    ).sort_values("timestamp")
    frame["day"] = frame["timestamp"].dt.date
    frame["absolute_power_change"] = (
        frame.groupby("day")["target"].diff().abs()
    )
    valid_changes = frame["absolute_power_change"].dropna().to_numpy()
    threshold = (
        float(np.quantile(valid_changes, CONFIG.ramp_event_quantile))
        if len(valid_changes)
        else float("nan")
    )
    frame["is_ramp_event"] = frame["absolute_power_change"] >= threshold
    frame["ramp_threshold"] = threshold
    return frame


if CONFIG.run_daywise_ramp_analysis and main_results:
    day_rows: List[Dict[str, Any]] = []
    ramp_rows: List[Dict[str, Any]] = []

    for model_key in ("cnn_lstm_baseline", "temporal_vit_proposed"):
        for seed in CONFIG.main_seeds:
            result = main_results.get((model_key, seed))
            if result is None:
                continue
            frame = prediction_frame_from_result(result)
            frame["day"] = frame["timestamp"].dt.date
            for day, group in frame.groupby("day"):
                metrics = regression_metrics(
                    group["actual"].to_numpy(), group["predicted"].to_numpy()
                )
                changes = np.abs(np.diff(group.sort_values("timestamp")["actual"]))
                day_rows.append(
                    {
                        "model_key": model_key,
                        "model": result.model_label,
                        "seed": seed,
                        "day": str(day),
                        "samples": len(group),
                        "mean_absolute_power_change": float(np.mean(changes))
                        if len(changes)
                        else 0.0,
                        "maximum_absolute_power_change": float(np.max(changes))
                        if len(changes)
                        else 0.0,
                        **metrics,
                    }
                )

            labels = build_ramp_labels(result.target_indices, result.targets)
            prediction_frame = pd.DataFrame(
                {
                    "target_index": result.target_indices,
                    "prediction": result.predictions,
                }
            )
            merged = labels.merge(prediction_frame, on="target_index", how="inner")
            for event_label, event_group in merged.groupby("is_ramp_event"):
                if len(event_group) == 0:
                    continue
                metrics = regression_metrics(
                    event_group["target"].to_numpy(),
                    event_group["prediction"].to_numpy(),
                )
                ramp_rows.append(
                    {
                        "model_key": model_key,
                        "model": result.model_label,
                        "seed": seed,
                        "event_type": "Rapid-ramp" if event_label else "Non-ramp",
                        "samples": len(event_group),
                        "ramp_quantile": CONFIG.ramp_event_quantile,
                        "ramp_threshold": float(event_group["ramp_threshold"].iloc[0]),
                        **metrics,
                    }
                )

    daywise_long = pd.DataFrame(day_rows)
    if not daywise_long.empty:
        daywise_long.to_csv(
            TABLE_DIR / "daywise_metrics_all_seeds.csv", index=False
        )
        daywise_summary = (
            daywise_long.groupby(["model_key", "model", "day"], as_index=False)
            .agg(
                samples=("samples", "first"),
                mean_absolute_power_change=("mean_absolute_power_change", "first"),
                mae_mean=("mae", "mean"),
                mae_std=("mae", "std"),
                rmse_mean=("rmse", "mean"),
                rmse_std=("rmse", "std"),
                r2_mean=("r2", "mean"),
                r2_std=("r2", "std"),
            )
            .fillna(0.0)
        )
        daywise_summary.to_csv(
            TABLE_DIR / "daywise_metrics_mean_std.csv", index=False
        )

    ramp_long = pd.DataFrame(ramp_rows)
    if not ramp_long.empty:
        ramp_long.to_csv(
            TABLE_DIR / "ramp_event_results_all_seeds.csv", index=False
        )
        ramp_summary = (
            ramp_long.groupby(["model_key", "model", "event_type"], as_index=False)
            .agg(
                samples=("samples", "first"),
                ramp_threshold=("ramp_threshold", "first"),
                mae_mean=("mae", "mean"),
                mae_std=("mae", "std"),
                rmse_mean=("rmse", "mean"),
                rmse_std=("rmse", "std"),
                r2_mean=("r2", "mean"),
                r2_std=("r2", "std"),
            )
            .fillna(0.0)
        )
        ramp_summary.to_csv(
            TABLE_DIR / "ramp_event_results_mean_std.csv", index=False
        )

        event_order = ["Non-ramp", "Rapid-ramp"]
        model_order = ["cnn_lstm_baseline", "temporal_vit_proposed"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        x = np.arange(len(event_order))
        width = 0.35
        for offset, model_key in zip((-width / 2, width / 2), model_order):
            frame = (
                ramp_summary[ramp_summary["model_key"] == model_key]
                .set_index("event_type")
                .reindex(event_order)
            )
            label = (
                "CNN-LSTM baseline"
                if model_key == "cnn_lstm_baseline"
                else "Proposed TemporalViT"
            )
            axes[0].bar(
                x + offset,
                frame["mae_mean"],
                width,
                yerr=frame["mae_std"],
                capsize=3,
                label=label,
            )
            axes[1].bar(
                x + offset,
                frame["rmse_mean"],
                width,
                yerr=frame["rmse_std"],
                capsize=3,
                label=label,
            )
        for axis, ylabel, title in zip(
            axes,
            ["MAE", "RMSE"],
            ["(a) MAE by power-change regime", "(b) RMSE by power-change regime"],
        ):
            axis.set_xticks(x)
            axis.set_xticklabels(event_order)
            axis.set_ylabel(ylabel)
            axis.set_title(title)
            axis.grid(True, axis="y", alpha=0.3)
            axis.legend()
        save_figure(fig, "Fig15_ramp_event_error_analysis")


# ============================================================
# 22. DAY-BLOCK BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================

day_block_ci_frame = pd.DataFrame()


def day_block_bootstrap_intervals(
    result: RunResult,
    repetitions: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    frame = prediction_frame_from_result(result)
    frame["day"] = frame["timestamp"].dt.date
    days = list(frame["day"].unique())
    if not days:
        return {}

    rng = np.random.default_rng(seed)
    values = {name: [] for name in ("mae", "rmse", "r2")}
    grouped = {day: frame[frame["day"] == day] for day in days}

    for _ in range(repetitions):
        sampled_days = rng.choice(days, size=len(days), replace=True)
        sampled = pd.concat([grouped[day] for day in sampled_days], ignore_index=True)
        metrics = regression_metrics(
            sampled["actual"].to_numpy(), sampled["predicted"].to_numpy()
        )
        for metric in values:
            values[metric].append(metrics[metric])

    output: Dict[str, Dict[str, float]] = {}
    for metric, samples in values.items():
        low, high = np.percentile(samples, [2.5, 97.5])
        output[metric] = {
            "estimate": float(result.metrics[metric]),
            "ci_low": float(low),
            "ci_high": float(high),
        }
    return output


if CONFIG.run_day_block_bootstrap and main_results:
    ci_rows: List[Dict[str, Any]] = []
    for (model_key, seed), result in main_results.items():
        intervals = day_block_bootstrap_intervals(
            result,
            CONFIG.bootstrap_repetitions,
            seed + 1000,
        )
        for metric, interval in intervals.items():
            ci_rows.append(
                {
                    "model_key": model_key,
                    "model": result.model_label,
                    "seed": seed,
                    "metric": metric,
                    **interval,
                    "bootstrap_unit": "day",
                    "repetitions": CONFIG.bootstrap_repetitions,
                }
            )
    day_block_ci_frame = pd.DataFrame(ci_rows)
    day_block_ci_frame.to_csv(
        TABLE_DIR / "day_block_bootstrap_95_confidence_intervals.csv",
        index=False,
    )


# ============================================================
# 23. WRITING-READY SECTION 4 NUMERICAL SUMMARY
# ============================================================


def safe_percent_change(old: float, new: float) -> float:
    return 100.0 * (old - new) / max(abs(old), 1e-12)


writing_lines = [
    "SECTION 4: WRITING-READY NUMERICAL SUMMARY",
    "=" * 55,
    "Values below are generated directly from the experiment tables.",
    "Check the corresponding CSV files before copying them into the paper.",
]

if not main_summary.empty:
    writing_lines.extend(["", "1. Main benchmark results"])
    writing_lines.append(main_summary.to_string(index=False))
    baseline_rows = main_summary[
        main_summary["model_key"] == "cnn_lstm_baseline"
    ]
    proposed_rows = main_summary[
        main_summary["model_key"] == "temporal_vit_proposed"
    ]
    if not baseline_rows.empty and not proposed_rows.empty:
        baseline = baseline_rows.iloc[0]
        proposed = proposed_rows.iloc[0]
        writing_lines.extend(
            [
                "",
                f"Proposed vs CNN-LSTM MAE reduction: "
                f"{safe_percent_change(baseline['mae_mean'], proposed['mae_mean']):.3f}%",
                f"Proposed vs CNN-LSTM RMSE reduction: "
                f"{safe_percent_change(baseline['rmse_mean'], proposed['rmse_mean']):.3f}%",
                f"Proposed minus CNN-LSTM R²: "
                f"{proposed['r2_mean'] - baseline['r2_mean']:.6f}",
            ]
        )

if not ablation_summary.empty:
    writing_lines.extend(["", "2. Ablation study", ablation_summary.to_string(index=False)])
    variants = ablation_summary[
        ablation_summary["model_key"] != "temporal_vit_proposed"
    ]
    if not variants.empty and "mae_increase_percent" in variants:
        largest = variants.sort_values("mae_increase_percent", ascending=False).iloc[0]
        writing_lines.append(
            f"Largest MAE deterioration: {largest['model']} "
            f"({largest['mae_increase_percent']:.3f}%)."
        )

if not multi_horizon_summary.empty:
    writing_lines.extend(
        ["", "3. Multi-horizon forecasting", multi_horizon_summary.to_string(index=False)]
    )

if not data_size_summary.empty:
    writing_lines.extend(
        ["", "4. Data-size study", data_size_summary.to_string(index=False)]
    )

if not forecast_skill_frame.empty:
    writing_lines.extend(
        ["", "5. Skill relative to persistence", forecast_skill_frame.to_string(index=False)]
    )

if not ramp_summary.empty:
    writing_lines.extend(
        ["", "6. Rapid-ramp analysis", ramp_summary.to_string(index=False)]
    )

if FAILED_RUNS:
    writing_lines.extend(
        [
            "",
            "WARNING: Some runs failed. Inspect tables/failed_runs.csv before final reporting.",
        ]
    )

section4_summary_path = RUN_ROOT / "section4_numerical_summary.txt"
section4_summary_path.write_text("\n".join(writing_lines), encoding="utf-8")

manifest_rows = [
    ("Main benchmark", "main_benchmark_results_mean_std.csv", "Table: overall performance"),
    ("Forecast skill", "forecast_skill_relative_to_persistence.csv", "Table: improvement over persistence"),
    ("Paired tests", "paired_statistical_tests.csv", "Table: CNN-LSTM versus proposed"),
    ("Efficiency", "computational_efficiency_results.csv", "Table/Figure 12"),
    ("Ablation", "ablation_results_mean_std.csv", "Table/Figure 13"),
    ("Ablation tests", "ablation_paired_statistical_tests.csv", "Statistical support for ablations"),
    ("Multi-horizon", "multi_horizon_results_mean_std.csv", "Table/Figure 9"),
    ("Horizon tests", "multi_horizon_paired_statistical_tests.csv", "Per-horizon statistical tests"),
    ("Data size", "data_size_results_mean_std.csv", "Table/Figure 14"),
    ("Day-wise", "daywise_metrics_mean_std.csv", "Error analysis"),
    ("Ramp events", "ramp_event_results_mean_std.csv", "Table/Figure 15"),
    ("Block bootstrap", "day_block_bootstrap_95_confidence_intervals.csv", "95% confidence intervals"),
]
manifest = pd.DataFrame(manifest_rows, columns=["analysis", "file", "use_in_section_4"])
manifest["exists"] = manifest["file"].apply(lambda name: (TABLE_DIR / name).exists())
manifest.to_csv(TABLE_DIR / "section4_output_manifest.csv", index=False)


# ============================================================
# 24. FINAL RUN SUMMARY
# ============================================================

summary_lines = [
    "TEMPORALVIT ROBUST PV FORECASTING STUDY",
    "========================================",
    f"Run name: {RUN_NAME}",
    f"Device: {CONFIG.device}",
    f"Sampling interval (minutes): {SAMPLING_INTERVAL_MINUTES:.4f}",
    f"Sequence length: {CONFIG.sequence_length}",
    f"Primary forecast horizon: {CONFIG.forecast_horizon_minutes} minute(s)",
    f"Multi-horizon values: {CONFIG.horizon_minutes}",
    f"Epochs: {CONFIG.epochs}",
    f"Early stopping patience: {CONFIG.early_stop_patience}",
    f"Main seeds: {CONFIG.main_seeds}",
    f"Ablation seeds: {CONFIG.ablation_seeds}",
    f"Multi-horizon seeds: {CONFIG.multi_horizon_seeds}",
    f"Data-size seeds: {CONFIG.data_size_seeds}",
    f"Train sequences: {len(main_split_starts['train'])}",
    f"Validation sequences: {len(main_split_starts['val'])}",
    f"Test sequences: {len(main_split_starts['test'])}",
    f"Chronological split purge days: {CONFIG.purge_days_between_splits}",
    f"Figure folder: {FIGURE_DIR}",
    f"Table folder: {TABLE_DIR}",
    f"Failed runs: {len(FAILED_RUNS)}",
    "",
    "Proposed architecture metadata:",
    json.dumps(architecture_metadata, indent=2),
]

for heading, frame in [
    ("Main benchmark mean ± std", main_summary),
    ("Computational efficiency", efficiency_frame),
    ("Ablation mean ± std", ablation_summary),
    ("Multi-horizon mean ± std", multi_horizon_summary),
    ("Data-size mean ± std", data_size_summary),
    ("Ramp-event mean ± std", ramp_summary),
]:
    if isinstance(frame, pd.DataFrame) and not frame.empty:
        summary_lines.extend(["", heading + ":", frame.to_string(index=False)])

summary_path = RUN_ROOT / "run_summary.txt"
summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
save_pipeline_heartbeat("Pipeline completed")

print("\n" + "=" * 90)
print("PIPELINE COMPLETED")
print("Results folder:", RUN_ROOT)
print("Paper-figure folder:", FIGURE_DIR)
print("Tables folder:", TABLE_DIR)
print("Section 4 numerical summary:", section4_summary_path)
print("Run summary:", summary_path)
if FAILED_RUNS:
    print("WARNING: Some runs failed. Inspect:", TABLE_DIR / "failed_runs.csv")
else:
    print("All scheduled runs completed without a recorded exception.")
print("=" * 90)
