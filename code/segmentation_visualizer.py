from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from path_config import DEFAULT_NPC_DATASET_DIR, resolve_project_path


AXES = ("axial", "coronal", "sagittal")
MASK_COLORS = np.array(
    [
        [0, 0, 0],
        [239, 68, 68],
        [34, 197, 94],
        [59, 130, 246],
        [245, 158, 11],
        [168, 85, 247],
        [20, 184, 166],
        [236, 72, 153],
    ],
    dtype=np.uint8,
)


def resolve_dataset_dir(dataset_dir: str | Path | None = None) -> Path:
    """Resolve a user-provided dataset path, defaulting to the bundled NPC data."""
    if dataset_dir is None or str(dataset_dir).strip() == "":
        return DEFAULT_NPC_DATASET_DIR
    return Path(resolve_project_path(dataset_dir)).resolve()


def list_available_cases(dataset_dir: str | Path | None = None) -> list[str]:
    """Return case IDs that contain both image_processed.npy and mask_processed.npy."""
    root = resolve_dataset_dir(dataset_dir)
    if not root.exists():
        return []

    cases: list[str] = []
    for case_dir in root.iterdir():
        if not case_dir.is_dir():
            continue
        if (case_dir / "image_processed.npy").exists() and (case_dir / "mask_processed.npy").exists():
            cases.append(case_dir.name)
    return sorted(cases)


def _case_dir(case_id: str, dataset_dir: str | Path | None = None) -> Path:
    if not case_id:
        raise ValueError("请先选择病例编号。")
    root = resolve_dataset_dir(dataset_dir)
    case_path = root / str(case_id)
    if not case_path.exists():
        raise FileNotFoundError(f"找不到病例目录: {case_path}")
    return case_path


def load_case(case_id: str, dataset_dir: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load a preprocessed image volume and its ground-truth mask."""
    case_path = _case_dir(case_id, dataset_dir)
    image_path = case_path / "image_processed.npy"
    mask_path = case_path / "mask_processed.npy"
    if not image_path.exists() or not mask_path.exists():
        raise FileNotFoundError(f"病例 {case_id} 缺少 image_processed.npy 或 mask_processed.npy。")

    image = np.load(image_path)
    mask = np.load(mask_path)
    if image.ndim != 3 or mask.ndim != 3:
        raise ValueError("当前可视化模块要求 image/mask 都是 3D 数组。")
    if image.shape != mask.shape:
        raise ValueError(f"image 与 mask 尺寸不一致: {image.shape} vs {mask.shape}")
    return image, mask


def load_optional_mask(case_id: str, mask_type: str, dataset_dir: str | Path | None = None) -> np.ndarray | None:
    """Load a real, pseudo, or prediction mask if available."""
    case_path = _case_dir(case_id, dataset_dir)
    mask_type = (mask_type or "真实 mask").strip()
    mask_type_lower = mask_type.lower()

    if "伪" in mask_type or "pseudo" in mask_type_lower:
        mask_path = case_path / "pseudo_mask.npy"
    elif "模型" in mask_type or "预测" in mask_type or "pred" in mask_type_lower or "model" in mask_type_lower:
        candidates = [
            case_path / "prediction_mask.npy",
            case_path / "prediction_medsam2.npy",
            case_path / "prediction_medsam.npy",
            case_path / "prediction_medsam_slice.npy",
        ]
        mask_path = next((path for path in candidates if path.exists()), None)
        if mask_path is None:
            return None
    else:
        mask_path = case_path / "mask_processed.npy"

    if not mask_path.exists():
        return None

    mask = np.load(mask_path)
    if mask.ndim != 3:
        return None
    return mask


def slice_count(volume_shape: Iterable[int], axis: str) -> int:
    """Return the number of valid slices for an axis."""
    shape = tuple(volume_shape)
    axis = normalize_axis(axis)
    if axis == "axial":
        return shape[2]
    if axis == "coronal":
        return shape[1]
    return shape[0]


def normalize_axis(axis: str) -> str:
    axis = (axis or "axial").lower()
    if axis not in AXES:
        return "axial"
    return axis


def clamp_slice_index(volume_shape: Iterable[int], axis: str, index: int | float | None) -> int:
    max_index = max(0, slice_count(volume_shape, axis) - 1)
    if index is None:
        return max_index // 2
    return max(0, min(max_index, int(index)))


def get_slice(volume: np.ndarray, axis: str, index: int | float | None) -> np.ndarray:
    """Extract a 2D slice from a 3D volume."""
    axis = normalize_axis(axis)
    index = clamp_slice_index(volume.shape, axis, index)
    if axis == "axial":
        sliced = volume[:, :, index]
    elif axis == "coronal":
        sliced = volume[:, index, :]
    else:
        sliced = volume[index, :, :]
    return np.rot90(sliced)


def normalize_to_uint8(image_slice: np.ndarray) -> np.ndarray:
    """Convert a medical image slice into display-friendly uint8 grayscale."""
    image = np.asarray(image_slice, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)

    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)

    image = np.clip((image - low) / (high - low), 0, 1)
    return (image * 255).astype(np.uint8)


def grayscale_to_rgb(gray: np.ndarray) -> np.ndarray:
    gray = normalize_to_uint8(gray) if gray.dtype != np.uint8 else gray
    return np.stack([gray, gray, gray], axis=-1)


def colorize_mask(mask_slice: np.ndarray) -> np.ndarray:
    """Colorize an integer mask slice."""
    mask = np.asarray(mask_slice)
    if mask.size == 0:
        return np.zeros((*mask.shape, 3), dtype=np.uint8)

    labels = np.rint(mask).astype(np.int32)
    labels = np.where(labels < 0, 0, labels)
    color_index = labels % len(MASK_COLORS)
    colored = MASK_COLORS[color_index]
    colored[labels == 0] = 0
    return colored.astype(np.uint8)


def overlay_mask(image_slice: np.ndarray, mask_slice: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Create an image + transparent mask overlay."""
    base = grayscale_to_rgb(normalize_to_uint8(image_slice)).astype(np.float32)
    mask_rgb = colorize_mask(mask_slice).astype(np.float32)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    foreground = np.any(mask_rgb > 0, axis=-1, keepdims=True)
    overlay = np.where(foreground, base * (1 - alpha) + mask_rgb * alpha, base)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute binary Dice score on non-zero mask voxels."""
    pred_bin = np.asarray(pred) > 0
    gt_bin = np.asarray(gt) > 0
    pred_sum = int(pred_bin.sum())
    gt_sum = int(gt_bin.sum())
    if pred_sum + gt_sum == 0:
        return 1.0
    intersection = int(np.logical_and(pred_bin, gt_bin).sum())
    return 2.0 * intersection / (pred_sum + gt_sum)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute binary IoU on non-zero mask voxels."""
    pred_bin = np.asarray(pred) > 0
    gt_bin = np.asarray(gt) > 0
    union = int(np.logical_or(pred_bin, gt_bin).sum())
    if union == 0:
        return 1.0
    intersection = int(np.logical_and(pred_bin, gt_bin).sum())
    return intersection / union


def render_case(
    case_id: str,
    axis: str = "axial",
    index: int | float | None = None,
    mask_type: str = "真实 mask",
    alpha: float = 0.45,
    dataset_dir: str | Path | None = None,
) -> dict[str, object]:
    """Load a case and return display images plus metrics."""
    image, gt_mask = load_case(case_id, dataset_dir)
    selected_mask = load_optional_mask(case_id, mask_type, dataset_dir)
    if selected_mask is None:
        selected_mask = np.zeros_like(gt_mask)
        missing_mask = True
    else:
        missing_mask = False

    axis = normalize_axis(axis)
    index = clamp_slice_index(image.shape, axis, index)
    image_slice = get_slice(image, axis, index)
    gt_slice = get_slice(gt_mask, axis, index)
    selected_slice = get_slice(selected_mask, axis, index)

    whole_dice = dice_score(selected_mask, gt_mask)
    whole_iou = iou_score(selected_mask, gt_mask)
    slice_dice = dice_score(selected_slice, gt_slice)
    slice_iou = iou_score(selected_slice, gt_slice)

    return {
        "image": grayscale_to_rgb(normalize_to_uint8(image_slice)),
        "mask": colorize_mask(selected_slice),
        "overlay": overlay_mask(image_slice, selected_slice, alpha=alpha),
        "index": index,
        "max_index": slice_count(image.shape, axis) - 1,
        "shape": image.shape,
        "missing_mask": missing_mask,
        "metrics": {
            "volume_dice": whole_dice,
            "volume_iou": whole_iou,
            "slice_dice": slice_dice,
            "slice_iou": slice_iou,
        },
    }
