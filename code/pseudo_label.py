from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from scipy.ndimage import binary_closing, binary_opening, generate_binary_structure, label

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

from segmentation_visualizer import load_case, resolve_dataset_dir


def _largest_components(mask: np.ndarray, min_size: int = 64, keep_components: int = 1) -> np.ndarray:
    """Keep the largest connected components after removing tiny regions."""
    if not SCIPY_AVAILABLE:
        return mask if int(mask.sum()) >= max(1, int(min_size)) else np.zeros_like(mask, dtype=bool)

    structure = generate_binary_structure(mask.ndim, 1)
    labeled, num_features = label(mask, structure=structure)
    if num_features == 0:
        return np.zeros_like(mask, dtype=bool)

    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    valid_labels = np.where(counts >= max(1, int(min_size)))[0]
    if valid_labels.size == 0:
        return np.zeros_like(mask, dtype=bool)

    ranked = valid_labels[np.argsort(counts[valid_labels])[::-1]]
    selected = ranked[: max(1, int(keep_components))]
    return np.isin(labeled, selected)


def generate_threshold_pseudo_mask(
    image: np.ndarray,
    percentile: float = 85,
    min_size: int = 64,
    keep_components: int = 1,
) -> np.ndarray:
    """Generate a simple pseudo mask from image intensity and morphology."""
    volume = np.asarray(image, dtype=np.float32)
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return np.zeros(volume.shape, dtype=np.uint8)

    positive = finite[finite > 0]
    threshold_source = positive if positive.size > 0 else finite
    percentile = float(np.clip(percentile, 1, 99))
    threshold = np.percentile(threshold_source, percentile)

    mask = volume > threshold
    if SCIPY_AVAILABLE:
        structure = generate_binary_structure(3, 1)
        mask = binary_opening(mask, structure=structure, iterations=1)
        mask = binary_closing(mask, structure=structure, iterations=1)
    mask = _largest_components(mask, min_size=min_size, keep_components=keep_components)
    return mask.astype(np.uint8)


def save_pseudo_mask_for_case(
    case_id: str,
    dataset_dir: str | Path | None = None,
    percentile: float = 85,
    min_size: int = 64,
    keep_components: int = 1,
) -> tuple[Path, np.ndarray]:
    """Generate and save pseudo_mask.npy for a case."""
    image, _ = load_case(case_id, dataset_dir)
    pseudo_mask = generate_threshold_pseudo_mask(
        image,
        percentile=percentile,
        min_size=min_size,
        keep_components=keep_components,
    )

    case_path = resolve_dataset_dir(dataset_dir) / str(case_id)
    output_path = case_path / "pseudo_mask.npy"
    np.save(output_path, pseudo_mask)
    return output_path, pseudo_mask
