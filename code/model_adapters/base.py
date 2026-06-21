from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class BaseSegmenter:
    name = "base"

    def available(self) -> bool:
        """Return whether model code and weights are ready."""
        return False

    def predict(self, image_volume: np.ndarray, prompt: Any | None = None) -> np.ndarray:
        """Predict a segmentation mask for a 3D volume."""
        raise NotImplementedError


@dataclass
class UnconfiguredSegmenter(BaseSegmenter):
    name: str = "unconfigured"
    message: str = "模型权重未配置，请先配置模型权重后再运行预测。"

    def available(self) -> bool:
        return False

    def predict(self, image_volume: np.ndarray, prompt: Any | None = None) -> np.ndarray:
        raise RuntimeError(self.message)
