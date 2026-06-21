from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from path_config import PART_B_DIR

from .base import BaseSegmenter


class MedSAMAdapter(BaseSegmenter):
    """MedSAM 2D slice adapter.

    This adapter follows the official MedSAM inference flow:
    1. resize a 2D image to 1024 x 1024,
    2. encode the image with SAM ViT-B,
    3. use a bounding-box prompt,
    4. decode a binary mask and resize it back to the original slice size.
    """

    name = "medsam_2d"

    def __init__(self, checkpoint_path: str | Path | None = None, device: str | None = None):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else (
            PART_B_DIR / "model_weights" / "medsam" / "medsam_vit_b.pth"
        )
        self.device_name = device
        self._model = None
        self._device = None
        self._last_error = ""

    def dependency_status(self) -> tuple[bool, str]:
        if not self.checkpoint_path.exists():
            return False, f"未找到 MedSAM 权重：{self.checkpoint_path}"
        try:
            import torch  # noqa: F401
            from segment_anything import sam_model_registry  # noqa: F401
        except Exception as exc:
            return False, f"MedSAM 依赖未安装完整：{exc}"
        return True, "MedSAM 权重与依赖已就绪。"

    def available(self) -> bool:
        ok, message = self.dependency_status()
        self._last_error = "" if ok else message
        return ok

    @property
    def last_error(self) -> str:
        if self._last_error:
            return self._last_error
        ok, message = self.dependency_status()
        return "" if ok else message

    def _load_model(self):
        ok, message = self.dependency_status()
        if not ok:
            raise RuntimeError(message)

        if self._model is not None:
            return self._model, self._device

        import torch
        from segment_anything import sam_model_registry

        device = self.device_name or ("cuda:0" if torch.cuda.is_available() else "cpu")
        model = sam_model_registry["vit_b"](checkpoint=None)
        state_dict = torch.load(str(self.checkpoint_path), map_location=torch.device(device))
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        self._model = model
        self._device = torch.device(device)
        return self._model, self._device

    @staticmethod
    def _normalize_slice(image_slice: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def box_from_mask(mask_slice: np.ndarray, margin: int = 4) -> list[int] | None:
        mask = np.asarray(mask_slice) > 0
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        h, w = mask.shape
        x_min = max(0, int(x_min) - margin)
        y_min = max(0, int(y_min) - margin)
        x_max = min(w - 1, int(x_max) + margin)
        y_max = min(h - 1, int(y_max) + margin)
        if x_max <= x_min:
            x_max = min(w - 1, x_min + 1)
        if y_max <= y_min:
            y_max = min(h - 1, y_min + 1)
        return [x_min, y_min, x_max, y_max]

    @staticmethod
    def default_box(image_slice: np.ndarray, margin_ratio: float = 0.08) -> list[int]:
        h, w = image_slice.shape
        mx = max(1, int(w * margin_ratio))
        my = max(1, int(h * margin_ratio))
        return [mx, my, max(mx + 1, w - mx - 1), max(my + 1, h - my - 1)]

    def predict_slice(self, image_slice: np.ndarray, box: list[int] | None = None) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        model, device = self._load_model()

        image_u8 = self._normalize_slice(image_slice)
        if image_u8.ndim != 2:
            raise ValueError("MedSAM 2D demo 需要单张 2D 切片。")

        h, w = image_u8.shape
        if box is None:
            box = self.default_box(image_u8)

        image_rgb = np.repeat(image_u8[:, :, None], 3, axis=-1)
        image_1024 = np.asarray(
            Image.fromarray(image_rgb).resize((1024, 1024), resample=Image.Resampling.BICUBIC),
            dtype=np.float32,
        )
        denom = max(float(image_1024.max() - image_1024.min()), 1e-8)
        image_1024 = (image_1024 - image_1024.min()) / denom
        image_tensor = torch.tensor(image_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)

        box_np = np.array([box], dtype=np.float32)
        box_1024 = box_np / np.array([w, h, w, h], dtype=np.float32) * 1024
        box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]

        with torch.no_grad():
            image_embedding = model.image_encoder(image_tensor)
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
            low_res_logits, _ = model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            low_res_pred = torch.sigmoid(low_res_logits)
            low_res_pred = F.interpolate(
                low_res_pred,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            pred = low_res_pred.squeeze().detach().cpu().numpy()
        return (pred > 0.5).astype(np.uint8)

    def predict(self, image_volume: np.ndarray, prompt: Any | None = None) -> np.ndarray:
        raise NotImplementedError("当前接入的是 MedSAM 2D slice demo，请调用 predict_slice。")
