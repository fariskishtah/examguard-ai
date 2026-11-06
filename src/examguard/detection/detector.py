"""YOLO person / cell-phone detector (Ultralytics) with MPS -> CPU fallback."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PERSON = "person"
PHONE = "cell phone"
COCO_IDS = {PERSON: 0, PHONE: 67}
PERSON_TRACKER_FLOOR = 0.10  # persons are detected down to this score so the tracker can use low-score boxes


@dataclass(frozen=True)
class Detection:
    label: str
    class_id: int
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in ORIGINAL frame pixels
    frame_index: int
    timestamp_s: float
    detector_version: str

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    def validate(self, width: int, height: int) -> bool:
        x1, y1, x2, y2 = self.bbox
        return 0 <= x1 < x2 <= width + 1 and 0 <= y1 < y2 <= height + 1


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def merge_detections(primary: list[Detection], extra: list[Detection], iou_thresh: float = 0.5) -> list[Detection]:
    """Union of two detection lists; an `extra` box overlapping a kept box (IoU > thresh) is a duplicate.
    The higher-confidence one wins."""
    from examguard.events.geometry import iou
    kept = list(primary)
    for e in sorted(extra, key=lambda d: -d.confidence):
        dup = next((i for i, k in enumerate(kept) if k.label == e.label and iou(k.bbox, e.bbox) > iou_thresh), None)
        if dup is None:
            kept.append(e)
        elif e.confidence > kept[dup].confidence:
            kept[dup] = e
    return kept


def resolve_device(requested: str) -> str:
    import torch

    if requested == "cpu":
        return "cpu"
    if requested in ("auto", "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class YoloDetector:
    """Wraps a pretrained COCO YOLO model restricted to person + cell phone.

    NOTE: weights are generic COCO-pretrained, not fine-tuned for exams.
    """

    def __init__(self, weights: str | Path, person_conf: float = PERSON_TRACKER_FLOOR, phone_conf: float = 0.25,
                 nms_iou: float = 0.5, imgsz: int = 640, device: str = "auto"):
        from ultralytics import YOLO
        import ultralytics

        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(
                f"Model weights not found: {self.weights}. Run `python scripts/fetch_model.py`.")
        self.model = YOLO(str(self.weights))
        self.person_conf, self.phone_conf = person_conf, phone_conf
        self.nms_iou, self.imgsz = nms_iou, imgsz
        self.device = resolve_device(device)
        self.version = (f"{self.weights.name}@sha256:{file_sha256(self.weights)[:12]}"
                        f"|ultralytics-{ultralytics.__version__}")
        self._class_ids = [COCO_IDS[PERSON], COCO_IDS[PHONE]]
        names = self.model.names
        if names.get(0) != PERSON or names.get(67) != PHONE:
            raise RuntimeError("Weights do not use COCO class ids (person=0, cell phone=67)")

    def detect_phones_in_crops(self, frame_bgr: np.ndarray, boxes: list[tuple[float, float, float, float]],
                               frame_index: int, timestamp_s: float, imgsz: int = 416, margin: float = 0.15,
                               max_crops: int = 12) -> list[Detection]:
        """Second-stage "person-crop assist": re-run the phone detector on each tracked person's crop.

        A phone that is only ~20 px in a full 1080p frame becomes relatively larger inside a person crop,
        which raises recall for small phones at the cost of extra inference (one batch per frame).
        Boxes are mapped back to ORIGINAL frame coordinates. Only phones are returned.
        """
        H, W = frame_bgr.shape[:2]
        crops, offs = [], []
        for x1, y1, x2, y2 in boxes[:max_crops]:
            w, h = x2 - x1, y2 - y1
            a, b = max(0, int(x1 - margin * w)), max(0, int(y1 - margin * h))
            c, d = min(W, int(x2 + margin * w)), min(H, int(y2 + margin * h))
            if c - a >= 16 and d - b >= 16:
                crops.append(frame_bgr[b:d, a:c])
                offs.append((a, b))
        if not crops:
            return []
        kw = dict(imgsz=imgsz, conf=self.phone_conf, iou=self.nms_iou, classes=[COCO_IDS[PHONE]], verbose=False)
        try:
            results = self.model.predict(crops, device=self.device, **kw)
        except Exception:
            if self.device == "cpu":
                raise
            self.device = "cpu"
            results = self.model.predict(crops, device="cpu", **kw)
        out: list[Detection] = []
        for r, (ox, oy) in zip(results, offs):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for bx, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                if cf >= self.phone_conf:
                    out.append(Detection(PHONE, COCO_IDS[PHONE], float(cf),
                                         (float(bx[0] + ox), float(bx[1] + oy), float(bx[2] + ox), float(bx[3] + oy)),
                                         frame_index, timestamp_s, self.version))
        return out

    def detect(self, frame_bgr: np.ndarray, frame_index: int, timestamp_s: float) -> list[Detection]:
        floor = min(self.person_conf, self.phone_conf)
        try:
            res = self.model.predict(frame_bgr, device=self.device, imgsz=self.imgsz, conf=floor,
                                     iou=self.nms_iou, classes=self._class_ids, verbose=False)[0]
        except Exception:
            if self.device == "cpu":
                raise
            self.device = "cpu"  # reliable fallback if MPS misbehaves
            res = self.model.predict(frame_bgr, device="cpu", imgsz=self.imgsz, conf=floor,
                                     iou=self.nms_iou, classes=self._class_ids, verbose=False)[0]
        out: list[Detection] = []
        b = res.boxes
        if b is None or len(b) == 0:
            return out
        xyxy, conf, cls = b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)
        for box, c, k in zip(xyxy, conf, cls):
            label = PERSON if k == 0 else PHONE
            if c < (self.person_conf if label == PERSON else self.phone_conf):
                continue
            out.append(Detection(label, int(k), float(c), tuple(float(v) for v in box),
                                 frame_index, timestamp_s, self.version))
        return out
