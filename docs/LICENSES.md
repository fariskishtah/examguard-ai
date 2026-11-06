# Licences

* **ExamGuard AI source code: AGPL-3.0-or-later** (see `/LICENSE`). Ultralytics YOLO and its tracker code are
  AGPL-3.0, and this application imports and bundles them; the project is therefore released under the same
  licence. Anyone who modifies it and offers it over a network must publish their source.
* **Model weights** `yolo11n.pt` are the official Ultralytics COCO-pretrained weights (AGPL-3.0). They are **not**
  committed (`models/*.pt` is git-ignored); `scripts/fetch_model.py` downloads them from the official release.
  Development weights SHA-256: `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
  Ultralytics also offers a commercial Enterprise licence for closed-source use.
* **Dependencies** (checked from installed package metadata): torch (BSD-style/Apache-2.0 family), OpenCV headless
  (Apache-2.0), NumPy (BSD-3), SQLAlchemy and Alembic (MIT), Pillow (MIT-CMU), psutil (BSD-3), lap (BSD-2),
  PyInstaller (GPLv2+ with an exception for distributing built apps). All are compatible with AGPL-3.0 distribution.
* **Sample media**: see `DATA_SOURCES.md`. CC BY-SA material is used locally only and is not redistributed.

* **Evaluation-only data** (not distributed): MOT15 is CC BY-NC-SA 3.0 (research use); COCO annotations are CC BY 4.0 with mixed
  image licences; neither is redistributed or used for training. Extra evaluation tools: py-motmetrics (MIT), remotezip (MIT),
  pandas (BSD-3), pyarrow (Apache-2.0).
* **Optional model** `yolo11s.pt` (official Ultralytics weights, AGPL-3.0, downloaded on request by `scripts/fetch_model.py yolo11s`).
