# Evaluation (only what was actually measured)

Hardware: Apple M2, 8 GB, macOS, PyTorch MPS. Model: generic COCO-pretrained YOLO11n (not fine-tuned).

## Not evaluated (and why)
| Metric | Status | Reason |
|---|---|---|
| Person / phone mAP50, precision, recall on real exam data | **NOT EVALUATED** | no labelled exam-room data |
| Tracking IDF1 / HOTA / MOTA / ID switches | **NOT EVALUATED** | no ground-truth identities |
| Event precision/recall, false alerts per hour on real footage | **NOT EVALUATED** | no labelled real events; only 25 s of normal footage exist |
| Fine-tuning | **NOT PERFORMED** | no authorised labelled data; no specialist model is claimed |

## Measured (a) normal footage, zero phones present (real model)
| Clip | Frames | Tracks | Alerts |
|---|---|---|---|
| Classroom (CC BY-SA) | 620 | 19-21 | **0** |
| Street (CC0) | 375 | 62-67 | **0** |
25 s total: it shows no false alert was observed there, not a false-alert rate.

## Measured (b) tracking fragmentation PROXY (not IDF1)
Seated classroom: >= 11 people visible simultaneously; IDs created 32 (ByteTrack defaults) -> 19 (defaults used here:
high 0.5, buffer 60, low-score boxes fed to the tracker). Street (people legitimately enter/leave): 88 -> 62 IDs.
So IDs still fragment (about 1.7x the simultaneous count on a static scene). Expect ID switches after occlusion.
Raw sweep: `artifacts/metrics/tracking_proxy.json`.

## Measured (c) phone events on the SYNTHETIC fixture only
Ground truth = exact pasted box, 199 frames over 8 s. IoU >= 0.3. Default settings (640 px, conf 0.25, support 5):
| | frame precision | frame recall | alerts (correct/false) | first-alert delay | 8 s window covered | duplicates |
|---|---|---|---|---|---|---|
| full-frame only | 1.00 | 0.10 | 1 / 0 | 4.1 s | 1 % | 0 |
| + person-crop assist (default) | 0.91 | 0.30 | 3 / 0 | 3.0 s | 39 % | 2 |
At conf 0.10 with assist, recall reaches 0.49-0.53 and coverage 55-76 % at precision ~0.77-0.80.
**Phone recall is weak**, mainly because the phone is small. The numbers come from one synthetic scene and the
thresholds were explored on that same scene, so they are optimistic and not a real-world estimate.
At 1280 px the full-frame detector got *worse* (precision 0.33-0.56). Rows: `artifacts/metrics/phone_fixture_eval.json`.

## Measured (d) performance (`artifacts/metrics/benchmark.json`, 640 px)
| Clip | device | crop assist | e2e FPS | model ms/frame | peak RSS |
|---|---|---|---|---|---|
| street 1080p | MPS | off | 63.1 | 14.2 | 684 MB |
| street 1080p | MPS | on | 19.0 | 50.5 | 787 MB |
| street 1080p | CPU | off | 36.0 | 26.1 | 700 MB |
| classroom 1080p | MPS | off | 61.4 | 14.7 | 477 MB |
| classroom 1080p | MPS | on | 15.3 | 62.9 | 341 MB |
| classroom 1080p | CPU | off | 36.9 | 25.4 | 555 MB |
| classroom 360p | MPS | on | 16.2 | 60.5 | 465 MB |
Writing the annotated video costs ~1-2 FPS. Video is streamed frame by frame; memory did not grow with length (clips
are short, so this is not a long-video soak test).

## Error analysis
* Phones are missed when small; the same phone was labelled `remote` by the model in several configurations.
* Larger input is not monotonically better for phones (1280 px added false boxes).
* Two incidents were produced for one staged phone because the target's track ID changed at the frame edge.
* Heavy motion blur and crowding fragment IDs (street clip).
