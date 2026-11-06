# PROJECT_STATUS — ExamGuard AI

Last updated: 2026-11-07.

**Assignment implementation complete; real-world validation remains limited.**

The three things below are *not* equivalent and are reported separately.

## 1. Assignment feature completeness — done
Detect people + phones (YOLO11), track (BoT-SORT, anonymous IDs), observable events with temporal confirmation, RED/GREEN
(+AMBER) boxes, evidence screenshots (raw + annotated), SQLite + Alembic persistence, non-blocking notifications, Tkinter desktop
app (upload, processing with pause/resume/stop, preview, alert centre, incident history, evidence viewer, human review,
settings, region editor), exports (CSV, JSON, screenshots, annotated H.264 video, HTML report), privacy defaults, macOS bundle.

## 2. Engineering validation and packaged-app readiness — done, with caveats
| Item | Result |
|---|---|
| Automated tests | **119 passed, 0 failed, 0 skipped** (6 min 25 s): db 12, events 46, export 5, real-model integration 6, processor 16, tracking 9, GUI/controller 11, video/evidence 14 |
| Packaged `ExamGuard AI.app` | verified as a bundle with the repo `models/` hidden: launch, bundled weights, real inference (torchvision fix holds), BoT-SORT, processing, evidence, SQLite integrity ok / 0 FK violations, review, CSV/JSON/screenshots/report exports, H.264 export (375/375 frames), restart persistence, clean exit |
| 10-minute soak (15 000 frames, 1080p) | 46.05 min wall time (4.6× real time), 5.4 fps, RSS 524→peak 858→495 MB (no growth), 40/40 staged events, pause/resume + cancel correct, 0 errors |
| Tracker state | bounded over 60 000 frames (removed list capped at 1000) |
| Performance cost of the new default | ≈2.7× slower than the earlier shipped config (5–8 fps on 1080p); native resolution alone is free; crop assist is the cost |
| Caveats | app is ad-hoc signed, **not notarised**; macOS only; the README window screenshots are from the *previous* packaged build (final build could not be captured — the Mac's screen was locked); longest soak = 10 min of **looped** footage |

## 3. Real-world detection accuracy — NOT established
* Detector: generic COCO-pretrained YOLO11 — **not fine-tuned for exams**; no training, no checkpoints.
* **No consented, labelled exam footage exists** → phone-event accuracy on real exams, false-alert rate on real exams, and tracking
  HOTA are **NOT EVALUATED**.
* Real labelled evidence that does exist (general scenes, not exams): COCO val phone AP50 (native resolution ≈ 2× the old
  640-px default on the small-phone stress test); MOT15 IDF1 0.77–0.80 for all sensible trackers (**no demonstrated tracking
  improvement**); region-exit on real pedestrians vs rule-derived ground truth (test P 0.92 / R 0.92).
* Synthetic composites only: untouched F3 → **1/3** staged events (earlier config 0/3), 0 clean-twin false alerts. This does
  **not** show reliable real-world phone detection.
* Two real false alerts found (face mask — fixed with a 0.5 s minimum duration; 6×10 px head blob — **not fixed**).

## Final configuration (defaults)
native resolution (auto, ≤1920) · YOLO11n · 640-px person-crop assist **on** · BoT-SORT (no GMC/appearance) · phone conf 0.25 ·
≥5 frames **and** ≥0.5 s · cross-track merge (6 s, 2.5 phone sizes, old track must be gone) · resolve 2 s · cooldown 10 s.
Options in Settings: "fast" (crop assist off), yolo11s ("accurate"), ByteTrack, resolution, stride.
The default was kept because the rule registered before F2/F3 (recall ≥ shipped and clean-twin false alerts no worse) was met —
on thin evidence (9 staged events: 5 vs 4).

## Findings worth remembering
* Root cause of missed small phones = input resolution (1080p shrunk 3× to 640). Crop assist alone is *worse* than full-frame on
  ordinary images; it only helps as a union and for tiny phones in small frames.
* The merge rule's first version **lost a real event** when two people stood side by side; fixed by requiring the old track to
  have disappeared (regression tests added; verified they fail without the fix).
* A frame-count-only confirmation raised an alert from a 0.13 s blip on 60 fps video → minimum duration in seconds.
* The earlier "fewer IDs created" tracking gain was a proxy; it did not improve IDF1.
* Tk 9.0.4/macOS segfaults if a second `Tk()` root is created after one was destroyed → GUI tests run per process.
* BoT-SORT's GMC had been silently disabled by a detection shim without `.xyxy` (fixed + regression test).
* COCO's host has a TLS certificate mismatch → the Hugging Face mirror was used (certificate verification never bypassed).

## Not done / blockers (exact)
1. **Real-world validation**: needs consented, labelled exam-style footage with real phone use and ground-truth identities.
2. No fine-tuning (no authorised labelled data); no specialist model is claimed.
3. Known unfixed false-alert pattern (tiny head/hand blobs); no size-vs-person filter (would be tuned on one example).
4. Re-capture README window screenshots from the final build (needs an unlocked screen); public Hugging Face demo not built.
5. Windows/Linux packaging untested; macOS app not notarised; no soak > 10 min; no real-footage soak.

## Resume instructions
```bash
git clone https://github.com/fariskishtah/examguard-ai && cd examguard-ai
uv venv --python 3.12 --managed-python .venv && source .venv/bin/activate
uv pip install -r requirements.txt && uv pip install -e .
python scripts/fetch_model.py yolo11n yolo11s && python scripts/fetch_samples.py
python scripts/make_synthetic_fixture.py data/samples/Video_Codec_Test_pedestrian_area_1080p25.y4m.webm \
    data/fixtures/synthetic_phone_pedestrian.mp4 --start 3 --end 11
pytest tests            # expect 119 passed
python -m examguard
```
Evaluation scripts (`scripts/eval_*.py`, `make_heldout_fixtures.py`, `soak_test.py`, `benchmark.py`) additionally need
`uv pip install -e .[eval]` and the local COCO/MOT15 data described in `docs/DATA_SOURCES.md`.
