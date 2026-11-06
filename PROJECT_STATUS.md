# PROJECT_STATUS — ExamGuard AI

Last updated: 2026-11-06. Overall: **PARTIAL** (see "Not done / blockers"). The desktop system works end to end and was
verified; real-world accuracy could not be validated.

## Completed
| Milestone | State | Evidence |
|---|---|---|
| Setup, compat check | done | Python 3.12 (uv-managed, has Tk 9.0.4), torch 2.15 MPS, ultralytics 8.4.203, OpenCV 5.0.0 |
| Detection (YOLO11n person+phone) | done | real inference on MPS/CPU; `tests/test_integration_real.py` |
| Tracking (ByteTrack, anonymous IDs) | done | `tests/test_tracking.py`; fragmentation proxy in `artifacts/metrics/tracking_proxy.json` |
| Event engine (phone near track, region exit) | done | 18 tests; lifecycle, cooldown, ambiguity skip |
| Evidence capture | done | raw + annotated, atomic, never overwrite, 0600 |
| SQLite + Alembic | done | 12 tests: FK, CHECK, migration==models, restart persistence |
| Tkinter app (dashboard, monitor, alerts, history, viewer, review, settings, region editor) | done | screenshots in `artifacts/screenshots/` taken from the packaged app |
| Exports (CSV/JSON/screenshots/video/HTML report) | done | verified on the packaged app's DB; H.264 1920x1080 375 frames |
| Tests | 82 passed / 0 failed / 0 skipped | `pytest tests` (~2 min) |
| macOS packaging | done (un-notarised) | `packaging/build_macos.sh`; bundle self-test with repo models hidden |
| Docs | done | README, docs/*.md |

## Models, data, checkpoints
* Weights: official `yolo11n.pt`, sha256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` (not committed;
  `scripts/fetch_model.py`). **No training / fine-tuning performed; no checkpoints exist.**
* Data: `data/samples/MANIFEST.json` (3 openly licensed files, provenance + sha256). Synthetic fixture:
  `data/fixtures/` (git-ignored, rebuildable). See `docs/DATA_SOURCES.md`.

## Decisions & findings worth knowing
* Person threshold is ByteTrack's *high* threshold (0.5); persons are fed to the tracker down to 0.10 so low-score boxes
  help (classroom IDs 21→19, street 67→62).
* Person-crop phone assist added after measuring weak full-frame recall (0.095→0.30 at conf 0.25); costs ~3x inference.
* Packaging bug found only by testing the bundle: torchvision ops missing → fixed with `--collect-all torchvision`.
* Tk 9.0.4/macOS crashes if a second `Tk()` root is created after one was destroyed in one process → GUI tests run per
  process (`tests/gui_scenarios.py`). An earlier wrong diagnosis (GC / OpenCV) was reverted.
* Container metadata can lie (a WebM claimed 1000 fps / 15125 frames; truth 25 fps / 375) → `probe_video` scans timestamps.

## Not done / blockers (exact)
1. **Real-world accuracy: NOT EVALUATED.** Needs consented, labelled exam-style video (phone events, ground-truth
   identities). Then: `scripts/run_headless.py`, extend `scripts/eval_*`.
2. **No fine-tuning** (no authorised labelled data).
3. Public Hugging Face demo: not built (optional; a desktop requirement took priority).
4. Windows/Linux packaging untested; macOS app not notarised.
5. Long-video soak test not run (longest clip 15 s).
6. Region-exit rule is covered by unit tests and a pipeline test with a scripted detector only — no real-video validation.

## Resume instructions
```bash
git clone https://github.com/fariskishtah/examguard-ai && cd examguard-ai && uv venv --python 3.12 --managed-python .venv && source .venv/bin/activate
uv pip install -r requirements.txt && uv pip install -e .
python scripts/fetch_model.py && python scripts/fetch_samples.py
python scripts/make_synthetic_fixture.py data/samples/Video_Codec_Test_pedestrian_area_1080p25.y4m.webm \
    data/fixtures/synthetic_phone_pedestrian.mp4 --start 3 --end 11
pytest tests            # expect 82 passed
python -m examguard
```
