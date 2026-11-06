# ExamGuard AI

**Intelligent Exam Monitoring & Suspicious Behavior Analysis Platform** — a local-first Tkinter desktop app that
detects people and phone-like objects in exam videos with YOLO, tracks people with ByteTrack, raises *observable-event*
alerts with screenshot evidence, stores everything in SQLite, and routes every alert to a **human reviewer**.

> **Automated alerts identify observable events requiring human review. They are not proof of academic misconduct.**
> ExamGuard never decides that anyone cheated. It does no face recognition and stores no identities — people are
> temporary, anonymous track numbers.

Built by **Faris Mohamed Kishtah** as Project 3 (Cheating Tracking and Alert System), Sprint 2 of the
INSTANT AI & Data Science training programme.

| | |
|---|---|
| ![Alert](artifacts/screenshots/03_flagged_event_and_alert_notification.jpg) | ![Tracking](artifacts/screenshots/02_multi_person_tracking.jpg) |

*Left: a flagged observable event (RED box, review notification, Alert Center). Right: multi-person tracking (GREEN).
**These frames use a clearly SYNTHETIC test composite** — a real CC0 phone photo pasted into CC0 street footage — see
[Honest status](#honest-status). They are not real footage of phone use.*

## What it does

1. **Upload** a video (mp4/avi/mov/mkv/m4v/webm; 4 GiB cap; corrupt/empty files rejected before anything is stored).
2. **Detect** `person` and `cell phone` with a pretrained **YOLO11n** at **native video resolution** (auto, capped at 1920) plus an optional
   person-crop phone pass (MPS on Apple Silicon, CPU fallback; optional `yolo11s` for higher accuracy).
3. **Track** people with **BoT-SORT** (no appearance model, no face recognition) → anonymous IDs (`Track #31`).
4. **Analyse events** with explicit rules and temporal confirmation (≥5 frames **and** ≥0.5 s, one incident per continuous event,
   cooldown, started → active → resolved lifecycle, and a guarded merge of incidents split by a tracker ID switch).
5. **Show status** on the video: GREEN normal · AMBER evidence accumulating · RED active event.
6. **Capture evidence**: the real frame saved twice (original + annotated), unique names, atomic writes.
7. **Persist** sessions / tracks / incidents / review history in SQLite (Alembic migrations, FK + CHECK constraints).
8. **Notify** with a non-blocking in-app toast (+ optional native macOS banner) and an Alert Center.
9. **Review**: Confirm observation · Dismiss (false alert) · Needs more evidence, with notes and history.
10. **Export** CSV, JSON, screenshots, annotated video (H.264), HTML session report — all labelled as machine-generated.

### Events (observable only)
| Event | Meaning |
|---|---|
| `PHONE_VISIBLE_NEAR_TRACK` | a phone-like object was detected near that tracked person in ≥5 of 15 processed frames spanning ≥0.5 s |
| `MONITORED_REGION_EXIT` | a track previously inside a user-drawn region has been outside it ≥1.5 s |

Not implemented (no reliable method/data here): gaze, head-turn, whispering, collaboration. See
[docs/EVENT_RULES.md](docs/EVENT_RULES.md).

## Architecture
![Architecture](artifacts/screenshots/08_system_architecture.png)

```
src/examguard/
  detection/  YOLO wrapper, Detection schema, person-crop phone assist
  tracking/   ByteTrack wrapper -> anonymous TrackedPerson
  events/     geometry, association, temporal engine, lifecycle
  video/      streamed reader (true timestamps), annotate, processor
  evidence/   screenshot capture, safe deletion
  database/   SQLAlchemy models, Alembic migrations, repository
  notifications/  alert model, native notifier
  reports/    CSV / JSON / HTML / video export
  ui/         controller (no Tk), worker thread + queue, widgets, views, app
```
The worker never touches Tk: it posts to a thread-safe queue plus a "latest frame" slot that the UI thread polls
(stale preview frames are dropped, never queued). Overlays are stored in *normalised* coordinates, so resizing
the window cannot make boxes or the monitored region drift.

**Database choice:** SQLite, not Firebase. It needs no credentials, cost or network, keeps sensitive exam evidence
on the machine, and is transactional. The assignment suggests Firestore as an option; a cloud store is not required.

## Screens
| Dashboard | Incident history + evidence viewer |
|---|---|
| ![Dashboard](artifacts/screenshots/04_dashboard.jpg) | ![Incidents](artifacts/screenshots/05_incident_history_and_evidence_viewer.jpg) |
| **Human review** | **Settings** |
| ![Review](artifacts/screenshots/06_human_review_workflow.jpg) | ![Settings](artifacts/screenshots/07_settings.jpg) |

## Honest status — three different things

| | Status |
|---|---|
| **1. Assignment features** (detect, track, flag, red/green boxes, screenshots, database, notifications, Tkinter app, upload, history, review, exports) | **Implemented and working** |
| **2. Engineering validation / packaged app** | **119 automated tests pass (0 failed, 0 skipped)**; packaged macOS app verified end to end; 10-minute soak test clean |
| **3. Real-world detection accuracy** | **Not validated.** No consented, labelled exam footage exists for this project |

**Assignment implementation complete; real-world validation remains limited.**

* The detector is **generic COCO-pretrained YOLO11 — not fine-tuned for exams.** No custom training was done and none is claimed.
* All phone-*event* numbers come from **synthetic composites** (a real CC0 phone photo pasted into real MOT15 footage). On the
  untouched fixture set F3 the final configuration found **1 of 3** staged events (the earlier shipped configuration: 0 of 3) with
  0 false alerts on the clean twins. **That is not evidence of reliable real-world phone detection.** Phone-event accuracy on real
  exam footage is **NOT EVALUATED**.
* Real *labelled* evidence does exist for the detector and tracker in general scenes (COCO val, MOT15) — see below.
* **Two real false alerts were found** on footage without phones: a face mask read as a phone (fixed with a 0.5 s minimum
  duration) and a 6×10 px head blob read as a phone (**not fixed**). Expect occasional false alerts; every one goes to a human.

### Before → after (what this improvement pass measured)
| | Before | After | Evidence |
|---|---|---|---|
| Small-phone detection, real labelled images (COCO, 'far' stress, test AP50) | 0.21 (640 px) / 0.35 (shipped crop416) | **0.43** native 1920 / 0.47 with 640 crops | `docs/EVALUATION.md` §2 |
| Staged phone events, untouched F3 (synthetic) | 0/3 | **1/3** (0 clean-twin false alerts) | §3 — thin evidence |
| One continuous event split by an ID switch (F2) | 4 incidents | **1 incident** (no false merge of neighbours, regression-tested) | §4 |
| Tracking IDF1, real MOT15 ground truth (3 test seqs) | 0.779–0.800 | 0.802 | §5 — **within noise, no demonstrated gain** |
| Region-exit on real footage (rule-derived GT, test) | P 0.91 / R 0.92 | P 0.92 / R 0.92 | §6 |
| Processing time (1080p) | 1.0× | **≈2.7× slower** by default (5–8 fps); "fast" mode ≈ same speed as before | §8 |
| Long-video stability | longest test 15 s | **10 min video, 46 min processing, no memory growth, 40/40 events, 0 errors** | §9 |

Tracking note: the earlier "fewer IDs created" improvement did **not** raise IDF1 on held-out sequences; I report it as
no demonstrated improvement. Region-exit "ground truth" is the same rule applied to ground-truth boxes, not human labels.

Everything measured is in [docs/EVALUATION.md](docs/EVALUATION.md) and `artifacts/metrics/*.json`.

### Performance (Apple M2, 8 GB, 1080p, controlled benchmark)
| Config | street | ADL-Rundle-8 | classroom | peak RSS |
|---|---|---|---|---|
| earlier shipped (640 + crop416) | 19.2 fps | 15.4 | 13.4 | ≤700 MB |
| **default** (native + crop640 + BoT-SORT) | **8.0** | **5.0** | **5.0** | ≤500 MB |
| fast (native, no crop assist) | 21.4 | 16.5 | 16.0 | ≤525 MB |
| accurate (yolo11s + crop640) | 2.8 | 2.2 | 2.2 | ≤580 MB |
The default is accuracy-first for an *offline review* tool: a 1-hour 1080p recording takes several hours (use "fast" or
`frame_stride` to trade accuracy for time).

![Evaluation](artifacts/screenshots/09_evaluation_overview.png)

## Install & run (macOS, Apple Silicon)
Use a Python with Tk. The Homebrew Python has no `_tkinter`; **uv's managed Python includes it**.
```bash
git clone https://github.com/fariskishtah/examguard-ai && cd examguard-ai
uv venv --python 3.12 --managed-python .venv && source .venv/bin/activate
uv pip install -r requirements.txt && uv pip install -e .
python scripts/fetch_model.py          # downloads official yolo11n.pt (AGPL-3.0); add `yolo11s` for the accurate option
python -m examguard                    # launch the desktop app
```
Runtime data lives in `~/Library/Application Support/ExamGuardAI` (override with `--home DIR` or `EXAMGUARD_HOME`).

Try it on openly licensed footage:
```bash
python scripts/fetch_samples.py        # CC0 / CC BY-SA clips from Wikimedia Commons (not redistributed here)
python scripts/make_synthetic_fixture.py data/samples/Video_Codec_Test_pedestrian_area_1080p25.y4m.webm \
       data/fixtures/synthetic_phone_pedestrian.mp4 --start 3 --end 11
python scripts/run_headless.py data/fixtures/synthetic_phone_pedestrian.mp4 --home /tmp/eg-demo   # no GUI
```

### Desktop packaging
```bash
./packaging/build_macos.sh             # -> dist/ExamGuard AI.app (~1 GB with torch; can take 15+ min on a busy Mac)
"dist/ExamGuard AI.app/Contents/MacOS/ExamGuard AI" --selftest some_video.mp4   # real GUI end to end + review + all exports, prints JSON
"dist/ExamGuard AI.app/Contents/MacOS/ExamGuard AI" --verify-persistence        # second launch: data survived a restart
```
Verified **as a bundle** (repo `models/` hidden): launches, bundled YOLO weights, real inference (torchvision issue stays fixed),
BoT-SORT, upload/processing, evidence screenshots, SQLite (integrity ok, 0 FK violations), human review, CSV/JSON/screenshot/
report exports, **H.264 export** (ffprobe: h264 1920×1080, 375/375 frames), restart persistence, clean exit. It is **ad-hoc signed,
not notarised** (right-click → Open on another Mac). Windows/Linux packaging was **not** tested. The window screenshots above come
from the previous packaged build (same layout; Settings has since gained model/tracker/duration fields); capturing the final build
was blocked by a locked screen.

## Testing
`pytest` → **119 passed, 0 failed, 0 skipped**: database/migrations, events (incl. merge, duration, **false-merge regression**),
processor with a *scripted fake detector*, exports, tracking, **6 real-model integration tests on real videos**, and 2 GUI scenarios
that include a genuine process restart. GUI scenarios run in separate processes because Tk 9.0.4 on macOS segfaults when a second
`Tk()` root is created after one was destroyed in the same process; the shipped app creates exactly one root.
Evaluation scripts (`scripts/eval_*.py`, needs `pip install -e .[eval]` and local data) are not part of the test suite.

## Privacy
Local-first; no uploads, telemetry or cloud DB; no face recognition; authorisation tick-box before processing; data
folder mode 0700; configurable retention with safe deletion (never outside the data folder); videos processed in
place, not copied. Details: [docs/PRIVACY.md](docs/PRIVACY.md). **Only process video you are authorised to process.**

## Limitations
* Real-world phone-event accuracy is **unknown**; synthetic composites cannot establish it.
* Small phones are still often missed (F3: 2 of 3 staged events missed with the default model); phones can be read as remotes;
  small dark/rectangular objects (masks, head/hand blobs) cause false alerts.
* Tracker IDs still switch in crowds; tracking differences between configurations are within noise on the data available.
* The accuracy-first default is ≈2.7× slower than before (5–8 fps on 1080p); the incident-merge rule can, in rare cases,
  merge two different people's events if one leaves view and another shows a phone in nearly the same place within 6 s.
* Region events assume a static camera; macOS-only, un-notarised packaging; longest soak test 10 minutes (looped footage).
See [docs/EVALUATION.md](docs/EVALUATION.md).

## Future work
Consented, labelled exam footage → fine-tune a phone/calculator detector with video-level splits and report mAP/IDF1;
appearance-based re-ID *without* identity; calibrated thresholds; optional Firestore sync; notarised builds.

## Licences & data
Code: **AGPL-3.0-or-later** (Ultralytics is AGPL-3.0 — see [docs/LICENSES.md](docs/LICENSES.md)). Sample media and
attribution: [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) (CC0 pedestrian clip by Taurus Media Technik, CC0 phone photo
by Dennis Cortés; the CC BY-SA classroom clip was used locally only).

## Acknowledgement
Developed as part of the **INSTANT AI & Data Science Training** programme (Sprint 2, Project 3).
