# Evaluation (only what was actually measured)

Hardware: Apple M2, 8 GB, macOS, PyTorch MPS. Detector: generic COCO-pretrained YOLO11 (**not fine-tuned for exams**).
Everything here is reproducible with the scripts named in each section; raw numbers are in `artifacts/metrics/*.json`.

## 0. How to read this document — three different kinds of evidence

| Kind | What it is | What it can support |
|---|---|---|
| **Real labelled data** | COCO val2017 images (phone boxes, held out from YOLO's training); MOT15 sequences with ground-truth identities | detector/tracker quality in *general* scenes |
| **Synthetic composites** | a real CC0 phone photograph pasted into real MOT15 footage at a known place/time | the software chain (detect → track → associate → confirm → evidence → DB) and relative comparisons between settings. **Not** real phone use |
| **Real-world exam validation** | consented, labelled exam-room footage with real phone use | **does not exist for this project → NOT EVALUATED** |

Nothing in this file is a claim about accuracy in real examinations.

## 1. Not evaluated (and why)
| Metric | Status | Reason |
|---|---|---|
| Phone events on **real** phone-use / exam footage | **NOT EVALUATED** | no consented, labelled footage |
| Phone-event false alerts per hour on real exam footage | **NOT EVALUATED** | same; only 127 s of *clean* composite twins + 25 s of normal clips were run (0 alerts), far too little for a rate |
| Fine-tuning / custom training | **NOT PERFORMED** | no authorised labelled data; no specialist model is claimed |
| Tracking HOTA | **NOT EVALUATED** | only IDF1/MOTA/ID switches were computed |

## 2. Why small phones were missed (root cause) and what helps
Dataset: all 214 COCO val2017 images containing a cell phone (262 instances) + 400 sampled person images without a phone,
split **dev/test by md5(image_id) parity, frozen before any experiment** (`artifacts/metrics/phone_coco_split.json`).
All choices were made on **dev**; **test** is reported next to it. COCO images are used locally only (mixed Flickr licences).
Two stress conditions shrink the *same real images* into a larger frame: **far** (1280×720) and **far1080** (1920×1080),
so phones become tiny relative to the frame, like a camera far from the desk.

**Root cause: input resolution.** A 1080p frame fed to YOLO at 640 px is shrunk ~3×; phones of 15–25 px vanish. Evidence
(yolo11n, AP50 at IoU 0.5; dev / test):

| Strategy | far | far1080 | ms/img |
|---|---|---|---|
| full-frame @640 (original default) | 0.229 / 0.212 | 0.212 / 0.204 | 14–16 |
| union full@640 + person-crop@416 (shipped earlier) | 0.376 / 0.353 | 0.364 / 0.338 | 30–48 |
| full-frame @1920 (native) | 0.483 / 0.428 | 0.495 / 0.469 | 47–70 |
| union full@1280 + person-crop@640 | 0.476 / 0.465 | 0.477 / 0.472 | 49–69 |
| yolo11s full-frame @1920 | 0.565 / 0.590 | 0.589 / 0.581 | 104–180 |
| yolo11s union full@1280 + crop@640 | 0.536 / 0.596 | 0.575 / 0.613 | 100–157 |

* The dev-selected choice held up on test (no overfitting to the selection).
* On ordinary 640 px COCO images (no shrinkage) **person-crop alone is worse than full-frame** (AP50 0.25 vs 0.47); it only
  helps as a *union* with the full-frame result, and 640-px crops beat 416/320/960.
* Tiling (SAHI-style) raised recall but produced many false positives (precision 0.57–0.61) at 4–10× the cost.
* Alternative pretrained detectors were **not** automatically better: yolov8n and yolo26n were comparable to yolo11n;
  only the larger yolo11s/m clearly helped (and cost 2–6× more).
* Person-proximity filtering lowers COCO AP because many COCO phones are not held by a visible person; the event layer
  applies it anyway.

Script: `scripts/eval_phone_coco.py` (`--set round1|round2`).

## 3. Phone events on synthetic composites (software-level — NOT real phone use)
Real MOT15 footage + real CC0 phone photos pasted at 15 % of the person's height (11–23 px). Fixture sets: **F1** (3 MOT15
test sequences) and **F2** (3 MOT15 dev sequences) are *design data* — their results shaped the final rules. **F3**
(ETH-Bahnhof, KITTI-13, **ADL-Rundle-8 = genuine 1920×1080**) was never used before the final run. The decision rule was
registered before F2/F3 ran: *adopt the new default only if event recall ≥ the previously shipped configuration and
clean-twin false alerts are no worse.* Clean twins = the same footage with no phone added.

Diagnosis on F1 (frame-level recall on the tiny phones): full-frame 0 % at 640 and ≤22 % even at 1920; **full + 640-px crops 55–59 %**
(`artifacts/metrics/fixture_diagnosis_F1.json`).

Events found out of 3 staged events per set (the baselines ran before the duration rule existed and have no merge):

| Configuration | F1 | F2 | **F3 (untouched)** | F3 clean-twin false alerts | F3 FPS | F3 peak RSS |
|---|---|---|---|---|---|---|
| v0: 640, no assist, ByteTrack | 0/3 | 1/3 | **0/3** | 0 | 59.7 | 746 MB |
| shipped earlier: 640 + crop416, ByteTrack | 3/3 | 1/3 | **0/3** | 0 | 21.2 | 580 MB |
| C0: native resolution, no assist, BoT-SORT | 0/3 | 1/3 | **0/3** | 0 | 36.9 | 249 MB |
| FINAL before the duration rule (native + crop640 + BoT-SORT + merge) | 2/3 † | 2/3 | **1/3** | 0 | 8.0 | 360 MB |
| FINAL without merge (ablation, same build) | – | – | 1/3 | 0 | 8.0 | 368 MB |
| FINAL with yolo11s | 1/3 ‡ | 3/3 | 1/3 (a different one) | 0 | 3.4 | 381 MB |
| **FINAL v2 = shipped default (+ 0.5 s minimum duration)** | **2/3** | **2/3 (0 duplicates)** | **1/3** | **0** | 6.7 | 345 MB |

† F1 predates the merge rule. ‡ the F1 yolo11s run had crop assist OFF (it predates that default), so it is not comparable.
FPS varies by a few tenths between runs on the same code (8.0 vs 6.7 for FINAL on F3); the controlled numbers are in §8.

**Verdict against the pre-registered rule (applied to the final v2 build):** on the untouched F3, recall is 1/3 vs 0/3 for the
earlier shipped configuration, and clean-twin false alerts are 0 vs 0 → **the new default is kept.**
**The evidence is thin.** Over the 9 staged events (F1+F2+F3), FINAL finds 5 and the earlier shipped configuration 4. F3's
1/3 is a single event (ETH-Bahnhof); the two 1080p-and-wide-format events (ADL-Rundle-8, KITTI-13) were missed with the
default model (yolo11s found ADL-Rundle-8 instead). **This does not demonstrate reliable phone detection, and nothing here is
real-world exam accuracy.** The pasted phones are tiny, have visible rectangular edges, and one is face-on.

### 3b. Real false alerts that were found (two, both on real footage with no phone)
1. **Face mask classified as a phone** (real classroom clip, 60 fps): confirmed from a **0.13 s** detection because confirmation
   counted frames only (5 frames = 83 ms at 60 fps). Caught by the real-model integration test. **Fixed**: confirmation now also
   requires the supporting detections to span ≥ 0.5 s (`phone_min_duration_seconds`; the window widens automatically at high
   frame rates). Unit tests: a 0.13–0.3 s burst never alerts at 7/14/25/30/60 fps; a 1 s burst always does. Classroom clip: 0 alerts again.
2. **Head/hand blob classified as a phone** (F1 PETS09 clean twin): a 6×10 px "phone" on a person's head for 1.9 s. **Not fixed**
   (not a blip, so the duration rule cannot catch it; a size-vs-person filter would be tuned on one example). Expect occasional
   false alerts of this kind; every alert goes to a human reviewer for exactly this reason.
Clean footage actually run with the final build: F1 twins 146 s (1 false alert), F2 twins 77 s (0), F3 twins 127 s (0),
classroom 10 s and street 15 s (0 after the fix). That is ~6 min in total — **not enough to state a false-alert rate.**

## 4. Duplicate incidents and the cross-track merge rule
Cause (inspected directly): on ETH-Pedcross2 one staged phone produced 4 incidents on **4 different track IDs**
(#106/#108/#110/#113) — ID switches in a crowd, not detector flicker.

Rule: a new phone incident is merged into a recent one (no new alert) if it starts within 6 s, the phones are within
2.5 phone-sizes, **and the earlier incident's track is no longer visible**.

* **First version had no visibility condition and was unsafe.** Two *separate* events on neighbouring people about one
  person-width apart (PETS09-S2L1): merge ON **lost event A** (it was folded into the neighbour's incident).
* **Fix:** require the old track to have disappeared (an ID switch removes the old ID; a neighbour stays visible).
  Re-test, two separate nearby events: PETS09 → A and B both reported (merge ON = OFF); ADL-Rundle-8 → both reported.
* What the merge still buys (ETH-Pedcross2, design data): **1 incident spanning 30.0–39.6 s** (staged window 29.7–39.7 s)
  instead of **4 fragments**. Residual risk: if a person leaves view and a different person shows a phone at nearly the same
  place within 6 s, the events would be merged. A duplicate alert is judged safer than a lost one.

Pipeline-level regression tests (scripted detector, `tests/test_processor.py`): (1) a forced ID switch yields **one** incident, **one**
alert, a start time equal to the first confirmation, an end time extended across the switch, a consistent track/session reference, and a
human review made while processing **intact**; (2) the same scenario with merging disabled yields two incidents (so the test has teeth);
(3) a neighbour whose track stays visible keeps a separate incident with unstretched timestamps. Engine-level tests cover re-opening a
resolved incident, the 6 s gap limit, the distance limit, and disabling the rule.
Effect on the fixtures (development runs): F2 duplicates 3 → 0; F3 duplicates 0.

Scripts: `scripts/make_two_phone_fixture.py`, `scripts/eval_events.py`; results `artifacts/metrics/merge_safety_twophone.json` (post-fix).

## 5. Tracking (real footage, ground-truth identities — MOT15)
MOT15 train sequences (CC BY-NC-SA 3.0, local research use, not redistributed). Split frozen first: **dev** ETH-Pedcross2,
TUD-Campus, KITTI-17; **test** PETS09-S2L1, TUD-Stadtmitte, ETH-Sunnyday. 54 ByteTrack settings + BoT-SORT variants were
swept on dev only. Detector: our YOLO11n; metrics: py-motmetrics (IoU ≥ 0.5).

| Config | dev IDF1 | **test IDF1** | test MOTA | test ID switches | IDs created / GT ids | tracker ms/frame |
|---|---|---|---|---|---|---|
| old ByteTrack (.35/.4, buffer 30, floor .35) | 0.717 | 0.800 | 0.743 | 31 | 1.86 | 0.4 |
| ByteTrack .5/.5, buffer 60, floor .1 (shipped earlier) | 0.726 | 0.779 | 0.758 | 36 | 1.61 | 0.7 |
| **BoT-SORT, no appearance, no GMC (new default)** | **0.746** | **0.802** | 0.759 | **30** | **1.55** | 0.5 |
| BoT-SORT + camera-motion compensation | 0.723 | 0.793 | 0.759 | 33 | 1.55 | 4.9 |
| BoT-SORT + generic appearance embeddings (yolo11n-cls) | 0.700 | 0.773 | 0.758 | 38 | 1.57 | 46–51 |

**Honest reading:** all sensible configurations land within ~0.03 IDF1 on three short test sequences; **I cannot claim a
demonstrated tracking improvement.** The earlier "fewer IDs created" gain (32 → 19 on one classroom clip) was a proxy and did
**not** translate into higher IDF1 (the shipped-earlier config scored *below* the old one on test). BoT-SORT-no-GMC is kept
as the default only because it was best/tied on both dev and test and costs nothing. Appearance embeddings were worse and
~100× slower; they add no face recognition but were not worth it. Script: `scripts/eval_tracking_mot.py`.
Bug found on the way: the detection shim lacked `.xyxy`, so BoT-SORT's GMC had been silently disabled (fixed + regression test).

## 6. MONITORED_REGION_EXIT on real footage (rule-derived ground truth)
Real MOT15 pedestrians; three fixed regions per sequence (left half, right half, centre band; not tuned). Expected events =
the same event rule applied to the **ground-truth** boxes; predicted events come from detector + tracker. This measures how
much detection/tracking degrades the event; it is **not** human-annotated exit labelling.

| Sequences | Config | GT events | Precision | Recall | Start-time error | Duplicates | False alerts/h (evaluated) |
|---|---|---|---|---|---|---|---|
| test (PETS09, TUD-Stadtmitte, ETH-Sunnyday) | old tracker | 65 | 0.909 | 0.923 | −0.03 s | 0 | 49.3 (7.3 min) |
| test | shipped / new tracker | 65 | 0.923 | 0.923 | −0.03 s | 0 | 41.1 (7.3 min) |
| dev (ETH-Pedcross2, TUD-Campus, KITTI-17) | old | 40 | 0.677 | 0.525 | +0.06 s | 1 | 155.6 (3.9 min) |
| dev | shipped / new | 40 | 0.657 | 0.575 | +0.06 s | 1 | 186.7 (3.9 min) |

Dev is worse because it contains a crowded scene and a moving-camera clip, where a fixed image-space region is a weak
assumption. The tracker variants are indistinguishable on this metric. The false-alerts/hour figures come from minutes of
footage and are high-variance.

## 7. Error analysis
* **Two real false alerts** (§3b): a face mask read as a phone (fixed by the minimum-duration rule) and a 6×10 px head blob read
  as a phone for 1.9 s (not fixed). Reviewers must expect occasional false alerts from small dark/rectangular objects.
* **Phone vs remote:** the same phone was classed as `remote` in several configurations (COCO confusion).
* **Missed tiny phones** when the frame is small *and* the phone is <25 px (KITTI-13, ADL-Rundle-8 with the default model).
* **Larger input is not monotonic:** at 1280 px the full-frame detector produced extra false boxes on some data.
* **ID switches** in crowds fragment incidents (mitigated by the merge rule) and fragment IDs generally.

## 8. Performance — before/after, same clips, idle machine (`artifacts/metrics/benchmark_final.json`)
Apple M2 8 GB, MPS, annotated video **not** written, one run per cell. "Time ×" = wall-clock relative to the earlier shipped
configuration on the same clip. Latency = model time per frame (full-frame pass + crop pass). Memory = whole-process peak RSS.

| Clip (1080p) | Config | FPS | Time × | Median / p95 model ms | Peak RSS |
|---|---|---|---|---|---|
| street (375 f) | **BEFORE** 640 + crop416 (shipped earlier) | 19.2 | 1.00 | 47.9 / 61.2 | 700 MB |
| | BEFORE-fast 640, no assist | 62.9 | 0.31 | 13.3 / 19.5 | 418 MB |
| | **AFTER default** native(1920) + crop640 | **8.0** | **2.39** | 119.2 / 152.2 | 495 MB |
| | AFTER-fast native, no assist | 21.4 | 0.90 | 43.8 / 45.0 | 375 MB |
| | AFTER-accurate yolo11s + crop640 | 2.8 | 6.77 | 349.7 / 455.0 | 580 MB |
| ADL-Rundle-8 (654 f) | BEFORE | 15.4 | 1.00 | 62.3 / 80.7 | 537 MB |
| | **AFTER default** | **5.0** | **3.07** | 199.9 / 218.0 | 381 MB |
| | AFTER-fast | 16.5 | 0.93 | 57.1 / 62.8 | 524 MB |
| | AFTER-accurate | 2.2 | 7.15 | 440.0 / 581.4 | 386 MB |
| classroom (620 f) | BEFORE | 13.4 | 1.00 | 72.4 / 78.7 | 397 MB |
| | **AFTER default** | **5.0** | **2.66** | 206.9 / 229.7 | 373 MB |
| | AFTER-fast | 16.0 | 0.83 | 59.2 / 65.6 | 317 MB |
| | AFTER-accurate | 2.2 | 6.12 | 456.0 / 549.8 | 407 MB |

**The cost is verified: the default is 2.4–3.1× (mean ≈ 2.7×) slower than the earlier shipped configuration**, i.e. 5–8 fps on
1080p. Memory stays under ~700 MB in every configuration. A 1-hour 1080p/30 fps exam recording (108 000 frames) would take
roughly 3.5–6 h at the default; `frame_stride` (e.g. 3) cuts that by the same factor but can miss short events, and "fast" mode
runs at about the earlier shipped speed.

**Does the improvement justify the cost?** It splits into two parts:
* *Native resolution* costs **nothing** (AFTER-fast ≈ BEFORE speed, ×0.83–0.93) and is the best-supported change: on real
  labelled images it roughly doubled small-phone AP50 (0.21 → 0.43–0.47 test) — §2.
* *Crop assist* is the ~3× part. Its benefit is real but narrower: tiny phones in small frames (F1 diagnosis: 0 % → ~55 % frame
  recall), +recall at the price of some precision on real images (§2), and a 1/3 vs 0/3 difference on the untouched F3 that is
  within noise. For an **offline review tool where a missed alert is worse than extra waiting**, accuracy-first is a defensible
  default under the pre-registered rule — but it is a judgement, not a proven win. Users who need speed should pick "fast"
  (crop assist off) in Settings.

## 9. Long-video stability (`scripts/soak_test.py` → `artifacts/metrics/soak_test.json`)
Footage: the CC0 street clip + synthetic phone fixture **looped 40×** (a long video, not a diverse one — this tests resource
stability, not accuracy). Default configuration (native 1920 + crop640 + BoT-SORT + merge), annotated video written, GUI worker
path, 1080p / 25 fps.

| Item | Result |
|---|---|
| Video duration processed | **10.0 min (15 000 frames)** |
| Wall-clock processing time | **46.05 min** (4.6× real time) |
| FPS: overall / per-minute median / min–max | 5.40 / 5.40 / 3.8–7.3 (dips coincide with my other CPU work during the run) |
| Memory (RSS) | start 524 MB, first-third mean 651, last-third mean 640, **peak 858**, end 495 → **no growth** |
| Mean process CPU | 108.9 % |
| Incidents / alerts | **40 / 40 for 40 staged phone windows** (one per window), minimum spacing 15.0 s, all RESOLVED |
| Evidence screenshots | all 40 incidents had both annotated and raw files |
| Database | session COMPLETED, 15 000 frames, 2 848 tracks recorded; `integrity_check` ok in the packaged run |
| Exported annotated video | playable, **15 000 frames** (= input) |
| Pause / resume | paused for 6 s → **1** frame advanced (the one in flight); resumed and completed |
| Cancellation (separate run) | STOPPED after 254 frames / 41.6 s; DB status STOPPED; partial video playable with 254 frames; its incident closed and evidence present |
| Errors / crashes | none |

Tracker state: tracker-only probe over **60 000 frames** with constantly churning IDs — the removed-track list rises to the
library cap (1000) and stays there (`removed_buffer`), tracked ≤ 6, lost ≤ 3, per-frame time flat
(`artifacts/metrics/tracker_cleanup_60k_frames.json`). Not tested: videos longer than 10 minutes, or an actual 1-hour recording.

## 10. Automated tests and packaged application
`pytest tests` → **119 passed, 0 failed, 0 skipped** (6 min 25 s): database 12, events 46, export 5, **real-model integration 6**,
processor/pipeline 16, tracking 9, GUI/controller 11 (2 GUI scenarios in separate processes incl. a real restart), video/evidence 14.
Packaged `ExamGuard AI.app` (1.0 GB, ad-hoc signed, not notarised), run with the repo's `models/` folder hidden:
`--selftest` drives the real GUI end to end inside the bundle, then a **second launch** (`--verify-persistence`) re-opens the data folder.

| Check | Result |
|---|---|
| Launches, frozen, bundled YOLO weights found | yes (`Contents/Frameworks/models/yolo11n.pt`) |
| torchvision packaging issue | **still resolved** — real inference ran (imgsz 1920) |
| Tracker | BoT-SORT (`botsort(ultralytics,gmc=none,reid=off)`) recorded in the session |
| Upload + processing | 375 frames, status COMPLETED |
| Preview frames / boxes | drawn by the worker and shown in the canvas (see visual note below) |
| Events + screenshots | 1 incident, annotated + raw evidence files present |
| SQLite | integrity ok, 0 foreign-key violations, 5 tables |
| Human review | recorded (NEEDS_MORE_EVIDENCE + note) and present in JSON history |
| Exports | CSV 856 B, JSON (1 incident, review history 1), 2 screenshot files, HTML report 54 KB |
| H.264 export | ffprobe: h264 High 1920×1080 25 fps **375 frames** |
| Restart | records, review note, evidence image all present; dashboard cards correct; exit code 0 (clean shutdown) |

*Visual check:* the README window screenshots (tracking overlays, red-box alert + toast, dashboard, incident viewer, review, settings)
were captured from this final packaged build. The red-box frame is the synthetic composite fixture.
