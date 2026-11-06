# Event rules and thresholds

All thresholds are **engineering defaults, not calibrated values** (`configs`/Settings; validated on save).

## Pipeline
decode (streamed) -> YOLO11 person + cell phone at **native resolution (auto, capped at 1920)** + person-crop phone assist
(640-px crops, optional) -> BoT-SORT (no appearance, no GMC) on persons -> EventEngine (association, temporal confirmation,
cross-track merge, lifecycle) -> evidence + DB + alert -> UI. Why these defaults: `docs/EVALUATION.md` §2, §5, §8.

## PHONE_VISIBLE_NEAR_TRACK
1. A phone is associated with a person only if >= 15 % of the phone box lies inside that person's box expanded by 15 %.
2. If two people overlap the phone almost equally (< 0.10 apart) and the phone centre does not clearly favour one,
   the frame is skipped (no guess).
3. Confirmation: >= 5 associated frames inside a sliding window of 15 processed frames **and** the supporting detections must
   span >= **0.5 s** (`phone_min_duration_seconds`). The window is widened automatically for fast video (e.g. 60 fps) so the
   duration is satisfiable. (Added after a 0.13 s face-mask blip was confirmed on a 60 fps clip.)
4. One incident per continuous event. It resolves after 2 s without support; a 10 s cooldown then suppresses
   re-triggers for the same track+event.
5. **Cross-track merge** (`phone_merge_gap_seconds` = 6, `phone_merge_distance` = 2.5 phone-sizes; 0 = off): a new phone
   incident is folded into a recent one (no new alert; a resolved incident is re-opened) only if it starts within the gap,
   the phones are within the distance, **and the earlier incident's track is no longer visible** (an ID switch removes the old
   ID; a neighbour stays visible, so is never merged). Residual risk: a different person showing a phone at nearly the same place
   within 6 s of someone leaving view would be merged — a duplicate alert is judged safer than a lost one.
6. Confidence = mean phone-detection confidence over the supporting frames.
It means a phone-like object was seen near that track. It does not mean use. COCO's `cell phone` class is confusable with
remotes, masks and small dark objects, and recall for small phones is limited (measured, `docs/EVALUATION.md`).

## MONITORED_REGION_EXIT
Optional polygon drawn on a real frame (stored normalised, so resizing cannot drift). A track alerts only if it
was previously inside, is now detected outside for >= 1.5 s. A track that never was inside, or vanishes (e.g. detector
dropout), does not alert. Resolves after 2 s back inside / absent.

## Frame sampling
`frame_stride = N` runs inference on every Nth frame only. Window sizes count *processed* frames, so a larger stride
lengthens the real-time confirmation delay and can miss short events. Preview/annotated-video boxes between processed
frames repeat the last result.

## Box colours
GREEN normal; AMBER = phone evidence accumulating but not confirmed; RED = active event. Red clears when the
incident resolves.

## Not implemented (future research, no reliable method/data here)
gaze/head-turn, whispering/talking, collaboration, object classification beyond phone.
