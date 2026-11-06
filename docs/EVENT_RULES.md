# Event rules and thresholds

All thresholds are **engineering defaults, not calibrated values** (`configs`/Settings; validated on save).

## Pipeline
decode (streamed) -> YOLO11n person + cell phone (+ optional person-crop phone assist) -> ByteTrack on persons ->
EventEngine (association, temporal confirmation, lifecycle) -> evidence + DB + alert -> UI.

## PHONE_VISIBLE_NEAR_TRACK
1. A phone is associated with a person only if >= 15 % of the phone box lies inside that person's box expanded by 15 %.
2. If two people overlap the phone almost equally (< 0.10 apart) and the phone centre does not clearly favour one,
   the frame is skipped (no guess).
3. Confirmation: >= 5 associated frames inside a sliding window of 15 processed frames.
4. One incident per continuous event. It resolves after 2 s without support; a 10 s cooldown then suppresses
   re-triggers for the same track+event.
5. Confidence = mean phone-detection confidence over the supporting frames.
It means a phone-like object was seen near that track. It does not mean use, and COCO's `cell phone` class is
confusable with remotes/calculators.

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
