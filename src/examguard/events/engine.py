"""Temporal event engine: observable events only, never "cheating".

Events
------
PHONE_VISIBLE_NEAR_TRACK  A phone-like object was detected near a tracked person in
                          >= N of the last W processed frames.
MONITORED_REGION_EXIT     A tracked person who was previously inside the configured
                          region has been detected outside it for >= T seconds.

Lifecycle per (track, event): [pending] -> ACTIVE -> RESOLVED, then cooldown.
One continuous event == one incident. Thresholds are engineering defaults
(see docs/EVENT_RULES.md); they are not scientifically calibrated.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from examguard.config import Settings
from examguard.detection.detector import Detection
from examguard.events.geometry import Box, center, containment, expand_box, point_in_polygon
from examguard.tracking.tracker import TrackedPerson

PHONE_EVENT = "PHONE_VISIBLE_NEAR_TRACK"
REGION_EVENT = "MONITORED_REGION_EXIT"
EVENT_LABELS = {
    PHONE_EVENT: "Phone-like object detected near track",
    REGION_EVENT: "Monitored region exit detected",
}
ACTIVE, RESOLVED = "ACTIVE", "RESOLVED"
MAX_EVIDENCE_FRAMES = 50


@dataclass
class Incident:
    event_type: str
    track_id: int
    started_ts: float
    started_frame: int
    last_ts: float
    last_frame: int
    person_bbox: Box
    phone_bbox: Box | None = None
    region_id: str | None = None
    conf_sum: float = 0.0
    conf_n: int = 0
    peak_conf: float = 0.0
    person_conf_sum: float = 0.0
    evidence_frames: list[int] = field(default_factory=list)
    state: str = ACTIVE
    ended_ts: float | None = None
    db_id: int | None = None  # filled by persistence layer

    @property
    def confidence(self) -> float:
        """Mean confidence of the detections that supported this event."""
        return self.conf_sum / self.conf_n if self.conf_n else 0.0

    @property
    def person_confidence(self) -> float:
        return self.person_conf_sum / self.conf_n if self.conf_n else 0.0

    def add_support(self, frame: int, ts: float, conf: float, person_conf: float,
                    person_bbox: Box, phone_bbox: Box | None = None) -> None:
        self.last_ts, self.last_frame = ts, frame
        self.person_bbox, self.phone_bbox = person_bbox, phone_bbox
        self.conf_sum += conf
        self.person_conf_sum += person_conf
        self.conf_n += 1
        self.peak_conf = max(self.peak_conf, conf)
        if len(self.evidence_frames) < MAX_EVIDENCE_FRAMES:
            self.evidence_frames.append(frame)

    def message(self) -> str:
        t = _fmt_ts(self.started_ts)
        if self.event_type == PHONE_EVENT:
            return f"Review Required: Phone-like object detected near Track #{self.track_id} at {t}."
        return f"Review Required: Monitored region exit detected for Track #{self.track_id} at {t}."


def _fmt_ts(ts: float) -> str:
    m, s = divmod(int(ts), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


@dataclass
class EngineOutput:
    started: list[Incident] = field(default_factory=list)
    resolved: list[Incident] = field(default_factory=list)
    reopened: list[Incident] = field(default_factory=list)  # resolved incidents continued by a new track (no new alert)
    active: dict[int, list[Incident]] = field(default_factory=dict)  # track_id -> active incidents


@dataclass
class _PhoneHit:
    phone: Detection
    person: TrackedPerson
    overlap: float


class EventEngine:
    def __init__(self, settings: Settings, fps: float = 30.0):
        self.s = settings
        self.fps = fps
        self._incidents: dict[tuple[int, str], Incident] = {}
        self._last_support_ts: dict[tuple[int, str], float] = {}
        self._cooldown_until: dict[tuple[int, str], float] = {}
        self._phone_win: dict[int, deque] = {}
        self._region: dict[int, dict] = {}
        self._poly = [tuple(p) for p in settings.region_polygon]
        self._last_seen: dict[int, float] = {}
        self._alias: dict[int, int] = {}              # new track id -> track id that owns the continued phone incident
        self._visible: set[int] = set()               # track ids present in the CURRENT frame
        self._phone_incidents: list[Incident] = []    # recent phone incidents (merge candidates)
        self.frame_size: tuple[int, int] | None = None  # (w, h) needed for region polygon

    # ---- association ---------------------------------------------------
    def associate_phones(self, tracks: list[TrackedPerson], phones: list[Detection]) -> dict[int, _PhoneHit]:
        """Assign each phone to AT MOST one person: the one whose (expanded) box contains
        most of the phone. Ambiguous frames (two people nearly tied) are skipped rather than guessed."""
        best_per_track: dict[int, _PhoneHit] = {}
        for ph in phones:
            scores = []
            for tr in tracks:
                ov = containment(ph.bbox, expand_box(tr.bbox, self.s.phone_assoc_expand))
                if ov >= self.s.phone_assoc_min_overlap:
                    scores.append((ov, tr))
            if not scores:
                continue
            scores.sort(key=lambda x: -x[0])
            if len(scores) > 1 and scores[0][0] - scores[1][0] < 0.10:
                # tie-break only if centres clearly favour one person, else skip (ambiguous)
                pc = center(ph.bbox)

                def dist2(item):
                    c = center(item[1].bbox)
                    return (pc[0] - c[0]) ** 2 + (pc[1] - c[1]) ** 2

                near, far = sorted(scores[:2], key=dist2)
                if dist2(far) < 1.5 * dist2(near):
                    continue  # centres do not clearly favour one person -> do not guess
                ov, winner = near
            else:
                ov, winner = scores[0]
            cur = best_per_track.get(winner.track_id)
            if cur is None or ph.confidence > cur.phone.confidence:
                best_per_track[winner.track_id] = _PhoneHit(ph, winner, ov)
        return best_per_track

    # ---- main entry ----------------------------------------------------
    def process(self, frame_index: int, ts: float, tracks: list[TrackedPerson],
                phones: list[Detection]) -> EngineOutput:
        out = EngineOutput()
        for tr in tracks:
            self._last_seen[tr.track_id] = ts
        by_id = {t.track_id: t for t in tracks}
        self._visible = set(by_id)
        hits = self.associate_phones(tracks, phones)
        self._phone_rule(frame_index, ts, by_id, hits, out)
        self._region_rule(frame_index, ts, by_id, out)
        self._resolve_stale(ts, out)
        self._gc(ts)
        for (tid, _), inc in self._incidents.items():
            out.active.setdefault(tid, []).append(inc)
        for new_tid, old_tid in self._alias.items():  # the red box follows the person's CURRENT track
            inc = self._incidents.get((old_tid, PHONE_EVENT))
            if inc is not None and new_tid in by_id and inc not in out.active.get(new_tid, []):
                out.active.setdefault(new_tid, []).append(inc)
        return out

    # ---- rule A --------------------------------------------------------
    def _phone_rule(self, frame, ts, by_id, hits, out):
        s = self.s
        for tid in set(hits) | set(self._phone_win):
            win = self._phone_win.setdefault(tid, deque(maxlen=self._window_frames()))
            h = hits.get(tid)
            win.append((frame, ts, h.phone.confidence, h.person.confidence, h.person.bbox, h.phone.bbox) if h else None)
            if len(win) == win.maxlen and all(w is None for w in win):
                del self._phone_win[tid]
                continue
            key = (self._alias.get(tid, tid), PHONE_EVENT)
            inc = self._incidents.get(key)
            if inc:
                if h:
                    inc.add_support(frame, ts, h.phone.confidence, h.person.confidence, h.person.bbox, h.phone.bbox)
                    self._last_support_ts[key] = ts
                continue
            support = [w for w in win if w]
            if (len(support) >= s.phone_min_support_frames and h is not None
                    and support[-1][1] - support[0][1] >= s.phone_min_duration_seconds - 1e-9):
                old = self._merge_candidate(key[0], ts, h.phone.bbox)
                if old is not None:  # same phone, different track id (ID switch): continue the incident
                    okey = (old.track_id, PHONE_EVENT)
                    self._alias[tid] = old.track_id
                    if okey not in self._incidents:  # it had been resolved -> re-open instead of a duplicate alert
                        old.state, old.ended_ts = ACTIVE, None
                        self._incidents[okey] = old
                        self._cooldown_until.pop(okey, None)
                        out.reopened.append(old)
                    old.add_support(frame, ts, h.phone.confidence, h.person.confidence, h.person.bbox, h.phone.bbox)
                    self._last_support_ts[okey] = ts
                    continue
            if (len(support) >= s.phone_min_support_frames and ts >= self._cooldown_until.get(key, -1)
                    and h is not None and support[-1][1] - support[0][1] >= s.phone_min_duration_seconds - 1e-9):
                f0, t0 = support[0][0], support[0][1]
                inc = Incident(PHONE_EVENT, tid, t0, f0, ts, frame, h.person.bbox, h.phone.bbox)
                for w in support:  # fold in evidence that confirmed the event
                    inc.conf_sum += w[2]; inc.person_conf_sum += w[3]; inc.conf_n += 1
                    inc.peak_conf = max(inc.peak_conf, w[2])
                    if len(inc.evidence_frames) < MAX_EVIDENCE_FRAMES:
                        inc.evidence_frames.append(w[0])
                inc.last_ts, inc.last_frame = ts, frame
                inc.started_ts, inc.started_frame = ts, frame  # alert time = confirmation time
                self._incidents[key] = inc
                self._last_support_ts[key] = ts
                self._phone_incidents.append(inc)
                del self._phone_incidents[:-64]
                out.started.append(inc)

    def _window_frames(self) -> int:
        """Sliding-window length in processed frames: the configured size, widened so the window can span the minimum
        duration (otherwise fast video, e.g. 60 fps, could never satisfy it)."""
        import math
        need = math.ceil(self.s.phone_min_duration_seconds * self.fps * 1.5)
        return max(self.s.phone_window_frames, need)

    def _merge_candidate(self, canon_tid: int, ts: float, phone_bbox: Box) -> Incident | None:
        """A recent phone incident on ANOTHER track whose phone is close in time and space (<= N phone sizes).

        An ID switch means the old track has DISAPPEARED. If the incident's track is still visible it is a different
        person (e.g. a neighbour), so it is never merged: a duplicate alert is safer than a lost one.
        """
        gap = self.s.phone_merge_gap_seconds
        if gap <= 0:
            return None
        pw = max(phone_bbox[2] - phone_bbox[0], phone_bbox[3] - phone_bbox[1])
        pc = center(phone_bbox)
        for old in reversed(self._phone_incidents):
            if old.track_id == canon_tid or old.phone_bbox is None or old.track_id in self._visible:
                continue
            last = old.last_ts if old.state == ACTIVE else (old.ended_ts or old.last_ts)
            if ts - last > gap:
                continue
            ow = max(old.phone_bbox[2] - old.phone_bbox[0], old.phone_bbox[3] - old.phone_bbox[1])
            oc = center(old.phone_bbox)
            if ((pc[0] - oc[0]) ** 2 + (pc[1] - oc[1]) ** 2) ** 0.5 <= self.s.phone_merge_distance * max(pw, ow):
                return old
        return None

    # ---- rule B --------------------------------------------------------
    def _region_rule(self, frame, ts, by_id, out):
        if not self._poly or self.frame_size is None:
            return
        w, h = self.frame_size
        poly = [(x * w, y * h) for x, y in self._poly]
        for tid, tr in by_id.items():
            st = self._region.setdefault(tid, {"seen_inside": False, "outside_since": None})
            inside = point_in_polygon(center(tr.bbox), poly)
            key = (tid, REGION_EVENT)
            inc = self._incidents.get(key)
            if inside:
                st["seen_inside"], st["outside_since"] = True, None
                continue
            if not st["seen_inside"]:
                continue  # never inside -> not an "exit"
            if st["outside_since"] is None:
                st["outside_since"] = ts
            if inc:
                inc.add_support(frame, ts, tr.confidence, tr.confidence, tr.bbox)
                self._last_support_ts[key] = ts
            elif (ts - st["outside_since"] >= s_exit(self.s) and ts >= self._cooldown_until.get(key, -1)):
                inc = Incident(REGION_EVENT, tid, ts, frame, ts, frame, tr.bbox, region_id=self.s.region_id)
                inc.add_support(frame, ts, tr.confidence, tr.confidence, tr.bbox)
                self._incidents[key] = inc
                self._last_support_ts[key] = ts
                out.started.append(inc)

    # ---- lifecycle -----------------------------------------------------
    def _resolve_stale(self, ts, out):
        for key in list(self._incidents):
            if ts - self._last_support_ts.get(key, ts) >= self.s.resolve_after_seconds:
                self._end(key, ts, out)

    def _end(self, key, ts, out):
        inc = self._incidents.pop(key)
        inc.state, inc.ended_ts = RESOLVED, self._last_support_ts.get(key, ts)
        self._cooldown_until[key] = ts + self.s.event_cooldown_seconds
        out.resolved.append(inc)

    def pending_tracks(self) -> set[int]:
        """Tracks with some (but not yet confirmed) phone evidence -> shown AMBER."""
        return {tid for tid, win in self._phone_win.items()
                if any(w for w in win) and (tid, PHONE_EVENT) not in self._incidents}

    def finalize(self, ts: float) -> EngineOutput:
        """Close every active incident at end of video / stop."""
        out = EngineOutput()
        for key in list(self._incidents):
            self._end(key, ts, out)
        return out

    def _gc(self, ts):
        for tid in [t for t, last in self._last_seen.items() if ts - last > 120]:
            self._last_seen.pop(tid, None)
            self._alias.pop(tid, None)
            self._region.pop(tid, None)


def s_exit(s: Settings) -> float:
    return s.region_exit_min_seconds
