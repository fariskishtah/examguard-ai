# Privacy and safety design

* **Observable events, not verdicts.** Alerts say "phone-like object detected near Track #N" or "monitored region
  exit detected". Nothing says "cheating". Every alert starts `PENDING_REVIEW`; a human may mark it
  Confirmed Observation, Dismissed or Needs More Evidence (with a note and a retained history). There is no
  "guilty" state and no automatic decision. The disclaimer is shown in the sidebar, dashboard, evidence images,
  CSV/JSON and report.
* **Anonymous tracks.** Track IDs are temporary integers from a tracker. No face recognition, biometrics, identity
  matching or inference of personal traits exists in the code. No student names are stored.
* **Local-first.** Processing, storage and notifications are local. The only network use is the optional,
  user-initiated model download. Nothing is uploaded; there is no telemetry and no cloud database.
* **Authorisation gate.** The Start button stays disabled until the user ticks "I am authorised to process this
  video".
* **Storage.** Runtime data lives in the per-user app-support folder (`~/Library/Application Support/ExamGuardAI`,
  mode 0700; DB and evidence 0600), outside the repository and app bundle. Evidence filenames are sanitised, unique
  and never overwritten (atomic write). Uploaded videos are processed in place, **not copied**.
* **Retention.** Configurable (default 30 days, 0 = keep). Expired sessions, incidents and their evidence files are
  deleted at start-up or on demand. Deletion only touches files inside the data folder, so a tampered DB path
  cannot delete other files.
* **Input hardening.** Extension allow-list, 4 GiB size cap, empty/corrupt video rejection before any DB record,
  CSV formula-injection neutralisation, HTML-escaped reports.
* **Publishing.** Do not publish real exam footage or screenshots of identifiable people without consent.
