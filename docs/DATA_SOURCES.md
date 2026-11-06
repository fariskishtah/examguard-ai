# Data sources, licences and consent

No student, exam-room or private footage was collected, used or published. All media below is openly
licensed (Wikimedia Commons). Files are **downloaded locally by `scripts/fetch_samples.py` and are not
redistributed in this repository**; provenance (licence, author, URL, SHA-256) is recorded in
`data/samples/MANIFEST.json`.

| File | Licence | Author (per Commons) | Use |
|---|---|---|---|
| Video Codec Test pedestrian area 1080p25 (1920x1080, 375 frames, 25 fps, 15 s) | CC0 | Taurus Media Technik | multi-person tracking, "normal" scenario, base of the synthetic fixture |
| Student studies in classroom during face to face learning in Padang, Indonesia (1920x1080, 60 fps, 10.4 s) | CC BY-SA 4.0 | Rhmtdns | "normal monitoring" scenario in a real classroom. **Local testing only.** It shows school pupils, so none of its frames are published |
| Black smartphone in hand (Unsplash) (6000x4000 photo) | CC0 | Dennis Cortes | the phone photograph pasted into the synthetic fixture |

Rejected: a "smartphone" clip that turned out to show a flag (no people/phones), and search results that were
crime-scene or news footage (inappropriate and not exam-like).

## The synthetic fixture (`data/fixtures/`, git-ignored)
`scripts/make_synthetic_fixture.py` pastes the CC0 phone photo (feathered, outline-masked) into the pedestrian
clip at a fixed position inside one tracked person's box for 3.0-11.0 s and records the exact pasted box per
frame (199 frames). **It is not real phone-use footage.** It exists to (a) test the full detect -> track ->
associate -> confirm -> evidence -> DB chain with the real model and (b) give *computable*, clearly-labelled
frame/event metrics. Do not read them as real-world accuracy.

## Gap
No real, consented, labelled exam-room video with phone use was available, so real-world phone-event accuracy is
**NOT EVALUATED**. To close it: record a short staged clip with willing participants (written consent), label
event start/end, and run `scripts/run_headless.py`.

## About the published screenshots
`artifacts/screenshots/` contains frames from the **CC0 street clip** (passers-by in a public square, published to the
public domain by its author) with the synthetic phone composite, plus UI/figures. It contains **no exam footage and no
school footage**. If you replace these with your own, only use footage you are authorised to publish.
