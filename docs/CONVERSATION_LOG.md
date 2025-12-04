# Conversation Log (assistant <> user)

Chronological record of prior interactions in this workspace (latest request excluded).

## 2025-12-03

- Reviewed repo contents (`app.py`, `detector.py`, `requirements.txt`, `README.md`) and identified a syntax error in the punctuation regex plus a flawed GPT-2 perplexity computation.
- Fixed `detector.py` by escaping the punctuation character class correctly and reworking the GPT-2 sliding-window perplexity to mask overlapping context tokens and guard against empty sequences.
- Added comprehensive documentation (`README.md`, `docs/ARCHITECTURE.md`) describing layout, setup, heuristics, testing, deployment, and future work.
- Ran smoke checks: `py -m compileall` on `detector.py`/`app.py`, basic `detect_ai` call without transformers, and verified Streamlit import behavior.
- Initialized git, configured identity (`g114064015@smail.nchu.edu.tw`), rebased onto remote `origin/main`, and pushed commit `bb51c0e` with the fixes and docs.
- Enhanced the Streamlit sidebar with presets, stateful toggles, verdict recap, cheat sheet, and helper actions; removed emoji to keep ASCII-only. Pushed commits `c3150be` and `5628adb` to `main`.
