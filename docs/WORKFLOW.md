# Engineering Workflow

Practical steps to develop, test, and ship changes to the AI Text Detector. Assumes Python 3.10+ (tested on 3.13), Streamlit, and optional transformers/torch for GPT-2 perplexity.

## 1) Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Notes:
- For fast iterations without GPT-2, you may comment out `torch`/`transformers` in `requirements.txt` and rely on heuristics only.
- Add `C:\Users\<user>\AppData\Local\Programs\Python\Python313\Scripts` to PATH if Windows warns about missing script locations.

## 2) Local dev loop

1. Edit `detector.py` (heuristics, GPT-2 handling) or `app.py` (UI/UX).
2. Syntax guard:
   ```powershell
   py -m compileall detector.py app.py
   ```
3. Quick smoke without GPT-2:
   ```powershell
   py -c "from detector import detect_ai; print(detect_ai('Sample text', use_gpt2=False))"
   ```
4. Run UI:
   ```powershell
   streamlit run app.py
   ```
   Use sidebar samples, toggle GPT-2 if installed, and observe probability + breakdown.
5. Optional GPT-2 check (slow/first-time download):
   ```powershell
   py -c "from detector import detect_ai; print(detect_ai('Longer text for perplexity test', use_gpt2=True))"
   ```

## 3) Calibration guidelines

- Favor **diversity signals** (`type_token_ratio`, punctuation) to pull probability down for human-like text.
- Favor **template signals** (repetition, low sentence-length variance, low TTR) to push probability up for AI-like text.
- Adjust thresholds (`Likely AI` ≥ 0.70, `Likely Human` ≤ 0.50) only when adding/retuning features.
- For stronger accuracy, collect labelled examples (human/AI) and fit a small calibration model (logistic regression) on exported features; swap the heuristic combiner with that mapping.

## 4) Git hygiene

```powershell
git status -sb
git add <files>
git commit -m "Your message"
git push
```

- Repository remote: `https://github.com/g114064015lab/AIOT_DA_HW5.git`
- Identity: `g114064015@smail.nchu.edu.tw`
- If remote diverges, `git fetch origin` then `git rebase origin/main` before pushing.

## 5) Deployment (Streamlit Cloud)

1. Ensure `requirements.txt` is up-to-date (remove heavy deps if heuristics-only).
2. Push to `main`.
3. On https://share.streamlit.io, create/update the app pointing to `g114064015lab/AIOT_DA_HW5`, branch `main`, file `app.py`.
4. If GPT-2 is needed, configure secrets/env (`TRANSFORMERS_CACHE`, HF token if private models).
5. Verify the app loads and sidebar samples render; run a few AI-like and human-like passages to confirm spread.

## 6) Release checklist

- [ ] `py -m compileall detector.py app.py`
- [ ] Manual smoke tests with/without GPT-2 toggle
- [ ] Sidebar samples show diverse probabilities
- [ ] README + docs updated if behavior changes
- [ ] `git status` clean; `git push` succeeds

## 7) Troubleshooting

- **Low variance in probabilities**: increase weights on repetition/variance features; add bias for template patterns; enable GPT-2.
- **Import errors for Streamlit/transformers**: reinstall via `pip install -r requirements.txt`; verify virtualenv active.
- **Slow GPT-2 load**: set `TRANSFORMERS_CACHE` to a persistent path; use smaller `distilgpt2` by changing `model_name` in `_ensure_gpt2`.
- **Encoding on Windows**: keep ASCII-only UI strings; files are UTF-8.
