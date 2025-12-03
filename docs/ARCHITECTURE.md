# Architecture & Design Notes

## Goals

- Provide a transparent baseline detector that combines fast, explainable heuristics with an optional transformer-based perplexity score.
- Keep the footprint friendly for classroom or hackathon deployments (no heavy databases or frameworks).
- Make it easy to deploy the same code both locally and on Streamlit Cloud with minimal configuration.

## Module responsibilities

| Module | Responsibility |
|--------|----------------|
| `detector.py` | Feature extraction, GPT‑2 management, scoring heuristics, and the `detect_ai` API contract. |
| `app.py` | Streamlit UI layer, user interaction plumbing, and presentation of detection results and prompts. |

`push_to_github.bat`, `.gitignore`, and `requirements.txt` support CI/CD hygiene rather than runtime logic.

## Detection flow

1. `app.py` gathers the text input, determines whether GPT‑2 is enabled, and calls `detect_ai`.
2. `detect_ai` trims whitespace, returns early for empty texts, and calls `stylometric_features`.
3. Stylometric features include:
   - average word length & sentence length (indicator of verbosity / uniformity),
   - punctuation density,
   - trigram repetition rate,
   - short word ratio,
   - basic counts (`num_words`, `num_sentences`).
4. If requested and available, `gpt2_perplexity` loads GPT‑2 once and computes a sliding-window perplexity over the entire sequence. Tokens that belong to the contextual overlap are masked so that each token contributes once.
5. Heuristic scores are derived from each signal, normalized, weighted, and combined into a single probability.
6. The result dictionary (probability, label, breakdown) is returned to the UI for display.

## Scoring heuristics

| Signal | Intuition | Weight |
|--------|-----------|--------|
| GPT‑2 perplexity | Low perplexity often correlates with AI-generated prose. Values are clamped between 10 and 1000 and scaled to 0‒1. | 0.60 (only when available) |
| Repetition rate | High reuse of short 3-grams is a common feature of AI writing. | 0.15 |
| Short word ratio | Some models over-use function words and short tokens. | 0.10 |
| Punctuation ratio | Reduced punctuation variety may indicate templated content. | 0.15 |

The final label thresholds are ≥0.75 = *Likely AI*, ≤0.35 = *Likely Human*, else *Unclear*. These values were chosen heuristically and should be adjusted whenever better calibration data becomes available.

## Streamlit UI

- Uses `st.sidebar` to host the GPT‑2 toggle and caveats.
- The main panel contains the input textarea, detect button, result summary, feature breakdown, and suggested prompt templates (system + user) so that reviewers can request corroborating evidence from other LLMs.
- `st.spinner` provides feedback while heuristics/GPT‑2 run.
- The breakdown uses two columns to keep features and weighted parts readable even on small screens.

## Model management

- `_ensure_gpt2` lazily loads tokenizer + model once per process and reuses them.
- The device is auto-detected (CUDA if possible, else CPU). No gradients are computed; inference runs within `torch.no_grad()`.
- Perplexity implements the Hugging Face sliding window recipe with masking to avoid over-counting overlapping context tokens. Errors fall back to `nan`, which automatically skips the GPT‑2 weight.

## Testing strategy

- Unit-level: direct invocation of `stylometric_features`/`detect_ai` via `py -c` scripts.
- Integration-level: `streamlit run app.py` plus manual spot checks with known AI/human samples.
- Static: `py -m compileall` keeps syntax regressions out of deployments.
- Future work: add pytest modules with fixtures around short/long texts and mocking of GPT‑2 to assert weights and labels.

## Extensibility ideas

1. **Richer features** – include POS tag ratios, readability scores, or embedding cosine similarity to known corpora.
2. **Model adapters** – let users pick alternative models (e.g., `distilgpt2`, `llama-3` perplexity) through Streamlit select boxes.
3. **Batch API** – expose `detect_ai` via FastAPI/Flask for integrations beyond the Streamlit UI.
4. **Calibration set** – store labelled samples and fit a logistic regression to map scores to probabilities.
5. **Caching layer** – hash inputs to avoid recomputing GPT‑2 perplexity on repeated checks.

These upgrades can live in new modules under `core/` or `services/` should the project grow; the current structure keeps separation of concerns clear enough for HW5 deliverables.
