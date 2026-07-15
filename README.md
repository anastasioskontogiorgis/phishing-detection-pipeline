# Phishing Email Detection — end-to-end

From a leakage-audited dataset to a containerised prediction service:
**notebook → versioned artifact → FastAPI → Docker**.

| Path | What it is |
|---|---|
| `phishing_v2.ipynb` | The analysis (shipped **executed**): data audit, leakage-free protocol with quantified duplicate inflation, model benchmark, honest held-out evaluation, source-artefact ablation, error analysis, artifact export. |
| `model/` | `phishing_pipeline.joblib` (TF-IDF + logistic regression) + `metadata.json` (version, protocol, held-out metrics). |
| `app/main.py` | Serving API: `POST /predict`, browser UI at `/`, `/health`, `/stats`, structured JSON request logs (metrics only — never email content). |
| `Dockerfile`, `requirements.txt` | The container. Pinned `scikit-learn` matches the training environment. |

**Held-out results (dedupe-first protocol):** accuracy 0.979 · F1 0.972 · ROC-AUC 0.997
on 3,505 test emails — see the notebook's *Skeptic's corner* for why these numbers
are trusted and what their limits are.

## Run the API locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
# UI:      http://localhost:8000
# predict: curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
#               -d '{"text":"Click here to claim your free reward"}'
```

## Run it in Docker

```bash
docker build -t phishing-detector .
docker run -p 8000:8000 phishing-detector
```

## Re-run the analysis (optional — outputs are included)

Place `Phishing_Email.csv` (Kaggle: subhajournal/phishingemails — not committed
here) next to the notebook, then:

```bash
pip install pandas scikit-learn matplotlib jupyter
jupyter execute --inplace phishing_v2.ipynb
```

## Monitoring

The training data's legitimate class has narrow provenance (largely Enron-era
corporate and academic mail — measured and ablated in the notebook), so live
traffic **will** differ from the benchmark. The service makes that visible: every request is logged as a JSON line (verdict, probability, latency, text length), and `/stats` tracks the running predictiondistribution against the held-out baseline.
