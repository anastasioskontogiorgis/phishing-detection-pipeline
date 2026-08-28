# Phishing Email Detection

A binary text classifier for phishing email detection, served through a
containerised API. The repository contains the analysis notebook, the versioned
model artefact it exports, and the service that loads and serves it.

| Path | Contents |
| :--- | :--- |
| `phishing_v2.ipynb` | The analysis, shipped executed. Corpus audit, leakage-free evaluation protocol with the correction quantified, model benchmark, held-out evaluation with calibration, source-artefact ablation, error analysis, out-of-corpus probe study, artefact export. |
| `probes/modern_probes.jsonl` | The thirty probe emails used in Section 8, labelled and tagged by era. |
| `model/` | `phishing_pipeline.joblib`, a TF-IDF and logistic regression pipeline, with `metadata.json` recording the library version, the training protocol and the held-out metrics. |
| `app/main.py` | The service. `POST /predict`, a browser interface at `/`, `/health` for orchestration, `/stats` for monitoring, and structured JSON request logs recording metrics only. |
| `Dockerfile`, `requirements.txt` | The container. The pinned `scikit-learn` version matches the training environment. |

**Held-out performance** under the deduplicate-first protocol, measured on 3,505
test emails: accuracy 0.979, F1 0.972, ROC-AUC 0.997, Brier 0.029.

Those figures describe mail resembling the training corpus. Section 8 of the
notebook measures what happens outside it, and the result materially qualifies
them.

## Findings

**Corpus quality.** Of 18,650 rows, 552 carry
no usable text, including 533 holding the literal string `"empty"` that a model
would memorise as a class signal. A further 3.2% duplicate another row exactly, so
a random split places verbatim copies of training emails into the test set.
Correcting the protocol costs approximately half a percentage point of accuracy.
The leakage is therefore real but modest, and the resulting claim rests on a clean
split rather than a favourable one.

**Provenance dependence.** The strongest negative
coefficients are *enron*, *vince*, *louise*, *linguistics* and the years 2000 to
2002. The legitimate class is drawn overwhelmingly from the Enron corpus and an
academic mailing list collected several years before the phishing samples, so the
classifier is partly determining whether a message is Enron-era corporate or
academic correspondence. Masking those source-identifying tokens costs little,
which establishes that the in-corpus signal is not carried by a handful of proper
nouns.

**Out-of-corpus behaviour.** The ablation does not establish robustness beyond the corpus. A thirty-email probe
set, comprising twenty-four emails in a contemporary register and six era
controls, scores 1.000 on the controls and 0.833 on modern mail. Decomposing the
log-odds identifies the mechanism: vocabulary coverage barely falls, but evidence
supporting the true class approximately halves in both directions while evidence
supporting the incorrect class rises. Predictions contract toward the intercept, so
the model does not become confidently wrong outside its corpus. It ceases to
distinguish between the classes, in precisely the region where the deployed
threshold sits.

## Model selection

Linear SVM leads the benchmark at 0.980 cross-validated F1 against logistic
regression at 0.972, a margin several times the fold-to-fold spread rather than a
tie. Logistic regression is nonetheless the model that ships. The service returns a
calibrated probability, on which the operating threshold, the browser interface and
the `/stats` monitor all depend, whereas `LinearSVC` exposes only
decision-function margins. Its coefficients are also directly interpretable, which
the ablation in Section 5 requires. Wrapping the support vector machine in
`CalibratedClassifierCV` would recover probabilities from the stronger model; that
option is recorded in the notebook and deferred, since the binding constraint on
this model is corpus provenance rather than the final fraction of in-corpus F1.

## Running the service

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The browser interface is then at `http://localhost:8000`. A single prediction:

```bash
curl -X POST localhost:8000/predict \
     -H 'Content-Type: application/json' \
     -d '{"text":"Click here to claim your free reward"}'
```

In a container:

```bash
docker build -t phishing-detector .
docker run -p 8000:8000 phishing-detector
```

## Reproducing the analysis

The notebook outputs are committed, so this step is optional. `Phishing_Email.csv`
is not redistributed here; it is available from Kaggle as
`subhajournal/phishingemails` and should be placed alongside the notebook.

```bash
pip install pandas scikit-learn matplotlib jupyter
jupyter execute --inplace phishing_v2.ipynb
```

The probe set in `probes/` is committed, so Section 8 reproduces without any
additional download.

## Monitoring

The legitimate class has narrow provenance, measured and ablated in Sections 5 and
8, so live traffic will differ from the benchmark distribution. The service makes
that difference visible. Every request is logged as a JSON line recording the
verdict, the probability, the latency and the text length, never the text itself,
and `/stats` tracks the running prediction distribution against the held-out
baseline.

Section 8 identifies what to watch for. Traffic unlike the training corpus produces
predictions clustered around the threshold rather than at the extremes, so a rising
proportion of mid-range probabilities is the earliest available indication that the
operating point requires revision.

## Limitations

The evaluation rests on a single corpus whose phishing-class provenance is
undocumented. The probe study is a diagnostic instrument rather than a measurement:
thirty hand-written emails support no confidence interval, and they were written by
someone already aware of the model's weaknesses.

Establishing the magnitude of the out-of-corpus gap requires an external labelled
corpus, preceded by a near-duplicate audit against this training set, since any
overlap would inflate the result. A threshold study is the second extension. The
0.5 default is a property of this corpus, and the asymmetry between quarantining
legitimate mail and admitting a phishing message is what should determine the
operating point in deployment.
