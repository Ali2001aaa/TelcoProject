# Telco Customer Churn Prediction

Educational end-to-end ML project: predict which customers of a **fictional** telecom company are likely to churn, then serve the model in a Streamlit app.
It is not validated on real operational data and must not be used to make real customer decisions.

> **Status:** code is complete; **results below are placeholders** until you run the notebook on the real dataset.
> Running it writes `models/results_summary.json` (and the final cell prints the same numbers); copy them into the tables here.

## Business problem
A retention team wants to prioritise customers at risk of leaving.
- False negative: a churner is missed (lost revenue). False positive: a loyal customer gets an unnecessary offer.
- **Primary metric: F1 of the churn class** (chosen before modelling). Secondary: precision, recall, ROC-AUC, Average Precision. Accuracy is not used for selection.

## Dataset
- IBM Telco Customer Churn (fictional company): https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
- Source repo: https://github.com/IBM/telco-customer-churn-on-icp4d (archived, code under Apache-2.0)
- **License:** no explicit license for the CSV was found, so it is not committed; the notebook downloads it. Credit IBM.
- Documented size 7,043 x 21; the notebook prints and checks the real shape. TBD: verified shape, churn rate.

## Method
| Step | Choice |
|---|---|
| Split | Stratified 80/20, seed 42; test set untouched until the end; 5-fold stratified CV on train |
| Cleaning | Drop `customerID`; `TotalCharges` text -> float, blanks tested against `tenure == 0` and set to 0 |
| Features | Original vs `original_no_total` vs engineered (`avg_monthly_spend`, `n_services`, `tenure_band`, `autopay`), chosen by predefined CV rule |
| Models | Logistic Regression, Random Forest, HistGradientBoosting (+ random baseline), `class_weight="balanced"` |
| Tuning | GridSearchCV / RandomizedSearchCV on F1 |
| Selection | Permutation importance + full vs reduced model (L1 selector inside the pipeline) |
| Decision | Calibration (sigmoid/isotonic) compared by Brier score, then F1-optimal threshold on out-of-fold predictions |

## Results (fill in after running)
**Selected configuration:** TBD (feature set / model / reduced? / calibration / threshold)

| Model (default params, CV on train) | Train F1 | Val F1 | Gap | Precision | Recall | ROC-AUC | Avg. Precision |
|---|---|---|---|---|---|---|---|
| TBD | | | | | | | |

**Full vs reduced features:** TBD  
**Final test performance (threshold TBD):** F1 TBD, precision TBD, recall TBD, AP TBD, ROC-AUC TBD, 95% bootstrap interval TBD

## Limitations
- Fictional single-snapshot data with no timestamps: no test on future customers; `tenure`/`TotalCharges` may partly reflect the outcome.
- One split of ~7k rows: small score differences between models are not meaningful.
- Threshold maximises F1, not campaign cost.
- Associations only, no causal claims.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-notebook.txt
jupyter lab notebooks/model_development.ipynb           # Run All; creates models/final_model.joblib
streamlit run app.py
```
After training, pin the versions the model was built with (a pickled model needs the same scikit-learn):
```bash
pip freeze | grep -iE "scikit-learn|pandas|numpy|joblib"   # copy into requirements.txt with ==
```

## Deploy on Streamlit Community Cloud
1. Make sure `models/final_model.joblib`, `app.py`, `churn_utils.py` and `requirements.txt` (with pinned versions) are committed and pushed to a **public** GitHub repo.
2. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**.
3. Choose the repo, branch `main`, main file `app.py`, then **Deploy**.
4. No secrets are needed. If you later add any, use the app's *Secrets* settings, never commit them.
5. If the app warns about a scikit-learn version mismatch, fix the pin in `requirements.txt` and reboot.

## GitHub setup
```bash
git init
git add .
git commit -m "Telco churn project: notebook, app, model"
git branch -M main
git remote add origin https://github.com/<your-user>/<repo-name>.git
git push -u origin main
```
Create the empty repository on GitHub first (no README/license), and commit the notebook **with outputs** after running it.

## Structure
```
├── notebooks/model_development.ipynb
├── churn_utils.py          # feature engineering + pipeline builder (shared by notebook and app)
├── app.py
├── models/final_model.joblib   # created by the notebook
├── data/README.md          # how to get the CSV (not committed)
├── requirements.txt / requirements-notebook.txt
├── README.md
└── .gitignore
```
