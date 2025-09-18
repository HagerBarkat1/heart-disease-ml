# 💓 Heart Disease Prediction — Comprehensive ML Pipeline

**Project:** Heart\_Disease\_Project

**Author:** Hager Mahmoud Barkat — SprintUP Graduation Project

**Language:** English

**Status:** Complete — Notebooks, model, Streamlit UI, and deployment notes included

---

## 🚀 Project Summary

This repository contains a full end-to-end machine learning pipeline applied to the UCI Heart Disease dataset. It covers data cleaning, EDA, dimensionality reduction (PCA), feature selection, supervised and unsupervised modeling, hyperparameter tuning, model export, and a Streamlit-based UI for quick demonstrations. The primary goal is to provide a reproducible, well-documented workflow that can be used for learning, evaluation, or further research.

---

## 📁 File Structure

```
Heart_Disease_Project/
│── data/
│   ├── heart_disease.csv
│── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   ├── 06_hyperparameter_tuning.ipynb
│── models/
│   ├── final_model.pkl
│── ui/
│   ├── app.py (Streamlit UI)
│── deployment/
│   ├── ngrok_setup.txt
│── results/
│   ├── evaluation_metrics.txt
│── README.md
│── requirements.txt
│── .gitignore
```

---

## 🧭 Quick Highlights

* Cleaned and preprocessed the UCI Heart Disease dataset.
* Dimensionality reduction and visualization using PCA.
* Feature selection using Random Forest importances, RFE, Chi-Square.
* Supervised models trained: Logistic Regression, Decision Tree, Random Forest, SVM.
* Unsupervised learning: K-Means and Hierarchical Clustering.
* Hyperparameter optimization with `GridSearchCV` and `RandomizedSearchCV`.
* Final model selection and export to `models/final_model.pkl`.
* Streamlit UI (`ui/app.py`) for live inference + deployment notes for Ngrok.

---

## 📦 Requirements & Setup

**Recommended:** Python 3.8+ and a virtual environment.

1. Clone the repository:

```bash
git clone <your-repo-url>
cd Heart_Disease_Project
```

2. Create & activate virtual environment (venv example):

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows (PowerShell)
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. (Optional) If `requirements.txt` is missing or you want a minimal list, install these core packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit imbalanced-learn joblib
```

---

## 🗂 Data

**Source:** UCI Heart Disease Dataset
(Original link: `https://archive.ics.uci.edu/ml/datasets/heart+Disease`)
Place the cleaned or original CSV file in `data/heart_disease.csv`.

**Note on provided data:** The notebooks expect a file path `/content/processed.cleveland.data` in the notebooks — if you're running locally, confirm that `data/heart_disease.csv` or `processed.cleveland.data` exists and update the path in the notebook(s) accordingly.

---

## 🛠 Notebooks — Order & Purpose

Open the notebooks in the order below to reproduce the entire analysis and model training flow:

1. `01_data_preprocessing.ipynb` — data import, cleaning, missing-value handling, encoding, scaling, EDA.
2. `02_pca_analysis.ipynb` — PCA, explained variance, scatter visualizations.
3. `03_feature_selection.ipynb` — feature importance (Random Forest), RFE, Chi-Square.
4. `04_supervised_learning.ipynb` — model training (Logistic, DT, RF, SVM), evaluation metrics, ROC curves.
5. `05_unsupervised_learning.ipynb` — K-Means elbow method, hierarchical clustering (dendrogram).
6. `06_hyperparameter_tuning.ipynb` — GridSearchCV and RandomizedSearchCV to optimize models.

Each notebook is annotated with code cells and explanations. If you use Colab, upload the `data/` folder or mount Google Drive and adjust file paths accordingly.

---

## 🔬 Model Pipeline & Artifacts

* The training pipeline uses StandardScaler on numeric inputs, one-hot encoding for selected categorical features, and feature selection to build the final `X_final` used for modeling.
* The saved model artifact is `models/final_model.pkl` (pickle). This file contains the best estimator selected at the end of the tuning stage.

**Important:** For production or safer reproducibility, prefer exporting a full pipeline (preprocessing + model) (e.g. `sklearn.pipeline.Pipeline`) so inputs are automatically preprocessed before prediction. Current repo contains a model saved alone — be sure to replicate preprocessing steps in any deployment code (the provided `ui/app.py` prepares features in the expected order).

---

## 🧩 Streamlit UI — `ui/app.py`

This lightweight UI demonstrates model inference and explains expected input mapping.

**How to run locally:**

```bash
cd ui
streamlit run app.py
```

**What it expects:**

* `models/final_model.pkl` must be present in the same directory where `streamlit run` is executed (or modify the path in `app.py`).
* `app.py` expects these features in this exact order (matching model training):

  ```python
  final_features = ['age', 'chol', 'thalach', 'oldpeak', 'cp_4.0']
  ```

  Where `cp_4.0` is a binary flag (1 if chest-pain type == 4, else 0).

**UI Notes & Disclaimers:**

* The app displays both predicted class and probability (if model supports `predict_proba`).
* A medical disclaimer is shown in the UI: this is an educational demo only and should not be used for real clinical decision-making.

---

## 🌐 Deploying with Ngrok (Local Preview)

A simple deployment recipe is included at `deployment/ngrok_setup.txt`.

Typical steps:

1. Run the Streamlit app: `streamlit run ui/app.py`.
2. In a new terminal, run `ngrok http 8501` (Streamlit default port is `8501`).
3. Use the provided public URL from ngrok to share the running app temporarily.

**Security note:** Ngrok exposes a tunnel to your local machine. Do not expose sensitive data or credentials. Use ngrok for demo/testing only.

---

## ✅ Reproduce Training & Create New `final_model.pkl`

If you want to retrain and export a new model:

1. Run all notebooks in order or execute the training cells in `06_hyperparameter_tuning.ipynb` to obtain `tuned_models`.
2. Evaluate `final_results_df` and choose the best model by F1-score (or AUC).
3. Save the best model with pickle or joblib:

```python
import pickle
with open('models/final_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
```

***Preferable approach:*** Save a pipeline including preprocessing:

```python
from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(scaler, selector_if_any, best_model)
pickle.dump(pipeline, open('models/pipeline_model.pkl', 'wb'))
```

This prevents mismatch between training preprocessing and runtime inputs.

---

## 📊 Evaluation & Results

* Evaluation metrics and detailed model comparison are stored in `results/evaluation_metrics.txt` and also printed in notebooks.
* The notebooks include ROC curves, confusion matrices, precision/recall/F1, and cluster comparison tables.

If you plan to present results, export figures from the notebooks as PNG or PDF and include them in a `docs/` folder.

---

## 🧰 Development Notes & Tips

* Ensure consistent feature ordering between training and inference — the Streamlit UI depends on that order.
* The original dataset contains placeholder values (`'?'`) for `ca` and `thal` in some rows. Notebooks convert them to `NaN` and impute.
* Outlier removal is performed in the notebooks using IQR; re-evaluate if you change this step.
* SMOTE is used before hyperparameter tuning to address class imbalance — if you change resampling, validate the model performance again.

---

## ⚠️ Troubleshooting

**`ModuleNotFoundError`** — install the missing package with `pip install <package>` or `pip install -r requirements.txt`.
**`FileNotFoundError: final_model.pkl`** — make sure `models/final_model.pkl` is saved and the path in `app.py` points to it.
**Mismatch in feature columns** — verify `final_features` defined in notebooks matches `app.py`. If you change feature selection, update `app.py` accordingly.

---

## 🔁 How to Update the Model in the UI

1. Retrain or load a new `pipeline_model.pkl` that contains preprocessing + estimator.
2. Replace `models/final_model.pkl` or update `app.py` to load `pipeline_model.pkl`.
3. Restart Streamlit (`Ctrl+C` to stop and re-run `streamlit run ui/app.py`).

---

## 🧾 Requirements (example `requirements.txt`)

```
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
imbalanced-learn
joblib
pickle-mixin
```

Adjust versions as needed (pin versions for reproducibility, e.g. `scikit-learn==1.2.2`).

---

## 📜 License

Add a `LICENSE` file as appropriate. For academic/demo projects, consider `MIT License` for permissive reuse.

---

## 🧑‍💻 Contribution

Contributions and improvements are welcome. Suggested workflow:

1. Fork repository
2. Create a feature branch
3. Submit a pull request with clear descriptions and tests (if any)

---

## 📬 Contact

If you have questions or want collaboration, add your contact details here (email, LinkedIn).

---

## FAQ (Short)

**Q: Is this app clinically validated?**
A: No. This is a demo/educational project only. Do not use for real medical decisions.

**Q: How to extend to other datasets?**
A: Replace `data/heart_disease.csv`, re-run preprocessing and feature-selection notebooks, retrain models, and export a new `final_model.pkl`.

---

## ✨ Final Notes (Best Practices)

* For productionize-ready systems, wrap preprocessing + prediction in a single serialized artifact (Pipeline), add unit tests, logging, and input validation.
* Keep sensitive patient data out of public repositories. Use synthetic or anonymized samples for demos.

---

*If you want, I can also:*

* Generate a compact `README` badge header and a short project poster.
* Produce a `requirements.txt` with exact pinned versions based on your environment.
* Convert this README to Arabic or another language.
