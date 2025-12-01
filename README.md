# Autoencoder-based Network Intrusion Detection (UNSW-NB15)

This project builds and compares **supervised classifiers** and an **unsupervised autoencoder** to detect network intrusions using the **UNSW-NB15** dataset.  
The goal is to see how a classical ML approach (Decision Tree, Naive Bayes, Random Forest) stacks up against an autoencoder used for anomaly detection on network traffic.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Problem Formulation](#problem-formulation)
- [Methodology](#methodology)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Data Preprocessing](#data-preprocessing)
  - [Supervised Models](#supervised-models)
  - [Autoencoder for Anomaly Detection](#autoencoder-for-anomaly-detection)
- [Results](#results)
  - [Supervised Models vs Autoencoder](#supervised-models-vs-autoencoder)
  - [Interpretation](#interpretation)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Limitations & Future Work](#limitations--future-work)
- [References](#references)

---

## Project Overview

Modern networks generate huge volumes of traffic, and manually spotting intrusions (attacks) is basically impossible.  
This project builds a **binary intrusion detection system** that labels flows as:

- `0` – normal traffic  
- `1` – malicious / attack traffic  

I train and evaluate:

- **Decision Tree**
- **Naive Bayes**
- **Random Forest**
- **Tuned Random Forest (with GridSearchCV)**
- **Autoencoder (MLPRegressor) for anomaly detection**

The interesting part is comparing **supervised models** (which need labeled attacks) with an **unsupervised autoencoder** trained only on benign traffic.

---

## Dataset

The project uses the **UNSW-NB15** dataset, a modern network intrusion dataset containing both normal and multiple attack types.

In the notebook, the provided **train** and **test** splits are used:

- Training set shape: `(175341, 44)`  
- Test set shape: `(82332, 44)`

Features include:

- **Network flow statistics** (bytes, packets, duration, rate)
- **Connection state features** (e.g., `ct_state_ttl`)
- **Protocol and service fields**
- A binary **`label`** indicating normal vs malicious traffic

You’ll need to download the UNSW-NB15 CSV files (train and test) from the official source and place them in the appropriate folder before running the notebook.

---

## Problem Formulation

- **Input:** Tabular network flow data from UNSW-NB15  
- **Output:** Binary label (0 = normal, 1 = attack)  

Two perspectives are studied:

1. **Supervised classification**  
   - Train on labeled normal + attack traffic  
   - Use metrics like Accuracy, Precision, Recall, F1-score, ROC-AUC

2. **Unsupervised anomaly detection (Autoencoder)**  
   - Train only on **benign** traffic  
   - Use **reconstruction error** (MSE) to flag anomalies (attacks)

---

## Methodology

### Exploratory Data Analysis (EDA)

Some of the EDA steps in the notebook:

- **Class distribution** of the target label (normal vs malicious)
- **Histograms** of numerical features (e.g. bytes, rate, duration)
- **Boxplots** and comparisons of key features such as:
  - `sbytes` vs label
  - `dbytes` vs label
  - `ct_state_ttl` vs label
  - `rate` vs label

These visualizations help show that:

- The dataset is **imbalanced**.
- Certain numerical features have very skewed distributions.
- Attack traffic often has distinct patterns in bytes, rates, and connection state.

---

### Data Preprocessing

The preprocessing pipeline includes:

1. **Handling missing values**  
   - Checked for missing values in both train and test sets.
   - Dataset had no problematic NaNs, so no heavy imputation was required.

2. **Outlier detection / inspection**  
   - Numeric distributions were plotted to understand skewness and outliers.

3. **Encoding categorical variables**  
   - Categorical columns (like protocol, service, state) were encoded into numeric form (e.g. one-hot encoding).
   - After encoding, the feature space expands significantly (around 195 features in the encoded train/test matrices).

4. **Train–Test split**  
   - The dataset came pre-split into train and test; those were used for all experiments.

5. **Class balancing with SMOTE (for supervised models)**  
   - The train set had **more malicious than normal** samples, so **SMOTE** was applied on the training data to balance the classes.
   - This helps the classifiers not be biased toward the majority class.

6. **Scaling / Normalization**  
   - A `StandardScaler` is fit on the training set and applied to both train and test.
   - This is especially important for distance-based behavior and for the autoencoder.

---

### Supervised Models

Four supervised models are trained on the **balanced and scaled** training data:

1. **Decision Tree**
2. **Naive Bayes**
3. **Random Forest**
4. **Tuned Random Forest**
   - Hyperparameters tuned using **GridSearchCV** with cross-validation.
   - Parameters searched include: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`.
   - Tuning is done using **F1-score** as the main metric.

Each model is evaluated on the **held-out test set** using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- (Plus extra metrics like MCC, Cohen’s Kappa, Balanced Accuracy in a summary table.)

---

### Autoencoder for Anomaly Detection

An **autoencoder** is used as an unsupervised anomaly detector:

- Implemented using `MLPRegressor` from scikit-learn.
- Architecture: a **symmetric feed-forward network**  
  Example layer configuration:
  - Input → 128 → 64 → 32 → 64 → 128 → Output  
- Activation: **ReLU**
- Optimizer: **Adam**
- Trained for a limited number of iterations (e.g. 80 epochs) on **only benign (label = 0) traffic**.

**Workflow:**

1. Filter training data to keep only normal flows.
2. Encode and scale features.
3. Train the autoencoder to reconstruct its input.
4. On the test set:
   - Compute reconstruction error (MSE) per sample.
   - Use ROC curve to pick an **optimal threshold** for the reconstruction error.
   - If `MSE > threshold` → classify as **attack (1)**, else **normal (0)**.

The intuition:

- The autoencoder learns “what normal traffic looks like”.
- Attack traffic is “weird”, so it reconstructs poorly and gets flagged as anomalous.

---

## Results

### Supervised Models vs Autoencoder

A combined evaluation table (from the notebook) compares all models on the same test set:

| Model              | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|--------------------|----------|-----------|--------|----------|---------|
| Decision Tree      | 0.530    | 0.618     | 0.385  | 0.475    | 0.546   |
| Naive Bayes        | 0.603    | 0.970     | 0.289  | 0.445    | 0.689   |
| Random Forest      | 0.703    | 0.751     | 0.690  | 0.719    | 0.824   |
| Tuned Random Forest| 0.695    | 0.750     | 0.670  | 0.708    | 0.826   |
| Autoencoder        | 0.582    | 0.582     | 0.856  | 0.693    | 0.407   |

Key observations:

- **Best supervised model:**  
  - **Random Forest** has the **highest overall performance** among supervised models (accuracy ≈ 0.703, F1 ≈ 0.719, ROC-AUC ≈ 0.823).
  - The **tuned Random Forest** slightly improves ROC-AUC (≈ 0.826) but with a tiny drop in accuracy/F1 due to less overfitting.

- **Autoencoder behavior:**
  - Very high **recall** (~0.856) for attacks.
  - Competitive **F1-score** (~0.693) but lower overall accuracy (~0.582) and low ROC-AUC (~0.407, because of how the anomaly scores are used).
  - This means it catches most attacks but raises more false positives.

---

### Interpretation

**Supervised models:**

- **Decision Tree**
  - Simple but struggles with high-dimensional data.
  - Lower recall and accuracy → prone to overfitting and weaker generalization.

- **Naive Bayes**
  - Very high precision but poor recall on attacks.
  - Strong independence assumptions between features hurt performance on complex network behavior.

- **Random Forest (and tuned RF)**
  - Handles mixed numerical + categorical features well.
  - Captures nonlinear relationships and interactions.
  - Provides the **best trade-off** between precision and recall.
  - Works great **when attack patterns in training are representative of test attacks**.

**Autoencoder:**

- Trained only on **benign** data → specializes in “normal” behavior.
- Excellent at detecting **unknown or novel attacks** that deviate from normal patterns.
- High recall makes it very attractive for **security monitoring**, where missing attacks is more costly than having some extra alerts.
- However, lower accuracy and higher false positive rate mean it’s not ideal as a standalone detector.

**Big picture:**

- **Random Forest** is best for **known attacks**.
- **Autoencoder** is best for **unknown or emerging attacks**.
- A **hybrid system** (RF + Autoencoder) would be more robust:
  - RF handles “known” threats learned from labels.
  - Autoencoder flags weird, unseen behavior for further investigation.

---

## How to Run

### 1. Clone the repository

```bash
git clone <your-repo-url>.git
cd <your-repo-name>
```

### 2. Install dependencies

Create a virtual environment (optional but recommended), then install Python packages.

Example (conda):

```bash
conda create -n ids-autoencoder python=3.10
conda activate ids-autoencoder
pip install -r requirements.txt
```

If you don’t have a `requirements.txt` yet, you can include packages like:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `imbalanced-learn`
- `jupyter` / `notebook`

### 3. Download the data

1. Download the **UNSW-NB15** train/test CSV files from the official source.
2. Place them in a folder such as:

```
data/
  UNSW_NB15_training-set.csv
  UNSW_NB15_testing-set.csv
```

3. Update the paths in the notebook if needed.

### 4. Run the notebook

Launch Jupyter and open the notebook:

```bash
jupyter notebook
```

Then open:

- `Autoencoder.ipynb`

Run all cells in order to:

1. Load and preprocess the data.
2. Train supervised models.
3. Train the autoencoder.
4. Evaluate and compare results.

---

## Project Structure

A simple structure for the repo could be:

```text
.
├── Autoencoder.ipynb        # Main notebook with all experiments
├── data/                    # Folder where UNSW-NB15 CSVs live (not committed if large)
├── figures/                 # (Optional) Saved plots
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## Limitations & Future Work

**Limitations:**

- Dataset is **imbalanced** and complex; class imbalance still exists in the test set.
- Binary label collapses many attack types into one class → less granularity.
- Autoencoder threshold selection is based on one test split; could be improved.
- Random Forest can be computationally heavy with many trees and features.

**Possible future improvements:**

- Try **other anomaly detection models** (Isolation Forest, One-Class SVM, deep autoencoders in PyTorch/Keras).
- Use **multi-class attack labels** instead of binary, to see which types are hardest to detect.
- Experiment with **different autoencoder architectures**, dropout, and more epochs.
- Deploy as a simple **API** (e.g. FastAPI) that takes flow features and returns a prediction.
- Add **explainability** (feature importance, SHAP) to better understand why attacks are flagged.

---

## References

- UNSW-NB15 dataset (network intrusion dataset)  
- scikit-learn documentation (DecisionTreeClassifier, RandomForestClassifier, MLPRegressor, GridSearchCV)  
- imbalanced-learn documentation (SMOTE)
