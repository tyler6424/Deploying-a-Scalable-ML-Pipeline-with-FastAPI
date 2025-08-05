# Model Card

For additional context see the original [Model Card paper](https://arxiv.org/pdf/1810.03993.pdf).

---

## Model Details
* **Algorithm** RandomForestClassifier (200 trees, default depth, `random_state=42`)  
* **Input representation**  
  * Continuous features: `age, fnlgt, education-num, capital-gain, capital-loss, hours-per-week` (used as is)  
  * Categorical features (8): one-hot encoded with `handle_unknown="ignore"`  
* **Output** Binary label  
  * `0 → "<=50K"`  
  * `1 → ">50K"`  
* **Pre-processing** `OneHotEncoder` for categoricals, `LabelBinarizer` for the target.  
* **Framework/Version** scikit-learn 1.3, Python 3.10.18  
* **Repository** <https://github.com/tyler6424/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/actions>

---

## Intended Use
The model is an educational example for Udacity’s *Deploying a Scalable ML Pipeline with FastAPI* project.  
It predicts whether a person’s annual income exceeds \$50 000 based on 1994 US Census features.  
It **is not** intended for real hiring, lending, or welfare decisions.

---

## Training Data
* Source: UCI Adult / Census Income dataset (extracted by Udacity starter kit)  
* Rows: 32 561  
* Time period: 1994 Census survey  
* After cleaning: “?” placeholders replaced with “Unknown”; no rows were dropped.  
* Class distribution: 24 % `>50K`, 76 % `<=50K`

---

## Evaluation Data
A stratified 20 % hold-out split (6,512 rows) from the same dataset.  
No records from the evaluation set are seen during training or hyper-parameter selection.

---

## Metrics
| Metric | Overall | Slice range (min‒max across all categorical values) |
|--------|---------|-----------------------------------------------------|
| Precision | **0.74** | 0.50 – 0.82 |
| Recall    | **0.63** | 0.48 – 0.82 |
| F1-score  | **0.68** | 0.49 – 0.78 |

*Slice metrics* were computed for every unique value in each categorical feature and saved to `slice_output.txt`.  
The lowest F1 (≈ 0.49) occurs for `workclass = Never-worked` (only 1 sample).  
The highest F1 (≈ 0.78) occurs for `workclass = Local-gov`.

---

## Ethical Considerations
* **Bias / Fairness** Income is strongly correlated with race, sex, and marital status.  
  Unequal slice performance indicates the model could systematically under-predict certain groups.  
* **Privacy** Data comes from a public, de-identified dataset; no PII is included.  
* **Misuse potential** Using this model for employment, lending, or insurance decisions could reinforce historical biases and is not recommended.

---

## Caveats and Recommendations
* Trained on 1994 data; socioeconomic trends have shifted. Expect **concept drift** if applied today.  
* Performance may degrade on populations outside the U.S. or younger cohorts.  
* Before any production use, conduct a dedicated fairness audit, update the training data, and monitor slice metrics in real time.  
* For decision-critical applications, pair the model with human oversight and provide explanations of feature contributions (e.g., SHAP values).

---