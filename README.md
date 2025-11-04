# Predicting Next Exam Paper Topics Using Machine Learning

## 1. Problem Definition

The objective of this project was to predict the topic distribution of the next exam paper based on previous years’ exams. Each exam consisted of three sections:

* **1-mark questions:** 6 topics
* **3-mark questions:** 3 topics
* **5-mark questions:** 2 questions with an “OR” option (4 topics total)

Each exam also had an associated difficulty level.
The central goal was to model how topics transition from one exam to the next, enabling the system to learn topic recurrence patterns and generate likely topic combinations for the next paper.

---

## 2. Dataset Preparation

The dataset was loaded from a CSV file containing columns for exam number, difficulty, and topic strings under each section.

**Preprocessing Steps:**

1. **Topic Splitting:**
   Each topic string was split by commas into individual topic names.

2. **Reshaping (Melt and Explode):**
   The dataset was converted to a long format using `melt` and `explode`, ensuring each topic became a separate row.

3. **One-Hot Encoding:**
   Topics were one-hot encoded to produce binary columns (`Topic_<name>`), where 1 indicated the topic’s presence in an exam.

4. **Reaggregation:**
   Data was regrouped by exam number, category, and difficulty to form a wide-format dataset — one row per exam-category pair.

---

## 3. Creating Sequential Exam Pairs

To predict the next exam (*n+1*) from the previous exam (*n*), data was paired sequentially.
For each exam category:

* Features (`X_`) represented topics in exam *n*
* Labels (`Y_`) represented topics in exam *n+1*

This transformation created a **multi-label classification** problem, where each target column denoted the presence or absence of a topic in the next exam.

---

## 4. K-Nearest Neighbors (KNN)

The **KNN model** was trained using cosine distance to measure topic similarity between exams.

### Results:

```
Best K found: 1
Exact Match Accuracy: 0.111
Hamming Loss: 0.284
F1 Score (micro): 0.685
```

### Predicted Exam (KNN-based):

**1-Mark Questions (6):** AboutML, FSS, KNN, LDA, Linear, ROC
**3-Mark Questions (3):** BSS, K-Cross Fold, Linear
**5-Mark Questions (2, each with OR):**
Q1: KNN OR Linear
Q2: Logistic OR ROC

KNN achieved moderate F1 but low exact match accuracy. It often failed to generalize due to the sparse, high-dimensional topic space, though it produced a reasonable next-exam pattern when trained on all available data.

---

## 5. Logistic Regression (Multi-Label)

**Logistic Regression** using `MultiOutputClassifier` remained the strongest performer overall.

### Results:

```
Exact Match Accuracy: 0.100
Hamming Loss: 0.233
F1 Score (micro): 0.753
```

### Predicted Exam (Logistic-based):

**1-Mark Questions (6):** ROC, KNN, Linear, AboutML, FSS, LDA
**3-Mark Questions (3):** K-Cross Fold, Logistic, BSS
**5-Mark Questions (2, each with OR):**
Q1: ROC OR Linear
Q2: Logistic OR LDA

Logistic regression performed consistently, producing interpretable probabilities and realistic predictions. It remained stable and efficient despite limited data.

---

## 6. Support Vector Machine (SVM)

The **SVM model** used the RBF kernel with probabilistic outputs via `MultiOutputClassifier`.

### Results:

```
Exact Match Accuracy: 0.100
Hamming Loss: 0.289
F1 Score (micro): 0.690
```

### Predicted Exam (SVM-based):

**1-Mark Questions (6):** AboutML, ROC, Linear, KNN, Logistic, FSS
**3-Mark Questions (3):** K-Cross Fold, Logistic, LDA
**5-Mark Questions (2, each with OR):**
Q1: ROC OR Linear
Q2: Logistic OR LDA

SVM captured non-linear topic relationships but did not significantly outperform logistic regression. It was slower to train and yielded similar predictions.

---

## 7. Linear Discriminant Analysis (LDA)

LDA was introduced as a dimensionality reduction–driven classifier capable of identifying linear class separability.

### Results:

```
Exact Match Accuracy: 0.100
Hamming Loss: 0.189
F1 Score (micro): 0.790
```

### Predicted Exam (LDA-based):

**1-Mark Questions (6):** ROC, AboutML, KNN, FSS, Linear, LDA
**3-Mark Questions (3):** K-Cross Fold, BSS, Logistic
**5-Mark Questions (2, each with OR):**
Q1: ROC OR Logistic
Q2: Linear OR LDA

LDA achieved the **highest F1 score**, showing it effectively captured topic dependencies despite limited data. Its probabilistic structure made its predictions interpretable and stable.

---

## 8. Random Forest Classifier

The **Random Forest** model aimed to test ensemble-based prediction over sparse features.

### Results:

```
Exact Match Accuracy: 0.200
Hamming Loss: 0.256
F1 Score (micro): 0.723
```

### Predicted Exam (Random Forest-based):

**1-Mark Questions (6):** N/A, N/A, N/A, N/A, N/A, N/A
**3-Mark Questions (3):** N/A, N/A, N/A
**5-Mark Questions (2, each with OR):**
Q1: N/A OR N/A
Q2: N/A OR N/A

Despite a fair F1 score, Random Forest failed to output valid predictions due to its inability to produce meaningful probabilities across sparse binary labels.

---

## 9. Observations and Conclusion

### Overall Model Comparison:

| Model               | Exact Match Accuracy | Hamming Loss | F1 Score (micro) | Remarks                                     |
| ------------------- | -------------------- | ------------ | ---------------- | ------------------------------------------- |
| KNN                 | 0.111                | 0.284        | 0.685            | Struggled with sparse data                  |
| Logistic Regression | 0.100                | 0.233        | 0.753            | Best balance of accuracy & interpretability |
| SVM                 | 0.100                | 0.289        | 0.690            | Similar to Logistic, slower                 |
| Random Forest       | 0.200                | 0.256        | 0.723            | Overfit and output N/A predictions          |
| LDA                 | 0.100                | 0.189        | **0.790**        | Highest F1, stable and interpretable        |

---

## 10. Final Conclusion

Among all tested models, **LDA** achieved the highest F1 score, indicating the best balance of precision and recall in predicting next-exam topics.
However, **Logistic Regression** remained the most practical and interpretable model overall, offering stability, clarity, and consistent probabilistic predictions.

**Final Insight:**
When predicting sequential topic patterns from small, sparse educational datasets, simple linear models (Logistic Regression, LDA) outperform more complex or distance-based models (KNN, SVM, Random Forest).
