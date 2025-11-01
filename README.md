# Predicting Next Exam Paper Topics Using Machine Learning

## 1. Problem Definition
The objective of this project was to predict the topic distribution of the next exam paper based on previous years’ exams. Each exam consisted of three sections:  
- **1-mark questions:** 6 topics  
- **3-mark questions:** 3 topics  
- **5-mark questions:** 2 questions with an “OR” option (4 topics total)  

Each exam also had an associated difficulty level.  
The central idea was to model how topics transition from one exam to the next, allowing the system to learn patterns and generate a likely topic set for the next paper.

---

## 2. Dataset Preparation
The dataset was read from a CSV file containing columns for exam number, difficulty, and topic strings for each category.

**Preprocessing Steps:**
1. **Topic Splitting:**  
   Topic strings were split by commas to form clean lists of individual topics under each category.

2. **Reshaping (Melt and Explode):**  
   The dataset was reshaped into a long format using the `melt` function, and the topic lists were exploded so every topic became a separate row.

3. **One-Hot Encoding:**  
   Topics were converted into binary indicator columns using one-hot encoding. Each topic was represented as `Topic_<name>` with a value of 1 if that topic appeared in the given exam.

4. **Reaggregation:**  
   The data was grouped again by exam number, category, and difficulty, producing a clean wide-format dataset where each row represented one exam-category combination.

---

## 3. Creating Sequential Exam Pairs
To predict the next exam from the previous one, the dataset was organized into consecutive exam pairs.  
For each category, the input features corresponded to exam *n* and the target labels corresponded to exam *n+1*.

Each feature column was prefixed with `X_`, and each target column with `Y_`, resulting in rows that represented transitions from one exam to the next.  
The dataset thus became a **multi-label classification problem**, where each output column (topic) represented a binary label to predict whether that topic would appear in the next exam.

---

## 4. K-Nearest Neighbors (KNN)
The first approach used **K-Nearest Neighbors (KNN)** with cosine distance as the metric.  
Although KNN could handle the multi-label structure, the performance was inconsistent due to the small dataset and the high-dimensional, sparse feature space.  

---

## 5. Logistic Regression (Multi-Label)
The next approach used **Logistic Regression** with `MultiOutputClassifier`, which trains one logistic model per topic.

**Evaluation Metrics:**
- Exact Match Accuracy  
- Hamming Loss  
- Micro-Averaged F1 Score  

Logistic Regression provided the most stable and interpretable results, performing significantly better than KNN.  
To generate the predicted next exam, the model took the last known exam as input, computed probabilities for each topic, and selected the top-N topics per category:
- 6 for 1-mark  
- 3 for 3-mark  
- 4 for 5-mark (2 OR pairs)

The final output resembled realistic exam topic combinations.

---

## 6. Support Vector Machine (SVM)
**SVM** was implemented using `SVC(probability=True)` wrapped in `MultiOutputClassifier` to allow probabilistic outputs.  
While the structure and prediction flow were identical to logistic regression, SVM was computationally heavier and provided no significant improvement in accuracy.  
Both models performed comparably, but Logistic Regression was faster, more stable, and easier to interpret for small datasets.

---

## 7. Random Forest Classifier
A **Random Forest-based MultiOutputClassifier** was implemented to test whether a non-linear ensemble model could capture more complex dependencies between topics.  

**Results:**
- Exact Match Accuracy: ~0.1  
- High Hamming Loss  
- All predictions returned “N/A”  

The poor performance was due to:
1. Small dataset with few sequential pairs for training  
2. Extremely sparse topic columns  
3. Random Forest’s reliance on larger and more diverse datasets  

Even after parameter tuning, the model continued to underperform.

---

## 8. Observations and Conclusion
Across all models, **Logistic Regression** emerged as the most effective approach.  
It was well-suited for small datasets, provided interpretable coefficients, and generated meaningful probability-based topic predictions.

**Model Performance Summary:**
- **KNN:** Struggled due to sparse, high-dimensional input.  
- **Logistic Regression:** Stable, accurate, and interpretable.  
- **SVM:** Similar accuracy but higher computation cost.  
- **Random Forest:** Failed due to sparse and limited data.  

The experiment demonstrated that a carefully prepared dataset and a simple, interpretable linear model can outperform more complex algorithms when data is limited.  
**Logistic Regression remains the preferred model** for predicting next exam topics based on previous patterns.

---
