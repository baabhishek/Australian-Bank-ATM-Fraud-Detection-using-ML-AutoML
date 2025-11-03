# Australian-Bank-ATM-Fraud-Detection-using-ML-AutoML
![Banking Project_Banner](https://res.cloudinary.com/momentum-media-group-pty-ltd/image/upload/c_fill,q_auto:good,f_auto,e_unsharp_mask:80,w_828,h_400/The%20Adviser%2FNews_stories%2FBank-Australia-ta)

##  Problem Statement
PredCatch Analytics’ Australian banking client is facing severe profitability and reputation loss due to fraudulent ATM transactions.
They want PredCatch to help them detect and prevent such fraudulent transactions in real time using a predictive model.

## Key Challenge:
The dataset is highly imbalanced, i.e., the number of fraudulent transactions is very small compared to normal transactions.

---
## Data Description

- Training Data: Contains masked variables for each transaction.
- Target Variable: Target
- 1 → Fraudulent Transaction
- 0 → Clean Transaction

## Dataset Structure:

- Each row → One transaction
- Each column → Information (features) about that transaction
- geo_scores,
- Lambda_wts
- Qset_tats
- instance_scores → External supporting datasets
---
# Approach & Methodology
## Exploratory Data Analysis (EDA)

- Conducted extensive data cleaning and inspection using SweetViz for automated EDA reporting.
- Analyzed distribution of key features and correlation between transaction-level indicators.
- Detected and handled missing values and data inconsistencies.

## Data Preprocessing

- Feature scaling and transformation.
- Encoding categorical variables.
- Addressed data imbalance using SMOTE (Synthetic Minority Oversampling Technique).

## Model Building

- Baseline Models: Random Forest
- Overfitting Detection:
**Cross-validation used to address overfitting.**
**After applying cross-validation, model performance stabilized —**

- Train vs Test difference < 0.01% → statistically negligible.

- Low standard deviation → stable & consistent across all folds.

- Cross-val training accuracy: 99.95%
---
## AutoML with PyCaret

- Implemented PyCaret to automate model selection and hyperparameter tuning.

- Best model identified: Quadratic Discriminant Analysis (QDA)

---
## Tools & Technologies Used

- Languages: Python

- Libraries: **Pandas, NumPy, Scikit-learn, PyCaret, SweetViz, Matplotlib, Seaborn**

- Techniques: EDA, Feature Engineering, SMOTE, Model Evaluation, Cross Validation, AutoML

--
## Insights & Observations

- Fraudulent transactions have unique behavioral patterns captured by the model.

- Geo-score & Lambda weights play a major role in fraud likelihood.

- AutoML significantly accelerated model tuning and improved recall without compromising stability.
