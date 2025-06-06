
# 1. Introduction

data link: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

Stroke remains one of the leading causes of death and long‑term disability worldwide. Early identification of high‑risk individuals is critical for implementing timely interventions, reducing healthcare costs, and improving patient outcomes. Traditional risk‑scoring systems often rely on a limited set of covariates and yield “black‑box” predictions, which can be difficult for clinicians to interpret and trust. To address these challenges, we leverage a rich, real‑world dataset and a probabilistic modeling framework that balances predictive power with transparency.

The Stroke Prediction Dataset compiles anonymized patient information spanning demographics (age, gender, marital status, residence), clinical measurements (BMI, average glucose level), lifestyle factors (smoking status, work type), and comorbidities (hypertension, heart disease). The binary target indicates whether a patient has experienced a stroke. This diversity of features offers a comprehensive view of the biological, socioeconomic, and behavioral drivers that contribute to cerebrovascular risk.

Our objective is to build a Bayesian Network that estimates the probability of stroke for individual patients, while simultaneously revealing the conditional dependencies among risk factors. By combining structure learning (to uncover the network of relationships) with parameter estimation (to quantify those relationships), we create an interpretable model suited for clinical decision support. During inference, clinicians can query the network to assess how changes in modifiable factors—such as smoking cessation or weight management—alter a patient’s stroke risk, thereby guiding personalized prevention strategies.

Using a dataset from Kaggle, “Stroke Prediction Dataset”, we aim to make a prediction model that predicts whether a patient will get a stroke or not. This dataset has a 5,110 rows and 12 columns of real patient data. This includes each patient’s demographic information, biological and socioeconomical attributes such as gender, age, whether or not he/she is married, BMI, hypertension, and disease history. We set our outcome variable to be whether a patient will have a stroke or not.

# 2. Data profile & preprocessing & EDA 

### 2.1 Data Profile

#### Dataset Overview
- **Rows:** 5,110  
- **Columns:** 12  
- **Description:** Each record describes a patient’s demographics, lifestyle, clinical measurements, and a binary `stroke` outcome (0 = no stroke, 1 = stroke).

#### Missing Values
- **`bmi`**: 201 missing (3.94% of rows)  
- All other columns are complete.

#### Attribute Information

| Column               | Type        | Description                                                           |
|----------------------|-------------|-----------------------------------------------------------------------|
| `id`                 | Integer     | Unique identifier (drop before modeling)                              |
| `gender`             | Categorical | Male, Female, or Other                                                |
| `age`                | Numeric     | Age of the patient (years)                                            |
| `hypertension`       | Binary      | 0 = no hypertension, 1 = has hypertension                             |
| `heart_disease`      | Binary      | 0 = no heart disease, 1 = has heart disease                           |
| `ever_married`       | Categorical | Yes or No                                                             |
| `work_type`          | Categorical | Private, Self‑employed, Govt_job, children, or Never_worked           |
| `Residence_type`     | Categorical | Urban or Rural                                                        |
| `avg_glucose_level`  | Numeric     | Average blood glucose level (mg/dL)                                   |
| `bmi`                | Numeric     | Body mass index (kg/m²)                                               |
| `smoking_status`     | Categorical | never smoked, formerly smoked, smokes, or Unknown                     |
| `stroke`             | Binary      | 0 = no stroke, 1 = had a stroke                                       |

#### Categorical Distributions
- **gender** (n=5,110)  
  - Female: 2,994 (58.6%)  
  - Male:   2,115 (41.4%)  
  - Other:     1 (< 0.1%)

- **ever_married**  
  - Yes: 3,353 (65.6%)  
  - No:  1,757 (34.4%)

- **work_type**  
  - Private:       2,925 (57.2%)  
  - Self‑employed:   819 (16.0%)  
  - children:        687 (13.4%)  
  - Govt_job:        657 (12.9%)  
  - Never_worked:     22 (0.4%)

- **Residence_type**  
  - Urban: 2,596 (50.8%)  
  - Rural: 2,514 (49.2%)

- **smoking_status**  
  - never smoked:    1,892 (37.0%)  
  - Unknown:         1,544 (30.2%)  
  - formerly smoked:   885 (17.3%)  
  - smokes:            789 (15.4%)

### 2.2 Key Observations
- **Imbalanced target:** Only ~5% of patients had a stroke. Consider SMOTE, class weights, or focal loss in modeling.  
- **Missing BMI values:** Impute (e.g., median by age/gender) or treat “Unknown” as its own category.  
- **Outliers present:** Extremely high `avg_glucose_level` and `bmi` values warrant further review.  
- **ID column:** Drop before modeling—no predictive value.  
- **Feature engineering ideas:**  
  - Bin `age` into categories (e.g., child, adult, senior).  
  - Create a “missing” indicator for `smoking_status = Unknown`.  

# 3. EDA

In EDA, we explored the potential relationships between each variable and the risk of stroke. We checked the conditional probability of stroke given each variable’s status, and drew a KDE plot for the continuous variables. Key findings include that categorical variables like having hypertension and heart disease appear to be strongly related to the risk of stroke, and continuous variables like age show different distributions depending on whether a person has had a stroke or not. Based on these findings, we applied them to our data preprocessing and formulated hypotheses about which factors most significantly affect the likelihood of stroke.

### Categorical Variable
#### Are strokes more common among people with heart disease or hypertension?

##### Yes
  
<img width="213" alt="image" src="https://github.com/user-attachments/assets/9c47ffa7-239b-4f84-a488-bc11b3fa7a73" /> <img width="223" alt="image" src="https://github.com/user-attachments/assets/c74903e1-68f7-4ada-adf6-8869407115bc" />

#### Is the risk of stroke related to gender, marriage status, or residence type?

##### No

<img width="183" alt="image" src="https://github.com/user-attachments/assets/14b41d58-f71d-4f83-ab74-f2d4a0750c72" /> <img width="206" alt="image" src="https://github.com/user-attachments/assets/f8eedc94-3868-498f-9237-867dec11c5c1" /> <img width="212" alt="image" src="https://github.com/user-attachments/assets/2b6d817b-5d75-4f05-8eac-baedd5f96df3" /> 
 
### Continuous Variable

#### Is age related to the risk of stroke?

##### Age appears to be strongly related to the risk of stroke.
<img width="345" alt="image" src="https://github.com/user-attachments/assets/5c9e1eeb-34fa-410d-825a-77d2d8f99101" /> 

#### Is BMI related to the risk of stroke?

##### BMI doesn't seem to be strongly related to the risk of stroke.
<img width="331" alt="image" src="https://github.com/user-attachments/assets/2edf0002-2862-4a87-8fe6-f5ba7ed62a8b" />

#### How is glucose level related to the risk of stroke?
##### The average glucose level seems to affect the risk of stroke depending on its interval.
<img width="349" alt="image" src="https://github.com/user-attachments/assets/bda9ec1c-55a8-4bb7-b554-61531c0354c0" />


# 4. Data preprocessing:

### 1. Bin the continuous variables
To feed this data into the Bayesian Network, continuous variables must be converted into categorical format. We identified age, BMI, and average glucose level are continuous variables. Therefore, we bin these variables into categorical variables by grouping them into categorical intervals.

### 2. Bin the categorical variables with low frequencies
For analytical convenience, we handle fragmented values with very low frequency by either removing them or merging them with the nearest appropriate category. For example, gender has a three values like male, female, and the other, with only one frequency. Also, we simplified job categories with several values into employed or not.


### 3. Binary into catgorical variable
For ease of interpretation, we transform binary values into "Yes" and "No" categories. This makes the Bayesian net easier to read and understand.


# 5. Methodology

## 5.1 BayesNet Analysis

## Methodology
- HillClimb
- Tree Search

## BayesNet

The Bayesian Network analysis reveals the dependency structure among various variables related to stroke prediction. We implemented two models to learn this structure: one using **Hill Climb Search** and the other using **Tree Search**. The attached images visualize the network structures generated by each model. For each node, we calculated the Conditional Probability Distributions (CPDs). 

### HillClimb

<img width="997" alt="image" src="https://github.com/user-attachments/assets/8edf6598-28e1-47d5-acba-5c1b55f0f4c6" />

This Bayesian Network, learning using the Hill Climb algorithm, shows that the age_group is the central node. It is connected to many other variables, and knowing whether an individual has hypertension helps infer their age group. Focusing on strike risk, the model shows that younger individuals have about a 1% chance of stroke, middle-aged individuals 6%, and older individuals 14%. In short, the older the individual, the higher the risk of stroke.


### Tree Search
<img width="1201" alt="image" src="https://github.com/user-attachments/assets/fd727535-15c0-4b06-8fea-2e5308e55b40" />

The Tree Search model presents a similar structure where age_group is again the central node influencing stroke, hypertension, glucose level, and smoking status.
Stroke risk increases with age — from 1% for the young to 14% for the elderly.

Both models identified age as the most influential factor in predicting stroke. This aligns with our EDA findings, which showed that older individuals have a higher risk of stroke compared to younger ones.


### Compare BIC score - the lower the better

We also computed the **Bayesian Information Criterion (BIC)** scores to evaluate and compare the models.

The Tree Search model achieved a better BIC score, indicating a better fit. While the Hill Climb model captures more complex relationships through additional connections, the Tree model presents a simpler and more interpretable hierarchical structure.


#### Hill Climb BIC: -5984.04
#### Tree Search BIC: -6708.90 (Best Model)


### Conclusion from Bayes Net Analysis 

Tree Search is the best model.


### Inference
<img width="430" alt="image" src="https://github.com/user-attachments/assets/00bedc76-5db8-4711-b4ef-3ba4b864612a" /> 
Query inferencing of getting a stroke

<img width="252" alt="image" src="https://github.com/user-attachments/assets/7873e5dc-98c5-4454-9383-8b913fdd583b" /> Distribution of strokes among the Young age group


<img width="248" alt="image" src="https://github.com/user-attachments/assets/ed65a5ce-303e-428b-97b8-2f909597579a" /> Distribution of getting stroke of the Old age group



## 5.2  Baseline Analysis & Advanced Prediction
###  Preprocess Data

To prepare the data for modeling, we applied the following preprocessing steps:

#### 1. Separate Features and Target
- The target variable `stroke` was separated from the feature set.

#### 2. Ordinal Encoding for Categorical Variables
- We used `OrdinalEncoder` to convert all categorical variables into numerical format.
- Unknown categories during transformation were handled with a dedicated encoding (`unknown_value=-1`).

#### 3. Handle Missing Values
- Any missing values were filled using the most frequent value (`mode`) for each column.

#### 4. SMOTE Oversampling
- To address class imbalance in the `stroke` label, we applied **SMOTE** (Synthetic Minority Over-sampling Technique).
- SMOTE synthetically generates new samples from the minority class to balance the dataset.

#### 5. Train/Test Split
- The resampled dataset was split into training and testing sets using an 80/20 ratio.
- The split was stratified based on the `stroke` label to preserve class distribution.


###  Naive Bayes Classifier

We implemented a **Categorical Naive Bayes** classifier to establish a lightweight probabilistic baseline.

#### 1. Model Training
- Trained using `CategoricalNB()` from `sklearn.naive_bayes`.
- The model was fit on the training set (`X_train`, `y_train`) and used to predict the labels for the test set (`X_test`).

#### 2. Label Conversion
- For evaluation purposes, both true and predicted labels were mapped from `{'No': 0, 'Yes': 1}`.

#### 3. Evaluation Metrics
- A **classification report** was generated to provide precision, recall, f1-score, and support for each class.
- A **confusion matrix** was plotted using a heatmap (`seaborn`) to visualize prediction performance.

#### 4. Regression Metrics (for consistency)
- We also computed `MSE` (Mean Squared Error) and `RMSE` (Root Mean Squared Error) to provide a unified metric comparison across different models.


<img width="577" alt="Naive Bayes V3" src="https://github.com/user-attachments/assets/1fefad43-e6e8-4ae3-8b7e-98d57325f2a2" />



### Random Forest

We implemented a **Random Forest** classifier to predict stroke occurrence and evaluate its performance.

#### Model Training
We trained a `RandomForestClassifier` from `sklearn.ensemble` using the training data. A fixed `random_state=42` was used to ensure reproducibility.

#### Prediction & Evaluation
Predictions were made on the test set. We generated a classification report including **precision**, **recall**, and **F1-score** for both stroke and no-stroke cases.

#### Confusion Matrix
We visualized the confusion matrix using a heatmap (`seaborn.heatmap`) to better understand the model’s prediction patterns.

#### MSE & RMSE
We also computed **Mean Squared Error (MSE)** and **Root Mean Squared Error (RMSE)** as numerical performance metrics. Labels were converted from `"Yes"`/`"No"` to `1`/`0` respectively for calculation.

#### Result
See below for the classification report and confusion matrix.

<img width="546" alt="RF V3" src="https://github.com/user-attachments/assets/d1895c45-7803-4141-a9dc-046fa5cf9cda" />


### XGBoost

We implemented an **XGBoost** classifier to predict stroke occurrence and evaluate its performance.

#### Model Training
We trained an `XGBClassifier` from `xgboost` using the training data. We disabled the default label encoder with `use_label_encoder=False`, used `'logloss'` as the evaluation metric, and set a fixed `random_state=42` for reproducibility.

#### Prediction & Evaluation
Predictions were made on the test set. We generated a classification report including **precision**, **recall**, and **F1-score** for both stroke and no-stroke cases.

#### Confusion Matrix
We visualized the confusion matrix using a heatmap (`seaborn.heatmap`) to better understand the model’s prediction patterns.

#### MSE & RMSE
We also computed **Mean Squared Error (MSE)** and **Root Mean Squared Error (RMSE)** as numerical performance metrics. Labels were converted from `"Yes"`/`"No"` to `1`/`0` respectively.

#### Result
See below for the classification report and confusion matrix.


<img width="546" alt="XGB V3" src="https://github.com/user-attachments/assets/5d224409-fac2-423e-8c6b-acd9d6dfbc1c" />



### Logistic Regression

We implemented a **Logistic Regression** model to predict the likelihood of stroke occurrence and evaluate its performance.

#### Model Training  
We used the `LogisticRegression` class from `sklearn.linear_model` to train the model on the prepared dataset. A fixed `random_state=42` and `max_iter=1000` were used to ensure convergence and reproducibility.

#### Prediction & Evaluation  
Predictions were generated on the test dataset. A classification report was generated, including **precision**, **recall**, and **F1-score** to assess model performance for both stroke and non-stroke cases.

#### Confusion Matrix  
We visualized the confusion matrix using a heatmap (`seaborn.heatmap`) to observe how well the model distinguishes between the classes.

#### MSE & RMSE  
We calculated **Mean Squared Error (MSE)** and **Root Mean Squared Error (RMSE)** to numerically evaluate prediction performance.

#### Result  
See below for the classification report and confusion matrix.  

<img width="592" alt="logit V3" src="https://github.com/user-attachments/assets/92c50293-7bf1-4187-8dcf-c0e1ff39f7a3" />



### Bayesian Linear Regression (PyMC)

We implemented a **Bayesian Linear Regression** model using the `PyMC` probabilistic programming library. This approach allows us to model uncertainty in the predictions by estimating the posterior distributions of the regression coefficients.

#### Model Setup

We used `pm.Model()` from the `pymc` library to define the probabilistic structure:
- **Priors** were assigned to the regression coefficients and intercept (Normal distributions).
- **Likelihood** was defined using a Normal distribution for the observed target variable, conditional on the linear predictor.

#### Inference with MCMC

To estimate the posterior distributions of the parameters, we used **Markov Chain Monte Carlo (MCMC)** sampling via `pm.sample()`:
- We drew 1,000 posterior samples with a 1,000-sample tuning phase.
- The `nuts` sampler was automatically selected by PyMC.

#### Posterior Analysis

We analyzed the posterior distributions using:
- `arviz.plot_trace()` to visualize trace plots and convergence of the chains.
- `arviz.summary()` to examine mean estimates and credible intervals of each parameter.

#### Prediction

Posterior predictive sampling was performed using `pm.sample_posterior_predictive()` to generate predicted values and quantify uncertainty in predictions.

#### Visualization

We visualized the uncertainty around predictions using:
- **Posterior predictive plots** with confidence intervals
- **Trace plots** to check sampler behavior
- **Summary statistics** of the posterior to interpret coefficient effects

#### Result

The Bayesian Linear Regression model allowed us to estimate not only point predictions but also the uncertainty around them, offering more interpretable insights compared to frequentist regression models.


<img width="553" alt="Bayesian RegressionV3" src="https://github.com/user-attachments/assets/af8177c2-8146-48f4-b093-6a9f952ad5ca" />



### Model Comparison and Selection

To evaluate the performance of different models in predicting stroke occurrence, we compared five baseline classifiers:

- **Naive Bayes**
- **Random Forest**
- **XGBoost**
- **Logistic Regression**
- **Bayesian Logistic Regression (PyMC)**

We used the **F1-score (Macro Average)** as the key evaluation metric, which balances precision and recall across both classes (stroke / no stroke).

#### F1-Score Comparison

| Model                          | F1-score (Macro Avg) |
|-------------------------------|----------------------|
| Naive Bayes                   | 0.76                 |
| Random Forest                 | 0.95                 |
| **XGBoost**                   | **0.95**             |
| Logistic Regression           | 0.67                 |
| Bayesian Logistic Regression (PyMC) | 0.68          |

Based on the comparison above, **XGBoost** achieved the highest macro-averaged F1-score of **0.95**, making it the best-performing model for our task.


---

### Feature Importance with XGBoost

After identifying **XGBoost** as the best-performing model, we used its built-in feature importance method to interpret the factors contributing most significantly to stroke prediction.

We used the `get_score()` function from the trained XGBoost model to extract importance scores based on the **F Score (weight)**, and visualized the top 10 features with the highest impact.

#### Visualization

The importance scores reveal which features were most frequently used to split data across all boosted trees in the model. Higher scores indicate stronger influence on stroke prediction.

![Top10](https://github.com/user-attachments/assets/eb26df90-63a7-4467-8ace-c78b71f1b195)




These top features offer insight into the key risk indicators of stroke based on our dataset and model training.


# 6. Conclusion & Future Work

## Summary of Results: 

The Bayesian network experiments and downstream classifiers together paint a coherent picture of stroke risk and modeling trade‑offs:

### Bayesian network structure learning
• The tree‑search model, with a BIC of –6708.90, proved more parsimonious than the hill‑climb graph (BIC –5984.04) while preserving key dependencies.

### Classifier performance and interpretation
• Categorical naive Bayes offered a quick baseline (macro‑F1 ≈ 0.76), underscoring the value of simple probabilistic models when interpretability and speed are priorities.

• Logistic regression improved calibration and interpretability, with coefficients directly indicating log‑odds shifts, but its macro‑F1 (≈ 0.67) lagged behind more flexible learners.

• Random forest and XGBoost captured complex, nonlinear interactions—both achieved a macro‑F1 of approximately 0.95. Their near‑identical performance suggests diminishing returns beyond ensemble tree methods for this dataset.

### Feature importance and actionable insights
• The most influential predictors across models were age, average glucose level, hypertension, and heart disease, followed by BMI, marital status, and smoking behavior.

• This hierarchy confirms that efforts to screen and manage blood pressure and glucose—especially in older populations—should remain cornerstones of stroke prevention programs.

• Lifestyle interventions (smoking cessation, weight control) have measurable impact, but their benefit is modulated by the patient’s age and comorbidity profile.

### Trade‑offs and combined use
• The Bayesian network excels at revealing how risk factors interconnect and supports “what‑if” queries that quantify the benefit of specific interventions.

• XGBoost and random forest deliver superior out‑of‑sample accuracy, making them preferable when prediction alone is the objective.

## Real World Use-Cases:
Our Bayesian Network model demonstrates that both non‑modifiable factors (age, gender, genetic predispositions reflected in comorbidities) and modifiable behaviors (smoking status, BMI, average glucose level, physical activity proxies) jointly shape an individual’s stroke risk. From a public health perspective, these insights translate into actionable guidance:

Hypertension and Diabetes Management: Regular blood pressure and blood sugar monitoring—coupled with adherence to prescribed medications—can substantially lower stroke probability. Community health programs should continue expanding free screening services and patient education on medication compliance.

Weight Control and Nutrition: Elevated BMI and glucose levels emerged as key intermediate risk factors. Public campaigns that promote balanced diets rich in fruits, vegetables, and whole grains, alongside accessible weight‑management resources (e.g., subsidized fitness memberships or nutritional counseling), can help shift population‑level risk profiles.

## Potential Future Work:

Temporal Validation and Dynamic Modeling: Incorporating longitudinal patient records to capture how risk trajectories evolve over time, enhancing the network’s ability to forecast stroke onset months or years in advance.

Integration of Additional Biomarkers: Adding variables such as lipid profiles, genetic markers (e.g., APOE genotype), and measures of inflammation to refine conditional dependencies and improve predictive accuracy.

By translating probabilistic insights into clear, evidence‑based recommendations and by continuously refining our model with richer data, we aim to empower both healthcare providers and the public to make informed decisions that reduce the global burden of stroke.




