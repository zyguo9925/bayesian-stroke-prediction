
# 1. Introduction (Alex)

data link: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

Stroke remains one of the leading causes of death and long‑term disability worldwide. Early identification of high‑risk individuals is critical for implementing timely interventions, reducing healthcare costs, and improving patient outcomes. Traditional risk‑scoring systems often rely on a limited set of covariates and yield “black‑box” predictions, which can be difficult for clinicians to interpret and trust. To address these challenges, we leverage a rich, real‑world dataset and a probabilistic modeling framework that balances predictive power with transparency.

The Stroke Prediction Dataset compiles anonymized patient information spanning demographics (age, gender, marital status, residence), clinical measurements (BMI, average glucose level), lifestyle factors (smoking status, work type), and comorbidities (hypertension, heart disease). The binary target indicates whether a patient has experienced a stroke. This diversity of features offers a comprehensive view of the biological, socioeconomic, and behavioral drivers that contribute to cerebrovascular risk.

Our objective is to build a Bayesian Network that estimates the probability of stroke for individual patients, while simultaneously revealing the conditional dependencies among risk factors. By combining structure learning (to uncover the network of relationships) with parameter estimation (to quantify those relationships), we create an interpretable model suited for clinical decision support. During inference, clinicians can query the network to assess how changes in modifiable factors—such as smoking cessation or weight management—alter a patient’s stroke risk, thereby guiding personalized prevention strategies.

Using a dataset from Kaggle, “Stroke Prediction Dataset”, we aim to make a prediction model that predicts whether a patient will get a stroke or not. This dataset has a 5,110 rows and 12 columns of real patient data. This includes each patient’s demographic information, biological and socioeconomical attributes such as gender, age, whether or not he/she is married, BMI, hypertension, and disease history. We set our outcome variable to be whether a patient will have a stroke or not.

# 2. Data profile & preprocessing & EDA (Alex & Sam)

## Data Profile: (Alex)

## Dataset Overview:

- Rows: 5,110

- Columns: 12

Description: Each record represents a patient’s demographic, lifestyle, and clinical information, with a binary target stroke indicating whether they’ve had a stroke or not.

## Missing Values
bmi: 201 missing (3.94% of rows)

All other columns are complete.

## Attribute Information
#### 1) id: unique identifier
#### 2) gender: "Male", "Female" or "Other"
#### 3) age: age of the patient
#### 4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
#### 5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
#### 6) ever_married: "No" or "Yes"
#### 7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
#### 8) Residence_type: "Rural" or "Urban"
#### 9) avg_glucose_level: average glucose level in blood
#### 10) bmi: body mass index
#### 11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
#### 12) stroke: 1 if the patient had a stroke or 0 if not

## Categorical Distributions
gender (n=5,110)

Female: 2,994 (58.6%)

Male: 2,115 (41.4%)

Other: 1 ( <0.1%)

ever_married

Yes: 3,353 (65.6%)

No: 1,757 (34.4%)

work_type

Private: 2,925 (57.2%)

Self-employed: 819 (16.0%)

children: 687 (13.4%)

Govt_job: 657 (12.9%)

Never_worked: 22 (0.4%)

Residence_type

Urban: 2,596 (50.8%)

Rural: 2,514 (49.2%)

smoking_status

never smoked: 1,892 (37.0%)

Unknown: 1,544 (30.2%)

formerly smoked: 885 (17.3%)

smokes: 789 (15.4%)

## Key Observations 
Imbalance: Only ~5% positive cases—may need SMOTE, class weights, or focal loss.

Missing BMI: Impute (e.g., with median by age/gender) or treat “Unknown” category.

Outliers: Extremely high avg_glucose_level and bmi values warrant further inspection.

ID Column: Drop before modeling.

Feature Engineering:

Binning age into categories (child, adult, senior).

Combining smoking_status = Unknown with a “missing” indicator.

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


# 4. Data preprocessing: (Sam)

### 1. Bin the continuous variables
To feed this data into the Bayesian Network, continuous variables must be converted into categorical format. We identified age, BMI, and average glucose level are continuous variables. Therefore, we bin these variables into categorical variables by grouping them into categorical intervals.

### 2. Bin the categorical variables with low frequencies
For analytical convenience, we handle fragmented values with very low frequency by either removing them or merging them with the nearest appropriate category. For example, gender has a three values like male, female, and the other, with only one frequency. Also, we simplified job categories with several values into employed or not.


### 3. Binary into catgorical variable
For ease of interpretation, we transform binary values into "Yes" and "No" categories. This makes the Bayesian net easier to read and understand.


# 5. Methodology

## 5.1 BayesNet Analysis (Sam)

## Methodology
- HillClimb
- Tree Search

## BayesNet
### HillClimb

<img width="997" alt="image" src="https://github.com/user-attachments/assets/8edf6598-28e1-47d5-acba-5c1b55f0f4c6" />


### Tree Search
<img width="1201" alt="image" src="https://github.com/user-attachments/assets/fd727535-15c0-4b06-8fea-2e5308e55b40" />

The Bayesian Network analysis reveals the dependency structure among various variables related to stroke prediction. We implemented two models to learn this structure: one using **Hill Climb Search** and the other using **Tree Search**. Each method has its own strengths.

Both models identified age as the most influential factor in predicting stroke. This aligns with our EDA findings, which showed that older individuals have a higher risk of stroke compared to younger ones.

The attached images visualize the network structures generated by each model. For each node, we calculated the Conditional Probability Distributions (CPDs). We also computed the **Bayesian Information Criterion (BIC)** scores to evaluate and compare the models.

The Tree Search model achieved a better BIC score, indicating a better fit. While the Hill Climb model captures more complex relationships through additional connections, the Tree model presents a simpler and more interpretable hierarchical structure.
### Compare BIC score - the lower the better

Hill Climb BIC: -5984.04
Tree Search BIC: -6708.90 (Best Model)


### Conclusion from Bayes Net Analysis 

Tree Search is the best model.


### Inference
- Plan to execute the inference analysis from the BayesNet.



## 5.2  Baseline Analysis & Advanced Prediction (Rebecca)
3.2 Baseline Prediction Model
BayesNet
Logistic Regression (Baseline)
Bayesian linear Regression
+ MCMC (Bayesian linear regression -> Estimate using PyMC Sampling)
Naive Bayes Classifier
XGBoost
Random forest
Comparison across the models and pick the best model.
XGboost(winner from comparison) - Importance Score
Conclusion:which features contribute to stroking the most from XGboost importance scores


# 5. Conclusion & Future Work (Alex)

## Summary of Results: 
Combine result from 3.1&3.2 together

## Real World Use-Cases:
Our Bayesian Network model demonstrates that both non‑modifiable factors (age, gender, genetic predispositions reflected in comorbidities) and modifiable behaviors (smoking status, BMI, average glucose level, physical activity proxies) jointly shape an individual’s stroke risk. From a public health perspective, these insights translate into actionable guidance:

Hypertension and Diabetes Management: Regular blood pressure and blood sugar monitoring—coupled with adherence to prescribed medications—can substantially lower stroke probability. Community health programs should continue expanding free screening services and patient education on medication compliance.

Weight Control and Nutrition: Elevated BMI and glucose levels emerged as key intermediate risk factors. Public campaigns that promote balanced diets rich in fruits, vegetables, and whole grains, alongside accessible weight‑management resources (e.g., subsidized fitness memberships or nutritional counseling), can help shift population‑level risk profiles.

## Potential Future Work:

Temporal Validation and Dynamic Modeling: Incorporating longitudinal patient records to capture how risk trajectories evolve over time, enhancing the network’s ability to forecast stroke onset months or years in advance.

Integration of Additional Biomarkers: Adding variables such as lipid profiles, genetic markers (e.g., APOE genotype), and measures of inflammation to refine conditional dependencies and improve predictive accuracy.

By translating probabilistic insights into clear, evidence‑based recommendations and by continuously refining our model with richer data, we aim to empower both healthcare providers and the public to make informed decisions that reduce the global burden of stroke.




