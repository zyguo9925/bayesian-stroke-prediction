Work still in progress (by Samuel)


# 1. Introduction (Alex)
data link: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

Using a dataset from Kaggle, “Stroke Prediction Dataset”, we aim to make a prediction model that predicts whether a patient will get a stroke or not. This dataset has a 5,110 rows and 12 columns of real patient data. This includes each patient’s demographic information, biological and socioeconomical attributes such as gender, age, whether or not he/she is married, BMI, hypertension, and disease history. We set our outcome variable to be whether a patient will have a stroke or not.


Project objective: Predict the probability of stroke using a Bayesian Network.
Motivation: Medical applications require not just prediction, but also interpretability.
Methodology overview: Structure learning + parameter estimation + inference.

2. Data profile & preprocessing & EDA (Alex & Sam)

-

- Data Profile: (Alex)

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



- EDA

- Categorical Variable

- Yes

  
<img width="213" alt="image" src="https://github.com/user-attachments/assets/9c47ffa7-239b-4f84-a488-bc11b3fa7a73" /> <img width="223" alt="image" src="https://github.com/user-attachments/assets/c74903e1-68f7-4ada-adf6-8869407115bc" />

- No
  
<img width="183" alt="image" src="https://github.com/user-attachments/assets/14b41d58-f71d-4f83-ab74-f2d4a0750c72" /> <img width="206" alt="image" src="https://github.com/user-attachments/assets/f8eedc94-3868-498f-9237-867dec11c5c1" /> <img width="212" alt="image" src="https://github.com/user-attachments/assets/2b6d817b-5d75-4f05-8eac-baedd5f96df3" /> 

 
-Continuous Variable 
-Yes
<img width="345" alt="image" src="https://github.com/user-attachments/assets/5c9e1eeb-34fa-410d-825a-77d2d8f99101" /> 


-No
<img width="331" alt="image" src="https://github.com/user-attachments/assets/2edf0002-2862-4a87-8fe6-f5ba7ed62a8b" />


-depends on the interval
<img width="349" alt="image" src="https://github.com/user-attachments/assets/bda9ec1c-55a8-4bb7-b554-61531c0354c0" />











- Data preprocessing: (Sam)


1) Bin the “continuous variable”: age, avg_glucose_level, bmi.
2) Bin the “categorical variables” with low frequency to simplify. e.g.) gender (Other), or simplify jobs (work or not), 
3) Remove null: bmi has some null values









3. Methodology
3.1 Bayes Net Analysis - Samuel
HillClimb
Tree Search
Compare BIC score
Conclusion from Bayes Net Analysis 

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


4. Conclusion & Future Work (Alex)
Combine result from 3.1&3.2 together
Give a real-world wording conclusion (e.g what kind of advice can give to the public from public health viewpoints)




