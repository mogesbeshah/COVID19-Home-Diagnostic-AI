# COVID-19 Home Diagnostic AI

## Project Objective

This project develops a Python-based AI system to estimate the probability of PCR-positive COVID-19 using information available to a patient at home before visiting a clinic or emergency room.

The project is completed entirely in Python and includes both machine-learning models and a Bayesian network.

## Data Sources

The project uses:

- COVIDCARE Phase II participant data
- COVIDCARE survey dictionary
- COVIDCARE DEMI knowledgebase

## Models

Two feature sets are evaluated:

### Model A — Home Information Only

Uses information available at home before a clinic or emergency-room visit, including:

- Symptoms
- Exposure history
- Vaccination information
- Demographics
- Health history
- Other eligible home-available variables

### Model B — Home Information + At-Home Test

Includes all Model A information plus an at-home COVID-19 test result collected before PCR testing.

The following predictive models are compared:

- Logistic Regression
- LASSO Logistic Regression
- Random Forest

Model performance is evaluated using five-fold stratified cross-validation.

Primary evaluation measures include:

- ROC-AUC
- Accuracy
- Sensitivity
- Specificity

## Bayesian Network

A Bayesian network is constructed entirely in Python using:

- LASSO feature selection
- DEMI temporal relationships
- NetworkX
- pgmpy

The network estimates:

P(PCR Positive | Home Evidence)

Bayesian inference is used to compare estimated PCR-positive probability with and without at-home COVID-19 test information.

## Streamlit Application

A Streamlit application will provide an interactive interface where users can enter home-available information and receive an estimated probability of PCR-positive COVID-19.

## Repository Structure

```text
COVID19-Home-Diagnostic-AI/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── COVIDCARE_FORSUBMISSION_MIT_CLEANED_Phase_II_2021-12-03.csv
│   ├── COVIDCARE_survey_dictionary_v2_ForSubmission_MIT_Phase_II_2021-12-26.csv
│   └── COVIDCARE_DEMI_knowledgebase_v4.csv
│
└── notebooks/
    └── COVID19_Home_Diagnostic_AI.ipynb
```

##Main Python Libraries

-pandas
-NumPy
-scikit-learn
-Matplotlib
-NetworkX
-pgmpy
-Streamlit

##Important Note

This system is a predictive AI prototype based on the COVIDCARE dataset. It is intended for academic and research purposes and should not be used as a substitute for professional medical diagnosis or clinical testing.
