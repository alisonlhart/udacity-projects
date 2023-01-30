# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity
- Author: alisonlhart (Alison Hart)
- Date: Jan 29 2023


## Project Description
This project calculates customer churn based on a dataset, but the goal of this project is to produce clean, refactored, efficient code adhering to PEP8 Guidelines and Pylint standards. 


## Files and data description

- `churn_library.py`: Contains functions for the data calculations, image processing, plotting, and model training. 

- `churn_script_logging_and_tests.py`: Performs tests and appropriate logging for all functions in churn_library.py. Log can be found under logs/churn_library.log.

- `images/eda/` : EDA plots and histograms produced from the input data. 

- `images/results/` : Feature importance plot, classification reports, and ROC plots. 

- `models/` : Models produced from the input data. 

- `data/` : Contains the Bank Data CSV for calculating customer churn. 

- `requirements.txt`: Required libraries for running the project. 


## Required Libraries and Installing Requirements

- `scikit-learn==0.24.1`
- `shap==0.40.0`
- `joblib==1.0.1`
- `pandas==1.2.4`
- `numpy==1.20.1`
- `matplotlib==3.3.4`
- `seaborn==0.11.2`
- `pylint==2.7.4`
- `autopep8==1.5.6`

Install these libraries with the `requirements.txt` file by running this command in the `predict-customer-churn/` directory:

`pip install -r requirements.txt`


## Running Files

To calculate the customer churn using the `churn_library.py` script, run this on the command line:

`python churn_library.py`

This will produce the images and models displaying the customer churn calculations. 

NOTE: Before the calculations start, the `images/` and `models/` directories will be emptied to remove previous runs' data. 

----

To run the tests and logger, run this on the command line:

`python churn_script_logging_and_tests.py`

This will run tests on each function in `churn_library.py` and will output logs into `logs/churn_library.log`. 
