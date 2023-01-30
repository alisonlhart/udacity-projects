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

- `requirements_py3.8.txt`: Use this to install requirements for running this script locally. 

## Running Files

To calculate the customer churn using the `churn_library.py` script, run this on the command line:

`python churn_library.py`

This will produce the images and models displaying the customer churn calculations. 

NOTE: Before the calculations start, the `images/` and `models/` directories will be emptied to remove previous runs' data. 



To run the tests and logger, run this on the command line:

`python churn_script_logging_and_tests.py`

This will run tests on each function in `churn_library.py` and will output logs into `logs/churn_library.log`. 
