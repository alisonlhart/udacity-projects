"""
churn_script_logging_and_tests.py

Description: This file is used to test functions in churn_library.py and log info and errors.
Author: Alison Hart (alisonlhart)
Date: Jan 29 2023
"""

import os
import logging
from churn_library import (
    import_data,
    perform_eda,
    encoder_helper,
    perform_feature_engineering,
    train_models,
    clean_up_dirs)


def test_import(import_data_function):
    '''
    test data import - this example is completed for you to assist with the other test functions

    input:
            import_data_function: import_data() function from churn_library.py

    output:
            dataframe: dataframe to pass to other tests
    '''
    try:
        dataframe = import_data_function("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")

    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return dataframe


def test_eda(dataframe, perform_eda_function):
    '''
    Test perform_eda function

    input:
            dataframe: Passed from test_import
            perform_eda_function: perform_eda() function from churn_library.py

    output:
                None
    '''
    try:
        perform_eda_function(dataframe)

        assert os.path.exists('images/eda/churn_diagram.png')
        logging.info("Testing test_eda: images/eda/churn_diagram.png exists")

        assert os.path.exists('images/eda/age_diagram.png')
        logging.info("Testing test_eda: images/eda/age_diagram.png exists")

        assert os.path.exists('images/eda/heatmap_diagram.png')
        logging.info("Testing test_eda: images/eda/heatmap_diagram.png exists")

        assert os.path.exists('images/eda/marital_plot.png')
        logging.info("Testing test_eda: images/eda/marital_plot.png exists")

        assert os.path.exists('images/eda/total_diagram.png')
        logging.info("Testing test_eda: images/eda/total_diagram.png exists")

        logging.info("Testing perform_eda: SUCCESS")

    except AssertionError as err:
        logging.error("Testing perform_eda: EDA Diagram is missing.")
        raise err


def test_encoder_helper(dataframe, encoder_helper_function):
    '''
    Test encoder_helper function

    input:
            dataframe: Passed from test_import
            encoder_helper_function: encoder_helper() function from churn_library.py

    output:
            None
    '''
    category_list = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    try:
        dataframe = encoder_helper_function(dataframe, category_list)

        assert dataframe.any
        logging.info("Testing test_encoder_helper: dataframe exists")

        logging.info("Testing test_encoder_helper: SUCCESS")

    except NameError as err:
        logging.error(
            "Testing test_encoder_helper: Dataframe wasn't returned from encoder_helper")
        raise err


def test_perform_feature_engineering(
        dataframe,
        perform_feature_engineering_function):
    '''
    Test perform_feature_engineering

    input:
            dataframe: Passed from test_import
            perform_feature_engineering_function:
                    perform_feature_engineering() function from churn_library.py

    output:
            X_train, X_test, y_train, y_test: Training and test data
    '''
    try:
        X_train, X_test, y_train, y_test, X = perform_feature_engineering_function(
            dataframe)
        assert X_train.any
        logging.info(
            "Testing test_perform_feature_engineering: X_train exists with data")

        assert X_test.any
        logging.info(
            "Testing test_perform_feature_engineering: X_test exists with data")

        assert y_train.any
        logging.info(
            "Testing test_perform_feature_engineering: y_train exists with data")

        assert y_test.any
        logging.info(
            "Testing test_perform_feature_engineering: y_test exists with data")

        assert X.any
        logging.info(
            "Testing test_perform_feature_engineering: X exists with data")

        logging.info("Testing test_perform_feature_engineering: SUCCESS")

    except AssertionError as err:
        logging.error(
            "Testing test_perform_feature_engineering: Values were not returned.")
        raise err

    return X_train, X_test, y_train, y_test, X


def test_train_models(
        X_train,
        X_test,
        y_train,
        y_test,
        X,
        train_models_function):
    '''
    Test train_models

    input:
            X_train, X_test, y_train, y_test: Training and test data

    output:
            None

    '''
    try:
        train_models_function(X_train, X_test, y_train, y_test, X)

        assert os.path.exists(
            "images/results/random_forest_classification_report.png")
        logging.info(
            "Testing test_train_models: "
            "images/results/random_forest_classification_report.png exists")

        assert os.path.exists(
            "images/results/logistic_regression_classification_report.png")
        logging.info(
            "Testing test_train_models: "
            "images/results/logistic_regression_classification_report.png exists")

        assert os.path.exists("images/results/roc_lrc_plot_rfc.png")
        logging.info(
            "Testing test_train_models: images/results/roc_lrc_plot_rfc.png exists")

        assert os.path.exists("images/results/roc_lrc_plot.png")
        logging.info(
            "Testing test_train_models: images/results/roc_lrc_plot.png exists")

        assert os.path.exists('./models/rfc_model.pkl')
        logging.info("Testing test_train_models: models/rfc_model.pkl exists")

        assert os.path.exists('./models/logistic_model.pkl')
        logging.info(
            "Testing test_train_models: models/logistic_model.pkl exists")

        assert os.path.exists("images/results/feature_importance_plot.png")
        logging.info(
            "Testing test_train_models: images/results/feature_importance_plot.png exists")

        logging.info("Testing test_train_models: SUCCESS")

    except AssertionError as err:
        logging.error("Testing test_train_models: Report image not found")
        raise err


if __name__ == "__main__":

    # Clean up images and logs
    clean_up_dirs(["./images/eda", "images/results", "models", "logs"])

    logging.basicConfig(
        filename='./logs/churn_library.log',
        level=logging.INFO,
        filemode='w',
        format='%(name)s - %(levelname)s - %(message)s')

    logging.info("STARTING TESTS")

    df = test_import(import_data)
    test_eda(df, perform_eda)
    test_encoder_helper(df, encoder_helper)
    X_train, X_test, y_train, y_test, X = test_perform_feature_engineering(
        df, perform_feature_engineering)
    test_train_models(X_train, X_test, y_train, y_test, X, train_models)
