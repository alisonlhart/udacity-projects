"""
churn_library.py

Description: This file contains all functions used to calculate customer churn from a dataset.
Author: Alison Hart (alisonlhart)
Date: Jan 29 2023
"""


import os
import seaborn as sns
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import shap
import joblib
import numpy as np
import pandas as pd

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

sns.set()

# pylint: disable=invalid-name


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    dataframe = pd.read_csv(pth)
    return dataframe


def perform_eda(dataframe):
    '''
    perform eda on dataframe and save figures to images folder
    input:
            dataframe: pandas dataframe

    output:
            None
    '''

    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    plt.figure(figsize=(20, 10))
    plt.title("Customer Churn")
    plt.xlabel("Likelyhood of Churn")
    plt.ylabel("Customers")
    churn_diagram = dataframe['Churn'].hist()
    figure = churn_diagram.get_figure()
    figure.savefig('images/eda/churn_diagram.png')

    plt.figure(figsize=(20, 10))
    plt.title("Customer Ages")
    plt.xlabel("Age")
    plt.ylabel("Number of Customers")
    age_diagram = dataframe['Customer_Age'].hist()
    figure = age_diagram.get_figure()
    figure.savefig('images/eda/age_diagram.png')

    plt.figure(figsize=(20, 10))
    plt.title("Marital Status Percentages")
    plt.xlabel("Marital Status")
    plt.ylabel("Percentage of Customers (out of 1)")
    marital_plot = dataframe.Marital_Status.value_counts(
        'normalize').plot(kind='bar')
    figure = marital_plot.get_figure()
    figure.savefig('images/eda/marital_plot.png')

    plt.figure(figsize=(20, 10))
    plt.title("Total Trans CT Density")
    total_diagram = sns.histplot(
        dataframe['Total_Trans_Ct'],
        stat='density',
        kde=True)
    figure = total_diagram.get_figure()
    figure.savefig('images/eda/total_diagram.png')

    plt.figure(figsize=(20, 10))
    plt.title("Feature Heatmap")
    heatmap_diagram = sns.heatmap(
        dataframe.corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    figure = heatmap_diagram.get_figure()
    figure.savefig('images/eda/heatmap_diagram.png')
    plt.close('all')


def encoder_helper(dataframe, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            dataframe: pandas dataframe
            category_lst: list of columns that contain categorical features

    output:
            dataframe: pandas dataframe with new columns for
    '''
    for feature in category_lst:
        lst = []
        feature_groups = dataframe.groupby(feature).mean()['Churn']

        for val in dataframe[feature]:
            lst.append(feature_groups.loc[val])
        name_plus_churn = feature + "_Churn"

        dataframe[name_plus_churn] = lst

    return dataframe


def perform_feature_engineering(dataframe):
    '''
    input:
              dataframe: pandas dataframe

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    y = dataframe['Churn']
    X = pd.DataFrame()

    dataframe = encoder_helper(dataframe, category_lst)

    X[keep_cols] = dataframe[keep_cols]
    X.head()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test, X


def classification_report_image(y_train,                # pylint: disable=too-many-arguments
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    plt.clf()
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig("images/results/random_forest_classification_report.png")
    plt.clf()

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig("images/results/logistic_regression_classification_report.png")
    plt.clf()
    plt.close('all')


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    importances = model.best_estimator_.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)
    plt.close('all')


def train_models(X_train, X_test, y_train, y_test, X):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
              X: pd dataframe
    output:
              None
    '''

    rfc = RandomForestClassifier(random_state=42)

    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    lrc_plot = plot_roc_curve(lrc, X_test, y_test)

    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    lrc_plot.plot(ax=axis, alpha=0.8)
    plt.savefig("images/results/roc_lrc_plot.png")
    plt.close('all')

    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')

    lrc_plot = plot_roc_curve(lr_model, X_test, y_test)

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(  # pylint: disable=unused-variable
        rfc_model,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig("images/results/roc_lrc_plot_rfc.png")
    plt.close('all')

    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)

    feature_importance_plot(
        cv_rfc, X, "images/results/feature_importance_plot.png")
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)


def clean_up_dirs(dirlist):
    """Deletes files from directories.

    input:
            dirlist: Array of string directory names

    output:
            None
    """
    for directory in dirlist:
        for file in os.listdir(directory):
            os.remove(os.path.join(directory, file))


if __name__ == "__main__":

    clean_up_dirs(["./images/eda", "images/results", "models"])

    df = import_data(r"./data/bank_data.csv")

    perform_eda(df)

    X_train_data, X_test_data, y_train_data, y_test_data, X_df = perform_feature_engineering(
        df)

    train_models(X_train_data, X_test_data, y_train_data, y_test_data, X_df)
