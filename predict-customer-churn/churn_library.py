# library doc string


# import libraries
import os
os.environ['QT_QPA_PLATFORM']='offscreen'

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, plot_roc_curve


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''	
    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    
    plt.figure(figsize=(20,10)) 
    churn_diagram = df['Churn'].hist()
    figure = churn_diagram.get_figure()
    figure.savefig('images/eda/churn_diagram.png')
        
    plt.figure(figsize=(20,10)) 
    age_diagram = df['Customer_Age'].hist()
    figure = age_diagram.get_figure()
    figure.savefig('images/eda/age_diagram.png')
    
    plt.figure(figsize=(20,10)) 
    marital_plot = df.Marital_Status.value_counts('normalize').plot(kind='bar')
    figure = marital_plot.get_figure()
    figure.savefig('images/eda/marital_plot.png')
    
    plt.figure(figsize=(20,10)) 
    # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained using a kernel density estimate
    total_diagram = sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    figure = total_diagram.get_figure()
    figure.savefig('images/eda/total_diagram.png')
    
    plt.figure(figsize=(20,10)) 
    heatmap_diagram = sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    figure = heatmap_diagram.get_figure()
    figure.savefig('images/eda/heatmap_diagram.png')


def encoder_helper(df, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for feature in category_lst:
        lst = []
        feature_groups = df.groupby(feature).mean()['Churn']

        for val in df[feature]:
            lst.append(feature_groups.loc[val])
        name_plus_churn = feature + "_Churn"
        
        df[name_plus_churn] = lst   
    
    return df


def perform_feature_engineering(df):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    category_lst = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
            'Total_Relationship_Count', 'Months_Inactive_12_mon',
            'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
            'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
            'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
            'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
            'Income_Category_Churn', 'Card_Category_Churn']
    
    y = df['Churn']
    X = pd.DataFrame()

    df = encoder_helper(df, category_lst)
    
    X[keep_cols] = df[keep_cols]
    X.head()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)
    return X_train, X_test, y_train, y_test, X



def classification_report_image(y_train,
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
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig("images/results/random_forest_classification_report.png")
    plt.clf()
    
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig("images/results/logistic_regression_classification_report.png")
    plt.clf()
    


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
    plt.figure(figsize=(20,5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)
  
    

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
    
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = { 
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }
    
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    # plots
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig("images/results/roc_lrc_plot.png")
    plt.show()
    
    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')
    
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')
    
    lrc_plot = plot_roc_curve(lr_model, X_test, y_test)
    
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(rfc_model, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig("images/results/roc_lrc_plot_rfc.png")
    plt.show()

    
    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    
    feature_importance_plot(cv_rfc, X, "images/results/feature_importance_plot.png")   
    classification_report_image(y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf)

    
    
    
if __name__ == "__main__":
    
    for f in os.listdir("./images/eda"):
        os.remove(os.path.join("./images/eda", f))
    for f in os.listdir("images/results"):
        os.remove(os.path.join("images/results", f))
    for f in os.listdir("models"):
        os.remove(os.path.join("models", f))
    
    df = import_data(r"./data/bank_data.csv")
    
    perform_eda(df)
    
    X_train, X_test, y_train, y_test, X = perform_feature_engineering(df)
    
    train_models(X_train, X_test, y_train, y_test, X)