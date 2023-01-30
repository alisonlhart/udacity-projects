import os
import logging
from churn_library import import_data, perform_eda, encoder_helper, perform_feature_engineering, train_models


def test_import(import_data):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = import_data("./data/bank_data.csv")
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err
	return df


def test_eda(df, perform_eda):
	'''
	test perform eda function
	'''
	try: 
		perform_eda(df)
		assert(os.path.exists('images/eda/churn_diagram.png'))
		logging.info("Testing test_eda: images/eda/churn_diagram.png exists")
		assert(os.path.exists('images/eda/age_diagram.png'))
		logging.info("Testing test_eda: images/eda/age_diagram.png exists")
		assert(os.path.exists('images/eda/heatmap_diagram.png'))
		logging.info("Testing test_eda: images/eda/heatmap_diagram.png exists")
		assert(os.path.exists('images/eda/marital_plot.png'))
		logging.info("Testing test_eda: images/eda/marital_plot.png exists")
		assert(os.path.exists('images/eda/total_diagram.png'))
		logging.info("Testing test_eda: images/eda/total_diagram.png exists")
		logging.info("Testing perform_eda: SUCCESS")

	except AssertionError as err:
		logging.error("Testing perform_eda: EDA Diagram is missing.")
		raise err
		
  
def test_encoder_helper(df, encoder_helper):
	'''
	test encoder helper
	'''
	category_list =['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
	try:
		df = encoder_helper(df, category_list)
		assert(df)
		logging.info("Testing test_encoder_helper: df exists")
		logging.info("Testing test_encoder_helper: SUCCESS")
	except NameError as err:
		logging.error("Testing test_encoder_helper: Dataframe wasn't returned from encoder_helper")  
		raise err


def test_perform_feature_engineering(df, perform_feature_engineering):
	'''
	test perform_feature_engineering
	'''
	try:
		X_train, X_test, y_train, y_test, X = perform_feature_engineering(df)
		assert(X_train.any)
		logging.info("Testing test_perform_feature_engineering: X_train exists with data")
		assert(X_test.any)
		logging.info("Testing test_perform_feature_engineering: X_test exists with data")
		assert(y_train.any)
		logging.info("Testing test_perform_feature_engineering: y_train exists with data")
		assert(y_test.any)
		logging.info("Testing test_perform_feature_engineering: y_test exists with data")
		assert(X.any)
		logging.info("Testing test_perform_feature_engineering: X exists with data")
		logging.info("Testing test_perform_feature_engineering: SUCCESS")
	except AssertionError as err:
		logging.error("Testing test_perform_feature_engineering: Values were not returned.")
		raise err
	return X_train, X_test, y_train, y_test, X

def test_train_models(X_train, X_test, y_train, y_test, X, train_models):
	'''
	test train_models
	'''
	try:
		train_models(X_train, X_test, y_train, y_test, X)
		assert(os.path.exists("images/results/random_forest_classification_report.png"))
		logging.info("Testing test_train_models: images/results/random_forest_classification_report.png exists")
		assert(os.path.exists("images/results/logistic_regression_classification_report.png"))
		logging.info("Testing test_train_models: images/results/logistic_regression_classification_report.png exists")
		assert(os.path.exists("images/results/roc_lrc_plot_rfc.png"))
		logging.info("Testing test_train_models: images/results/roc_lrc_plot_rfc.png exists")
		assert(os.path.exists("images/results/roc_lrc_plot.png"))
		logging.info("Testing test_train_models: images/results/roc_lrc_plot.png exists")
		assert(os.path.exists('./models/rfc_model.pkl'))
		logging.info("Testing test_train_models: models/rfc_model.pkl exists")
		assert(os.path.exists('./models/logistic_model.pkl'))
		logging.info("Testing test_train_models: models/logistic_model.pkl exists")
		assert(os.path.exists("images/results/feature_importance_plot.png"))
		logging.info("Testing test_train_models: images/results/feature_importance_plot.png exists")
		logging.info("Testing test_train_models: SUCCESS")
	except AssertionError as err:
		logging.error("Testing test_train_models: Report image not found")
		raise err
	

if __name__ == "__main__":

	# Clean up images and logs
	for f in os.listdir("./images/eda"):
		os.remove(os.path.join("./images/eda", f))
	for f in os.listdir("images/results"):
		os.remove(os.path.join("images/results", f))
	for f in os.listdir("models"):
		os.remove(os.path.join("models", f))
	for f in os.listdir("logs"):
		os.remove(os.path.join("logs", f))  
  
	logging.basicConfig(
	filename='./logs/churn_library.log',
	level = logging.INFO,
	filemode='w',
	format='%(name)s - %(levelname)s - %(message)s')
 
	logging.info("STARTING TESTS") 
		
	df = test_import(import_data)
	test_eda(df, perform_eda)
	X_train, X_test, y_train, y_test, X = test_perform_feature_engineering(df, perform_feature_engineering)
	test_train_models(X_train, X_test, y_train, y_test, X, train_models)
	
	
	
	








