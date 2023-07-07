import os
import sys
import time
import warnings
import numpy as np
import joblib as jl
import sklearn as sk
import sklearn.ensemble as ske
import sklearn.model_selection as skm

warnings.filterwarnings("ignore")


def regression(input_dataframe, output_dataframe, n_folds, n_metrics_columns, n_reg_models):

    start_time = time.time()

    dummy = input_dataframe.to_numpy()
    y_train = output_dataframe.to_numpy()

    total_reg_metrics = n_metrics_columns * n_reg_models  # Sheet -Results- in Tiresias_Models total number of columns

    results = np.zeros([len(output_dataframe.columns), total_reg_metrics], dtype=float)  # Matrix displayed in -Results-
    alpha_tree = 0.0  # To quickly test Trees or SVR Regularization (ccp_alpha or C). Default 0
    alpha_reg = alpha_tree  # Same, for regularization of polynomials. Default 0
    print('Regression models fitting and evaluation in progress...')

    for n in range(len(output_dataframe.columns)):

        for Selected_Model in range(n_reg_models):

            model_name = "No model"  # In case of errors
            model_0 = "No model"
            x_train = dummy  # To reset the variables, as polynomial features modifies them

            # Regression model selection
            if Selected_Model == 0:
                model_0 = ske.RandomForestRegressor(n_estimators=100, ccp_alpha=alpha_tree)  # Default 100 , 0
                model_name = "Random Forest"
            if Selected_Model == 1:
                model_0 = sk.linear_model.Ridge(alpha=alpha_tree)  # Default 0}
                model_name = "Linear Regression"
            if Selected_Model == 2:
                model_0 = sk.svm.SVR(C=1.0 - alpha_tree)  # Default 1
                model_name = "Support Vector Regression (SVR)"
            if Selected_Model == 3:
                model_0 = ske.GradientBoostingRegressor(max_depth=3, ccp_alpha=alpha_tree)  # Default 3, 0
                model_name = "Gradient Boost Regression"
            if Selected_Model == 4:
                model_0 = ske.AdaBoostRegressor(sk.tree.DecisionTreeRegressor(max_depth=None,
                                                                              ccp_alpha=alpha_tree))  # Default None, 0
                model_name = "Ada Boost Regression"
            if Selected_Model == 5:
                model_0 = sk.tree.DecisionTreeRegressor(max_depth=None, ccp_alpha=alpha_tree)  # Default None , 0
                model_name = "Decision Tree Regression"
            if Selected_Model == 6:
                poly = sk.preprocessing.PolynomialFeatures(degree=2)  # Second order polynomial needs matrix transformation
                x_train = poly.fit_transform(x_train)
                model_0 = sk.linear_model.Ridge(alpha=alpha_reg)  # Default 0
                model_name = "Second Order Polynomial Regression"
            if Selected_Model == 7:
                poly = sk.preprocessing.PolynomialFeatures(degree=3)
                x_train = poly.fit_transform(x_train)
                model_0 = sk.linear_model.Ridge(alpha=alpha_reg)  # Default 0
                model_name = "Third Order Polynomial Regression"
            if Selected_Model == 8:
                poly = sk.preprocessing.PolynomialFeatures(degree=4)
                x_train = poly.fit_transform(x_train)
                model_0 = sk.linear_model.Ridge(alpha=alpha_reg)  # Default 0
                model_name = "Fourth Order Polynomial Regression"

            sys.stdout.write(
                "\r\rVariable %i of %i (%s). Actual model: %s\033[K" % (n + 1, len(output_dataframe.columns),
                                                                        output_dataframe.columns[n], model_name))
            sys.stdout.flush()

            # Data normalization and Cross Validation with n_folds Folds
            model = model_0
            cv_skf = skm.KFold(n_splits=n_folds, shuffle=True, random_state=4)

            model_scores = skm.cross_validate(model, x_train, y_train[:, n], cv=cv_skf, scoring=[
                'neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'],
                                              return_train_score=True)

            model.fit(x_train, y_train[:, n])

            jl.dump(model, os.getcwd() + '\\Model Data' + '\\' + model_name + str(n))  # Saving model parameters in folder

            # Fills the metrics for the cross validation of the regression models in the Results matrix
            results[n, Selected_Model * n_metrics_columns + 0] = np.mean(abs(model_scores['train_neg_mean_squared_error'])) ** (1 / 2)
            results[n, Selected_Model * n_metrics_columns + 1] = np.std(abs(model_scores['train_neg_mean_squared_error'])) ** (1 / 2)
            results[n, Selected_Model * n_metrics_columns + 2] = np.mean(abs(model_scores['train_neg_mean_absolute_error']))
            results[n, Selected_Model * n_metrics_columns + 3] = np.std(abs(model_scores['train_neg_mean_absolute_error']))
            results[n, Selected_Model * n_metrics_columns + 4] = np.mean(abs(model_scores['train_r2']))
            results[n, Selected_Model * n_metrics_columns + 5] = np.mean(abs(model_scores['test_neg_mean_squared_error'])) ** (1 / 2)
            results[n, Selected_Model * n_metrics_columns + 6] = np.std(abs(model_scores['test_neg_mean_squared_error'])) ** (1 / 2)
            results[n, Selected_Model * n_metrics_columns + 7] = np.mean(abs(model_scores['test_neg_mean_absolute_error']))
            results[n, Selected_Model * n_metrics_columns + 8] = np.std(abs(model_scores['test_neg_mean_absolute_error']))
            results[n, Selected_Model * n_metrics_columns + 9] = np.mean(abs(model_scores['test_r2']))

    sys.stdout.write("\r\rFitting for all the regression models has finished!\033[K\n\n")
    sys.stdout.flush()

    time_reg = time.time() - start_time

    return time_reg, results

