import os

import lightgbm as lgb
import optuna
from keras.losses import mean_squared_error
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from DataBroker.DataBroker import DataBroker


def create_objective(X_train, X_test, y_train, y_test):
    def objective(trial):

        params = {
            'objective': 'regression',
            'metric': 'mse',
            'num_leaves': trial.suggest_int('num_leaves', 2, 50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, step=0.01),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.01, 1.0, step=0.01),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.01, 1.0, step=0.01),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 2, 50),
            'max_depth': trial.suggest_int('max_depth', -1, 10),
            'lambda_l1': 0.0,  # L1 regularization term on weights (default: 0.0)
            'lambda_l2': 0.0,  # L2 regularization term on weights (default: 0.0)
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.00, 1.0, step=0.01),
            'num_iterations': trial.suggest_int('num_iterations', 1, 200),
            'early_stopping_rounds': None,
            'verbose': -1  # level of verbosity (default: -1, silent)
        }

        training_dataset = lgb.Dataset(X_train, label=y_train)
        trained_model = lgb.train(params, training_dataset)

        y_pred = trained_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        return mse
    return objective


class LightGBMModel:

    def __init__(self, trials=100, skip_training=False):
        self._trained_model_file_path = "TrainedModels/model.txt"
        self.__show_plot = False
        self.__X_train = None
        self.__X_test = None
        self.__y_train = None
        self.__y_test = None
        if os.path.isfile('model.txt'):
            self.__gbm = lgb.Booster(model_file=self._trained_model_file_path)
        self.__broker = DataBroker()
        self.__training_data = self.__broker.assemble_random_training_data(1000, 15)
        self.__testing_data = None
        self.__split_training_data()
        self.__trials = trials
        if not skip_training:
            self.__train_model()
            self.__show_plot = False


    def assign_training_data(self, training_data):
        self.__training_data = training_data

    def assign_testing_data(self, testing_data):
        self.__testing_data = testing_data

    def __split_training_data(self):
        X = self.__training_data.drop(self.__training_data.columns[-1], axis=1)
        y = self.__training_data.iloc[:, -1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def __train_model(self):

        # Create LightGBM dataset objects for training and test sets
        raw_dataset = lgb.Dataset(self.X_train, label=self.y_train)

        my_objective = create_objective(self.X_train, self.X_test, self.y_train, self.y_test)
        study = optuna.create_study(direction='minimize')
        study.optimize(my_objective, n_trials=self.__trials)
        params = study.best_params

        self.__gbm = lgb.train(params, raw_dataset)

        # Make predictions on training and test set
        y_train_pred = self.__gbm.predict(self.X_train)
        y_test_pred = self.__gbm.predict(self.X_test)
        test_indexes = list(range(0, len(y_test_pred)))
        residuals = self.y_test - y_train_pred[test_indexes]

        if self.__show_plot:
            plt.scatter(y_train_pred[test_indexes], residuals)
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.axhline(y=0, color='r', linestyle='-')
            plt.title('Residual Plot')
            plt.show()

        # Calculate MSRE on training and test set
        msre_train = mean_squared_error(self.y_train, y_train_pred)
        msre_test = mean_squared_error(self.y_test, y_test_pred)
        print('MSE on training set: %.4f' % msre_train)
        print('MSE on test set: %.4f' % msre_test)


        self.__gbm.save_model(self._trained_model_file_path)


    def predict(self):
        prediction_data = self.__gbm.predict(self.X_train)
        return prediction_data




