import time

import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from keras_tuner import RandomSearch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from RegressionLSTM.DataBroker import DataBroker


class KerasNuralNetworkPredict:

    def __init__(self, timesteps=15):
        self.model = None
        self.batch_size = 32
        self.epochs = 3
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model_history = None
        self.timesteps = timesteps
        self.__gather_training_data()
        self.tuner = RandomSearch(
            self.__build_model,
            objective='val_loss',
            max_trials=5,
            executions_per_trial=2,
            directory='Tuner',
            project_name='RageArbit')
        self.__build_tuned_model()
        if self.model:
            self.model.save('Keras_1Layer_A.h5')
        else:
            self.model = load_model('Keras_1Layer_A.h5')

    def __gather_training_data(self) -> None:

        # Get the training data
        db = DataBroker()
        df = db.assemble_random_training_data(500, 30)
        time.sleep(1)

        # normalize the data
        mean = df.mean()
        std = df.std()
        df = (df - mean) / std

        # create empty lists to store the input and target data
        X = []
        y = []

        for i in range(self.timesteps, len(df) - self.timesteps):
            X.append(df.iloc[i - self.timesteps:i, :-1].values)

            # create a target sample by taking X values in a segment
            y.append(df.iloc[i + self.timesteps, -1])

        # convert the lists to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    # Gets a new sample for tests.

    def __build_model(self, hp) -> Sequential:
        # use hp.Choice to define the batch_size hyperparameter
        self.batch_size = hp.Choice('batch_size', values=[32, 64, 128])
        # use hp.Int to define the epochs hyperparameter
        self.epochs = hp.Int('epochs', min_value=10, max_value=50, step=5)

        model = Sequential()
        # define the LSTM model
        model.add(LSTM(hp.Int('units', min_value=8, max_value=64, step=8),
                       input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        # train the model
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        # model history may be useful to some, not currently used
        self.model_history = model.fit(self.X_train, self.y_train, epochs=self.epochs,
                                  validation_data=(self.X_test, self.y_test), callbacks=[early_stopping])
        self.model = model
        return model

    def __build_tuned_model(self) -> None:
        self.tuner.search(
            self.X_train,
            self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(self.X_test, self.y_test)
        )

    def get_prediction(self, X_test=None, y_test=None) -> float:

        if not X_test or not y_test:
            X_test = self.X_test
            y_test = self.y_test

        # evaluate the model on the test data
        test_loss = self.model.evaluate(X_test, y_test)

        # print the test loss
        print(f'Test loss: {test_loss}')

        # generate predictions for the test data
        y_pred = self.model.predict(self.X_test)
        print("PREDICTIONS:\n", y_pred)

        errors = self.y_test - y_pred
        max_abs_error = np.max(np.abs(errors))
        print(f"Maximum absolute error: {max_abs_error}")

        # plot a histogram of the errors
        plt.hist(errors, bins=20)
        plt.xlabel("Error")
        plt.ylabel("Count")
        plt.show()

        # calculate performance metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)

        # print the performance metrics
        print(f'Mean squared error: {mse}')
        print(f'Mean absolute error: {mae}')

        return y_pred
