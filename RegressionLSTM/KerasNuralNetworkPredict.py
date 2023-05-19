import time

import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from keras_tuner import RandomSearch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from DataBroker.DataBroker import DataBroker


class KerasNuralNetworkPredict:

    def __init__(self, timesteps=15):
        self.__model = None
        self.__batch_size = 32
        self.__epochs = 3
        self.__X_train = None
        self.__y_train = None
        self.__X_test = None
        self.__y_test = None
        self.model_history = None
        self.__timesteps = timesteps
        self.__gather_training_data()
        self.__tuner = RandomSearch(
            self.__build_model,
            objective='val_loss',
            max_trials=5,
            executions_per_trial=2,
            directory='Tuner',
            project_name='RageArbit')
        self.__build_tuned_model()
        if self.__model:
            self.__model.save('Keras_1Layer_A.h5')
        else:
            self.__model = load_model('Keras_1Layer_A.h5')

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

        for i in range(self.__timesteps, len(df) - self.__timesteps):
            X.append(df.iloc[i - self.__timesteps:i, :-1].values)

            # create a target sample by taking X values in a segment
            y.append(df.iloc[i + self.__timesteps, -1])

        # convert the lists to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.__X_train = X_train
        self.__X_test = X_test
        self.__y_train = y_train
        self.__y_test = y_test

    # Gets a new sample for tests.

    def __build_model(self, hp) -> Sequential:
        # use hp.Choice to define the __batch_size hyperparameter
        self.__batch_size = hp.Choice('__batch_size', values=[32, 64, 128])
        # use hp.Int to define the __epochs hyperparameter
        self.__epochs = hp.Int('__epochs', min_value=10, max_value=50, step=5)

        model = Sequential()
        # define the LSTM __model.txt
        model.add(LSTM(hp.Int('units', min_value=8, max_value=64, step=8),
                       input_shape=(self.__X_train.shape[1], self.__X_train.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        # train the __model.txt
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        # __model.txt history may be useful to some, not currently used
        self.model_history = model.fit(self.__X_train, self.__y_train, epochs=self.__epochs,
                                       validation_data=(self.__X_test, self.__y_test), callbacks=[early_stopping])
        self.__model = model
        return model

    def __build_tuned_model(self) -> None:
        self.__tuner.search(
            self.__X_train,
            self.__y_train,
            batch_size=self.__batch_size,
            epochs=self.__epochs,
            validation_data=(self.__X_test, self.__y_test)
        )

    def get_prediction(self, X_test=None, y_test=None) -> float:

        if not X_test or not y_test:
            X_test = self.__X_test
            y_test = self.__y_test

        # evaluate the __model.txt on the test data
        test_loss = self.__model.evaluate(X_test, y_test)

        # print the test loss
        print(f'Test loss: {test_loss}')

        # generate predictions for the test data
        y_pred = self.__model.predict(self.__X_test)
        print("PREDICTIONS:\n", y_pred)

        errors = self.__y_test - y_pred
        max_abs_error = np.max(np.abs(errors))
        print(f"Maximum absolute error: {max_abs_error}")

        # plot a histogram of the errors
        plt.hist(errors, bins=20)
        plt.xlabel("Error")
        plt.ylabel("Count")
        plt.show()

        # calculate performance metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        mse = mean_squared_error(self.__y_test, y_pred)
        mae = mean_absolute_error(self.__y_test, y_pred)

        # print the performance metrics
        print(f'Mean squared error: {mse}')
        print(f'Mean absolute error: {mae}')

        return y_pred
