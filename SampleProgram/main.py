from RegressionLGBM.LgbmModel import LightGBMModel
from RegressionLSTM.KerasNuralNetworkPredict import KerasNuralNetworkPredict


def main():
    k = KerasNuralNetworkPredict()
    prediction = k.get_prediction()
    print(f"prediction: {prediction}")

    l = LightGBMModel()
    prediction = l.predict()
    print("prediction[0]:", prediction[0])


if __name__ == '__main__':
    main()
