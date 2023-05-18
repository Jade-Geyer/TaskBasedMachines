from RegressionLSTM.KerasNuralNetworkPredict import KerasNuralNetworkPredict


def main():
    k = KerasNuralNetworkPredict()
    prediction = k.get_prediction()
    print(f"prediction: {prediction}")


if __name__ == '__main__':
    main()
