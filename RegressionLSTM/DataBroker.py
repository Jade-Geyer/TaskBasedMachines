
import numpy as np
import pandas as pd


class DataBroker:

    def assemble_random_training_data(self, nrows: int, ncols: int) -> pd.DataFrame:
        # Generate random data with a recognizable pattern
        data = np.random.rand(nrows, ncols)
        mid = nrows // 2
        data[:mid, 0] += np.linspace(0, 1, mid)
        data[mid:, 0] -= np.linspace(0, 1, nrows - mid)

        # Create a DataFrame from the data
        df = pd.DataFrame(data, columns=[f'col_{i}' for i in range(ncols)])

        return df

