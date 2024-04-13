import base_generator
import pandas as pd
import numpy as np

class IndexGenerator:

    def __init__(self, generator : base_generator.BaseGenerator):
        self._generator = generator
        self._problematic_cols = []
        
    def generate_index(self, data : pd.DataFrame) -> pd.DataFrame:
        """Generates synthetic data for the whole index

        Args:
            data (pd.DataFrame): All data in the index

        Raises:
            ValueError: data must be a DataFrame

        Returns:
            pd.DataFrame : DataFrame containing all the columns
        """
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a DataFrame")
        
        generated_data = data.copy(deep=True)
        self._problematic_cols = []
        for col in data.columns:
            train_data = data.loc[:, col]
            mask = ~np.isnan(train_data)
            self._generator.fit_model(train_data[mask])
            n_samples = train_data[mask].shape[0]
            temp_series, _ = self._generator.generate_data(len=n_samples, burn=500)
            generated_data.loc[mask,col] = temp_series
            if np.isinf(temp_series).any():
                self._problematic_cols.append(col)
        
        print(f"Stocks containing infinity : {self._problematic_cols}.")
        
        return generated_data
        
        