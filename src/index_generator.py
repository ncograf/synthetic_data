import base_generator
import pandas as pd
import numpy as np
import numpy.typing as npt

class IndexGenerator:

    def __init__(self, generator : base_generator.BaseGenerator):
        self._generator = generator
        
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
        for col in data.columns:
            train_data = data.loc[:, col]
            mask = ~np.isnan(train_data)
            self._generator.fit_model(train_data[mask])
            n_samples = train_data[mask].shape[0]
            temp_series, return_series = self._generator.generate_data(len=n_samples, burn=500)
            generated_data.loc[mask,col] = temp_series
        
        return generated_data
        
        