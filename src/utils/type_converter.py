import re

import numpy as np
import torch


class TypeConverter:
    """Static class to convert strings to types"""

    @staticmethod
    def extract_dtype(dtype_str: str) -> str:
        """Extract type essentials from given string

        Args:
            dtype_str (str): String to extract the type from

        Raises:
            ValueError: If the input type does not match the supported types.

        Returns:
            str: Extracted dtype e.g. 'float32' or 'int8'
        """
        dtype_str = dtype_str.lower()

        match = re.match(
            r"(?:.*((?:(?:float|int)(\d\d?)|double|half|float|long|int)).*)", dtype_str
        )
        if match is None:
            raise ValueError(
                f'The given string: "{dtype_str}", cannot be converted to a python data type.'
            )

        type = match[1]
        number = match[2]

        if number is not None:
            number = int(number)
            base = np.log2(number)
            if base != int(base):
                raise ValueError(f"Number {number} in {type} should power of two.")

        return match[1]

    @staticmethod
    def str_to_numpy(dtype_str: str) -> np.dtype:
        """Convert type string to numpy type

        Args:
            dtype_str (str): string describing the type

        Returns:
            np.dtype: numpy type
        """

        str_type = TypeConverter.extract_dtype(dtype_str=dtype_str)
        return getattr(np, str_type)

    @staticmethod
    def str_to_torch(dtype_str: str) -> torch.dtype:
        """Convert type string to torch type

        Args:
            dtype_str (str): string describing the type

        Returns:
            torch.dtype: torch dtype
        """

        str_type = TypeConverter.extract_dtype(dtype_str=dtype_str)
        return getattr(torch, str_type)

    @staticmethod
    def type_to_str(type: torch.dtype | np.dtype) -> str:
        """Convert type to string

        Args:
            type (torch.dtype | np.dtype): numpy or torch dtype

        Returns:
            str: string of the type
        """

        str_type = str(type)
        str_type = TypeConverter.extract_dtype(str_type)
        return str_type


if __name__ == "__main__":
    print(TypeConverter.extract_dtype("torch.float16"))
    print(TypeConverter.extract_dtype("numpy.float32"))
    print(TypeConverter.extract_dtype("numpy.double"))
    print(TypeConverter.type_to_str(np.double))
