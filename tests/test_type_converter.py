import numpy as np
import torch
from type_converter import TypeConverter


class TestTypeConverter:
    def test_extract_dtype(self):
        running_tests = [
            "numpy.int8",
            "numpy.int16",
            "numpy.int32",
            "numpy.int64",
            "numpy.float16",
            "numpy.float32",
            "numpy.float64",
            "torch.int8",
            "torch.int16",
            "torch.int32",
            "torch.int64",
            "torch.float16",
            "torch.float32",
            "torch.float64",
        ]
        return_values = [
            "int8",
            "int16",
            "int32",
            "int64",
            "float16",
            "float32",
            "float64",
            "int8",
            "int16",
            "int32",
            "int64",
            "float16",
            "float32",
            "float64",
        ]

        for test_in, test_out in zip(running_tests, return_values):
            assert TypeConverter.extract_dtype(test_in) == test_out

        test_wong_index = "numpy.float46"
        try:
            TypeConverter.extract_dtype(test_wong_index)
        except ValueError as e:
            if "power of two" not in str(e):
                assert False, "Only powers of two must be accepted"
        except:  # noqa E722
            assert False, "Only powers of two must be accepted"

        test_wong_type = "numpy.bool"
        try:
            TypeConverter.extract_dtype(test_wong_type)
        except ValueError as e:
            if "data type" not in str(e):
                assert False, "Only int and float types are accepted"
        except:  # noqa E722
            assert False, "Only int and float types are accepted"

    def test_str_to_numpy(self):
        test_in = [
            "torch.float64",
            "torch.float32",
            "torch.float16",
            "....int16",
            "....int8",
        ]

        test_out = [
            np.float64,
            np.float32,
            np.float16,
            np.int16,
            np.int8,
        ]

        for t_in, t_out in zip(test_in, test_out):
            assert TypeConverter.str_to_numpy(t_in) == t_out

    def test_str_to_torch(self):
        test_in = [
            "numpy.float64",
            "numpy.float32",
            "numpy.float16",
            "....int16",
            "....int8",
        ]

        test_out = [
            torch.float64,
            torch.float32,
            torch.float16,
            torch.int16,
            torch.int8,
        ]

        for t_in, t_out in zip(test_in, test_out):
            assert TypeConverter.str_to_torch(t_in) == t_out

    def test_type_to_str(self):
        test_in = [
            torch.float64,
            torch.float32,
            torch.float16,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            np.float64,
            np.float32,
            np.float16,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
        ]

        test_out = [
            "float64",
            "float32",
            "float16",
            "int8",
            "int16",
            "int32",
            "int64",
        ]

        for t_in, t_out in zip(test_in, test_out):
            assert TypeConverter.type_to_str(t_in) == t_out
