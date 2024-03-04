import pandas as pd
data = pd.read_csv("tests/testdata/HUBB.csv")
print(data.dtypes)
data = pd.read_csv("tests/testdata/local/ABBV.csv")
print(data.dtypes)