import pandas as pd

df = pd.DataFrame({
    "name": ["Alice", "Bob", "Chen"],
    "age": [23, 31, 29],
    "city": ["Toronto", "Waterloo", "Ottawa"],
})

# From a CSV
df2 = pd.read_csv("path/to/file.csv")

# Peek / basics
df.head()
df.info()
df["age"].mean()