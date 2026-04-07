import pandas as pd

data = pd.read_csv("data/Restaurant_Reviews.tsv", delimiter="\t", quoting = 3)
print(data)