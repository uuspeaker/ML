import pandas as pd
import os

HOUSING_PATH = "../dataset/housing"

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    house = pd.read_csv(csv_path)
    return pd.read_csv(csv_path)

housing = load_housing_data()
print(housing.head(),housing.info())
print(housing.ocean_proximity.value_counts())
print(housing.describe())

import matplotlib.pyplot as plt
# housing.hist(bins=100, figsize=(20,15))
# plt.show()

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(train_set.shape, test_set.shape)

import numpy as np
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print(strat_train_set.shape, strat_test_set.shape)

# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
# s=housing["population"]/200, label="population",
# c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
# )
# plt.legend()
# plt.show()

corr_matrix = housing.corr()
relation = corr_matrix["median_house_value"].sort_values(ascending=False)
print(relation)

from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income",
"housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(8, 6))

housing.plot(kind="scatter", x="median_income", y="median_house_value",
alpha=0.1)
plt.show()