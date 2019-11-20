import numpy as np
from loadData import load_housing_data

housing = load_housing_data()
housing.plot(kind="scatter", x="longitude", y="latitude")