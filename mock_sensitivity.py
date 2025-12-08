# Step 0: imports and seed
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

np.random.seed(42)


# Step 1: create 60 individuals split into 4 groups
n = 60
per_group = n // 4  # 15 each

groups = []
ages = []
avg_incomes = []
for group_idx in range(4):
    for i in range(per_group):
        if group_idx == 0:  # low income, young
            ages.append(int(np.random.normal(30, 3)))
            avg_incomes.append(int(np.random.normal(35000, 4000)))
            groups.append("low_young")
        elif group_idx == 1:  # low income, old
            ages.append(int(np.random.normal(60, 4)))
            avg_incomes.append(int(np.random.normal(34000, 4000)))
            groups.append("low_old")
        elif group_idx == 2:  # high income, young
            ages.append(int(np.random.normal(33, 3)))
            avg_incomes.append(int(np.random.normal(110000, 8000)))
            groups.append("high_young")
        else:  # high income, old
            ages.append(int(np.random.normal(62, 4)))
            avg_incomes.append(int(np.random.normal(115000, 8000)))
            groups.append("high_old")

df = pd.DataFrame(
    {"id": np.arange(n), "group": groups, "age": ages, "avg_income": avg_incomes}
)

df.head()
