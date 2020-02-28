import pandas as pd
import numpy as np


def is_infeasible(pair):
    if restrictions[pair[0]][pair[1]] == -1:
        return True
    else:
        return False


data_path = "./ecoli_set.dat"
data = pd.read_csv(data_path, header=None)

# print(data.head())

restrictions_path = "./ecoli_set_const_10.const"
restrictions = pd.read_csv(restrictions_path, header=None)

# print(restrictions.head())

maxim = []

minim = data.min()
print(minim)
maxim = data.max()
print(maxim)
# for i in range(data.shape[1]):
#     print(i)
#     minim[i] = np.min(data.loc[:, i])
#     maxim[i] = np.max(data.loc[:, i])
#     print(minim[i], "  ", maxim[i])
#





