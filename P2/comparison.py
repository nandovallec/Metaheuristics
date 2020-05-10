import subprocess
import time
import numpy as np
from subprocess import PIPE

modes = ["steady", "generational"]
data_set_names = ["newthyroid","ecoli","rand", "iris"]
restr_level = ["10", "20"]
seeds = ["123","456","789","101112","131415"]
lambda_mod = "1"
n_population = "50"
mutation_prob = "0.001"
uniform = ["si","no","no"]
two_point = ["no","si","no"]

for mode in modes:
    for name in data_set_names:
        for restr in restr_level:
            for seed in seeds:
                for i in range(3):
                        process = subprocess.run(["python3", "main.py", mode, name, restr, seed, lambda_mod,
                                                  n_population, mutation_prob, uniform[i], two_point[i]], stdout=PIPE,
                                                 stderr=PIPE, universal_newlines=True)

modes = ["random", "best", "all"]
interval = "10"
subset = "0.1"
for mode in modes:
    for name in data_set_names:
        for restr in restr_level:
            for seed in seeds:
                for i in range(3):
                    process = subprocess.run(["python3", "memetic.py", mode, name, restr, seed, lambda_mod,
                                          n_population, mutation_prob, uniform[i], two_point[i], interval, subset], stdout=PIPE,
                                         stderr=PIPE, universal_newlines=True)
