import subprocess
import time
import numpy as np
from subprocess import PIPE

modes = ["greedy", "local"]
data_set_names = ["ecoli","rand", "iris"]
restr_level = ["10", "20"]
seeds = ["123","456","789","101112","131415"]

lambda_mod = "1"
mode = "local"

name_file = "local_results"
f = open("./" + name_file + ".csv", "w")
for name in data_set_names:

    for restr in restr_level:
        f.write(str(name+"  Restr: "+ restr+"%"))
        f.write('\n')
        f.write("Seed,Tasa_C,Tasa_inf,Agr.,T")
        f.write('\n')
        for seed in seeds:

            aver_time = []

            for i in range(10):
                process = subprocess.run(["python3", "main.py", mode, name, restr, seed, lambda_mod], stdout=PIPE, stderr=PIPE, universal_newlines=True)
                print("Mode:", mode, "   Dataset: ", name,"  with rest.level: ", restr, "  and seed: ", seed)

                ex = process.stdout.replace('Tasa C: ', '').replace('Tasa Inf: ', '').replace('Agr: ', '').replace('Time: ', '').split('\n')
                ex.pop()
                aver_time.append(ex[3])

            ex[3] = np.mean(np.array(aver_time).astype(np.float))
            ex = [seed]+ex
            # print(ex)

            f.write(', '.join(map(str, ex)))
            f.write('\n')

    f.write('\n')

mode = "greedy"
name_file = "greedy_results"
f = open("./" + name_file + ".csv", "w")
for name in data_set_names:

    for restr in restr_level:
        f.write(str(name+"  Restr: " + restr+"%"))
        f.write('\n')
        f.write("Seed,Tasa_C,Tasa_inf,T")
        f.write('\n')
        for seed in seeds:

            aver_time = []

            for i in range(10):
                process = subprocess.run(["python3", "main.py", mode, name, restr, seed, lambda_mod], stdout=PIPE, stderr=PIPE, universal_newlines=True)
                print("Mode:", mode, "   Dataset: ", name,"  with rest.level: ", restr, "  and seed: ", seed)

                ex = process.stdout.replace('Tasa C: ', '').replace('Tasa Inf: ', '').replace('Time: ', '').split('\n')
                ex.pop()
                aver_time.append(ex[2])

            ex[2] = np.mean(np.array(aver_time).astype(np.float))
            ex = [seed]+ex
            # print(ex)

            f.write(', '.join(map(str, ex)))
            f.write('\n')

    f.write('\n')

