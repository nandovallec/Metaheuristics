import subprocess
import time
import numpy as np
from subprocess import PIPE

data_set_names = ["rand"]
restr_level = ["10", "20"]
seeds = ["123"]

lambda_mod = "1"
alphas = ["0.3","0.5","0.75"]
betas = ["0.05", "0.25", "0.5", "0.75"]
n_attacks = ["50", "100", "150"]

name_file = "results1"
f = open("./" + name_file + ".csv", "w")
for name in data_set_names:
    for restr in restr_level:
        for seed in seeds:
            for alpha in alphas:
                for beta in betas:
                    for n_attack in n_attacks:
                        process = subprocess.run(["python3", "main.py", name, restr, seed, lambda_mod, alpha, beta, n_attack], stdout=PIPE, stderr=PIPE, universal_newlines=True)
                        # print("####################")
                        say = ("Dataset: "+ name+ restr+ " alpha: "+ alpha+ " beta: "+ beta+ " n_attacks: "+ n_attack)
                        ex = process.stdout.replace('Tasa C: ', '').replace('Tasa Inf: ', '').replace('Agr: ', '').replace('Time: ', '').split('\n')
                        # ex = process.stdout
                        ex.pop()
                        # aver_time.append(ex[3])

                        # ex[3] = np.mean(np.array(aver_time).astype(np.float))
                        # ex = [seed]+ex
                        print(say)
                        print(ex)
                        # print(ex)
                        f.write(say)
                        f.write('\n')

                        f.write(', '.join(map(str, ex)))
                        f.write('\n')

    f.write('\n')
