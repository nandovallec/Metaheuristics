import subprocess
import time
import numpy as np
from subprocess import PIPE

data_set_names = [ "iris", "ecoli","rand", "newthyroid"]
# data_set_names = [ "iris", "rand"]

restr_level = ["10", "20"]
seeds = ["123","456","789","101112","131415"]
programs = ["bmb", "ils"]
# programs = ["bmb"]

lambda_mod = "1"

for name_file in programs:
    f = open("./Results/" + name_file + ".csv", "w")
    program_name = str(name_file+".py")
    for restr in restr_level:
        f.write(",")
        for name in data_set_names:
            f.write(str(name + "  Restr: " + restr + "%"))
            f.write(",,,,")
        f.write('\n')
        f.write("Seed,")
        for name in data_set_names:
            f.write("Tasa_C,Tasa_inf,Agr.,T,")
        f.write('\n')
        for seed in seeds:
            f.write(str(seed+","))

            for name in data_set_names:
                aver_time = []

                for i in range(1):
                    process = subprocess.run(["python3", program_name, name, restr, seed, lambda_mod], stdout=PIPE, stderr=PIPE, universal_newlines=True)
                    print(" Dataset: ", name,"  with rest.level: ", restr, "  and seed: ", seed)

                    ex = process.stdout.replace('Tasa C: ', '').replace('Tasa Inf: ', '').replace('Agr: ', '').replace('Time: ', '').split('\n')
                    ex.pop()
                    aver_time.append(ex[3])

                f.write(', '.join(map(str, ex)))
                f.write(', ')
            f.write('\n')

        f.write('\n')
        f.write('\n')
        f.write('\n')

programs = ["annealing", "ils-annealing"]

for name_file in programs:
    f = open("./Results/" + name_file + ".csv", "w")
    program_name = str(name_file + ".py")
    for restr in restr_level:
        f.write(",")
        for name in data_set_names:
            f.write(str(name + "  Restr: " + restr + "%"))
            f.write(",,,,")
        f.write('\n')
        f.write("Seed,")
        for name in data_set_names:
            f.write("Tasa_C,Tasa_inf,Agr.,T,")
        f.write('\n')
        for seed in seeds:
            f.write(str(seed + ","))

            for name in data_set_names:
                aver_time = []

                for i in range(1):
                    process = subprocess.run(["python3", program_name, name, restr, seed, lambda_mod, "si", "0.95"],
                                             stdout=PIPE, stderr=PIPE, universal_newlines=True)
                    print("Dataset: ", name, "  with rest.level: ", restr, "  and seed: ", seed)

                    ex = process.stdout.replace('Tasa C: ', '').replace('Tasa Inf: ', '').replace('Agr: ', '').replace(
                        'Time: ', '').split('\n')
                    print(ex)
                    ex.pop()
                    aver_time.append(ex[3])

                f.write(', '.join(map(str, ex)))
                f.write(', ')
            f.write('\n')

        f.write('\n')
        f.write('\n')
        f.write('\n')

alfas = ["0.95", "0.98"]

for alfa in alfas:
    for name_file in programs:
        f = open("./Results/" + name_file+alfa + ".csv", "w")
        program_name = str(name_file+".py")
        for restr in restr_level:
            f.write(",")
            for name in data_set_names:
                f.write(str(name + "  Restr: " + restr + "%"))
                f.write(",,,,")
            f.write('\n')
            f.write("Seed,")
            for name in data_set_names:
                f.write("Tasa_C,Tasa_inf,Agr.,T,")
            f.write('\n')
            for seed in seeds:
                f.write(str(seed+","))

                for name in data_set_names:
                    aver_time = []

                    for i in range(1):
                        process = subprocess.run(["python3", program_name, name, restr, seed, lambda_mod, "no", alfa], stdout=PIPE, stderr=PIPE, universal_newlines=True)
                        print("Dataset: ", name,"  with rest.level: ", restr, "  and seed: ", seed)

                        ex = process.stdout.replace('Tasa C: ', '').replace('Tasa Inf: ', '').replace('Agr: ', '').replace('Time: ', '').split('\n')
                        print(ex)
                        ex.pop()
                        aver_time.append(ex[3])

                    f.write(', '.join(map(str, ex)))
                    f.write(', ')
                f.write('\n')

            f.write('\n')
            f.write('\n')
            f.write('\n')

