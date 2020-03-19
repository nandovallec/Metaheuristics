import subprocess
import time
from subprocess import PIPE

modes = ["greedy", "local"]
data_set_names = ["ecoli","rand", "iris"]
restr_level = ["10", "20"]
seeds = ["123","456","789","101112","131415"]

start_time = time.perf_counter()

# process = subprocess.run(["python3", "main.py", "local", "ecoli", "10", "50"], stdout=PIPE, stderr=PIPE,
#                          universal_newlines=True)
elapsed_time = time.perf_counter() - start_time

print(elapsed_time)
for mode in modes:
    for name in data_set_names:
        for restr in restr_level:
            for seed in seeds:
                name_file = (mode + "_" + name + "_" + restr + "_" + seed)
                # f = open("./results/" + name_file + ".csv", "w")
                for i in range(5):
                    start_time = time.perf_counter()
                    process = subprocess.run(["python3", "main.py", mode, name, restr, seed], stdout=PIPE, stderr=PIPE, universal_newlines=True)
                    elapsed_time = time.perf_counter() - start_time
                    print("Mode:", mode, "   Dataset: ", name,"  with rest.level: ", restr, "  and seed: ", seed, "      It: ",i, " Time:" ,elapsed_time)


                    # ex = process.stdout.replace('Tasa C: ', '').replace('Tasa Inf: ', '').replace('Aggr: ', '').split(
                    #     '\n')
                    # ex.pop()
                    # ex.append(elapsed_time)
                    # f.write(', '.join(map(str, ex)))
                    # f.write('\n')

            print("##############################################################################################")
        print()

# names2 = ['ecoli']
# for name in names2:
#     f = open("./lambdaComp"+name+"10n.csv", "w")
#     for i in range(20,31,1):
#         l = 1 + (i/10)
#         process = subprocess.run(["python3", "main.py", "local", name, "10", "456", str(l)], stdout=PIPE, stderr=PIPE,
#                                  universal_newlines=True)
#         ex = process.stdout.replace('For lambda var: ', '').replace('Iter: ', '').replace('Tasa C: ', '').replace('Tasa Inf: ', '').replace('Aggr: ', '').split('\n')
#         ex.pop()
#         f.write(', '.join(map(str, ex)))
#         f.write('\n')
#         print(ex)
#     #
#     # f = open("./lambdaComp"+name+"20n.csv", "w")
#     # for i in range(20,31,1):
#     #     l = 1 + (i/10)
#     #     process = subprocess.run(["python3", "main.py", "local", name, "20", "456", str(l)], stdout=PIPE, stderr=PIPE,
#     #                              universal_newlines=True)
#     #     ex = process.stdout.replace('For lambda var: ', '').replace('Iter: ', '').replace('Tasa C: ', '').replace('Tasa Inf: ', '').replace('Aggr: ', '').split('\n')
#     #     ex.pop()
#     #     f.write(', '.join(map(str, ex)))
#     #     f.write('\n')
#     #     print(ex)