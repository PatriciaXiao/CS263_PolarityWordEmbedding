import os
from random import random # usage: random()

'''
experiments = [
    "python3.7 main.py -avg > ./output/default_paramaters.out"
]
'''

experiments = list()
range_to_test = [1e-6, 1e-3]
for lambda_d in range_to_test:
    for lambda_e in range_to_test:
        # --debug
        experiments.append("python3.7 main.py -avg -e 1 --lambda_d {0} --lambda_e {1} > ./output/lambda_{0}_{1}.out".format(lambda_d, lambda_e))


def get_experiments():
    return experiments

for e in get_experiments():
    print(e)
    os.system(e)