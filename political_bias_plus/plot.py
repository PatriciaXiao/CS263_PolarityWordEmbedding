import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

training_file = "./output/sentence_baseline_train.csv"
testing_file = "./output/sentence_baseline_testing.csv"

# Data for plotting
'''
TITLE = "loss of GloVe-average sentence classification"
X = "# epoch"
Y = ""
df = pd.read_csv(training_file)
x = list(df["epoch"])
y = list(df["loss"])
'''

TITLE = "F1 of GloVe-average sentence classification"
X = "# epoch"
Y = "f1-score on training set"
df = pd.read_csv(training_file)
x = list(df["epoch"])
y = list(df["f1_score"])

fig, ax = plt.subplots()
ax.plot(x, y)

ax.set(xlabel=X, ylabel=Y,
       title=TITLE)
ax.grid()

fig.savefig("test.png")
plt.show()