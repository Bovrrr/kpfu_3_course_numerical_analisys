import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


def plot_graphic(fs, a, b, n):
    f, ax = plt.subplots(1, 1)
    f.set_size_inches(16, 9)
    xs = np.linspace(a, b, n)
    for f in fs:
        ys = np.array([f(x) for x in xs])
        sns.lineplot(x=xs, y=ys, ax=ax)
    plt.show()


def f(x):
    return 2 * np.log(x + 5 ** (1 / 2)) * np.sinh(x)


a = 1
b = 3

x_1 = 1.24
x_2 = 1.97
x_3 = 2.54
