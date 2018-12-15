import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import sys
import matplotlib.pyplot as plt

import pandas as pd

dat = pd.read_csv("reward.txt", sep=" ")

reward = dat['reward']
tim = dat['time']

plt.plot(reward, color='red', linewidth=0.5)

plt.xlabel("TIME")
plt.ylabel("REWARD")

plt.show()

