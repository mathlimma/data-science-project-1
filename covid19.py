import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

covid = pd.read_csv(r'caso.csv', encoding = "utf-8")

plt.plot(((covid['city'] == 'Recife'))/len(covid['city']))