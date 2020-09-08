import numpy as np
import pandas as pd
from pylab import *

covid = pd.read_csv(r'caso.csv', encoding = "utf-8")

plt.plot(covid['date'])