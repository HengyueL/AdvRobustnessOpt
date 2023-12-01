import argparse
import sys
import os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
sns.set()
COLORS = list(mcolors.TABLEAU_COLORS)