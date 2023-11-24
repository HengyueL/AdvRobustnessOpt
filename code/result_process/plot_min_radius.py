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

# ====
from config_setups.config_log_setup import clear_terminal_output


def main(args):
    root_dir = args.res_root
    norm = args.norm



if __name__ == "__main__":
    clear_terminal_output()
    print("This scripts generate the plot of min-radius comparision between FAB and PyGRANSO.")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--res_root", dest="res_root", type=str,
        default=os.path.join('..', 'paperRes', 'CIFAR-Min'),
        help="path to the folder containing FAB and PyGRANSO (min form) results."
    )
    parser.add_argument(
        "--norm", dest="norm", type=str,
        default="L1",
        help="Which distance metric is used to compare."
    )
    args = parser.parse_args()
    main(args)
    print("Completed.")
