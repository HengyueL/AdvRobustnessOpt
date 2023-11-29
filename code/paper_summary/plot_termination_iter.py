import sys
import os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
import argparse, os, torch, time
import numpy as np
import pandas as pd
from config_setups.config_log_setup import clear_terminal_output
from utils.general import load_json

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
# use seaborn plotting defaults
import seaborn as sns
import matplotlib.colors as mcolors
sns.set()
COLORS = list(mcolors.TABLEAU_COLORS)


def main(args):
    # === Filter result log dirs ===
    root_dir = os.path.join(
        "..", "resData", args.exp_name, args.case_name
    )
    distance_metric_str = args.case_name.split("-")[-1]
    exp_list = list(os.listdir(root_dir))
    exp_list.sort()
    print("Experiments list: ", exp_list)

    res_result_list = []
    # === Read Experimen Data ===
    for exp_name in exp_list:
        result_file_path = os.path.join(root_dir, exp_name, args.file_name)
        res_data = pd.read_csv(result_file_path)
        res_result_list.append(res_data)
    res_data = pd.concat(res_result_list, ignore_index=True)

    # === Read the config file ===
    conf_file_path = os.path.join(
        root_dir, exp_name, "Exp_Config.json"
    )
    confs = load_json(conf_file_path)
    primary_key = [key for key in confs.keys() if "params" in key][0]
    subkey = [key for key in confs[primary_key].keys() if "max_iter" in key and "early" not in key][0]
    max_x_tick = confs[primary_key][subkey]


    # Select Valid Data
    # distance_metric_str = distance_metric_str + "_distance"
    iter_key = "best_iter" if "best_iter" in res_data.keys() else "total_iter"
    radius_data = res_data.loc[res_data[iter_key]>0, iter_key ].tolist()

    len_data = len(radius_data)
    print("Total data point: ", len_data)
    min_x, max_x = np.amin(radius_data), np.amax(radius_data)
    n_bins = int((max_x - min_x) // args.bin_width)
    
    # === Plot ===
    tick_size = 25
    legend_size = 25
    fig,ax = plt.subplots(
        nrows=1, ncols=1, figsize=(6, 4)
    )
    n, bins, _, = ax.hist(
        radius_data, bins=n_bins, alpha=0.9
    )

    scale = 1.2
    if "granso" not in args.case_name.lower():
        ax.vlines(
            x=100, ymin=0, ymax=scale*max(n), colors=COLORS[1], lw=2, ls="dashed", alpha=0.9, 
            label=r"default stop"
        )
        ax.legend(
            loc="upper right",
            ncol=1, fancybox=True, shadow=False, fontsize=legend_size, framealpha=0.3
        )

    ax.set_xlim([0, max_x_tick])
    ax.set_ylim([0, scale*np.amax(n)])
    ax.set_xticks([0, max_x_tick // 2, max_x_tick])
    ax.set_yticks([np.amax(n)//2, np.amax(n)])
    ax.tick_params(axis='x', which='major', colors='black', labelsize=tick_size)
    ax.tick_params(axis='y', which='major', colors='black', labelsize=tick_size)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    clear_terminal_output()
    print("...Perform APGD Max Form (Lp distance only) ...")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name", dest="exp_name", type=str,
        default="ImageNet-Max",
        help="CIFAR/ImageNet - Min-Max experiment."
    )
    parser.add_argument(
        "--case_name", dest="case_name", type=str,
        default="Granso-Linf",
        help="APGD/Fab/Granso - L1/L2/Linf/PAT."
    )
    args = parser.parse_args()
    args.file_name = "opt_result.csv"
    args.bin_width = 50
    main(args)
    print("Complete")