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


def load_res_data(args, case_name):
    root_dir = os.path.join(
        "..", "resData", args.exp_name, case_name
    )
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
    return res_data


def main(args):
    # === Load FAB and Granso Experiment Data ===
    distance_metric_str = args.case_name + "_distance"
    fab_case_name = "Fab-" + args.case_name
    fab_res_data = load_res_data(args, fab_case_name)
    granso_case_name = "Granso-" + args.case_name
    granso_res_data = load_res_data(args, granso_case_name)
    print("Check data length: FAB - {} | Granso - {}".format(len(fab_res_data), len(granso_res_data)))
    assert len(fab_res_data) == len(granso_res_data), "The length of the res data must agree."

    filtered_fab_radius, filtered_granso_radius = [], []
    for i in range(len(fab_res_data)):
        fab_idx, fab_radius = fab_res_data["sample_idx"][i], fab_res_data[distance_metric_str][i]
        granso_idx, granso_radius = granso_res_data["sample_idx"][i], granso_res_data[distance_metric_str][i]

        if fab_idx == granso_idx and fab_radius > 0 and granso_idx > 0:  # throw away if fab falls
            filtered_fab_radius.append(fab_radius)
            filtered_granso_radius.append(granso_radius)

    start = 30
    total_plot_len = 40
    print("Length: ", total_plot_len)
    x = np.linspace(0, total_plot_len, total_plot_len, endpoint=False)
    y_max = np.amax(filtered_fab_radius[start:start+total_plot_len] + filtered_granso_radius[start:start+total_plot_len])
    # === Plot Figure ===
    tick_size = 20
    legend_size = 20
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        x,
        filtered_fab_radius[start:start+total_plot_len],
        alpha=0.4,
        width=0.8,
        align="edge",
        label=r"FAB",
        color=COLORS[0]
    )
    ax.bar(
        x,
        filtered_granso_radius[start:start+total_plot_len], 
        alpha=0.6,
        width=0.45,
        align="edge",
        label=r"PWCF",
        color=COLORS[1]
    )

    # ax.set_yticks([0.05, 0.1])
    ax.set_yticks([450, 900])
    ax.set_xticks([0, total_plot_len // 2, total_plot_len])
    ax.tick_params(axis='y', colors="black", labelsize=tick_size)
    ax.tick_params(axis='x', labelsize=tick_size)
    ax.set_ylim([0, 1.1 * y_max])
    ax.set_xlim([0-1, total_plot_len+1])
    ax.legend(
        loc="upper right",
        ncol=1, fancybox=True, shadow=False, fontsize=legend_size, framealpha=0.3
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    clear_terminal_output()
    print("... Plot  ...")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name", dest="exp_name", type=str,
        default="ImageNet-Min",
        help="CIFAR/ImageNet - Min experiment."
    )
    parser.add_argument(
        "--case_name", dest="case_name", type=str,
        default="L1",
        help="L1/L2/Linf."
    )
    args = parser.parse_args()
    args.file_name = "opt_result.csv"
    args.bin_width = 20
    main(args)
    print("Complete")