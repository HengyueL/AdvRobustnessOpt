import argparse
import sys
import os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from config_setups.config_log_setup import clear_terminal_output
sns.set()
COLORS = list(mcolors.TABLEAU_COLORS)


REF_TIME = {
    "CIFAR-APGD-L2": 19.2,
    "CIFAR-APGD-Linf": 18.2,
    "ImageNet-APGD-L2": 5.7,
    "ImageNet-APGD-Linf": 9.8,
    
    "CIFAR-FAB-L2": 389,
    "CIFAR-FAB-Linf": 395,
    "ImageNet-FAB-L2": 1900,
    "ImageNet-FAB-Linf": 1888,
}

def main(args):

    # === Parse the reference time by APGD or FAB ===
    dataset_name = args.exp_name.split("-")[0]
    min_max_name = args.exp_name.split("-")[-1]
    if "Max" in min_max_name:
        alg_name = "APGD"
    elif "Min" in min_max_name:
        alg_name = "FAB"
    else:
        None
    distance_name = args.case_name.split("-")[-1]
    key = dataset_name + "-" + alg_name + "-" + distance_name
    reference_time = REF_TIME[key] 

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

    # Select Valid Data "Time"
    iter_key = "time"
    time_data = res_data.loc[res_data[iter_key]>0, iter_key ].tolist()

    len_data = len(time_data)
    print("Total data point: ", len_data)
    min_x, max_x = np.amin(time_data), np.amax(time_data)
    n_bins = int((max_x - min_x) // args.bin_width)


    # === Plot ===
    tick_size = 25
    legend_size = 25
    fig,ax = plt.subplots(
        nrows=1, ncols=1, figsize=(6, 4)
    )
    n, bins, _, = ax.hist(
        time_data, bins=n_bins, alpha=0.9
    )
    
    #  Plot reference time
    scale = 1.2
    ax.vlines(
        x=reference_time, ymin=0, ymax=scale*max(n), colors=COLORS[1], lw=2, ls="dashed", alpha=0.9, 
        label=r"%s reference" % alg_name
    )
    ax.legend(
        loc="upper right",
        ncol=1, fancybox=True, shadow=False, fontsize=legend_size, framealpha=0.3
    )

    max_x_tick = max(max_x, scale * 1.1)
    ax.set_xlim([0, max_x_tick])
    ax.set_ylim([0, scale*np.amax(n)])
    # ax.set_xticks([0, max_x_tick // 2, max_x_tick])
    ax.set_xticks([0, 80, 160])
    ax.set_yticks([np.amax(n)//2, np.amax(n)])
    ax.tick_params(axis='x', which='major', colors='black', labelsize=tick_size)
    ax.tick_params(axis='y', which='major', colors='black', labelsize=tick_size)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    clear_terminal_output()
    print("...Plot the Granso Run Time analysis...")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name", dest="exp_name", type=str,
        default="CIFAR-Max",
        help="CIFAR/ImageNet - Min-Max experiment."
    )
    parser.add_argument(
        "--case_name", dest="case_name", type=str,
        default="Granso-L2",
        help="Granso - L1/L2/Linf/PAT."
    )
    args = parser.parse_args()
    args.file_name = "opt_result.csv"
    args.bin_width = 5
    main(args)
    print("Complete")