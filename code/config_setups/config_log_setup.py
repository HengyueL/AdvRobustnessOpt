import multiprocessing
import os, json
import numpy as np
import torch
import random
import platform
import pandas as pd


# ==== functions related to save log ====
def makedir(dir):
    os.makedirs(dir, exist_ok=True)


def makedirs(dir_list):
    for dir in dir_list:
        makedir(dir)


def create_save_file_name(dir, name):
    return os.path.join(dir, name)


def save_dict_to_json(dict, save_dir):
    with open(save_dir, "w") as outfile:
        json.dump(dict, outfile, indent=4)


def save_dict_to_csv(
    summary_dict,
    save_name,
    transpose=False
):
    """
        Save a dict into json file.
    """
    # csv format
    if not transpose:
        pd_data = pd.DataFrame.from_dict(summary_dict)
    else:
        pd_data = pd.DataFrame.from_dict(summary_dict).T
    pd_data.to_csv(save_name, index=False)


def save_exp_info(save_dir, config):
    """
        Create new log folder / Reuse the checkpoint folder for experiment.
    """
    # Save Exp Settings as Json File
    exp_config_file = os.path.join(save_dir, "Exp_Config.json")
    save_dict_to_json(config, exp_config_file)


def create_log_info(dir, name="experiment_log.txt"):
    log_file = os.path.join(
        dir,
        name
    )
    return log_file


# ==== functions related to config setup ====
def clear_terminal_output():
    system_os = platform.system()
    if "Windows" in system_os:
        cmd = "cls"
    elif "Linux" in system_os:
        cmd = "clear"
    os.system(cmd)


def set_random_seeds(config):
    """
        This function sets all random seed used in this experiment.
        For reproduce purpose.
    """
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_devices(config):
    """
        This function returns a list of available gpu devices.

        If GPUs do noe exist, return: None
    """
    n_gpu = torch.cuda.device_count()
    print("Total GPUs availbale: [%d]" % n_gpu)
    if n_gpu > 0:
        gpu_list = ["cuda:{}".format(i) for i in range(n_gpu)]
    else:
        gpu_list = None
    set_random_seeds(config)
    return gpu_list


def set_default_device(config):
    """
        This function set the default device to move torch tensors.

        If GPU is availble, default_device = torch.device("cuda:0").
        If GPU is not available, default_device = torch.device("cpu").
    """
    gpu_list = set_devices(config)
    if gpu_list is None:
        return torch.device("cpu"), gpu_list
    else:
        return torch.device(gpu_list[0]), gpu_list


def get_n_cpus():
    num_cores = multiprocessing.cpu_count()
    return num_cores


def create_summary_dict(keys):
    summary = {}
    for key in keys:
        summary[key] = []
    return summary


def create_cpk_dir_min_formulation(
    save_root, attack_type, attack_method,
    attack_bound, attack_config,
    attack_restart=None,
    attack_softmax_logits=None
):
    if attack_type != "PyGranso":
        experiment_name = "min-%s-%s-%.04f" % (
            attack_type, 
            attack_method, 
            attack_bound
        )
        check_point_dir = os.path.join(
            save_root, 
            experiment_name
        )
    else:
        granso_input_constraint = attack_config["granso_input_constraint_type"]
        ieq_rescale = attack_config["granso_rescale_ieq"]
        if attack_restart is not None:
            ieq_rescale += "-Restart%d" % attack_restart
        if attack_softmax_logits is not None:
            ieq_rescale += ("-" + str(attack_softmax_logits))
        experiment_name = "min-%s-%s-%.04f-%s-%s" % (
                attack_type, 
                attack_method,
                attack_bound,
                granso_input_constraint,
                ieq_rescale
            )
        check_point_dir = os.path.join(
            save_root, 
            experiment_name
        )
    return check_point_dir, experiment_name

