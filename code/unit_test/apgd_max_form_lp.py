import sys
import os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
import argparse, os, torch, time
import numpy as np
# ===========
from utils.general import load_json, print_and_log
from config_setups.config_log_setup import makedir
from config_setups.config_log_setup import clear_terminal_output, create_log_info_file,\
    makedir, save_dict_to_csv, save_exp_info, set_default_device, set_random_seeds
from utils.build import build_model, get_loss_func_eval, get_lp_attack, get_loader_clean, \
    generate_attack_lp, get_samples
from utils.general import tensor2img, rescale_array
# ==== 


def main(cfg, dtype=torch.float):
    set_random_seeds(cfg)
    # === Create Experiment Save Dirs ===
    save_root = os.path.join(
        "..", "log_folder", cfg["log_folder"]["save_root"]
    )
    makedir(save_root)
    # === Experiment Name ===
    img_start_idx, img_end_idx = cfg["curr_sample"], cfg["end_sample"]
    exp_name = "APGD-%s_%d-%d" % (cfg["granso_params"]["distance_metric"], img_start_idx, img_end_idx)
    ckpt_dir = os.path.join(save_root, exp_name)
    if cfg["continue"]:  # Load or save experiment dir
        ckpt_dir = cfg["ckpt_dir"]
    else:
        cfg["ckpt_dir"] = ckpt_dir
        makedir(ckpt_dir)
    
    # === Create Experiment Log File (for printing) and save the exp settings
    log_file = create_log_info_file(ckpt_dir)
    save_exp_info(ckpt_dir, cfg)
    device, _ = set_default_device()  # We only need 1 gpu for this project
    # === Setup the base classifier model ===
    classifier_model, msg = build_model(
        model_config=cfg["classifier_model"],
        num_classes=cfg["dataset"]["num_classes"],
        device=device
    )
    classifier_model = classifier_model.to(device, dtype=dtype)
    classifier_model.eval()
    print_and_log(msg, log_file, mode="w")
    # === Setup a dataloader ===
    _, val_loader, _ = get_loader_clean(
        cfg, only_val=True, shuffle_val=False
    )
    # === Format to summarize the final result
    opt_config = cfg["apgd_params"]
    loss_type = opt_config["loss"]
    attack_type = opt_config["distance_metric"]
    result_csv_dir = os.path.join(ckpt_dir, "opt_result.csv")

    radius_name = "%s_distance"
    result_summary = {
        "sample_idx": [],
        "restart": [],
        "true_label": [],

        "max_logit_before_opt": [],
        "max_logit_after_opt": [],

        radius_name: [],
        "eps": [],

        "box_constraint_violation": [],
        "time": [],
        "best_iter": [],
        "termination_code": []
    }


if __name__ == "__main__":
    clear_terminal_output()
    print("...Perform APGD Max Form (Lp distance only) ...")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", dest="config", type=str,
        default=os.path.join('config_file_examples', 'max-form-apgd-lp.json'),
        help="Path to the json config file."
    )
    args = parser.parse_args()
    cfg = load_json(args.config)
    cfg["dataset"]["batch_size"] = 1
    main(args)
    print("Complete")