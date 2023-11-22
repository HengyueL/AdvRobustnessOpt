import sys
import os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
import argparse, os, torch, time
import numpy as np
# ===========
from utils.build import build_model, get_loss_func_eval, get_loader_clean
from config_setups.config_log_setup import clear_terminal_output, create_log_info_file,\
    makedir, save_dict_to_csv, save_exp_info, set_default_device, set_random_seeds
    
from utils.general import load_json, print_and_log, save_image, tensor2img, get_samples
from attacks.target_fab import FABAttackPTModified
import numpy as np



def execute_granso_min_target(
    input_to_granso, label_to_granso,
    target_label,
    x0,
    classifier_model, device,
    print_opt, attack_config, mu0=1,
    H0_init=None,
    is_continue=False,
    granso_opt_log_file=None,
    max_iter=None
):
    attack_type = attack_config["attack_method"]
    if max_iter is None:
        max_iter = attack_config["granso_max_iter"]
    mem_size = attack_config["granso_mem"]
    input_constraint_type = attack_config["granso_input_constraint_type"]
    constraint_folding_type = attack_config["granso_constraint_folding_type"]

    # ==== how total violation and stationarity is determined ====
    stat_l2 = attack_config["granso_stat_l2"]
    steering_l1 = attack_config["granso_steering_l1"]
    granso_slack_variable = attack_config["granso_slack_variable"]
    wall_time = attack_config["granso_walltime"]
    ineq_tol = attack_config["granso_ieq_tol"]
    eq_tol = attack_config["granso_eq_tol"]
    opt_tol = attack_config["granso_opt_tol"]

    time_start = time.time()
    sol = granso_min_target(
        input_to_granso, label_to_granso,
        x0=x0,
        target_label=target_label,
        model=classifier_model, attack_type=attack_type,
        device=device,
        stat_l2=stat_l2,
        steering_l1=steering_l1,
        max_iter=max_iter,
        mu0=mu0,
        ineq_tol=ineq_tol,
        eq_tol=eq_tol, 
        opt_tol=opt_tol,
        input_constraint_type=input_constraint_type,
        constraint_folding_type=constraint_folding_type,
        granso_slack_variable=granso_slack_variable,
        mem_size_param=mem_size,
        print_log=print_opt,
        H0_init=H0_init,
        wall_time=wall_time,
        is_continue=is_continue,
        granso_opt_log_file=granso_opt_log_file
    )
    time_end = time.time()
    print("Execution time: [%.05f]" % (time_end - time_start))
    return sol



def main(cfg, dtype=torch.double):
    set_random_seeds(cfg)

    # === Create Experiment Save Dirs ===
    save_root = os.path.join(
        "..", "log_folder", cfg["log_folder"]["save_root"]
    )
    makedir(save_root)
    # === Experiment Name ===
    img_start_idx, img_end_idx = cfg["curr_sample"], cfg["end_sample"]
    exp_name = "Granso-%s_%d-%d" % (cfg["granso_params"]["distance_metric"], img_start_idx, img_end_idx)
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
    opt_config = cfg["granso_params"]
    attack_type = opt_config["distance_metric"]
    result_csv_dir = os.path.join(ckpt_dir, "opt_result.csv")
    attack_distance_key = "%s_distance" % attack_type
    result_summary = {
        "sample_idx": [],
        "restart": [],
        "true_label": [],
        "max_logit_before_opt": [],
        "max_logit_after_opt": [],
        
        attack_distance_key: [],
        "distance_to_decision_boundary": [],
        "box_constraint_violation": [],
        "time": []
    }

    # === Create some variables from the cfg file for convenience in OPT settings ===
    n_restart = opt_config["granso_n_restarts"]
    max_iter = opt_config["granso_max_iter"]
    init_scale = 0.1

    # List to save dataset before & after optimization
    orig_img_list, adv_img_list = [], []

    # === Main OPT
    for batch_idx, data in enumerate(val_loader):
        if batch_idx < cfg["curr_sample"]:
            pass  # Image not within the specified range
        else:
            if batch_idx >= cfg["end_sample"]:
                break  # Completed all images specified
            msg = "===== Testing sample [%d] =====" % batch_idx
            print_and_log(msg, log_file)

            inputs, labels = get_samples(cfg, data)
            inputs, labels = inputs.to(device, dtype=dtype), labels.to(device) 
            with torch.no_grad():
                pred_logits_before = classifier_model(inputs)
            pred_before = pred_logits_before.argmax(1)
            pred_correct_before = (pred_before == labels).sum().item()

            if pred_correct_before < 0.5:
                msg = "Sample [%d] - prediction wrong. Skip OPT >>>"
                print_and_log(msg, log_file)
                result_summary["sample_idx"].append(batch_idx)
                result_summary["true_label"].append(labels.item())
                result_summary["max_logit_before_opt"].append(pred_before.item())
                for key in result_summary.keys():
                    if key not in ["sample_idx", "true_label", "max_logit_brefore_opt"]:
                        result_summary[key].append(-100)  # Add a placeholder in the logger
            else:
                msg = "Sample [%d] - prediction correct. Begin PyGRANSO OPT >>>" % batch_idx
                print_and_log(msg, log_file)

                # OPT
                granso_adv_output = {}
                for restart_idx in range(n_restart):
                    applied_perturbation = init_scale * (2 * torch.rand_like(inputs).to(device) - 1)
                    x_init = (inputs + applied_perturbation).to(device, dtype=dtype)

                    try:
                        sol = execute_granso_min_target(
                            input_to_granso, label_to_granso, None, x_init, classifier_model, device,
                            print_opt, attack_config, max_iter=max_iter
                        )
                    except:
                        msg = "  Restart [%d] OPT Failure... Return original the original inputs... " % restart_idx
                        print_and_log(msg, log_file)
                        x_sol = inputs.clone()


if __name__ == "__main__":
    clear_terminal_output()
    print("Solving [Min Form] optimization with PyGRANSO.")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", dest="config", type=str,
        default=os.path.join('config_file_examples', 'min-form-fab.json'),
        help="Path to the json config file (FAB version)."
    )
    args = parser.parse_args()
    cfg = load_json(args.config)  # Load Experiment Configuration file
    cfg["dataset"]["batch_size"] = 1  # PyGRANSO only wants batch_size = 1 currently
    main(cfg, default_dtype=torch.float)  # Use double to compare with PyGRANSO
    print("Completed")