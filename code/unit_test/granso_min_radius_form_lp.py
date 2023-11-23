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
    
from utils.general import load_json, print_and_log, get_samples, get_granso_adv_output
from attacks.granso_min_lp import granso_min


def calc_min_dist_sample(
    adv_output_dict,  # A dict of fab results
    orig_input, 
    attack_type,  
    model, 
    target_label,
    log_file
):
    keys = adv_output_dict.keys()

    radius_dict = {}
    boundary_distance_dict = {}
    box_violation_dict = {}
    attacked_label_dict = {}

    best_boundary_distance, best_key = float("inf"), None

    for key in keys:
        adv_output = adv_output_dict[key]
        # Check Result 
        with torch.no_grad():
            pred = model(adv_output)
            true_label_logit = pred[0, target_label].item()

        pred_clone = pred.clone()
        pred_clone[0, target_label] = -1e12
        attacked_logit = torch.amax(pred_clone).item()
        attacked_label = pred_clone.argmax(1).item()
        distance_to_boundary = attacked_logit - true_label_logit
        boundary_distance_dict[key] = distance_to_boundary
        attacked_label_dict[key] = attacked_label
        if distance_to_boundary < best_boundary_distance:
            best_boundary_distance = distance_to_boundary
            best_key = key

        adv_output = adv_output.clone().reshape(1, -1)
        orig_input = orig_input.clone().reshape(1, -1)
        # Check [0, 1] box constraint
        greater_than_1 = torch.sum(torch.where(adv_output > (1 + 1e-4), 1, 0)).item()
        smaller_than_0 = torch.sum(torch.where(adv_output < (0 - 1e-4), 1, 0)).item()
        num_violation = greater_than_1 + smaller_than_0
        box_violation_dict[key] = num_violation
        
        err_vec = torch.abs(adv_output - orig_input)
        assert "L" in attack_type, "Norm character partition incorrect"

        if attack_type == "L1-Reform":
            attack_key_word = "L1"
        else:
            attack_key_word = attack_type
        p_norm = attack_key_word.split("L")[-1]
        if p_norm == "inf":
            p_distance = torch.amax(err_vec, dim=1).cpu().item()
        else:
            p_norm = float(p_norm)
            p_distance = (torch.sum(err_vec**p_norm, dim=1)**(1/p_norm)).cpu().item()
        
        radius_dict[key] = p_distance
        msg = "  >> Restart [%d] has  - radius [%.04f] - boundary distance [%.04f] - box violations [%d] >> " % (
            key, p_distance, distance_to_boundary, num_violation
        )
        print_and_log(msg, log_file)

    return radius_dict, boundary_distance_dict, box_violation_dict, attacked_label_dict, best_key


def execute_granso_min_target(
    input_to_granso, label_to_granso, target_label, x0,
    classifier_model, device, attack_config, 
    mu0=1, H0_init=None, max_iter=None, print_opt=True
):
    attack_type = attack_config["distance_metric"]
    if max_iter is None:
        max_iter = attack_config["granso_max_iter"]
    mem_size = attack_config["granso_mem"]

    # ==== how total violation and stationarity is determined ====
    stat_l2 = attack_config["granso_stat_l2"]
    steering_l1 = attack_config["granso_steering_l1"]
    ineq_tol = attack_config["granso_ieq_tol"]
    eq_tol = attack_config["granso_eq_tol"]
    opt_tol = attack_config["granso_opt_tol"]

    time_start = time.time()
    sol = granso_min(
        input_to_granso, label_to_granso, x0=x0, target_label=target_label,
        model=classifier_model, attack_type=attack_type, device=device,
        stat_l2=stat_l2,
        steering_l1=steering_l1,
        max_iter=max_iter,
        mu0=mu0,
        ineq_tol=ineq_tol,
        eq_tol=eq_tol, 
        opt_tol=opt_tol,
        mem_size_param=mem_size,
        print_log=print_opt,
        H0_init=H0_init
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
        "time": [],
        "termination_code": []
    }

    # === Create some variables from the cfg file for convenience in OPT settings ===
    n_restart = opt_config["granso_n_restarts"]
    max_iter = opt_config["granso_max_iter"]
    if attack_type == "Linf":
        init_scale = 0.03
    else:
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
                termination_code_dict = {}
                time_dict = {}
                for restart_idx in range(n_restart):
                    applied_perturbation = init_scale * (2 * torch.rand_like(inputs).to(device) - 1)
                    x_init = (inputs + applied_perturbation).to(device, dtype=dtype)
                    # x_init = init_scale * torch.rand_like(inputs).to(device, dtype=dtype)
                    t_start = time.time()
                    # try:
                    sol = execute_granso_min_target(
                        inputs, labels, None, x_init, classifier_model, device,
                        attack_config=cfg["granso_params"], max_iter=max_iter
                    )
                    x_sol = get_granso_adv_output(
                        sol, attack_type, inputs
                    )
                    termination_code = sol.termination_code
                    # except:
                    #     msg = "  Restart [%d] OPT Failure... Return original the original inputs... " % restart_idx
                    #     print_and_log(msg, log_file)
                    #     sol = None
                    #     x_sol = get_granso_adv_output(
                    #         sol, attack_type, inputs
                    #     )
                    #     termination_code = -100
                    t_end = time.time()
                    granso_adv_output[restart_idx] = x_sol
                    time_dict[restart_idx] = t_end - t_start
                    termination_code_dict[restart_idx] = termination_code

                radius_dict, boundary_distance_dict, box_violation_dict, attacked_label_dict, best_key = calc_min_dist_sample(
                    granso_adv_output, inputs, attack_type, classifier_model, labels.item(), log_file
                )
                # === Log the best result ===
                for key in granso_adv_output.keys():
                    result_summary["sample_idx"].append(batch_idx)
                    result_summary["restart"].append(key)
                    result_summary["true_label"].append(labels.item())
                    result_summary["max_logit_before_opt"].append(pred_before.item())
                    result_summary[attack_distance_key].append(radius_dict[key])
                    result_summary["max_logit_after_opt"].append(attacked_label_dict[key])
                    result_summary["distance_to_decision_boundary"].append(boundary_distance_dict[key])
                    result_summary["box_constraint_violation"].append(box_violation_dict[key])
                    result_summary["time"].append(time_dict[key])
                    result_summary["termination_code"].append(termination_code_dict[key])
            save_dict_to_csv(
                result_summary, result_csv_dir
            )

            if cfg["save_vis"]:
                vis_dir = os.path.join(ckpt_dir, "dataset_vis")
                makedir(vis_dir)
                orig_save_name, adv_save_name = os.path.join(vis_dir, "orig_imgs.npy"), os.path.join(vis_dir, "adv_imgs.npy")

                orig_img_np = inputs.detach().cpu().numpy()[0, :, :, :]
                orig_img_list.append(orig_img_np)
                adv_img_np = granso_adv_output[best_key].detach().cpu().numpy()[0, :, :, :]
                adv_img_list.append(adv_img_np)

                np.save(orig_save_name, np.asarray(orig_img_list))
                np.save(adv_save_name, np.asarray(adv_img_list))


if __name__ == "__main__":
    clear_terminal_output()
    print("Solving [Min Form] optimization with PyGRANSO.")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", dest="config", type=str,
        default=os.path.join('config_file_examples', 'min-form-granso.json'),
        help="Path to the json config file (FAB version)."
    )
    args = parser.parse_args()
    cfg = load_json(args.config)  # Load Experiment Configuration file
    cfg["dataset"]["batch_size"] = 1  # PyGRANSO only wants batch_size = 1 currently
    main(cfg)  # Use double to compare with PyGRANSO
    print("Completed")