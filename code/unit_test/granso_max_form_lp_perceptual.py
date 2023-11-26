import sys
import os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
import argparse, os, torch, time
import numpy as np
# ===========
from utils.build import build_model, get_loader_clean, get_granso_loss_func
from config_setups.config_log_setup import clear_terminal_output, create_log_info_file,\
    makedir, save_dict_to_csv, save_exp_info, set_default_device, set_random_seeds
from utils.general import load_json, print_and_log, get_samples
from models.model import AlexNetFeatureModel
from percept_utils.distance import LPIPSDistance
from attacks.granso_max import granso_max_attack


def calc_best_sample(
    adv_output_dict,  # A dict of adv results
    orig_input, 
    attack_type,  
    model, 
    target_label,
    log_file
):
    keys = adv_output_dict.keys()

    radius_dict = {}
    box_violation_dict = {}
    attacked_label_dict = {}
    best_key = list(keys)[0]

    for key in keys:
        adv_output = adv_output_dict[key]
        # Check Result 
        with torch.no_grad():
            pred = model(adv_output)
        attacked_label = pred.argmax(1).item()
        attacked_label_dict[key] = attacked_label

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
        msg = "  >> Restart [%d] has  - radius [%.04f] - box violations [%d] >> " % (
            key, p_distance, num_violation
        )
        print_and_log(msg, log_file)

        if attacked_label != target_label:
            best_key = key

    return radius_dict, box_violation_dict, attacked_label_dict, best_key




def get_granso_adv_output_maxform(sol, input_to_granso):
    """
        Retrive x' from granso sol.
    """
    if sol is None:
        granso_adv_output = -1 * torch.ones_like(input_to_granso)
    else:
        granso_adv_output = torch.reshape(
            sol.final.x,
            input_to_granso.shape
        )
    return granso_adv_output


def execute_granso_max_attack(
    input_to_granso, label_to_granso,
    x0,
    classifier_model,
    max_loss_function,
    device,
    granso_config, mu0=1,
    lpips_model=None,
    H0_init=None,
    max_iter=None,
    dtype=torch.double
):
    """
        Helper function to perform granso max with customized inputs.
    """
    attack_metric = granso_config["distance_metric"]
    eps = granso_config["attack_bound"]
    if max_iter is None:
        max_iter = granso_config["granso_max_iter"]
    mem_size = granso_config["granso_mem"]
    # ==== how total violation and stationarity is determined ====
    stat_l2 = granso_config["granso_stat_l2"]
    steering_l1 = granso_config["granso_steering_l1"]

    eq_tol = granso_config["granso_eq_tol"]
    ieq_tol = granso_config["granso_ieq_tol"]
    opt_tol = granso_config["granso_opt_tol"]

    sol = granso_max_attack(
        inputs=input_to_granso,
        labels=label_to_granso,
        x0=x0,
        model=classifier_model,
        attack_type=attack_metric,
        device=device,
        loss_func=max_loss_function,
        eps=eps,
        lpips_distance=lpips_model,
        max_iter=max_iter,
        eq_tol=eq_tol,
        ieq_tol=ieq_tol,
        opt_tol=opt_tol, 
        stat_l2=stat_l2,
        steering_l1=steering_l1,
        mu0=mu0,
        H0_init=H0_init,
        mem_size_param=mem_size,
        dtype=dtype
    )
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

    # === If distance_metric == "PAT", then init the perceptual distance
    if attack_type == "PAT":
        lpips_backbone = AlexNetFeatureModel(
            lpips_feature_layer=False, 
            use_clamp_input=False,
            device=device,
            dtype=dtype
        ).to(device=device, dtype=dtype)
        lpips_backbone.eval()
        lpips_model = LPIPSDistance(lpips_backbone).to(device=device, dtype=dtype)
        lpips_model.eval()
    else:
        lpips_model=None

    # ==== Get the loss used in maximization ====
    attack_loss_config = opt_config["granso_loss"]
    granso_attack_loss_func_name = attack_loss_config["type"]
    reduction = attack_loss_config["reduction"]
    use_clip_loss = attack_loss_config["use_clip_loss"]
    dataset_name = cfg["dataset"]["name"]
    granso_attack_loss_func, msg = get_granso_loss_func(
        granso_attack_loss_func_name, 
        reduction, 
        use_clip_loss,
        dataset_name=dataset_name
    )

    # === Create some variables from the cfg file
    n_restart = opt_config["granso_n_restarts"]
    max_iter = opt_config["granso_max_iter"]
    es_max_iter = opt_config["granso_early_max_iter"]  # For early stopping
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
                msg = "Sample [%d] - prediction wrong. Skip OPT >>>" % batch_idx
                print_and_log(msg, log_file)
                result_summary["sample_idx"].append(batch_idx)
                result_summary["true_label"].append(labels.item())
                result_summary["max_logit_before_opt"].append(pred_before.item())
                for key in result_summary.keys():
                    if key not in ["sample_idx", "true_label", "max_logit_before_opt"]:
                        result_summary[key].append(-100)  # Add a placeholder in the logger
            else:
                msg = "Sample [%d] - prediction correct. Begin PyGRANSO OPT >>>" % batch_idx
                print_and_log(msg, log_file)

                # OPT
                granso_interm_dict = {}
                granso_iter_dict = {}
                granso_interm_x_dict = {}
                granso_H_dict = {}
                granso_mu_dict = {}
                granso_iter_dict = {}
                granso_interm_termination_code_dict = {}
                granso_ieq_dict = {}
                time_dict = {}
                best_f, best_idx = float("inf"), 0

                # ==== Early Stopped
                for restart_idx in range(n_restart):
                    applied_perturbation = init_scale * (2 * torch.rand_like(inputs).to(device) - 1)
                    x_init = (inputs + applied_perturbation).to(device, dtype=dtype)

                    t_start = time.time()
                    # try:
                    sol = execute_granso_max_attack(
                        inputs, labels, x_init, classifier_model, granso_attack_loss_func,
                        device, opt_config, 
                        mu0=opt_config["granso_mu0"],
                        lpips_model=lpips_model,
                        max_iter=es_max_iter,
                        dtype=dtype
                    )
                    termination_code = sol.termination_code
                    total_violation = sol.final.tv
                    final_f = sol.final.f
                    # except:
                    #     msg = "  Restart [%d] OPT Failure... Return original the original inputs... " % restart_idx
                    #     print_and_log(msg, log_file)
                    #     sol = None
                    #     termination_code = -100
                    #     total_violation = float("inf")
                    t_end = time.time()
                    x_sol = get_granso_adv_output_maxform(sol, inputs)
                    # === Log Interm result ===
                    if sol is not None:
                        granso_interm_dict[restart_idx] = x_sol
                        granso_iter_dict[restart_idx] = sol.iters
                        granso_interm_x_dict[restart_idx] = sol.final.x
                        granso_H_dict[restart_idx] = sol.H_final
                        granso_mu_dict[restart_idx] = sol.final.mu
                        granso_interm_termination_code_dict[restart_idx] = termination_code
                        granso_ieq_dict[restart_idx] = total_violation
                        time_dict[restart_idx] = t_end - t_start

                        # == Check if attack successful
                        with torch.no_grad():
                            adv_pred = classifier_model(x_sol).argmax(1)
                        condition = (adv_pred == labels).sum().item()

                        if termination_code == 0 and condition < 0.5:
                            best_idx = restart_idx
                            best_f = -float("inf")
                            break
                        elif final_f < best_f and termination_code not in [6, 7]:
                            best_f = final_f
                            best_idx = restart_idx
                    else:
                        granso_interm_dict[restart_idx] = None
                        granso_iter_dict[restart_idx] = None
                        granso_interm_x_dict[restart_idx] = None
                        granso_H_dict[restart_idx] = None
                        granso_mu_dict[restart_idx] = None
                        granso_ieq_dict[restart_idx] = total_violation
                        granso_interm_termination_code_dict[restart_idx] = -100
                        time_dict[restart_idx] = -100
                # === Select the interm result with the lowest objective to continue ===
                best_termination_code = granso_interm_termination_code_dict[best_idx]
                if best_termination_code == 0:
                    msg = "   Best ES result converged. Skip continue."
                    print_and_log(msg, log_file)
                    # === ES stage already converge ===
                    x_sol = granso_interm_dict[best_idx]
                    termination_code = best_termination_code
                    final_iters = granso_iter_dict[best_idx]
                    final_time = 0
                else:
                    msg = "   Warm start with #[%d] run" % best_idx
                    print_and_log(msg, log_file)
                    H_continue = granso_H_dict[best_idx]
                    x_continue = granso_interm_x_dict[best_idx]
                    mu_continue = granso_mu_dict[best_idx]
                    # if H_continue["S"].shape[1] == H_continue["rho"].shape[1]:
                    t_start = time.time()
                    sol = execute_granso_max_attack(
                        inputs, labels, x_continue, classifier_model, granso_attack_loss_func,
                        device, opt_config, 
                        mu0=mu_continue,
                        lpips_model=lpips_model,
                        max_iter=max_iter-es_max_iter,
                        H0_init=H_continue,
                        dtype=dtype
                    )
                    final_time = time.time() - t_start
                    termination_code = sol.termination_code
                    x_sol = get_granso_adv_output_maxform(sol, inputs)
                    final_iters = sol.iters + granso_iter_dict[best_idx]

                granso_final_x_dict = {best_idx: x_sol}
                radius_dict, box_violation_dict, attacked_label_dict, best_key = calc_best_sample(
                    granso_final_x_dict, inputs, attack_type, classifier_model, labels.item(), log_file
                )
                eps = opt_config["attack_bound"]
                # === Log the best result ===
                total_time = np.sum(list(time_dict.values())) + final_time
                result_summary["sample_idx"].append(batch_idx)
                result_summary["restart"].append(best_key)
                result_summary["true_label"].append(labels.item())
                result_summary["max_logit_before_opt"].append(pred_before.item())
                result_summary["max_logit_after_opt"].append(attacked_label_dict[best_key])
                result_summary[radius_name].append(radius_dict[best_key])
                result_summary["eps"].append(eps)
                result_summary["box_constraint_violation"].append(box_violation_dict[best_key])
                result_summary["time"].append(total_time)
                result_summary["termination_code"].append(termination_code)
                result_summary["best_iter"].append(final_iters)
            save_dict_to_csv(
                result_summary, result_csv_dir
            )
            if cfg["save_vis"] and (pred_correct_before > 0.5):
                vis_dir = os.path.join(ckpt_dir, "dataset_vis")
                makedir(vis_dir)
                orig_save_name, adv_save_name = os.path.join(vis_dir, "orig_imgs.npy"), os.path.join(vis_dir, "adv_imgs.npy")

                orig_img_np = inputs.detach().cpu().numpy()[0, :, :, :]
                orig_img_list.append(orig_img_np)
                adv_img_np = granso_final_x_dict[best_key].detach().cpu().numpy()[0, :, :, :]
                adv_img_list.append(adv_img_np)

                np.save(orig_save_name, np.asarray(orig_img_list))
                np.save(adv_save_name, np.asarray(adv_img_list))


if __name__ == "__main__":
    clear_terminal_output()
    print("Solving [Max Form] optimization with PyGRANSO.")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", dest="config", type=str,
        default=os.path.join('config_file_examples', 'max-form-granso.json'),
        help="Path to the json config file (pygranso version)."
    )
    args = parser.parse_args()
    cfg = load_json(args.config)  # Load Experiment Configuration file
    cfg["dataset"]["batch_size"] = 1
    main(cfg, dtype=torch.double)  # Use double to compare with PyGRANSO
    print("Completed")