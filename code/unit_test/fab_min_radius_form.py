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

def calc_min_dist_sample_fab(
    fab_adv_output,  # A dict of fab results
    orig_input, 
    attack_type,  
    model, 
    target_label,
    log_file
):
    keys = fab_adv_output.keys()

    fab_distance_dict = {}
    boundary_distance_dict = {}
    box_violation_dict = {}
    attacked_label_dict = {}

    best_boundary_distance, best_key = float("inf"), None

    for key in keys:
        fab_output = fab_adv_output[key]
        # Check Result 
        with torch.no_grad():
            pred = model(fab_output)
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

        fab_output = fab_output.clone().reshape(1, -1)
        orig_input = orig_input.clone().reshape(1, -1)
        # Check [0, 1] box constraint
        greater_than_1 = torch.sum(torch.where(fab_output > (1 + 1e-4), 1, 0)).item()
        smaller_than_0 = torch.sum(torch.where(fab_output < (0 - 1e-4), 1, 0)).item()
        num_violation = greater_than_1 + smaller_than_0
        box_violation_dict[key] = num_violation
        
        err_vec = torch.abs(fab_output - orig_input)
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
        
        fab_distance_dict[key] = p_distance
        msg = "  >> Restart [%d] has  - radius [%.04f] - boundary distance [%.04f] - box violations [%d] >> " % (
            key, p_distance, distance_to_boundary, num_violation
        )
        print_and_log(msg, log_file)

    return fab_distance_dict, boundary_distance_dict, box_violation_dict, attacked_label_dict, best_key



def main(cfg, default_dtype=torch.double):
    dtype = default_dtype
    set_random_seeds(cfg)

    # === Create Experiment Save Dirs ===
    save_root = os.path.join(
        "..", "log_folder", cfg["log_folder"]["save_root"]
    )
    makedir(save_root)
    # === Experiment Name ===
    img_start_idx, img_end_idx = cfg["curr_sample"], cfg["end_sample"]
    exp_name = "FAB-%s_%d-%d" % (cfg["fab_params"]["distance_metric"], img_start_idx, img_end_idx)
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
    opt_config = cfg["fab_params"]
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

    # === Create some variables from the cfg file
    n_restart = opt_config["fab_n_restarts"]
    init_scale = opt_config["fab_init_scale"]
    max_iter = opt_config["fab_max_iter"]
    fab_opt_runner = FABAttackPTModified(
        classifier_model, n_restarts=n_restart, n_iter=max_iter, targeted=False, eps=init_scale,
        seed=cfg["seed"], norm=attack_type, verbose=False, device=device
    )

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
                msg = "Sample [%d] - prediction correct. Begin FAB OPT >>>" % batch_idx
                print_and_log(msg, log_file)
                
                # OPT
                fab_time_start = time.time()
                fab_adv_output = fab_opt_runner.perturb(
                    inputs, labels
                )
                fab_time_end = time.time()
                fab_time = fab_time_end - fab_time_start
                msg = "-- FAB Unterget %d restart total time [%.04f] -- " % (n_restart, fab_time)

                distance_dict, boundary_distance_dict, box_violation_dict, attacked_label_dict, best_key = calc_min_dist_sample_fab(
                    fab_adv_output, inputs, attack_type, classifier_model, labels.item(), log_file
                )
                # === Log the best result ===
                for key in fab_adv_output.keys():
                    result_summary["sample_idx"].append(batch_idx)
                    result_summary["restart"].append(key)
                    result_summary["true_label"].append(labels.item())
                    result_summary["max_logit_before_opt"].append(pred_before.item())
                    result_summary[attack_distance_key].append(distance_dict[key])
                    result_summary["max_logit_after_opt"].append(attacked_label_dict[key])
                    result_summary["distance_to_decision_boundary"].append(boundary_distance_dict[key])
                    result_summary["box_constraint_violation"].append(box_violation_dict[key])
                    result_summary["time"].append(fab_time)

            save_dict_to_csv(
                result_summary, result_csv_dir
            )
            
            if cfg["save_vis"]:
                vis_dir = os.path.join(ckpt_dir, "dataset_vis")
                makedir(vis_dir)
                orig_save_name, adv_save_name = os.path.join(vis_dir, "orig_imgs.npy"), os.path.join(vis_dir, "adv_imgs.npy")

                orig_img_np = inputs.detach().cpu().numpy()[0, :, :, :]
                orig_img_list.append(orig_img_np)
                adv_img_np = fab_adv_output[best_key].detach().cpu().numpy()[0, :, :, :]
                adv_img_list.append(adv_img_np)

                np.save(orig_save_name, np.asarray(orig_img_list))
                np.save(adv_save_name, np.asarray(adv_img_list))



if __name__ == "__main__":
    clear_terminal_output()
    print("Solving [Min Form] optimization with FAB.")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", dest="config", type=str,
        default=os.path.join('config_file_examples', 'min-form-fab.json'),
        help="Path to the json config file (FAB version)."
    )
    args = parser.parse_args()
    cfg = load_json(args.config)  # Load Experiment Configuration file
    cfg["dataset"]["batch_size"] = 1
    main(cfg, default_dtype=torch.float)  # Use double to compare with PyGRANSO
    print("Completed")