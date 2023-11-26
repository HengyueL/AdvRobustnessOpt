import sys
import os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
import argparse, os, torch, time
import numpy as np
# ===========
from utils.general import load_json, print_and_log
from config_setups.config_log_setup import makedir
from config_setups.config_log_setup import clear_terminal_output, create_log_info,\
    makedir, save_dict_to_csv, save_exp_info, set_default_device
from utils.build import build_model, get_loss_func_eval, get_lp_attack, get_loader_clean, \
    generate_attack_lp, get_samples
from utils.general import tensor2img
# ==== 
from unit_test.granso_max_loss_form import calc_restart_summary


if __name__ == "__main__":
    clear_terminal_output()
    print("...Perform APGD max-loss-form ...")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", dest="config", type=str,
        default=os.path.join('config_file_examples', 'max-loss-form.json'),
        help="Path to the json config file."
    )

    args = parser.parse_args()
    cfg = load_json(args.config)  # Load Experiment Configuration file
    assert cfg["dataset"]["batch_size"] == 1, "Only want to support batch size = 1 for comparison."

    default_dtype = torch.float
    granso_dtype = torch.double

    # Create Root Path to save Experiment logs
    save_root = os.path.join("..", "log_folder")
    root_name = cfg["log_folder"]["save_root"]
    save_root = os.path.join(save_root, root_name)
    makedir(save_root)

    # Experiment ID
    attack_config = cfg["test_attack"]
    attack_alg = attack_config["attack_type"]
    assert "APGD" in attack_alg, "This script only allows APGD experiments. Look for other files for PyGRANSO."
    start_sample = cfg["curr_batch"]
    end_sample = cfg["end_batch_num"]
    
    # Create Experiment Log Dir
    attack_param_config = cfg["apgd_params"]
    attack_metric = attack_param_config["distance_metric"]
    exp_name = "%s-%s-%d-%d" % (
        attack_alg, attack_metric,
        start_sample, end_sample,
    )
    check_point_dir = os.path.join(
        save_root, 
        exp_name
    )
    if cfg["continue"]:
        check_point_dir = cfg["checkpoint_dir"]
    else:
        cfg["checkpoint_dir"] = check_point_dir
        makedir(check_point_dir)
    # Create Experiment Log File and save settings
    log_file = create_log_info(check_point_dir)
    save_exp_info(check_point_dir, cfg)
    if cfg["save_vis"]:
        vis_dir = os.path.join(
            check_point_dir, "dataset_vis"
        )
        os.makedirs(vis_dir, exist_ok=True)

    device, _ = set_default_device(cfg)

    # Create save csv dir
    final_res_csv_dir = os.path.join(
        check_point_dir, "attack_result_log.csv"
    )
    final_summary = {
        "sample_id": [],
        "true_label": [],

        "adv_distance": [],
        "eps": [],
        "adv_loss": [],
        "box_violation": [],

        "time": []
    }
    
    # === Setup Classifier Model ===
    cls_model_config = cfg["classifier_model"]
    classifier_model, msg = build_model(
        model_config=cls_model_config, 
        global_config=cfg, 
        device=device
    )
    classifier_model.eval()
    print_and_log(msg, log_file, mode="w")

    # ==== Get either APGD-CE or APGD-Margin ====
    attack = get_lp_attack(
        attack_alg,
        cfg["apgd_params"], 
        classifier_model, 
        device
    )

    # ==== Construct the original unclipped margin loss for evaluating attack success ====
    eval_loss_func, msg = get_loss_func_eval(
        "Margin", 
        reduction="none", 
        use_clip_loss=False
    )

    # ==== Get Clean Data Loader ====
    _, val_loader, _ = get_loader_clean(
        cfg, only_val=True, shuffle_val=False
    )

    # attack params
    apgd_config = cfg["apgd_params"]
    attack_bound = apgd_config["attack_bound"]

    # === Lists to save dataset ===
    orig_image_list = []
    adv_image_list = []

    for batch_idx, data in enumerate(val_loader):
        if batch_idx < cfg["curr_batch"]:
            # Option to select image to test
            # do nothing because these batches have been tested.
            pass
        else:
            if batch_idx > (cfg["end_batch_num"]-1):
                break
            print_and_log(
                "===== Testing APGD on Sample [%d] =====" % batch_idx, log_file
            )
            inputs, labels = get_samples(
                cfg,
                data_from_loader=data
            )
            inputs = inputs.to(device, dtype=default_dtype)
            labels = labels.to(device)
            with torch.no_grad():
                pred_logits = classifier_model(inputs)
                pred = pred_logits.argmax(1)
            attack_target = labels
            pred_correct = (pred == attack_target).sum().item() > 0.5

            final_summary["sample_id"].append(batch_idx)

            if not pred_correct:
                print_and_log(
                    "    Sample [%d] predicted wrongly. Skip for RE." % batch_idx,
                    log_file
                )
                # === Write Dummy values in the exp log ===
                for key in final_summary.keys():
                    if key not in ["sample_id"]:
                        final_summary[key].append(-1e12)
            else:
                final_summary["true_label"].append(pred.item())
                print_and_log(
                    "Prediction Correct, now APT opt...",
                    log_file
                )

                
                adv_output_dict = {}
                iter_log = {}
                
                t_start = time.time()
                # ==== Perform APGD ====
                attacked_adv_input = generate_attack_lp(
                    inputs, labels, device, attack
                )
                time_end = time.time()

                # === Use dictionary format to use the granso summary function
                adv_output_dict[0] = attacked_adv_input
                iter_log[0] = 0
                feasibility_thres = 0.01
                final_attack_sample, best_loss, best_distance, _, box_violations, _, _ = calc_restart_summary(
                    adv_output_dict,
                    iter_log,
                    inputs,
                    attack_metric,
                    log_file,
                    classifier_model,
                    eval_loss_func,
                    labels,
                    None,
                    attack_bound,
                    feasibility_thres
                ) 
                # Record Result
                final_summary["adv_distance"].append(best_distance)
                final_summary["eps"].append(attack_bound)
                final_summary["adv_loss"].append(best_loss)
                final_summary["box_violation"].append(box_violations)
                final_summary["time"].append(time_end-t_start)
                print_and_log(
                    "    -- OPT total time -- %.04f \n" % (
                        time_end-t_start
                    ),
                    log_file
                )

                # ==== Save Visualization =====
                if cfg["save_vis"]:
                    vis_dir = os.path.join(
                        check_point_dir, "dataset_vis"
                    )
                    os.makedirs(vis_dir, exist_ok=True)

                    adv_image_np = tensor2img(final_attack_sample.to(device, dtype=torch.float))
                    orig_image_np = tensor2img(inputs)
                    orig_image_list.append(orig_image_np)
                    adv_image_list.append(adv_image_np)

                    orig_save_name = os.path.join(
                        vis_dir, "orig_img_list.npy"
                    )
                    adv_save_name = os.path.join(
                        vis_dir, "adv_img_list.npy"
                    )   

                    np.save(
                        orig_save_name, np.asarray(orig_image_list)
                    )
                    np.save(
                        adv_save_name, np.asarray(adv_image_list)
                    )
                    print("Check Save Array Shape: ", np.asarray(orig_image_list).shape)

            save_dict_to_csv(
                final_summary, final_res_csv_dir
            )