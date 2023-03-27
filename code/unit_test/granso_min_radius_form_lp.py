import sys
import os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
import argparse, os, torch, time
import numpy as np
# =======
from utils.build import build_model, get_loss_func_eval, get_loader_clean, \
    get_samples
from config_setups.config_log_setup import clear_terminal_output, create_log_info,\
    makedir, save_dict_to_csv, save_exp_info, set_default_device
from utils.general import load_json, print_and_log
from attacks.granso_min_attack import granso_min_target
from attacks.transform_functions import sine_transfor_box as sine_transform
# ======= The following functions should be synced ======


if __name__ == "__main__":
    clear_terminal_output()
    print("...Perform PyGRANSO min-radius-form with Lp distances ...")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", dest="config", type=str,
        default=os.path.join('config_file_examples', 'max-loss-form.json'),
        help="Path to the json config file."
    )

    args = parser.parse_args()
    cfg = load_json(args.config)  # Load Experiment Configuration file
    assert cfg["dataset"]["batch_size"] == 1, "Only want to support batch size = 1"

    default_dtype = torch.float
    granso_dtype = torch.double

    # Create Root Path to save Experiment logs
    save_root = os.path.join("..", "log_folder")
    root_name = cfg["log_folder"]["save_root"]
    save_root = os.path.join(save_root, root_name)
    makedir(save_root)

    # Experiment ID
    attack_config = cfg["test_attack"]
    attack_type = attack_config["attack_type"]
    assert attack_type == "PyGranso", "This script only allows PyGranso experiments. Look for other files for APGD."
    start_sample = cfg["curr_batch"]
    end_sample = cfg["end_batch_num"]

    # Create Experiment Log Dir 
    granso_config = cfg["granso_params"]
    attack_metric = granso_config["distance_metric"]
    exp_name = "GransoMin-%s-%d-%d" % (
        attack_metric,
        start_sample, end_sample,
    )
    check_point_dir = os.path.join(
        save_root, 
        exp_name
    )
    if cfg["continue"]:
        # For continue option
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

    # To save sample OPT trajectory
    granso_opt_traj_log_file = create_log_info(
        check_point_dir, "Granso_OPT_traj.txt"
    )
    # Path to save the experiment summary
    final_res_csv_dir = os.path.join(
        check_point_dir, "min_result_log.csv"
    )
    final_summary = {
        "sample_idx": [],
        "true_label": [],

        attack_metric: [],
        "adv_loss": [],

        "box_violation": [],
        "iters": [],
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
    
    # ==== Construct the original unclipped loss for evaluation ====
    eval_loss_func, msg = get_loss_func_eval(
        "Margin", 
        reduction="none", 
        use_clip_loss=False
    )

    # some param used 
    granso_init_scale = 0.1
    # ==== Get Clean Data Loader ====
    batch_size = cfg["dataset"]["batch_size"]
    assert batch_size == 1, "PyGRANSO currently only accept batch_size = 1. (One problem at a time)"
    _, val_loader, _ = get_loader_clean(
        cfg, only_val=True, shuffle_val=False
    )
    num_classes = cfg["dataset"]["num_classes"]

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
                "===== Testing Sample [%d] =====" % batch_idx, log_file
            )
            inputs, labels = get_samples(
                cfg,
                data_from_loader=data
            )
            # ==== Get samples and prepare OPT ====
            inputs = inputs.to(device, dtype=default_dtype)
            labels = labels.to(device)
            classifier_model = classifier_model.to(device, dtype=default_dtype)
            with torch.no_grad():
                pred_logits = classifier_model(inputs)
                pred = pred_logits.argmax(1)
            attack_target = labels
            pred_correct = (pred == attack_target).sum().item() > 0.5
            
            final_summary["sample_id"].append(batch_idx)

            if not pred_correct:
                print_and_log(
                    "    Sample [%d] predicted wrongly. Skip RE for this sample." % batch_idx,
                    log_file
                )
                # === Write Dummy values in the exp log ===
                for key in final_summary.keys():
                    if key not in ["sample_id"]:
                        final_summary[key].append(-1e12)
            else:
                final_summary["true_label"].append(pred.item())
                print_and_log(
                    "    Prediction Correct, now Granso opt...",
                    log_file
                )