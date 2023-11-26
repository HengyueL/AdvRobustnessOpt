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
from utils.build import build_model, get_lp_attack, get_loader_clean, \
    get_samples
# ==== 


def calc_sample_stas(
    adv_output,  # a tensor of adv_output
    orig_input, 
    attack_type,  
    model, 
    log_file
):

    # Check Result 
    with torch.no_grad():
        pred = model(adv_output)
    attacked_label = pred.argmax(1).item()

    adv_output = adv_output.clone().reshape(1, -1)
    orig_input = orig_input.clone().reshape(1, -1)
    # Check [0, 1] box constraint
    greater_than_1 = torch.sum(torch.where(adv_output > (1 + 1e-4), 1, 0)).item()
    smaller_than_0 = torch.sum(torch.where(adv_output < (0 - 1e-4), 1, 0)).item()
    num_violation = greater_than_1 + smaller_than_0

    
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
    radius = p_distance

    msg = "  >> Adv output  - radius [%.04f] - box violations [%d] >> " % (
        p_distance, num_violation
    )
    print_and_log(msg, log_file)

    return radius, num_violation, attacked_label


def main(cfg, dtype=torch.float):
    set_random_seeds(cfg)
    # === Create Experiment Save Dirs ===
    save_root = os.path.join(
        "..", "log_folder", cfg["log_folder"]["save_root"]
    )
    makedir(save_root)
    # === Experiment Name ===
    img_start_idx, img_end_idx = cfg["curr_sample"], cfg["end_sample"]
    exp_name = "APGD-%s_%d-%d" % (cfg["apgd_params"]["distance_metric"], img_start_idx, img_end_idx)
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

    opt_config = cfg["apgd_params"]
    loss_type = opt_config["loss"]
    eps = opt_config["attack_bound"]

    # ==== Get either APGD-CE or APGD-Margin ====
    if loss_type == "CE":
        attack_alg = "APGD"
    elif loss_type == "Margin":
        attack_alg = "APGD-Margin"
    else:
        raise RuntimeError("Unidentified APGD algorithm. Unsupported loss used.")
    attack = get_lp_attack(
        attack_alg,
        cfg["apgd_params"], 
        classifier_model, 
        device
    )

    # === Format to summarize the final result
    attack_type = opt_config["distance_metric"]
    result_csv_dir = os.path.join(ckpt_dir, "opt_result.csv")
    radius_name = "%s_distance" % attack_type
    result_summary = {
        "sample_idx": [],
        "true_label": [],

        "max_logit_before_opt": [],
        "max_logit_after_opt": [],

        radius_name: [],
        "eps": [],

        "box_constraint_violation": [],
        "time": []
    }

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
                msg = "Sample [%d] - prediction correct. Begin %s OPT >>>" % (batch_idx, "APGD-"+loss_type)
                print_and_log(msg, log_file)

                t_start = time.time()
                # === Perform APGD ===
                attacked_adv_output = attack.perturb(
                    x=inputs, y=labels, best_loss=True
                )
                t_end = time.time()

                radius, box_violations, attacked_label = calc_sample_stas(
                    attacked_adv_output, inputs, attack_type, classifier_model, log_file
                )
                # == Record Result ==
                result_summary["sample_idx"].append(batch_idx)
                result_summary["true_label"].append(labels.item())
                result_summary["max_logit_before_opt"].append(pred_before.item())
                result_summary["max_logit_after_opt"].append(attacked_label)
                result_summary[radius_name].append(radius)
                result_summary["eps"].append(eps)
                result_summary["box_constraint_violation"].append(box_violations)
                result_summary["time"].append(t_end - t_start)
            save_dict_to_csv(
                result_summary, result_csv_dir
            )
            if cfg["save_vis"] and (pred_correct_before > 0.5):
                vis_dir = os.path.join(ckpt_dir, "dataset_vis")
                makedir(vis_dir)
                orig_save_name, adv_save_name = os.path.join(vis_dir, "orig_imgs.npy"), os.path.join(vis_dir, "adv_imgs.npy")

                orig_img_np = inputs.detach().cpu().numpy()[0, :, :, :]
                orig_img_list.append(orig_img_np)
                adv_img_np = attacked_adv_output.detach().cpu().numpy()[0, :, :, :]
                adv_img_list.append(adv_img_np)

                np.save(orig_save_name, np.asarray(orig_img_list))
                np.save(adv_save_name, np.asarray(adv_img_list))


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
    main(cfg, dtype=torch.float)
    print("Complete")