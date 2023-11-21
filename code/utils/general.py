import os
import numpy as np
import torch, json
import matplotlib.pyplot as plt


def load_numpy_dict(file_path):
    """
        Load dict data saved in numpy
    """
    summary_data = np.load(file_path, allow_pickle=True)
    saved_dict = summary_data.item()
    return saved_dict


def check_lr_criterion(lr, target_lr):
    """
        True if meet lr criterion
    """
    return lr <= target_lr


def get_optimizer_lr(optimizer):
    for param_group in optimizer.param_groups:
        optimizer_lr = param_group["lr"]
    return optimizer_lr


def write_log_txt(file_name, msg, mode="a"):
    """
        Write training msg to file in case of cmd print failure in MSI system.
    """
    with open(file_name, mode) as f:
        f.write(msg)
        f.write("\n")


def print_and_log(msg, log_file_name, mode="a", terminal_print=True):
    """
        Write msg to a text file.
    """
    if type(msg) == str:
        if terminal_print == True:
            print(msg)
        write_log_txt(log_file_name, msg, mode=mode)
    elif type(msg) == list:
        for word in msg:
            print_and_log(word, log_file_name, mode=mode, terminal_print=terminal_print)
    else:
        assert RuntimeError("msg input only supports string / List input.")


def save_model_ckp(
    config, model, epoch, iter_num,
    optimizer, scheduler, save_dir, name=None
    ):
    """
        Save training realted checkpoints: 
            model_state_dict, scheduler, epoch, iter, optimizer
    """
    if config["data_parallel"]:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    if name is None:
        checkpoint = {
            "epoch": epoch,
            "iter": iter_num,
            "model_state_dict": model_state_dict,
            "optimizer": config["optimizer"]["type"],
            "scheduler": scheduler,
            "optimizer_state_dict": optimizer.state_dict()
            }
        torch.save(checkpoint, os.path.join(save_dir, "checkpoint.pth"))
        msg = "  >> Save Normal Training Checkpoint (for resume training purpose) ..."
    else:
        checkpoint = {
            "epoch": epoch,
            "iter": iter_num,
            "model_state_dict": model_state_dict,
        }
        torch.save(checkpoint, os.path.join(save_dir, "%s.pth" % name))
        msg = "  >> Save %s Model State Dict ..." % name
    return msg


def tensor2img(tensor):
    """
        Take into an array with shape [1, 3, x, x] and convert into an ndarray with shape [x, x, 3]
    """
    array = tensor.detach().cpu().numpy()[0, :, :, :]
    array = np.transpose(array, [1, 2, 0])
    return array


def save_image(image_array, name):
    """
        Save image (visualizaton) to path <name>
    """
    fig = plt.figure(figsize=(8,8))
    plt.imshow(image_array)
    plt.axis('off')
    plt.savefig(name)
    plt.close(fig)


def rescale_array(array):
    """
        Rescale an array to [0, 1]
    """
    ele_min, ele_max = np.amin(array), np.amax(array)
    array = (array - ele_min) / (ele_max - ele_min)
    return array


def load_json(json_path):
    content = json.load(open(json_path, "r"))
    return content


def plot_train_val_eval(
        summary_dict, save_dir, 
        save_name, save_every
    ):
    plot_val = True if "val_clean_loss" in summary_dict.keys() else False
    x = list(range(len(summary_dict["train_loss"])))
    fig, ax = plt.subplots(nrows=3, ncols=1,
                           figsize=(9, 6), sharex=True)
    ax[0].plot(x, summary_dict["train_loss"], label="train_adv_loss")
    if plot_val:
        ax[0].plot(x, summary_dict["val_clean_loss"], label="val_clean_loss")
    ax[0].set_ylabel("Loss")
    ax[0].set_xlabel("Train_Iter * %d" % save_every)
    ax[0].legend()

    ax[1].plot(x, summary_dict["train_acc"], label="adv_train_accuracy")
    if plot_val:
        ax[1].plot(x, summary_dict["val_clean_acc"], label="val_clean_acc")
    ax[1].set_ylabel("Acc (%)")
    ax[1].set_xlabel("Train_Iter * %d" % save_every)
    ax[1].legend()

    # ax[2].plot(summary_dict["train_radius_mean"], label="train_mean_radius")
    ax[2].errorbar(
        x, summary_dict["train_radius_mean"], yerr=summary_dict["train_radius_std"], 
        label='train mean (std) radius'
    )
    if plot_val:
        ax[2].errorbar(
            x, summary_dict["val_radius_mean"], yerr=summary_dict["val_radius_std"], 
            label='val mean (std) radius'
        )
    ax[2].set_ylabel("Radius")
    ax[2].set_xlabel("Train_Iter * %d" % save_every)
    ax[2].legend()
    save_dir = os.path.join(save_dir, "%s.png" % save_name)
    plt.savefig(save_dir)
    plt.close(fig)


def get_samples(config, data_from_loader):
    if config["dataset"]["name"] in ["imagenet", "cifar10"]:
        inputs = data_from_loader[0]
        labels = data_from_loader[1]
    else:
        raise RuntimeError("Unsupported Dataset")
    return inputs, labels


if __name__ == "__main__":
    pass

