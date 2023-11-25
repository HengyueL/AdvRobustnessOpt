import os, torch, time
import numpy as np
import robustness.data_augmentation as da
from torchvision.datasets import CIFAR100, CIFAR10
from torch.utils.data import DataLoader
from autoattack.autopgd_base import APGDAttack_targeted
from autoattack.fab_pt import FABAttack_PT
from autoattack.square import SquareAttack

# ===== Project import =====
from loss.losses import CELoss, MarginLossOrig, NegCELoss, NegMarginLoss
from models.model import CifarResNetFeatureModel, UnionRes18, DeepMindWideResNetModel,\
    ImageNetResNetFeatureModel, AlexNetFeatureModel
from dataset.transform import TRAIN_TRANSFORMS_DEFAULT, TEST_TRANSFORMS_DEFAULT
from dataset.dataset import ImageNetN
from attacks.auto_attacks import AutoL1Attack, AutoL2Attack,\
     AutoLinfAttack, APGDAttackMargin, APGDAttackCE
from attacks.target_fab import FABAttackPTModified

# ==== PAT attacks ====
from percept_utils.modified_attacks import FastLagrangePerceptualAttack_Revised, \
    LagrangePerceptualAttack_Revised_Lp, PerceptualPGDAttack


def build_model(
    model_config, 
    num_classes, 
    device
    ):
    model_type = model_config["type"]
    model_path = model_config["weight_path"]
    use_clamp_value = False
    
    if model_path is not None:
        model_path = eval(model_path)
    if "RobustUnion" in model_type:
        msg = " >> Init a UnionRes18 Model \n"
        model = UnionRes18(use_clamp_input=use_clamp_value).to(device)
        if model_path is not None:
            msg += " >> Weight from path: %s" % model_path
            state_dict = torch.load(
                model_path,
                map_location=device)
            model.model.load_state_dict(state_dict)
    elif "DeepMind" in model_type:
        msg = " >> Init a DeepMind WRN70(Cifar10) Model \n"
        model = DeepMindWideResNetModel(use_clamp_input=use_clamp_value).to(device)
        if model_path is not None:
            msg += " >> Weight from path: %s" % model_path
            state_dict = torch.load(
                model_path, map_location=device
            )
            model.model.load_state_dict(state_dict)
    elif "PAT-ImageNet" in model_type:
        msg = " >> Init a ImageNetResNetFeatureModel (PAT version) \n"
        model = ImageNetResNetFeatureModel(
            num_classes=num_classes,
            use_clamp_input=use_clamp_value,
            pretrained=False,
            lpips_feature_layer=None
        ).to(device)
        if model_path is not None:
            msg += " >> Weight from path: %s" % model_path
            state_dict = torch.load(
                model_path,
                map_location=device
            )["model"]
            model.model.load_state_dict(state_dict)
    elif "PAT-Cifar" in model_type:
        msg = " >> Init a Cifar-Preatrained (PAT version) \n"
        model = CifarResNetFeatureModel(
            num_classes=num_classes,
            use_clamp_input=use_clamp_value,
            pretrained=False,
            lpips_feature_layer=None
        ).to(device)
        if model_path is not None:
            msg += " >> Weight from path: %s" % model_path
            state_dict = torch.load(
                model_path,
                map_location=device
            )["model"]
            model.model.load_state_dict(state_dict)
    else:
        raise RuntimeError("The author is lazy and did not implement another model yet.")
    return model, msg


def build_lpips_model(lpips_model_config, device):
    lpips_name = lpips_model_config["type"]
    lpips_layer = eval(lpips_model_config["lpips_layer"])
    if lpips_name == "alexnet":
        model = AlexNetFeatureModel(
            lpips_feature_layer=lpips_layer
            ).to(device)
    else:
        raise RuntimeError("Unimplemented LPIPS Model.")
    model.eval()
    return model


def get_optimizer(optimizer_cfg, model):
    opt_name = optimizer_cfg["type"]
    lr = optimizer_cfg["lr"]
    weight_decay = optimizer_cfg["weight_decay"]

    if opt_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr, 
            weight_decay=weight_decay
            )
    elif opt_name == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise RuntimeError("The author did not implement other optimizers yet.")
    return optimizer


# === Normal Loss function for evaluating results ===
def get_loss_func_eval(name, reduction, use_clip_loss):
    if name == "CE":
        loss_func = CELoss(
            reduction=reduction, 
            use_clip_loss=use_clip_loss
        )
        msg = "  >> CE Loss Function."
    elif name == "Margin":
        loss_func = MarginLossOrig(
            reduction=reduction, 
            use_clip_loss=use_clip_loss
        )
        msg = "  >> Margin Loss Function"
    else:
        raise RuntimeError("Unimplemented Loss Type")
    return loss_func, msg


# === Get Negative loss function for PyGRANSO optimization ===
def get_granso_loss_func(name, reduction, use_clip_loss, dataset_name="imagenet"):
    if name == "CE":
        if use_clip_loss:
            if dataset_name  == "imagenet":
                clamp_value = 4.7
            elif dataset_name == "cifar10":
                clamp_value = 2.4
            else:
                raise RuntimeError("Need to specify dataset name for clipping")
        else:
            clamp_value = None
        loss_func = NegCELoss(
            reduction=reduction, 
            use_clip_loss=use_clip_loss,
            clamp_value=clamp_value
        )
        msg = "  >> Granso Neg-CE Loss"
    elif name == "Margin":
        loss_func = NegMarginLoss(
            reduction=reduction, 
            use_clip_loss=use_clip_loss
        )
        msg = "  >> Granso Neg-Margin Loss"
    else:
        raise RuntimeError("Unimplemented Granso Objective Function. Check 'granso_attacks.py'. ")
    return loss_func, msg


def get_train_transform(config):
    dataset_name = dataset_name = config["dataset"]["name"]
    transform_type = config["dataset"]["train_transform"]
    input_size = config["dataset"]["input_size"]
    if "imagenet" in dataset_name:
        if transform_type == "default":
            train_transform = da.TRAIN_TRANSFORMS_IMAGENET
        if transform_type == "plain":
            train_transform = da.TEST_TRANSFORMS_IMAGENET
        else:
            train_transform = da.TRAIN_TRANSFORMS_IMAGENET
            # raise RuntimeError("Unsupported Data Augmentation for Imagenet Training.")
    elif "cifar" in dataset_name:
        if transform_type == "default":
            train_transform = TRAIN_TRANSFORMS_DEFAULT(input_size)
        elif transform_type == "plain":
            train_transform = TEST_TRANSFORMS_DEFAULT(32)
        else:
            raise RuntimeError("Unsupported Data Augmentation for Imagenet Training.")
    else:
        raise RuntimeError("Error dataset name. Check input json files.")
    return train_transform


def get_loader_clean(config, only_val=False, shuffle_val=True):
    dataset_name = config["dataset"]["name"]
    data_path = config["dataset"]["dataset_path"]
    batch_size = config["dataset"]["batch_size"]
    num_worders = config["dataset"]["workers"]

    train_transform = get_train_transform(config)
    msg = " >> Train Transform: [%s]" % train_transform.__repr__()

    if "imagenet" in dataset_name:
        label_list = eval(config["dataset"]["label_list"])
        dataset = ImageNetN(data_path=data_path,
                            label_list=label_list,
                            train_transform=train_transform)
        train_loader, val_loader = dataset.make_loaders(
                                    workers=num_worders, 
                                    batch_size=batch_size,
                                    only_val=only_val,
                                    shuffle_val=shuffle_val
                                    )
        return train_loader, val_loader, msg
    elif "cifar100" in dataset_name:
        if only_val:
            train_loader = None
        else:
            train_dataset = CIFAR100(
                root=data_path, train=True, transform=train_transform, download=True
            )
            train_loader = DataLoader(
                dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worders
            )
        
        val_dataset = CIFAR100(
            root=data_path, train=False, transform=TEST_TRANSFORMS_DEFAULT(32), download=True
        )
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=batch_size, shuffle=shuffle_val, num_workers=num_worders
        )
        return train_loader, val_loader, msg
    elif "cifar10" in dataset_name:
        if only_val:
            train_loader = None
        else:
            train_dataset = CIFAR10(
                root=data_path, train=True, transform=train_transform, download=True
            )
            train_loader = DataLoader(
                dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worders
            )

        val_dataset = CIFAR10(
            root=data_path, train=False, transform=TEST_TRANSFORMS_DEFAULT(32), download=True
        )
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=batch_size, shuffle=shuffle_val, num_workers=num_worders
        )
        return train_loader, val_loader, msg
    else:
        raise RuntimeError("Unsupported Dataset Yet.")


def get_auto_attack(model, attack_config, dataset_config):
    dataset_name = dataset_config["name"]
    if "cifar100" in dataset_name:
        name = "cifar"
    elif "imagenet" in dataset_name:
        name = "imagenet"
    else:
        raise RuntimeError("Unsupported dataset name. Check input.")

    attack_name = attack_config["attack_type"]
    attack_bound = attack_config["attack_bound"]
    if attack_name == "L2":
        # default bound = 1200/255
        attack = AutoL2Attack(
            model, name, bound=attack_bound
        )  
    elif attack_name == "Linf":
        # default bound = 4/255
        attack = AutoLinfAttack(
            model, name, bound=attack_bound
        )
    elif attack_name == "L1":
        attack = AutoL1Attack(
            model, name, bound=attack_bound
        )
    else:
        raise RuntimeError("Unsupported Attack Type.")
    msg = " >> Getting Attack: [%s]" % attack_name
    return attack, msg


def get_lp_attack(
    attack_type,
    attack_config, 
    model, device):

    attack_type = attack_type
    norm_type = attack_config["distance_metric"]
    bound = attack_config["attack_bound"]
    
    if attack_type == "APGD":
        print("APGD-CE")
        n_iter = attack_config["apgd_max_iter"]
        n_restart = attack_config["apgd_n_restarts"]
        attack = APGDAttackCE(
            model, n_restarts=n_restart, n_iter=n_iter, verbose=False, 
            eps=bound, norm=norm_type, eot_iter=1, rho=0.75, 
            seed=None, device=device, logger=None
        )
    elif attack_type == "APGD-Margin":
        print("APGD-Margin")
        n_iter = attack_config["apgd_max_iter"]
        n_restart = attack_config["apgd_n_restarts"]
        attack = APGDAttackMargin(
            model, n_restarts=n_restart, n_iter=n_iter, verbose=False, 
            eps=bound, norm=norm_type, eot_iter=1, rho=0.75, 
            seed=None, device=device, logger=None
        )
    elif attack_type == "FAB":
        n_iter = attack_config["fab_max_iter"]
        n_restart = attack_config["fab_n_restarts"]
        attack =  FABAttackPTModified(
            model, n_restarts=n_restart, n_iter=n_iter,
            targeted=False,
            eps=bound, seed=0, norm=norm_type,
            verbose=False, device=device
        )
    elif attack_type == "FAB-Target":
        n_iter = attack_config["fab_max_iter"]
        n_restart = attack_config["fab_n_restarts"]
        attack = FABAttackPTModified(
            model, n_restarts=n_restart, n_iter=n_iter, targeted=True,
            eps=bound, seed=0, norm=norm_type,
            verbose=False, device=device
        )
    elif attack_type == "Square":
        attack = SquareAttack(
            model, p_init=0.8, n_queries=5000,
            eps=bound, norm=norm_type, n_restarts=1,
            seed=None, verbose=False, device=device,
            resc_schedule=False
        )
    else:
        raise RuntimeError("Not Implemented Attack.")
    return attack


def generate_attack_lp(
    inputs, labels, 
    device, attack,
    target_class=None
    ):
    inputs = inputs.to(device, dtype=torch.float)
    if labels is not None:
        labels = labels.to(device, dtype=torch.long)

    if type(attack) in [APGDAttackCE, APGDAttackMargin]:
        attack_adv_inputs = attack.perturb(
            x=inputs, y=labels, best_loss=False
        )
    elif type(attack) in [APGDAttack_targeted]:
        attack_adv_inputs = attack.perturb(
            x=inputs, y=labels
        )
    elif type(attack) in [SquareAttack, FABAttack_PT]:
        attack_adv_inputs = attack.perturb(
            x=inputs, y=labels
        )  # None for square attack is random target attack
    elif type(attack) in [FABAttackPTModified]:
        if attack.targeted == True:
            assert target_class is not None, "Need to specify a target class."
            attack_adv_inputs = attack.perturb(
                inputs, labels, target_class
            )
        else:
            attack_adv_inputs = attack.perturb(
                inputs, labels
            )
    else:
        raise RuntimeError("Undefined Attack Type.")
    return attack_adv_inputs


def get_perceptual_attack(
    config, model, lpips_model,
    loss_func):
    print("PAT attack use loss: ", loss_func.__repr__())
    print("Use Clip Loss:", loss_func.use_clip_loss)
    attack_name = config["test_attack"]["alg"]
    num_iter = config["test_attack"]["pat_iter"]
    attack_bound = config["test_attack"]["bound"]
    if attack_name == "FastLagrange":
        train_attack = FastLagrangePerceptualAttack_Revised(
            model,
            lpips_model,
            loss_func=loss_func,
            num_iterations=num_iter,
            bound=attack_bound
            )
    elif attack_name == "PPGD":
        train_attack = PerceptualPGDAttack(
            model,
            lpips_model,
            num_iterations=num_iter,
            bound=attack_bound
        )
    elif attack_name == "LagrangeLp":
        lpips_type = config["test_attack"]["lpips_type"]
        train_attack = LagrangePerceptualAttack_Revised_Lp(
            model,
            lpips_model,
            lpips_distance_norm=lpips_type,
            num_iterations=num_iter,
            bound=attack_bound
        )
    else:
        raise RuntimeError("Unsupported Perceptual Attack Type.")
    return train_attack


def get_samples(config, data_from_loader):
    if "imagenet" in config["dataset"]["name"]:
        inputs = data_from_loader[0]
        labels = data_from_loader[1]
    elif "cifar" in config["dataset"]["name"]:
        inputs = data_from_loader[0]
        labels = data_from_loader[1]
    else:
        raise RuntimeError("Unsupported Dataset")
    return inputs, labels


if __name__ == "__main__":
    pass

