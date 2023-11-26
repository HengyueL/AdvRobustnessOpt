# This file realizes the max attack by granso
import gc, torch
import numpy as np
# ==== Granso Import ====
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct


# ==== This is the realization of max-loss form solved by PyGRANSO ====
# ==== Untargeted Version ====
# ==== Note: PyGRANSO only support batch_size = 1 input currently ====

def granso_max_attack(
    inputs, labels, x0, model, attack_type, device,
    loss_func, eps,
    lpips_distance=None,
    max_iter=1000,
    eq_tol=1e-8,
    ieq_tol=1e-8,
    opt_tol=1e-8, 
    steering_c_viol=0.1,
    steering_c_mu=0.9,
    stat_l2=False,
    steering_l1=True,
    mu0=1,
    mem_size_param=10,
    print_log=True,
    dtype=torch.double,
    linesearch_maxit=None,
    maxclocktime=None,
    H0_init=None,
):
    """
        inputs --- input_image, torch.tensor with shape (1, 3, x, x)
        labels --- label of input_image, torch.tensor with shape (1, 1)
        model --- model to be attacked (f_\theta)
        attack_type --- a string indicate of the distance metric d. Valid input as ["L1", "L1.5", ...]
                        "Linf" default uses reformulation & constraint folding; "Linf-Orig" uses the original formulation;
                        "PAT" uses perceptual distnace.
        device --- torch.device("xxx")
        loss_func --- the objective loss function
        eps --- attack budget
        input_constraint_type ---  the [0, 1] box constraint uses "L2-folding"; uses original form if None.

        other parameters are corresponding to setup params for PyGRANSO
    """
    attack_type = attack_type

    # ==== pygranso ====
    opts = pygransoStruct()
    if dtype != torch.double:
        opts.double_precision = False
    opts.torch_device = device
    opts.maxit = max_iter
    opts.opt_tol = opt_tol
    opts.viol_ineq_tol = ieq_tol
    opts.viol_eq_tol = eq_tol
    opts.steering_c_viol = steering_c_viol
    opts.steering_c_mu = steering_c_mu
    opts.stat_l2_model = stat_l2
    opts.steering_l1_model = steering_l1

    # === Add Line Search Params ===
    if linesearch_maxit is not None:
        opts.linesearch_maxit = linesearch_maxit
    opts.print_frequency = 1
    if not print_log:
        opts.print_level = 0
    opts.limited_mem_size = mem_size_param
    opts.mu0 = mu0

    if maxclocktime is not None:
        opts.maxclocktime = maxclocktime  # Set granso wall time

    var_in = {"z": list(inputs.shape)}
    
    # ===== If attack is LPIPS, then this cannot be None =====
    if attack_type == "PAT":
        assert lpips_distance is not None, "Check Attack Type and LPIPS distance specifications"
        lpips_distance = lpips_distance.to(device, dtype=torch.float)

    comb_fn = lambda X_struct : user_fn_max(
        X_struct=X_struct, 
        inputs=inputs, 
        labels=labels,
        model=model,
        attack_type=attack_type,
        eps=eps,
        loss_func=loss_func,
        lpips_distance=lpips_distance
    )

    # ==== Init ====
    opts.x0 = torch.reshape(x0, (-1, 1))
    
    # === If H0 is given, then pygranso uses warm start ===
    if H0_init is not None:
        opts.limited_mem_warm_start = H0_init
        # ==== For exact warm restart, need to turn off scaling
        opts.scaleH0 = False
    
    # ==== Start Optimization ====
    try:
        soln = pygranso(
            var_spec=var_in,
            combined_fn=comb_fn,
            user_opts=opts
        )
    except:
        soln = None
    # ==== Calculation Done ====
    # Collect Garbage
    gc.collect()
    return soln


def user_fn_max(
    X_struct, 
    inputs, 
    labels,
    model,
    attack_type,
    eps,
    loss_func,
    lpips_distance=None
):
    adv_inputs = X_struct.z

    epsilon = eps
    logit_outputs = model(adv_inputs)
    f = loss_func(logit_outputs, labels)

    # normalizing factor if used 
    # num_pixels = torch.as_tensor(np.prod(adv_inputs.shape))
    # normalization_factor = num_pixels**0.5
    normalization_factor = 1

    # Equality constraint
    ce = None
    # Inequality constraint
    ci = pygransoStruct()

    delta_vec = (adv_inputs-inputs).reshape(-1)

    # ===== Norm Constraint =====
    if attack_type == "Linf":
        linf_dist_violation = torch.abs(delta_vec) - epsilon
        constr_vec = torch.clamp(linf_dist_violation, min=0)
        constr_vec = torch.linalg.vector_norm(constr_vec, ord=2)
    elif attack_type == "Linf-Orig":
        linf_dist_violation = torch.linalg.vector_norm(delta_vec, ord=float("inf")) - eps
        constr_vec = torch.clamp(linf_dist_violation, min=0)
    elif attack_type == "L1":
        l1_dist_violation = torch.linalg.vector_norm(delta_vec, ord=1) - eps
        constr_vec = torch.clamp(l1_dist_violation, min=0)
        constr_vec = torch.linalg.vector_norm(constr_vec, ord=2)
    elif attack_type == "L2":
        l2_dist_violation = torch.linalg.vector_norm(delta_vec, ord=2) - eps
        constr_vec = torch.clamp(l2_dist_violation, min=0)
        constr_vec = torch.linalg.vector_norm(constr_vec, ord=2)
    elif attack_type == "PAT":
        assert lpips_distance is not None, "need lpips distance function to compute the distance"
        lpips_dist_violation = lpips_distance(adv_inputs, inputs) - eps
        constr_vec = torch.clamp(lpips_dist_violation, min=0)
        constr_vec = torch.linalg.vector_norm(constr_vec, ord=2)
    elif "L" in attack_type:
        norm_p = float(attack_type.split("L")[-1])
        lp_dist = torch.sum(torch.abs(delta_vec)**norm_p) ** (1/norm_p)
        lp_dist_violation = lp_dist - eps
        constr_vec = torch.clamp(lp_dist_violation, min = 0)
        constr_vec = torch.linalg.vector_norm(constr_vec, ord=2)
    else:
        raise RuntimeError("Undefined Maximization Problem. Check Code or add function")
    ci.c1 = constr_vec

    # ==== [0, 1] Box Constraint ====
    box_constr = torch.hstack(
        (adv_inputs.reshape(inputs.numel()) - 1,
        -adv_inputs.reshape(inputs.numel()))
    )
    box_constr = torch.clamp(box_constr, min=0)
    folded_constr = torch.linalg.vector_norm(box_constr.reshape(-1), ord=2) / normalization_factor
    ci.c2 = folded_constr
    return [f,ci,ce]
