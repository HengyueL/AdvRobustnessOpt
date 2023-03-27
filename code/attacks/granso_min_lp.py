# This file realizes the min robust radius opt. by granso
import gc, torch
import numpy as np
# ==== Granso Import ====
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct


# ==== Target Min =====
def granso_min(
    inputs, labels,
    x0,
    target_label,
    model, attack_type,
    device,
    max_iter=1000,
    ineq_tol=1e-8,
    eq_tol=1e-8, 
    opt_tol=1e-8, 
    steering_c_viol=0.1,
    steering_c_mu=0.9,
    stat_l2=True,
    steering_l1=True,
    mu0=1,
    mem_size_param=100,
    linesearch_maxit=None,
    print_log=True,
    dtype="double",
    constraint_folding_type="L2-folding",
    granso_slack_variable=None,
    wall_time=None,
    H0_init=None,
    granso_opt_log_file=None
    ):

    attack_type = attack_type
    opts = pygransoStruct()
    if dtype == "double":
        opts.double_precision = True  # Do not use double precision
        dtype = torch.double
    elif dtype == "float":
        opts.double_precision = False
        dtype = torch.float
    else:
        raise RuntimeError("Specify the torch default dtype please.")

    if attack_type in ["Linf"]:
        var_in = {"z": list(inputs.shape),"t":[1, 1]}
    elif attack_type in ['L1-Reform']:
        var_in = {"z": list(inputs.shape), "t": list(inputs.shape)}
    else:
        var_in = {"z": list(inputs.shape)}
        
    comb_fn = lambda X_struct : user_fn_min_separate_constraint(
        X_struct=X_struct, 
        inputs=inputs, 
        labels=labels,
        model=model,
        attack_type=attack_type,
        target_label=target_label,
        constraint_folding_type=constraint_folding_type,
        granso_slack_variable=granso_slack_variable,
        granso_opt_log_file=granso_opt_log_file
    )

    opts.torch_device = device
    opts.maxit = max_iter
    opts.opt_tol = opt_tol
    opts.viol_ineq_tol = ineq_tol
    opts.viol_eq_tol = eq_tol
    opts.steering_c_viol = steering_c_viol
    opts.steering_c_mu = steering_c_mu
    opts.stat_l2_model = stat_l2
    opts.steering_l1_model = steering_l1

    # === Add Line Search Params ===
    if linesearch_maxit is not None:
        opts.linesearch_maxit = linesearch_maxit
    
    if wall_time is not None:
        # Set granso wall time
        opts.maxclocktime = wall_time  

    opts.print_frequency = 1
    if not print_log:
        opts.print_level = 0

    opts.limited_mem_size = mem_size_param
    opts.mu0 = mu0
    
    # ==== Assigned Initialization ====
    opts.x0 = torch.reshape(
        x0,
        (-1, 1)
    )

    # === Warm start option ===
    if H0_init is not None:
        opts.limited_mem_warm_start = H0_init
        # ==== For exact warm restart, need to turn off scaling
        opts.scaleH0 = False
    else:
        # ==== Init t based on x0 and x(input) ====
        init_err = torch.abs(x0 - inputs)
        # if not is_continue:
        if attack_type in ["Linf"]:
            t = torch.ones([1, 1]).to(device, dtype=dtype) * torch.amax(init_err)
            opts.x0 = torch.cat([opts.x0, t], dim=0)
            print("Check Init scaling [t]", torch.linalg.vector_norm(t.reshape(-1), ord=float("inf")).item())
        elif attack_type in ["L1-Reform"]:
            t = torch.ones_like(opts.x0).to(device, dtype=dtype) * init_err.clone().reshape(-1, 1)
            opts.x0 = torch.cat([opts.x0, t], dim=0)
            print("Check Init scaling [t]", torch.linalg.vector_norm(t.reshape(-1), ord=1).item())
        
    # ==== Start Optimization ====
    soln = pygranso(
        var_spec=var_in,
        combined_fn=comb_fn,
        user_opts=opts
    )
    # ==== Calculation Done ====
    # Collect Garbage
    gc.collect()
    return soln


def user_fn_min_separate_constraint(
    X_struct, 
    inputs, labels,
    model,
    attack_type,
    target_label=None,
    constraint_folding_type="L2-folding",
    granso_slack_variable=None
    ):

    z = X_struct.z
    labels = labels.item()
    if target_label is not None:
        target_label = target_label.item() if type(target_label) == torch.Tensor else target_label

    if attack_type in ["Linf", "L1-Reform"]:
        if granso_slack_variable == "t":
            t = X_struct.t
        elif granso_slack_variable == "t^2":
            t = (X_struct.t) ** 2
        else:
            raise RuntimeError("Slack variable input invalid.")
    
    adv_inputs = z
    # reshape delta vec
    delta_vec = (adv_inputs-inputs).reshape(-1)

    # normalizing factor 
    num_pixels = torch.as_tensor(np.prod(adv_inputs.shape))
    normalization_factor = num_pixels**0.5
    # normalization_factor = 1

    # objective
    if attack_type == 'L2':
        f = torch.linalg.vector_norm(delta_vec, ord=2)
    elif attack_type == 'L1':
        f = torch.linalg.vector_norm(delta_vec, ord=1)
    elif attack_type == "L1-Reform":
        t_vec = t.reshape(-1)
        f = torch.sum(t_vec)
    elif attack_type == 'Linf':
        f = t * normalization_factor
    elif attack_type == "Linf-Orig":
        f = torch.linalg.vector_norm(delta_vec, ord=float("inf"))
    else:
        # General norm
        order_number = float(attack_type.split("L")[-1])
        f = torch.sum(torch.abs(delta_vec)**(order_number))**(1 / order_number)
    
    # Equality constraint
    ce = None

    # Inequality constraint
    ci = pygransoStruct()
    
    logits_outputs = model(adv_inputs)
    fc = logits_outputs[:, labels] # true class output

    if target_label is not None:
        fl = logits_outputs[:, target_label] # attack target
        ci.c1 = (fc - fl)
    else:
        k = logits_outputs.shape[1]
        fl = torch.hstack(
            (
                logits_outputs[:, 0:labels],
                logits_outputs[:, labels+1:k]
            )
        )
        ci.c1 = fc - torch.amax(fl)
    
    box_constr = torch.hstack(
        (adv_inputs.reshape(inputs.numel()) - 1,
        -adv_inputs.reshape(inputs.numel()))
    )
    box_constr = torch.clamp(box_constr, min=0)
    if constraint_folding_type == "L2-folding":
        folded_constr = torch.linalg.vector_norm(box_constr.reshape(-1), ord=2)
    else:
        raise RuntimeError("Unimplemented Constraint Folding methods.")
    ci.c2 = folded_constr

    if attack_type in ["Linf", "L1-Reform"]:
        if attack_type == 'Linf':
            err_vec = torch.abs(delta_vec) - t
        elif attack_type == "L1-Reform":
            err_vec = torch.hstack(
                (delta_vec - t_vec,
                -delta_vec - t_vec)
            )
        constr_vec = torch.clamp(err_vec, min=0)
        
        if constraint_folding_type == "L2-folding":
            folded_constr = torch.linalg.vector_norm(constr_vec.reshape(-1), ord=2)
        ci.c3 = folded_constr
    
    return [f,ci,ce]

if __name__ == "__main__":
    print()

