import torch
from src.ebmUtils import Reg, RegLatent, loss_functional
def trainTeacher(model
                    , optimizer
                    , epochs
                    , times
                    , device 
                    , train_data
                    , alpha = 10
                    , beta = 10
                    , f_clk = 2
                    , scale = 1e-2
                    , bound_for_saving = 6000):
    print("Training Teacher Model")
    rem = []
    rem.append(torch.tensor(0.0))
    reg = []
    p_times = epochs//times # print times

    y = train_data

    # best model
    models = []

    p_p, p_c = [], []
    reg_proj_array = []
    reg_latent_array = []
    memory = []
    cost_array = []
    best_cost = torch.inf
    reg_latent = torch.tensor(0.0)
    for epoch in range(epochs):
        
        # forward
        y_pred = model(model.z)
        # add pump
        if epoch % f_clk == 0 and epoch != 0:
            y_pred_std = torch.randn(y_pred.shape)*memory[-1]*scale
            y_pred_std = y_pred_std.to(y_pred.device)
            y_pred = y_pred + y_pred_std

        reg_proj = Reg(y_pred) # regularize projection module
        if reg_proj == 0:
            reg_proj = torch.tensor(0.0)
        reg_proj_array.append(reg_proj) # save reg_proj

        if epoch > 1:
            reg_latent = RegLatent(model.z_l) # regularize latent space
            reg_latent_array.append(reg_latent.item()) # save reg_latent
        
        e = loss_functional(y_pred.cpu().detach().numpy(), y.cpu().detach().numpy(), model)
        e.requires_grad = True

        F, z = e.min(1) # get energy and latent
        memory.append(torch.sum(F).item()) # save energy to memory

        cost = torch.sum(F) + alpha*reg_proj + beta*reg_latent # add regularizers
        cost_array.append(cost.item()) # save cost

        # backward
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        if cost < best_cost:
            best_cost = cost
            best_model = model
            best_outputs = y_pred
            best_z = z
            best_lat = model.z_l
            best_epoch = epoch
        if cost < bound_for_saving:
            p_p.append(y_pred)
            p_c.append(cost)
        if (epoch + 1) % p_times == 0:
            # print
            print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                    "Training loss: {:.5f}.. ".format(cost.item()),
                    "Reg Proj: {:.5f}.. ".format(reg_proj.item()),
                    "Reg Latent: {:.5f}.. ".format(reg_latent.item()),
                    "Memory: {:.5f}.. ".format(torch.sum(F).item()),
                    "Cost: {:.5f}.. ".format(cost.item())
                    )
        
    return {
        "best_model": best_model,
        "best_outputs": best_outputs,
        "best_z": best_z,
        "best_lat": best_lat,
        "best_epoch": best_epoch,
        "p_p": p_p,
        "p_c": p_c,
        "reg_proj_array": reg_proj_array,
        "reg_latent_array": reg_latent_array,
        "memory": memory,
        "cost_array": cost_array
    }
