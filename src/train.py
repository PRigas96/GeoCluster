import torch

def trainTeacher(model
                    , criterion
                    , optimizer
                    , epochs
                    , times
                    , device 
                    , train_data
                    , f_clk = 2
                    , scale = 1e-2):
    print("Training Teacher Model")
    costs = []
    rem = []
    rem.append(torch.tensor(0.0))
    reg = []

    y = train_data

    # best model
    models = []

    p_p, p_c = [], []
    reg_let_r, reg_let_c = [], []
    
    for epoch in range(epochs):
        
        # forward
        y_pred = model(model.z)
        # add pump
        if epoch % f_clk == 0:
            model.pump()


