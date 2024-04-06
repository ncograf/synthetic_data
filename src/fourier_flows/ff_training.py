import torch
import numpy as np
from fourier_flow import FourierFlow

def train_fourier_flow(X : torch.Tensor, epochs : int, learning_rate : float):
    
    D = X.shape[0]
    T = X.shape[1]
    hidden_dim = T*2
    n_layer = 5

    model = FourierFlow(hidden_dim=hidden_dim, D=D, T=T, n_layer=n_layer)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    
    losses = []

    for epoch in range(epochs):
        
        optimizer.zero_grad()
        
        z, log_prob_z, log_jac_det = model(X)
        loss = torch.mean(-log_prob_z - log_jac_det)
        
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 10:
            print((f'Epoch: {epoch:>10d}, last loss {loss.item():>10.4f},'
                  f'aveage_loss {np.mean(losses):>10.4f}'))
        
        
    print(f'Finished training!')
    
    return model, losses