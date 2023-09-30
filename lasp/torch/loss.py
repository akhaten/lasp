import torch
import lasp.torch.regularization

def mumford_shah_loss(
    alpha: float, 
    beta0: float, 
    beta1: float, 
    y: torch.Tensor,
    y_pred: torch.Tensor,
    isotropic_mode: bool = True
) -> torch.Tensor:
    
    tv_reg = lasp.torch.regularization.tv_isotropic if isotropic_mode \
        else lasp.torch.regularization.tv_anisotropic
    
    fidelity_term = torch.sqrt( torch.sum( (y - y_pred)**2 ) ) 
    dirichlet = lasp.torch.regularization.dirichlet_energy(y_pred, detach=False)
    tv = tv_reg(y_pred, detach=False)
    
    return (alpha / 2) * fidelity_term \
        + (beta0 / 2) * dirichlet \
        + beta1 * tv

class MumfordShahLoss:

    def __init__(self, isotropic_mode: bool) -> None:
        self.isotropic_mode = isotropic_mode
        # self.tv_reg = tv_isotropic if isotropic_mode else tv_anisotropic

    def __call__(self, alpha: float, beta0:float, beta1: float, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # fidelity_term = torch.sum(torch.sqrt( (y_pred - y)**2 )) 
        # dirichlet = dirichlet_energy(y, detach=False)
        # tv = self.tv_reg(y, detach=False)
        # return alpha * fidelity_term + beta0 * dirichlet + beta1 * tv
        return mumford_shah_loss(
            alpha=alpha,
            beta0=beta0,
            beta1=beta1,
            y=y,
            y_pred=y_pred,
            isotropic_mode=self.isotropic_mode
        )