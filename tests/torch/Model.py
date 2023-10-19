import torch
import torch.nn

import lasp.torch.differential
import lasp.torch.thresholding

class TotalVariation(torch.nn.Module):

    class Layer(torch.nn.Module):

        def __init__(self,
            kernel_size: tuple[int, int],
            lamda: tuple[float, bool],
            sigma: tuple[float, bool]
        ) -> None:

            super(TotalVariation.Layer, self).__init__()
            
            self.inv = torch.nn.Conv2d(
                in_channels=1, 
                out_channels=1, 
                kernel_size=kernel_size,
                padding = 'same'
            )
            
            self.HT = torch.nn.Conv2d(
                in_channels=1, 
                out_channels=1, 
                kernel_size=kernel_size,
                padding = 'same'
            )
            
            # 5e-2
            self.lamda = torch.nn.Parameter(
                data = torch.tensor([lamda[0]], dtype=torch.float),
                requires_grad = lamda[1]
            )
            
            # 5e-3
            self.sigma = torch.nn.Parameter(
                data = torch.tensor([sigma[0]], dtype=torch.float),
                requires_grad = sigma[1]
            )
        
        def forward(self, 
            g: torch.Tensor, # (1, N, M)
            d_x: torch.Tensor, # (1, N, M)
            d_y: torch.Tensor, # (1, N, M)
            b_x: torch.Tensor, # (1, N, M)
            b_y: torch.Tensor # (1, N, M)
        ) -> torch.Tensor:
            
            a = self.sigma * (
                lasp.torch.differential.dxT2D_circ(d_x-b_x, detach=True)
                + lasp.torch.differential.dyT2D_circ(d_y-b_y, detach=True)
            ) # (1, N, M)

            b = a + self.HT(g) # (1, N, M)

            f = self.inv(a + b) # (1, N, M)

            f_dx = lasp.torch.differential.dx2D_circ(f, detach=True) # (1, N, M)
            f_dy = lasp.torch.differential.dy2D_circ(f, detach=True) # (1, N, M)

            d_x, d_y = lasp.torch.thresholding.multidimensional_soft(
                d = torch.cat(
                    tensors = [ f_dx, f_dy ],
                    dim = 0
                ),
                epsilon = self.lamda / self.sigma,
                gamma_zero = 1e-12
            ) # (1, N, M), (1, N, M)

            b_x += (f_dx - d_x) # (1, N, M)
            b_y += (f_dy - d_y) # (1, N, M)

            return f, d_x, d_y, b_x, b_y


    def __init__(self, 
            kernel_size: tuple[int, int], 
            nb_iterations: int,
            lamda: tuple[float, bool] = (5e-2, True),
            sigma: tuple[float, bool] = (5e-3, True)
        ) -> None:
        super(TotalVariation, self).__init__()
        self.layers = torch.nn.ModuleList(
            [ 
                TotalVariation.Layer(kernel_size, lamda, sigma) 
                for _ in range(0, nb_iterations)
            ]
        )

    def forward(self, 
        g: torch.Tensor # (1, N, M)
    ) -> torch.Tensor:
        
        d_x = torch.zeros_like(g) # (1, N, M)
        d_y = torch.zeros_like(g) # (1, N, M)
        b_x = torch.zeros_like(g) # (1, N, M)
        b_y = torch.zeros_like(g) # (1, N, M)

        for layer in self.layers:
            f, d_x, d_y, b_x, b_y = layer(g, d_x, d_y, b_x, b_y)

        return f
    
    @classmethod
    def from_config(self,
        config: dict
    ) -> 'TotalVariation':

        device = config['model']['device']

        kernel_size = config['model']['kernel_size']
        nb_iterations = config['model']['nb_iterations']

        # (5e-2, True)
        lamda_learn = config['model']['lamda']['requires_grad']
        lamda_value = config['model']['lamda']['initialize']

        sigma_learn = config['model']['sigma']['requires_grad']
        sigma_value = config['model']['sigma']['initialize']

        model = TotalVariation(
            kernel_size,
            nb_iterations,
            (lamda_value, lamda_learn),
            (sigma_value, sigma_learn)
        )

        return model.to(device)