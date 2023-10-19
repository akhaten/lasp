import torch

def multidimensional_soft(
    d: torch.Tensor, 
    epsilon: float, 
    gamma_zero: float=1e-12
) -> torch.Tensor:
    """ Thresholding soft for multidimensional array
    Use generalization of sign function
    
    Params:
        - d : multidimensional array
        - epsilon : threshold
        - gamma_zero : for zero value (prevent "Error detected in DivBackward0")

    Return:
        Array thresholded with dimesion equal to d
    """
    #print('d :', d.size())
    # l22 = 
    
    # s[s==0] = 
    #print('s :', s.size())
    s = torch.sqrt(torch.sum(d**2, axis=0)+gamma_zero)

    ss = torch.where(s > epsilon, (s-epsilon)/s, 0)
    output = torch.concat([(ss*d[i]).unsqueeze(0) for i in range(0, d.size()[0])], 0)
    #print('output :', output.size())
    #print(output.size())
    return output