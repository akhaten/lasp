

import numpy
import typing

import lasp.differential
import lasp.utils
import lasp.filters.linear
import lasp.thresholding


def mumford_shah_denoise_v1(
    y: numpy.ndarray,
    alpha: float,
    beta: float,
    sigma: float,
    nb_iterations: int,
    tolerance: float,
    error_history: list[float] = None
) -> numpy.ndarray:

    """Mumford Shah
    
    Solve argmin_{x} { (alpha/2) || y - x ||^2 + (beta/2) || nabla y ||^2 + || nabla y ||_1 }
    """

    Dx = lasp.differential.dx
    Dy = lasp.differential.dy
    Dxt = lasp.differential.dxT
    Dyt = lasp.differential.dyT

    # Build kernel
    uker = numpy.zeros_like(y)

    laplacian = lasp.filters.linear.laplacian()
    lap_diag = lasp.utils.fourier_diagonalization(
        kernel = laplacian,
        shape_out = y.shape 
    )
   

    uker = alpha + (beta+sigma) * lap_diag

    rhs1fft = alpha * numpy.fft.fft2(y)

    # Initialization
    u = numpy.copy(y) 
    d_x=numpy.zeros_like(y)
    d_y=numpy.zeros_like(y)
    b_x=numpy.zeros_like(y)
    b_y=numpy.zeros_like(y)

    for _ in range(0, nb_iterations):

        rhs2= sigma*Dxt(d_x-b_x)+sigma*Dyt(d_y-b_y)
        rhsfft = rhs1fft + numpy.fft.fft2(rhs2)

        u0=numpy.copy(u)
        
        u = numpy.real(numpy.fft.ifft2(rhsfft / uker))    

        err = numpy.linalg.norm(u-u0, 'fro') / numpy.linalg.norm(u, 'fro')
        
        if not(error_history is None):
            error_history.append(err)

        if err < tolerance:
            break
        
        u_dx, u_dy = Dx(u), Dx(y)

        d_x, d_y = lasp.thresholding.multidimensional_soft(
            d = numpy.array([ u_dx + b_x, u_dy + b_y ]),
            epsilon = 1/sigma
        )

        b_x += (u_dx - d_x)
        b_y += (u_dy - d_y)

    u_normalized = lasp.utils.normalize(u)

    return u_normalized


def mumford_shah_denoise_v2(
    y: numpy.ndarray,
    alpha: float,
    beta: float,
    sigma: float,
    nb_iterations: int,
    tolerance: float,
    error_history: list[float] = None
) -> numpy.ndarray:

    """Mumford Shah
    
    Solve argmin_{x} { (alpha/2) || y - x ||^2 + (beta/2) || nabla y ||^2 + || nabla y ||_1 }
    """

    Dx = numpy.zeros(y)
    Dx[0, 0:2] = numpy.array([1, -1])
    Dx = numpy.fft.fft2(Dx)
    Dxt = numpy.conj(Dx)

    Dy = numpy.zeros_like(y)
    Dy[0:2, 0] = numpy.transpose(numpy.array([1, -1]))
    Dy = numpy.fft.fft2(Dy)
    Dyt = numpy.conj(Dy)


    # Build kernel
    uker = numpy.zeros_like(y)

    laplacian = lasp.filters.linear.laplacian()
    lap_diag = lasp.utils.fourier_diagonalization(
        kernel = laplacian,
        shape_out = y.shape 
    )
   

    uker = alpha + (beta+sigma) * lap_diag

    rhs1fft = alpha * numpy.fft.fft2(y)

    # Initialization
    u = numpy.copy(y) 
    d_x=numpy.zeros_like(y)
    d_y=numpy.zeros_like(y)
    b_x=numpy.zeros_like(y)
    b_y=numpy.zeros_like(y)

    for _ in range(0, nb_iterations):

        rhs2fft = sigma*Dxt*numpy.fft.fft2(d_x-b_x)+sigma*Dyt*numpy.fft.fft2(d_y-b_y)
        rhsfft = rhs1fft + rhs2fft

        u0=numpy.copy(u)
        
        u_fft = rhsfft / uker
        u = numpy.real(numpy.fft.ifft2(u_fft))    

        err = numpy.linalg.norm(u-u0, 'fro') / numpy.linalg.norm(u, 'fro')
        
        if not(error_history is None):
            error_history.append(err)

        if err < tolerance:
            break
        
        u_dx = numpy.real(numpy.fft.ifft2(Dx * u_fft))
        u_dy = numpy.real(numpy.fft.ifft2(Dy * u_fft))

        d_x, d_y = lasp.thresholding.multidimensional_soft(
            d = numpy.array(
                [ 
                    u_dx + b_x,
                    u_dy + b_y 
                ]
            ),
            epsilon = 1/sigma
        )

        b_x += (u_dx - d_x)
        b_y += (u_dy - d_y)

    u_normalized = lasp.utils.normalize(u)

    return u_normalized


def mumford_shah_denoise_v3(
    y: numpy.ndarray,
    alpha: float,
    beta: float,
    sigma: float,
    nb_iterations: int,
    tolerance: float,
    error_history: list[float] = None
) -> numpy.ndarray:

    """Mumford Shah
    
    Solve argmin_{x} { (alpha/2) || y - x ||^2 + (beta/2) || nabla y ||^2 + || nabla y ||_1 }
    """

    Dx = numpy.zeros(y)
    Dx[0, 0:2] = numpy.array([1, -1])
    Dx = numpy.fft.fft2(Dx)
    Dxt = numpy.conj(Dx)

    Dy = numpy.zeros_like(y)
    Dy[0:2, 0] = numpy.transpose(numpy.array([1, -1]))
    Dy = numpy.fft.fft2(Dy)
    Dyt = numpy.conj(Dy)


    # Build kernel
    uker = numpy.zeros_like(y)

    laplacian = Dxt * Dx + Dyt * Dy
    lap_diag = lasp.utils.fourier_diagonalization(
        kernel = laplacian,
        shape_out = y.shape 
    )
   

    uker = alpha + (beta+sigma) * lap_diag

    rhs1fft = alpha * numpy.fft.fft2(y)

    # Initialization
    u = numpy.copy(y) 
    d_x=numpy.zeros_like(y)
    d_y=numpy.zeros_like(y)
    b_x=numpy.zeros_like(y)
    b_y=numpy.zeros_like(y)

    for _ in range(0, nb_iterations):

        rhs2fft = sigma*Dxt*numpy.fft.fft2(d_x-b_x)+sigma*Dyt*numpy.fft.fft2(d_y-b_y)
        rhsfft = rhs1fft + rhs2fft

        u0=numpy.copy(u)
        
        u_fft = rhsfft / uker
        u = numpy.real(numpy.fft.ifft2(u_fft))    

        err = numpy.linalg.norm(u-u0, 'fro') / numpy.linalg.norm(u, 'fro')
        
        if not(error_history is None):
            error_history.append(err)

        if err < tolerance:
            break
        
        u_dx = numpy.real(numpy.fft.ifft2(Dx * u_fft))
        u_dy = numpy.real(numpy.fft.ifft2(Dy * u_fft))

        d_x, d_y = lasp.thresholding.multidimensional_soft(
            d = numpy.array(
                [ 
                    u_dx + b_x,
                    u_dy + b_y 
                ]
            ),
            epsilon = 1/sigma
        )

        b_x += (u_dx - d_x)
        b_y += (u_dy - d_y)

    u_normalized = lasp.utils.normalize(u)

    return u_normalized



def mumford_shah_deconv_v1(
    y: numpy.ndarray, 
    h: numpy.ndarray, 
    alpha: float,
    beta: float,
    sigma: float,
    nb_iterations: int,
    tolerance: float,
    error_history: list[float] = None
) -> numpy.ndarray:

    """Mumford Shah
    
    Solve argmin_{x} { (alpha/2) || y - Hx ||^2 + (beta/2) || nabla y ||^2 + || nabla y ||_1 }
    """

    Dx = lasp.differential.dx
    Dy = lasp.differential.dy
    Dxt = lasp.differential.dxT
    Dyt = lasp.differential.dyT

    # Build kernel
    uker = numpy.zeros_like(y)

    laplacian = lasp.filters.linear.laplacian()
    lap_diag = lasp.utils.fourier_diagonalization(
        kernel = laplacian,
        shape_out = y.shape 
    )
   
    h_diag = lasp.utils.fourier_diagonalization(
        kernel = h,
        shape_out = y.shape
    )

    h2_diag = numpy.abs(h_diag)**2


    uker = alpha * h2_diag + (beta+sigma) * lap_diag

    rhs1fft = alpha * numpy.conj(h_diag) * numpy.fft.fft2(y)

    # Initialization
    u = numpy.copy(y) 
    d_x=numpy.zeros_like(y)
    d_y=numpy.zeros_like(y)
    b_x=numpy.zeros_like(y)
    b_y=numpy.zeros_like(y)

    for _ in range(0, nb_iterations):

        rhs2 = sigma*Dxt(d_x-b_x)+sigma*Dyt(d_y-b_y)
        rhsfft = rhs1fft + numpy.fft.fft2(rhs2)

        u0=numpy.copy(u)
        
        u = numpy.real(numpy.fft.ifft2(rhsfft / uker))    

        err = numpy.linalg.norm(u-u0, 'fro') / numpy.linalg.norm(u, 'fro')
        
        if not(error_history is None):
            error_history.append(err)

        if err < tolerance:
            break
        
        u_dx, u_dy = Dx(u), Dy(u)

        d_x, d_y = lasp.thresholding.multidimensional_soft(
            d = numpy.array([ u_dx + b_x, u_dy + b_y ]),
            epsilon = 1/sigma
        )

        b_x += (u_dx - d_x)
        b_y += (u_dy - d_y)

    u_normalized = lasp.utils.normalize(u)

    return u_normalized


def mumford_shah_deconv_v2(
    y: numpy.ndarray, 
    h: numpy.ndarray, 
    alpha: float,
    beta: float,
    sigma: float,
    nb_iterations: int,
    tolerance: float,
    error_history: list[float] = None
) -> numpy.ndarray:

    """Mumford Shah
    # TODO: make test
    Solve argmin_{x} { (alpha/2) || y - Hx ||^2 + (beta/2) || nabla y ||^2 + || nabla y ||_1

    Difference with v1 ?
    We use Derivation in fourier space
    """

    # Dx = lasp.differential.dx
    # Dy = lasp.differential.dy
    # Dxt = lasp.differential.dxT
    # Dyt = lasp.differential.dyT

    Dx = numpy.zeros(y)
    Dx[0, 0:2] = numpy.array([1, -1])
    Dx = numpy.fft.fft2(Dx)
    Dxt = numpy.conj(Dx)

    Dy = numpy.zeros_like(y)
    Dy[0:2, 0] = numpy.transpose(numpy.array([1, -1]))
    Dy = numpy.fft.fft2(Dy)
    Dyt = numpy.conj(Dy)

    # Build kernel

    laplacian = lasp.filters.linear.laplacian()
    lap_diag = lasp.utils.fourier_diagonalization(
        kernel = laplacian,
        shape_out = y.shape 
    )
    # lap_diag = Dxt * Dx + Dyt * Dy
    #lap_diag = numpy.abs(Dx)**2 + numpy.abs(Dy)**2
   
    h_diag = lasp.utils.fourier_diagonalization(
        kernel = h,
        shape_out = y.shape
    )

    h2_diag = numpy.abs(h_diag)**2


    uker = alpha * h2_diag + (beta+sigma) * lap_diag

    rhs1fft = alpha * numpy.conj(h_diag) * numpy.fft.fft2(y)

    # Initialization
    u = numpy.copy(y) 
    d_x=numpy.zeros_like(y)
    d_y=numpy.zeros_like(y)
    b_x=numpy.zeros_like(y)
    b_y=numpy.zeros_like(y)

    for _ in range(0, nb_iterations):

        rhs2fft = sigma*Dxt*numpy.fft.fft2(d_x-b_x)+sigma*Dyt*numpy.fft.fft2(d_y-b_y)
        rhsfft = rhs1fft + rhs2fft

        u_prev = numpy.copy(u)

        u_fft = rhsfft / uker
        u = numpy.real(numpy.fft.ifft2(u_fft))    

        err = numpy.linalg.norm(u-u_prev, 'fro') / numpy.linalg.norm(u, 'fro')
        
        if not(error_history is None):
            error_history.append(err)

        if err < tolerance:
            break
        
    
        u_dx = numpy.real(numpy.fft.ifft2(Dx * u_fft))
        u_dy = numpy.real(numpy.fft.ifft2(Dy * u_fft))

        d_x, d_y = lasp.thresholding.multidimensional_soft(
            d = numpy.array(
                [ 
                    u_dx + b_x,
                    u_dy + b_y 
                ]
            ),
            epsilon = 1/sigma
        )

        b_x += (u_dx - d_x)
        b_y += (u_dy - d_y)

    u_normalized = lasp.utils.normalize(u)

    return u_normalized

def mumford_shah_deconv_v3(
    y: numpy.ndarray, 
    h: numpy.ndarray, 
    alpha: float,
    beta: float,
    sigma: float,
    nb_iterations: int,
    tolerance: float,
    error_history: list[float] = None
) -> numpy.ndarray:

    """Mumford Shah
    # TODO: make test
    Solve argmin_{x} { (alpha/2) || y - Hx ||^2 + (beta/2) || nabla y ||^2 + || nabla y ||_1

    Difference with v1 ?
    We use Derivation in fourier space
    and we set lap_diag = Dxt * Dx + Dyt * Dy and not laplacia  filter
    """

    # Dx = lasp.differential.dx
    # Dy = lasp.differential.dy
    # Dxt = lasp.differential.dxT
    # Dyt = lasp.differential.dyT

    Dx = numpy.zeros(y)
    Dx[0, 0:2] = numpy.array([1, -1])
    Dx = numpy.fft.fft2(Dx)
    Dxt = numpy.conj(Dx)

    Dy = numpy.zeros_like(y)
    Dy[0:2, 0] = numpy.transpose(numpy.array([1, -1]))
    Dy = numpy.fft.fft2(Dy)
    Dyt = numpy.conj(Dy)

    # Build kernel

    laplacian = lasp.filters.linear.laplacian()
    lap_diag = lasp.utils.fourier_diagonalization(
        kernel = laplacian,
        shape_out = y.shape 
    )
    lap_diag = Dxt * Dx + Dyt * Dy
    #lap_diag = numpy.abs(Dx)**2 + numpy.abs(Dy)**2
   
    h_diag = lasp.utils.fourier_diagonalization(
        kernel = h,
        shape_out = y.shape
    )

    h2_diag = numpy.abs(h_diag)**2


    uker = alpha * h2_diag + (beta+sigma) * lap_diag

    rhs1fft = alpha * numpy.conj(h_diag) * numpy.fft.fft2(y)

    # Initialization
    u = numpy.copy(y) 
    d_x=numpy.zeros_like(y)
    d_y=numpy.zeros_like(y)
    b_x=numpy.zeros_like(y)
    b_y=numpy.zeros_like(y)

    for _ in range(0, nb_iterations):

        rhs2fft = sigma*Dxt*numpy.fft.fft2(d_x-b_x)+sigma*Dyt*numpy.fft.fft2(d_y-b_y)
        rhsfft = rhs1fft + rhs2fft

        u_prev = numpy.copy(u)

        u_fft = rhsfft / uker
        u = numpy.real(numpy.fft.ifft2(u_fft))    

        err = numpy.linalg.norm(u-u_prev, 'fro') / numpy.linalg.norm(u, 'fro')
        
        if not(error_history is None):
            error_history.append(err)

        if err < tolerance:
            break
        
    
        u_dx = numpy.real(numpy.fft.ifft2(Dx * u_fft))
        u_dy = numpy.real(numpy.fft.ifft2(Dy * u_fft))

        d_x, d_y = lasp.thresholding.multidimensional_soft(
            d = numpy.array(
                [ 
                    u_dx + b_x, 
                    u_dy + b_y 
                ]
            ),
            epsilon = 1/sigma
        )

        b_x += (u_dx - d_x)
        b_y += (u_dy - d_y)

    u_normalized = lasp.utils.normalize(u)

    return u_normalized


def mumford_shah_fsr(
    y: numpy.ndarray, 
    h: numpy.ndarray, 
    alpha: float,
    beta0: float,
    beta1: float,
    sigma: float,
    d: int,
    nb_iterations: int,
    tolerance: float,
    error_history: list[float] = None
) -> numpy.ndarray:

    def block_mm(nr, nc, Nb, x1, order: str) -> numpy.ndarray:

        block_shape = numpy.array([nr, nc])

        x1 = lasp.utils.blockproc_reshape(x1, block_shape, order)
        x1 = numpy.reshape(x1, newshape=(nr*nc, Nb), order=order)
        x1 = numpy.sum(x1, axis=1)
        x = numpy.reshape(x1, newshape=(nr, nc), order=order)

        return x

    """Mumford Shah
    # TODO: make test
    Solve $$argmin_{x} { (alpha/2) || y - Hx ||^2 + (beta/2) || nabla y ||^2 + || nabla y ||_1$$

    Params:
        - y: low resolution
        - h: deblur kernel
        - alpha: hyper parameter of data fidelity
        - beta0: hyper parameter of dirichlet energy
        - beta1: hyper parameter of total variation
        - sigma: split-bregman hyper parameter
        - d: decimation
        - nb_iterations: number of iteration
        - tolerance: tolerance
        - error_history: save errors of each iteration
    
    Returns:
        - high resolution of y
    """

    y_rows, y_cols = y.shape

    # Dx = lasp.differential.dx
    # Dy = lasp.differential.dy
    # Dxt = lasp.differential.dxT
    # Dyt = lasp.differential.dyT

    Dx = numpy.zeros(shape=(d*y_rows, d*y_cols))
    Dx[0, 0:2] = numpy.array([1, -1])
    Dx = numpy.fft.fft2(Dx)
    Dxt = numpy.conj(Dx)

    Dy = numpy.zeros(shape=(d*y_rows, d*y_cols))
    Dy[0:2, 0] = numpy.transpose(numpy.array([1, -1]))
    Dy = numpy.fft.fft2(Dy)
    Dyt = numpy.conj(Dy)


    # Build kernel

    # laplacian = lasp.filters.linear.laplacian()
    # lap_diag = lasp.utils.fourier_diagonalization(
    #     kernel = laplacian,
    #     shape_out = y.shape 
    # )
    #######
    lap_diag = Dxt * Dx + Dyt * Dy + 1e-8
    #lap_diag = numpy.abs(Dx)**2 + numpy.abs(Dy)**2
   
    h_diag = lasp.utils.fourier_diagonalization(
        kernel = h,
        shape_out = numpy.array([d*y_rows, d*y_cols])
    )

    h_diag_transp = numpy.conj(h_diag)

    h2_diag = numpy.abs(h_diag)**2


    #uker = alpha * h2_diag + (beta+sigma) * lap_diag
    lap_diag = (2*beta0+sigma) * lap_diag

    
    STy = numpy.zeros(shape=(d*y_rows, d*y_cols))
    STy[0::d, 0::d] = numpy.copy(y)
    rhs1fft = alpha * h_diag_transp * numpy.fft.fft2(STy)

    # Initialization
    import PIL.Image
    u = numpy.array(
        PIL.Image.Image.resize(
            PIL.Image.fromarray(y),
            (y_rows*d, y_cols*d),
            PIL.Image.Resampling.BICUBIC
        )
    )
    # u = numpy.copy(y) 
    d_x=numpy.zeros_like(u)
    d_y=numpy.zeros_like(u)
    b_x=numpy.zeros_like(u)
    b_y=numpy.zeros_like(u)

    for _ in range(0, nb_iterations):

        rhs2fft = sigma*Dxt*numpy.fft.fft2(d_x-b_x)+sigma*Dyt*numpy.fft.fft2(d_y-b_y)
        rhsfft = rhs1fft + rhs2fft

        u_prev = numpy.copy(u)

        # Inverse
        #u_fft = rhsfft / uker
        ## Parameters
        # fr = rhsfft
        # fb = h_diag
        # fbc = numpy.conj(h_diag)
        # f2b = h2_diag
        # nr, nc = y.shape
        # m = nr * nc
        # f2d = lap_diag
        # nb = d*d
        ##
        # x1 = h_diag*rhsfft / lap_diag
        x1 = h_diag*rhsfft / lap_diag
        fbr = block_mm(y_rows, y_cols, d*d, x1, order='F')
        # invW = block_mm(y.shape[0], y.shape[1], d*d, h2_diag / lap_diag, order='F')
        invW = block_mm(y_rows, y_cols, d*d, h2_diag / lap_diag, order='F')
        # invWBR = fbr / (invW + beta1*d*d)
        invWBR = fbr / (invW + beta1*d*d)
        fun = lambda block : block*invWBR
        FCBinvWBR = lasp.utils.blockproc(numpy.copy(h_diag_transp), numpy.array([y_rows, y_cols]), fun)
        ## Returns
        u_fft = (rhsfft - FCBinvWBR) / lap_diag
        # u_fft /= beta1
        ##########

        # Compute errors
        u = numpy.real(numpy.fft.ifft2(u_fft))    

        err = numpy.linalg.norm(u-u_prev, 'fro') / numpy.linalg.norm(u, 'fro')
        
        if not(error_history is None):
            error_history.append(err)

        if err < tolerance:
            break
        
        
        u_dx = numpy.real(numpy.fft.ifft2(Dx * u_fft))
        u_dy = numpy.real(numpy.fft.ifft2(Dy * u_fft))


        d_x, d_y = lasp.thresholding.multidimensional_soft(
            d = numpy.array(
                [ 
                    u_dx + b_x, 
                    u_dy + b_y 
                ]
            ),
            epsilon = beta1 / sigma
        )

        b_x += (u_dx - d_x)
        b_y += (u_dy - d_y)

    u_normalized = lasp.utils.normalize(u)

    return u_normalized