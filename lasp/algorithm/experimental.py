

import numpy
import typing

import lasp.differential
import lasp.utils
import lasp.filters.linear
import lasp.thresholding

def mumford_shah_v1(
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
    
    Solve argmin_{x} { (alpha/2) || y - Hx ||^2 + (beta/2) || nabla y ||^2 + || nabla y ||_1
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
        
        d_x, d_y = lasp.thresholding.multidimensional_soft(
            d = numpy.array([ Dx(u)+b_x, Dy(u)+b_y ]),
            epsilon = 1/sigma
        )

        b_x=b_x+Dx(u)-d_x
        b_y=b_y+Dy(u)-d_y

    u_normalized = lasp.utils.normalize(u)

    return u_normalized


def mumford_shah_v2(
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

    # laplacian = lasp.filters.linear.laplacian()
    # lap_diag = lasp.utils.fourier_diagonalization(
    #     kernel = laplacian,
    #     shape_out = y.shape 
    # )
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
        
    
        d_x, d_y = lasp.thresholding.multidimensional_soft(
            d = numpy.array(
                [ 
                    Dx * u_fft + b_x, 
                    Dy * u_fft + b_y 
                ]
            ),
            epsilon = 1/sigma
        )

        b_x += Dx * u_fft - d_x
        b_y += Dy * u_fft - d_y

    u_normalized = lasp.utils.normalize(u)

    return u_normalized

def blockproc_reshape(
    array: numpy.ndarray,
    block_size: numpy.ndarray,
    order: str
) -> numpy.ndarray:

    nb_rows, nb_cols = array.shape
    res = None

    step_r, step_c = block_size

    range(0, nb_rows, step_r)

    for j in range(0, nb_cols, step_c):

        column = None
        
        for i in range(0, nb_rows, step_r):
            
            bloc = numpy.reshape(
                array[i:i+step_r, j:j+step_c], 
                (step_r*step_c, 1),
                order=order
            )
            if column is None:
                column = numpy.copy(bloc)
            else:
                column = numpy.vstack([ column, bloc ])

        if res is None:
            res = numpy.copy(column)
        else:
            res = numpy.hstack([ res, column ])

    return res

def blockproc(
    array: numpy.ndarray, 
    block_shape: numpy.ndarray,
    fun: typing.Callable[[numpy.ndarray], numpy.ndarray]
) -> numpy.ndarray:
    
    nb_rows, nb_cols = array.shape
    nb_block_rows, nb_block_cols = block_shape
    
    for i in range(0, nb_rows, nb_block_rows):
        for j in range(0, nb_cols, nb_block_cols):
            array[i:i+nb_block_rows, j:j+nb_block_cols] = \
                fun(array[i:i+nb_block_rows, j:j+nb_block_cols])
    
    return array
                
def block_mm(nr, nc, Nb, x1, order: str) -> numpy.ndarray:

    block_shape = numpy.array([nr, nc])

    x1 = blockproc_reshape(x1, block_shape, order)
    x1 = numpy.reshape(x1, newshape=(nr*nc, Nb), order=order)
    x1 = numpy.sum(x1, axis=1)
    x = numpy.reshape(x1, newshape=(nr, nc), order=order)

    return x


def mumford_shah_fsr(
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

    # laplacian = lasp.filters.linear.laplacian()
    # lap_diag = lasp.utils.fourier_diagonalization(
    #     kernel = laplacian,
    #     shape_out = y.shape 
    # )
    #######
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
    import PIL.Image
    d = 1_000_000
    u = numpy.array(
        PIL.Image.Image.resize(
            PIL.Image.fromarray(y),
            (y.shape[1]*d, y.shape[0]*d),
            PIL.Image.Resampling.BICUBIC
        )
    )
    # u = numpy.copy(y) 
    d_x=numpy.zeros_like(y)
    d_y=numpy.zeros_like(y)
    b_x=numpy.zeros_like(y)
    b_y=numpy.zeros_like(y)

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
        x1 = h_diag*rhsfft / h2_diag
        fbr = block_mm(y.shape[0], y.shape[1], d*d, x1)
        invW = block_mm(y.shape[0], y.shape[1], d*d, h2_diag / lap_diag)
        # invWBR = fbr / (invW + beta1*d*d)
        invWBR = fbr / (invW + d*d)
        fun = lambda block : block*invWBR
        FCBinvWBR = blockproc(numpy.conj(h2_diag), numpy.array([y.shape[0], y.shape[1]]), fun)
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
        
    
        d_x, d_y = lasp.thresholding.multidimensional_soft(
            d = numpy.array(
                [ 
                    Dx * u_fft + b_x, 
                    Dy * u_fft + b_y 
                ]
            ),
            epsilon = 1/sigma
        )

        b_x += Dx * u_fft - d_x
        b_y += Dy * u_fft - d_y

    u_normalized = lasp.utils.normalize(u)

    return u_normalized