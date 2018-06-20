import numpy as np
import glob
from os.path import basename
from pycuda import driver, compiler, gpuarray, tools

# -- initialize the device
import pycuda.autoinit

kernel_code_template = """
__global__ void MatrixMulKernel(float *a, float *b, float *c)
{
    // 2D Thread ID (assuming that only *one* block will be executed)
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Pvalue is used to store the element of the matrix
    // that is computed by the thread
    float Pvalue = 0;

    // Each thread loads one row of M and one column of N, 
    //   to produce one element of P.
    for (int k = 0; k < %(MATRIX_SIZE)s; ++k) {
        float Aelement = a[ty * %(MATRIX_SIZE)s + k];
        float Belement = b[k * %(MATRIX_SIZE)s + tx];
        Pvalue += Aelement * Belement;
    }

    // Write the matrix to device memory;
    // each thread writes one element
    c[ty * %(MATRIX_SIZE)s + tx] = Pvalue;
}
"""

# define the (square) matrix size
#  note that we'll only use *one* block of threads here
#  as a consequence this number (squared) can't exceed max_threads,
#  see http://documen.tician.de/pycuda/util.html#pycuda.tools.DeviceData
#  for more information on how to get this number for your device
def dotp(a_cpu,b_cpu):

    # transfer host (CPU) memory to device (GPU) memory 
    a_gpu = gpuarray.to_gpu(a_cpu) 
    b_gpu = gpuarray.to_gpu(b_cpu)

    MATRIX_SIZE = len(a_cpu)
    # create empty gpu array for the result (C = A * B)
    c_gpu = gpuarray.empty((1, 1), np.float32)

    # get the kernel code from the template 
    # by specifying the constant MATRIX_SIZE
    kernel_code = kernel_code_template % {
            'MATRIX_SIZE': MATRIX_SIZE 
            }

    # compile the kernel code 
    mod = compiler.SourceModule(kernel_code)

    # get the kernel function from the compiled module
    matrixmul = mod.get_function("MatrixMulKernel")

    # call the kernel on the card
    matrixmul(
            # inputs
            a_gpu, b_gpu, 
            # output
            c_gpu, 
            # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
            block = (MATRIX_SIZE, MATRIX_SIZE, 1),
            )

    return c_gpu.get()

"""
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension M x K
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
"""
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - dotp(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - dotp(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break
    return P, Q.T

def read_quarter_mat(quarter):
    print('Processing: '+quarter)
    
    with open("./monthly-mat/"+quarter) as f:
        R = np.loadtxt(f)
        f.close()
 
    print('Initiate  matrix factorization...')
        
    R = np.array(R)
    N,M = R.shape
    K = 30
    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)
    nP, nQ = matrix_factorization(R, P, Q, K)

    print('Completed matrix factorization', nP.shape, nQ.shape)
 
    np.savetxt('monthly-mf/'+quarter+'-u', nP, fmt='%.3f')
    np.savetxt('monthly-mf/'+quarter+'-v', nQ, fmt='%.3f')
    print('Completed processing: '+quarter)

for my_file in sorted(glob.glob("./monthly-mat/*")):
    if (my_file != '1999-12'):
        read_quarter_mat(basename(my_file))
