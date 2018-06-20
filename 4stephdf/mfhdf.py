import h5py
import numpy as np
import glob
from os.path import basename
from pycuda import driver, compiler, gpuarray, tools
import sys

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

    # create empty gpu array for the result (C = A * B)
    c_gpu = gpuarray.empty((1, 1), np.float32)

    MATRIX_SIZE = len(a_cpu)
    
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

def matrix_factorization(R, P, Q, K, steps=50, alpha=0.002, beta=0.02):
    for step in range(steps):
        print("In step of matrix factorization : ", step)
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
                    e = e + pow(R[i][j] - dotp(P[i,:],Q[:,j]) ,  2)
                    for k in range(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break
    return P, Q

def read_quarter_mat(quarter):
    print('Processing: '+quarter)
    
    print('Initiate  matrix factorization...')
    
    file = h5py.File("./monthly-hd5/"+quarter, 'r')

    file2 = h5py.File("./monthly-mf/"+quarter, 'w') 
    D = file['dataset']
 
    print(D.shape)

    file2['W'] = np.random.random((len(D),20)) 
    file2['H'] = np.random.random((20,len(D))) 
    P = file2['W'] 
    Q = file2['H'] 
    P, Q = matrix_factorization(D, P, Q, 20)
    print(P.shape, Q.shape)

    file2['W'] = P
    file2['H'] = Q
    file.close()
    file2.close()

    print('Completed matrix factorization')
 
    print('Completed processing: '+quarter)

for my_file in sorted(glob.glob("./monthly-hd5/"+sys.argv[1]+"*")):
    read_quarter_mat(basename(my_file))
