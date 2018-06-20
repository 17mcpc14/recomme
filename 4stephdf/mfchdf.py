import numpy as np
import glob
from os.path import basename
from sklearn.cluster import KMeans
import sys
import h5py

def read_mf_mat(quarter):
    print('Processing: '+quarter)
    
    file = h5py.File("./mf/"+quarter, 'r')

    P = file['W']
    print(P.shape)
    nP = KMeans(n_clusters=30, random_state=0).fit_transform(P.T).T
    print('Completed matrix clustering', nP.shape)

    Q = file['H']
    print(Q.shape)
    nQ = KMeans(n_clusters=30, random_state=0).fit_transform(Q.T).T
    print('Completed matrix clustering' nQ.shape)

    nR = nP.T*nQ
    
    file2 = h5py.File("./mfc/"+quarter, 'w')
    file2['clustermat'] = nR
        
    print('Completed processing: '+quarter)

if(len(sys.argv) >0):
    read_mf_mat(sys.argv[1])
else:
    for my_file in sorted(glob.glob("./mf/*")):
        read_mf_mat(basename(my_file))
