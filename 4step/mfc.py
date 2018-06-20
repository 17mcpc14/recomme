import numpy as np
import glob
from os.path import basename
from sklearn.cluster import KMeans
import sys

def read_mf_mat(quarter):
    print('Processing: '+quarter)
    
    with open("./mf/"+quarter) as f:
        R = np.loadtxt(f)
        print(R.shape)
        f.close()
        nP = KMeans(n_clusters=30, random_state=0).fit_transform(R.T).T
        print('Completed matrix clustering', nP.shape)
        np.savetxt('mfc/'+quarter, nP, fmt='%.3f')
    
    print('Completed processing: '+quarter)

if(len(sys.argv) >0):
    read_mf_mat(sys.argv[1])
else:
    for my_file in sorted(glob.glob("./mf/*")):
        read_mf_mat(basename(my_file))
