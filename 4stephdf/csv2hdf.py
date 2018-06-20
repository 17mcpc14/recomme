import numpy as np
import pandas as pd
import glob
from os.path import basename
import h5py
import sys

for my_file in sorted(glob.glob("./monthly-mat/"+sys.argv[1]+"*")):
    f = open(my_file, 'r')
    R = np.loadtxt(f)
    f.close()

    file = h5py.File("./monthly-hd5/"+basename(my_file), 'w')
    file['dataset'] = R
    file.close()
