#import matplotlib.pyplot as plt
import numpy as np
from LoadData import load_rating_data, spilt_rating_dat
from sklearn.model_selection import train_test_split
from ProbabilisticMatrixFactorization import PMF
import glob
from os.path import exists, basename
import sys

if __name__ == "__main__":
    
    for file_path in sorted(glob.glob("./monthly-data-bkp/monthly-data/"+sys.argv[1]+"*")):
        print(file_path)
        if( not exists("./monthly-pmf/"+basename(file_path)+"_u")):
            print("Processing ...:"+basename(file_path));

            pmf = PMF()
            pmf.set_params({"num_feat": 30, "epsilon": 1, "_lambda": 0.1, "momentum": 0.8, "maxepoch": 30, "num_batches": 100,
                    "batch_size": 1000})
            ratings = load_rating_data(file_path)
            print(len(np.unique(ratings[:, 0])), len(np.unique(ratings[:, 1])), pmf.num_feat)
            train, test = train_test_split(ratings, test_size=0.0)  # spilt_rating_dat(ratings)
            pmf.fit(train, test, basename(file_path))

    # Check performance by plotting train and test errors
    #plt.plot(range(pmf.maxepoch), pmf.rmse_train, marker='o', label='Training Data')
    #plt.plot(range(pmf.maxepoch), pmf.rmse_test, marker='v', label='Test Data')
    #plt.title('The MovieLens Dataset Learning Curve')
    #plt.xlabel('Number of Epochs')
    #plt.ylabel('RMSE')
    #plt.legend()
    #plt.grid()
    #plt.show()
    #print("precision_acc,recall_acc:" + str(pmf.topK(test)))

