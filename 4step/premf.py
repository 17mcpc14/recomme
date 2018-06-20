import numpy as np
import glob
from os.path import basename
from sklearn.cluster import KMeans
import sys

def build_matrix(f):
    lines = f.readlines()
    lines.pop(0)
    f.close()
    users = []
    movies = []
    
    for line in lines :
        line = line.rstrip()
        
        individual_rating = line.split(",")
        movie_id = int(individual_rating[0])
        user_id = int(individual_rating[1])
        if(user_id not in users):
            users.append(user_id)
        if(movie_id not in movies):
            movies.append(movie_id)
     
    P = np.zeros((len(movies), len(users) ))
    
    for line in lines :
        line = line.rstrip()
        
        individual_rating = line.split(",")
        movie_id = int(individual_rating[0])
        user_id = int(individual_rating[1])
        rating = int(individual_rating[2])
    
        P[movies.index(movie_id), users.index(user_id)] =  rating

    return P          

def read_quarter_data(quarter):
    with open("./monthly-data/"+quarter) as f:
        
        print('Processing: '+quarter)
        R = build_matrix(f)
        print('Completed building matrix')
        
        np.savetxt('monthly-mat/'+quarter, R , fmt='%.3f')
        print('Completed processing: '+quarter)

for my_file in sorted(glob.glob("./monthly-data/2004*")):
   if(my_file != '2004-1' and my_file != '2004-10'):
       read_quarter_data(basename(my_file))
