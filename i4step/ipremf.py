import numpy as np
import glob
from os.path import basename
import datetime

quarter_map = {}

for my_file in sorted(glob.glob("../monthly-data/*")):
    if (basename(my_file) != "1999-11" and basename(my_file) != '1999-12' and basename(my_file) != '2006-1'):
        dt = datetime.datetime.strptime(basename(my_file), '%Y-%m') + datetime.timedelta(days=1)

        quarter = str(dt.year*100 + dt.month )
        quarter_map[quarter] = my_file
        
users = []
movies = []
P = np.matrix([[0,0],[0,0]]) 

def build_matrix(f):
    lines = f.readlines()
    lines.pop(0)
    f.close()

    global P
    
    print("shape of P:",p.shape)

    for line in lines :
        line = line.rstrip()

        individual_rating = line.split(",")
        movie_id = int(individual_rating[0])
        user_id = int(individual_rating[1])
        if(user_id not in users):
            ux =  bisect_left( sorted(users), user_id, 0 , len(users))
            users.append(user_id)
            P = np.insert(P, ux, 0, axis=1)      
        if(movie_id not in movies):
            ix =  bisect_left( sorted(movies), movie_id, 0 , len(movies))
            movies.append(movie_id)
	    P = np.insert(P, ix, 0, axis=0)   
        
        P[movies.index(movie_id), users.index(user_id)] =  rating

    print("shape of P:",p.shape)

    return P

for i in sorted(quarter_map.keys()):
    print(i)
    build_matrix(open(quarter_map[i]))
    
    np.savetxt('../monthly-mat2/'+i, P, fmt='%.3f')

