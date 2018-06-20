import glob
import sys

    
def read_raw():
    f = open('quarterly-data/2005-3')
    lines = f.readlines()
    lines.pop(0)
    f.close()
    new_file = 0
        
    for line in lines :
        individual_rating = line.split(",")
        userid = individual_rating[1]
        movieid = individual_rating[0]
        rating = individual_rating[2]
        if(not new_file):    
            new_file = open('quarterly-data/temp', 'w')
        new_file.write(str(movieid)+','+str(userid)+','+str(rating)+'\n')

read_raw()    
