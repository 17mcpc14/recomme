import datetime
import glob
import sys

quarter_map = {}
    
def read_raw(my_file):
    print("Processing : "+my_file)
    f = open(my_file)
    lines = f.readlines()
    lines.pop(0)
    f.close()
    movie_id = 0
        
    for line in lines :
        temp = line
        line = line.rstrip()
        
        if( line.endswith(':')):
            individual_movie = line.split(":")
            movie_id = int(individual_movie[0])
        else:
            individual_rating = line.split(",")
            dt = datetime.datetime.strptime(individual_rating[2], '%Y-%m-%d') + datetime.timedelta(days=1)
            quarter = str(dt.year) + '-' + str( dt.month ) 
            new_file = quarter_map.get(quarter)
            if(not new_file):    
                new_file = open('monthly-data/'+quarter, 'w')
                quarter_map[quarter] = new_file
            new_file.write(str(movie_id)+','+individual_rating[0]+','+individual_rating[1]+"\n")
    print("Processing completed for file :"+my_file);

read_raw('netflix-prize-data/combined_data_1.txt')
read_raw('netflix-prize-data/combined_data_2.txt')
read_raw('netflix-prize-data/combined_data_3.txt')
read_raw('netflix-prize-data/combined_data_4.txt')

for i in sorted(quarter_map.keys()):
    quarter_map[i].close()
