import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Reshape
import csv
from keras.layers import Dropout

def read_mmf(u):
    X = []
    Y = []

    with open('./xy.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            strX = row['X']
            strY = row['Y']
            print(strX, strY)
            with open("./monthly-pmf/"+strX+"_u") as myfile:    
                tempX = np.loadtxt(myfile)
                X.append(tempX)
                myfile.close()
        
            with open("./monthly-pmf/"+strY+"_u") as myfile2:    
                tempY = np.loadtxt(myfile2)
                Y.append(tempY)
                myfile2.close()

    X = np.array(X)
    Y = np.array(Y)   
    
    return train_test_split(X, Y, test_size = 0.2)


def train_predict(X_train, X_test, y_train, y_test, u):
    classifier = Sequential()

    # Adding the input layer and the first hidden layer
    classifier.add(Reshape((10,10), input_shape=(10, 10)))
    # Adding the second hidden layer
    classifier.add(Reshape((10,10)))
    # Adding the second hidden layer
    classifier.add(Reshape((10,10)))
    classifier.add(Dropout(0.5))
    # Adding the output layer
    classifier.add(Reshape((10,10)))

    # Compiling Neural Network
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Fitting our model 
    classifier.fit(X_train, y_train, batch_size = 1, nb_epoch = 100)

    score = classifier.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    with open("./monthly-pmf/2005-12_u") as myfile:    
        tempX = np.loadtxt(myfile)
        tempY = classifier.predict(tempX)
        print(tempY)    
        np.savetxt('monthly-pmf/2006-1_u', tempY, fmt='%0.2f')
    
X_train, X_test, y_train, y_test = read_mmf('u')
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
train_predict(X_train, X_test, y_train, y_test, 'u')

X_train, X_test, y_train, y_test = read_mmf('v')
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
train_predict(X_train, X_test, y_train, y_test, 'v')
