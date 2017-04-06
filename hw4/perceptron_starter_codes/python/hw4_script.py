import os
import csv
import numpy as np
import math

data_dir = os.path.join('..','data')
XTrain = np.genfromtxt(os.path.join(data_dir, 'XTrain.csv'), delimiter=',')
print XTrain[1,:]
