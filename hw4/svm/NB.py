import math
import numpy as np

# The logProd function takes a vector of numbers in logspace 
# (i.e., x[i] = log p[i]) and returns the product of those numbers in logspace
# (i.e., logProd(x) = log(product_i p[i]))

import run_svm as rs


(X,Y) = rs.getdat("hw4data1.mat")

C = 4


model = rs.run_svm(X, Y, C)