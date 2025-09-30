import numpy as np

array = [[1,2,3],[4,5,6],[7,8,9]]
array = np.array(array)
print((np.shape(array)))

array = array.reshape(-1,3)
print(array)
print((np.shape(array)))