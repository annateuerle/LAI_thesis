""""
Description of the solver function from the numpy
"""


import numpy as np

x = np.array([0, 1, 2, 3]*30)
y = np.array([-1, 0.2, 0.9, 2.1]*30)
x= x.ravel()
y= y.ravel()

A = np.vstack([x, np.ones(len(x))]).T
print(A)
print(y)
# print (np.linalg.__doc__)
lin_answer = np.linalg.lstsq(A, y, rcond=None)
print (lin_answer)
m, c = lin_answer[0]
print(f'm:{m} , c{c}')
y_pred = c+m*x
print(y_pred)

rmse = 0

for iCnt in range(x.size):
    print (iCnt, x[iCnt], y[iCnt], y_pred[iCnt])
    rmse+= (y-y_pred)**2

rmse= (rmse.mean())**0.5
print (rmse)