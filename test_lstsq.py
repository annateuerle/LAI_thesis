import numpy as np

x = np.array([0, 1, 2, 3]*30)
y = np.array([-1, 0.2, 0.9, 2.1]*30)
x= x.ravel()
y= y.ravel()

A = np.vstack([x, np.ones(len(x))]).T
print (np.linalg.__doc__)
lin_answer= np.linalg.lstsq(A, y, rcond= -1)
print (lin_answer)
m, c= lin_answer[0]
print (m, c)
y_pred= c+m*x
rmse= 0
for iCnt in range(x.size):
    print (iCnt, x[iCnt], y[iCnt], y_pred[iCnt])
    rmse+= (y-y_pred)**2
rmse= (rmse.mean())**0.5
print (rmse)