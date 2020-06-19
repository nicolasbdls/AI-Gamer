###############################################################################
#In this algorithm the Q-functions are directly approximated from the Q-tables#
#A plot shows how the Q-values are displayed and also the corresponding       #
#polynomial approximation.                                                   #
###############################################################################


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from matplotlib import cm
from numpy import linalg
x=np.linspace(0,20,20)
y=np.linspace(0,20,20)

fig = plt.figure(figsize=(20, 20))

for i in range(8990, 9000, 10):
    print(i)
    ax1 = fig.add_subplot(211, projection='3d')
    ax2 = fig.add_subplot(212, projection='3d')
    q_table = np.load(f"qtables/{i}-qtable.npy")
    a,b,c = np.dsplit(q_table, 3)
    z1=np.amin(a)
    z2=np.amin(b)
    z3=np.amin(c)
    z4=np.amax(a)
    z5=np.amax(b)
    z6=np.amax(c)
    a.shape=(20, 20)
    b.shape=(20, 20)
    c.shape=(20, 20)
    ind1 = np.unravel_index(np.argmin(a, axis=None), a.shape)
    ind2 = np.unravel_index(np.argmin(b, axis=None), b.shape)
    ind3 = np.unravel_index(np.argmin(c, axis=None), c.shape)
    ind4 = np.unravel_index(np.argmax(a, axis=None), a.shape)
    ind5 = np.unravel_index(np.argmax(b, axis=None), b.shape)
    ind6 = np.unravel_index(np.argmax(c, axis=None), c.shape)

    X, Y = np.meshgrid(x, y, copy=False)
    X=X.flatten()
    Y=Y.flatten()
    A = np.array([X*0+1, X, Y, X**2, X**2*Y, X**2*Y**2, Y**2, X*Y**2, X*Y]).T
    B=a.flatten()
    C=b.flatten()
    D=c.flatten()
    d, r, rank, s = np.linalg.lstsq(A, B, rcond=None)
    e, r, rank, s = np.linalg.lstsq(A, C, rcond=None)
    f, r, rank, s = np.linalg.lstsq(A, D, rcond=None)

    x1=np.linspace(0,20,20)
    y1=np.linspace(0,20,20)
    X1, Y1= np.meshgrid(x1, y1)

    z = d[0] + X1*d[1] + Y1*d[2] + X1**2*d[3] + X1**2*Y1*d[4] + X1**2*Y1**2*d[5] + Y1**2*d[6] + X1*Y1**2*d[7] + X1*Y1*d[8],
    ax1.scatter(X, Y, a, color='r')
    ax2.scatter(X1, Y1, z, color='r', marker='x')

    ax1.set_title("Q values after 900 episodes")
    ax2.set_title("approx poly : %f + %fx + %fy + %fx^2 + %fx^2y + %fx^2y^2 + %fy^2 + %fxy^2 + %fxy" %(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8]))

    plt.show()
