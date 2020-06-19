########################################################################################################
#This algorithm allows to visualize with ation has an optimal Q-value for each state (shown in green)  #
#It uses Q-tables produced by the q-learning algorithm.                                                # 
########################################################################################################


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')



def get_q_color(value, vals):
    if value == max(vals):
        return "green"
    else:
        return "red"


fig = plt.figure(figsize=(5, 12))


ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
i = 9000
q_table = np.load(f"qtables/{i}-qtable.npy")


for x, x_vals in enumerate(q_table):
    for y, y_vals in enumerate(x_vals):
        ax1.scatter(x, y, c=get_q_color(y_vals[0], y_vals), marker=",")
        ax2.scatter(x, y, c=get_q_color(y_vals[1], y_vals), marker=",")
        ax3.scatter(x, y, c=get_q_color(y_vals[2], y_vals), marker=",")

        ax1.set_ylabel("Action Left")
        ax2.set_ylabel("Action Nothing")
        ax3.set_ylabel("Action Right")

plt.grid(True)
plt.show()