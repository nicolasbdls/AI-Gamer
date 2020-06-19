########################################################################################################
#This algorithm allows to visualize with ation has an optimal Q-value for each state (shown in green)  #
#It uses Q-tables produced by the q-learning algorithm. By looking at which action has the best q-value#
#it is possible to retrace the path of the agent in the maze                                           # 
########################################################################################################


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')



def get_q_color(value, vals):
    if value == max(vals):
        return "green", 1.0
    else:
        return "red", 1.0


fig = plt.figure(figsize=(5, 12))


ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)
i = 100
q_table = np.load(f"qtablesmaze/{i}-qtable.npy")


for x, x_vals in enumerate(q_table):
    for y, y_vals in enumerate(x_vals):
        ax1.scatter(x, -y, c=get_q_color(y_vals[0], y_vals), marker=",")
        ax1.set_yticklabels([0, 2, 4, 6, 8, 10])
        ax1.set_xticklabels([0, 1, 3, 5, 7, 9])
        ax2.scatter(x, -y, c=get_q_color(y_vals[1], y_vals), marker=",")
        ax2.set_yticklabels([0, 2, 4, 6, 8, 10])
        ax2.set_xticklabels([0, 1, 3, 5, 7, 9])
        ax3.scatter(x, -y, c=get_q_color(y_vals[2], y_vals), marker=",")
        ax3.set_yticklabels([0, 2, 4, 6, 8, 10])
        ax3.set_xticklabels([0, 1, 3, 5, 7, 9])
        ax4.scatter(x, -y, c=get_q_color(y_vals[3], y_vals), marker=",")
        ax4.set_yticklabels([0, 2, 4, 6, 8, 10])
        ax4.set_xticklabels([0, 1, 3, 5, 7, 9])

        ax1.set_ylabel("Action N")
        ax2.set_ylabel("Action S")
        ax3.set_ylabel("Action E")
        ax4.set_ylabel("Action W")

plt.grid(True)
plt.show()