import numpy as np
from dataset.syn_data.create_syn_data import SynData
import matplotlib.pyplot as plt

seed = 1
n_g_group = 3
d_g = 5
d_z = 3
d_u = 1
d_x = 5

data = SynData(seed = seed, n_g_group = n_g_group, d_g = d_g, d_z = d_z, d_u = d_u, d_x = d_x)

Tmax = 50
arcoef_trans = 0.8
nsample = 1000
v_type = "hard"
data.sample(nsample = nsample, v_type = v_type, arcoef_trans=arcoef_trans, Tmax = Tmax)

# data.x
# data.z
# data.v
# data.s

z_seq = {i:[] for i in range(n_g_group)}
x_seq = {i:[] for i in range(n_g_group)}
g_seq = {i:[] for i in range(n_g_group)}

for i in range(data.nsample):
    ggroup = np.argmax(data.v[i])
    z_seq[ggroup].append(data.z_cat[i])
    x_seq[ggroup].append(data.x[i])
    g_seq[ggroup].append(data.g[i])

z_plot = []
x_plot = []
g_plot = []
x_flatten = []

z_plot_std = []
x_plot_std = []
for i in range(n_g_group):
    z_seq_ggroup = np.mean(np.array(z_seq[i]), axis=0)
    x_seq_ggroup = np.mean(np.array(x_seq[i]), axis=0)
    g_seq_ggroup = np.array(g_seq[i])

    z_plot.append(z_seq_ggroup)
    x_plot.append(x_seq_ggroup)
    g_plot.append(g_seq_ggroup)

    x_flatten.append(np.array(x_seq[i]).reshape(-1,x_seq[i][0].shape[-1]))

    z_plot_std.append(np.std(np.array(z_seq[i]), axis=0))
    x_plot_std.append(np.std(np.array(x_seq[i]), axis=0))

x_flatten = np.array(np.array(x_plot), (-1, x_plot[0].shape[-1]))

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure()
ticks = ["G{}".format(i) for i in range(d_g)]
bp1 = plt.boxplot(g_plot[0], positions=np.array(range(d_g))*3.0-0.6, sym='', widths=0.4)
bp2 = plt.boxplot(g_plot[1], positions=np.array(range(d_g))*3.0, sym='', widths=0.4)
bp3 = plt.boxplot(g_plot[2], positions=np.array(range(d_g))*3.0+0.6, sym='', widths=0.4)
set_box_color(bp1, 'blue') # colors are from http://colorbrewer2.org/
set_box_color(bp2, 'red')
set_box_color(bp3, 'green')
plt.plot([], c='blue', label='Group 1')
plt.plot([], c='red', label='Group 2')
plt.plot([], c='green', label='Group 3')
plt.xticks(range(0, len(ticks) * 3, 3), ticks)
plt.legend()
plt.ylabel("Values of G")
plt.title("The distribution of G in different genetic groups")
plt.tight_layout()
plt.show()
plt.savefig('Boxcompare_G.png')



plt.figure()
ticks = ["X{}".format(i) for i in range(d_x)]
bp1 = plt.boxplot(x_flatten[0], positions=np.array(range(d_x))*3.0-0.6, sym='', widths=0.4)
bp2 = plt.boxplot(x_flatten[1], positions=np.array(range(d_x))*3.0, sym='', widths=0.4)
bp3 = plt.boxplot(x_flatten[2], positions=np.array(range(d_x))*3.0+0.6, sym='', widths=0.4)
set_box_color(bp1, 'blue') # colors are from http://colorbrewer2.org/
set_box_color(bp2, 'red')
set_box_color(bp3, 'green')
plt.plot([], c='blue', label='Group 1')
plt.plot([], c='red', label='Group 2')
plt.plot([], c='green', label='Group 3')
plt.xticks(range(0, len(ticks) * 3, 3), ticks)
plt.legend()
plt.ylabel("Values of X")
plt.title("The distribution of X in different genetic groups")
plt.tight_layout()
plt.show()




z_dim = 0
plt.figure()
plt.plot(z_plot[0][:,z_dim], c='blue', label='Group 1')
plt.fill_between([i for i in range(Tmax)], z_plot[0][:,z_dim]-z_plot_std[0][:,z_dim],
                 z_plot[0][:,z_dim]+z_plot_std[0][:,z_dim], color='blue', alpha=0.25 )
plt.plot(z_plot[1][:,z_dim], c='red', label='Group 2')
plt.fill_between([i for i in range(Tmax)], z_plot[1][:,z_dim]-z_plot_std[1][:,z_dim],
                 z_plot[1][:,z_dim]+z_plot_std[1][:,z_dim], color='red', alpha=0.25 )
plt.plot(z_plot[2][:,z_dim], c='green', label='Group 3')
plt.fill_between([i for i in range(Tmax)], z_plot[2][:,z_dim]-z_plot_std[2][:,z_dim],
                 z_plot[2][:,z_dim]+z_plot_std[2][:,z_dim], color='green', alpha=0.25 )
plt.title("The mean value of Z[0] in different genetic groups,\n shaded region is 1 std range")
plt.xlabel("T")
plt.legend()
# plt.show()
plt.savefig('z.png')

x_dim = 0
plt.figure()
plt.plot(x_plot[0][:,x_dim], c='blue', label='Group 1')
plt.fill_between([i for i in range(Tmax)], x_plot[0][:,x_dim]-x_plot_std[0][:,x_dim],
                 x_plot[0][:,x_dim]+x_plot_std[0][:,x_dim], color='blue', alpha=0.25 )
plt.plot(x_plot[1][:,x_dim], c='red', label='Group 2')
plt.fill_between([i for i in range(Tmax)], x_plot[1][:,x_dim]-x_plot_std[1][:,x_dim],
                 x_plot[1][:,x_dim]+x_plot_std[1][:,x_dim], color='red', alpha=0.25 )
plt.plot(x_plot[2][:,x_dim], c='green', label='Group 3')
plt.fill_between([i for i in range(Tmax)], x_plot[2][:,x_dim]-x_plot_std[2][:,x_dim],
                 x_plot[2][:,x_dim]+x_plot_std[2][:,x_dim], color='green', alpha=0.25 )
plt.title("The mean value of X[2] in different genetic groups,\n shaded region is 1 std range")
plt.xlabel("T")
plt.legend()
# plt.show()
plt.savefig('x.png')