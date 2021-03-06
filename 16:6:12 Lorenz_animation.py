# Animation of the Lorenz model

import numpy as np
from scipy import integrate

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation

N_trajectories = 2 #labeled 1 and 2 accordingly

# Modify initial conditions here
x1_init = 1 
y1_init = 1.0
z1_init = 1.05

x2_init = 1.5
y2_init = 1.0
z2_init = 1.05


# Modify Parameter values in the following line
def lorentz_deriv((x, y, z), t0, sigma=10., beta=8./3, rho=14.0):
    # Compute the time-derivatives of a Lorentz system
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


# initial values put into an array
x0 = np.array([[x1_init, y1_init, z1_init], [x2_init, y2_init, z2_init]])

# Solve for the trajectories
t = np.linspace(0, 75, 6000)
x_t = np.asarray([integrate.odeint(lorentz_deriv, x0i, t)
                  for x0i in x0])

# Set up figure & 3D axis for animation
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
ax.axis('off')

# choose a different color for each trajectory
colors = plt.cm.jet(np.linspace(0, 1, N_trajectories))

# set up lines and points
lines = sum([ax.plot([], [], [], '-', c=c)
             for c in colors], [])
pts = sum([ax.plot([], [], [], 'o', c=c)
           for c in colors], [])

# prepare the axes limits
ax.set_xlim((-25, 25))
ax.set_ylim((-35, 35))
ax.set_zlim((5, 55))

# set point-of-view: specified by (altitude degrees, azimuth degrees)
ax.view_init(30, 0)

# initialization function: plot the background of each frame
def init():
    for line, pt in zip(lines, pts):
        line.set_data([], [])
        line.set_3d_properties([])

        pt.set_data([], [])
        pt.set_3d_properties([])
    return lines + pts

# animation function.  This will be called sequentially with the frame number
def animate(i):
    # we'll step two time-steps per frame.  This leads to nice results.
    i = (2 * i) % x_t.shape[1]

    for line, pt, xi in zip(lines, pts, x_t):
        x, y, z = xi[:i].T
        line.set_data(x, y)
        line.set_3d_properties(z)

        pt.set_data(x[-1:], y[-1:])
        pt.set_3d_properties(z[-1:])

    ax.view_init(30, 0.3 * i)
    fig.canvas.draw()
    return lines + pts

# instantiate the animator.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=1200, interval=30)


plt.show()

## Code adapted from Jake VanderPlas
## https://jakevdp.github.io/blog/2013/02/16/animating-the-lorentz-system-in-3d/