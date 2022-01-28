import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import jit

#code is very unoptimized pls no bulli

@jit
def optimum(input_arr,G):
    """The original Optimum Thoery CA. Takes a 2D array of values and returns
    the next iteration. G is the divisor."""

    bottom = np.roll(input_arr,-1,axis=0)
    top = np.roll(input_arr,1,axis=0)
    right = np.roll(input_arr,-1,axis=1)
    left = np.roll(input_arr,1,axis=1)
    upper_left = np.roll(left,1,axis=0)
    upper_right = np.roll(right,1,axis=0)
    bottom_left = np.roll(left,-1,axis=0)
    bottom_right = np.roll(right,-1,axis=0)

    neumann_sum = bottom + top + right + left
    symm_diff_neumann_moore_sum = (upper_left+upper_right+bottom_left+bottom_right)/2
    rand_offset = np.random.randint(-1000,1000,size=input_arr.shape)
    delta = input_arr+rand_offset

    return((neumann_sum + symm_diff_neumann_moore_sum + input_arr + delta)/G)

@jit
def optimum_quiet(input_arr,G):
    """A quiet version without the 'quantum foam'."""

    bottom = np.roll(input_arr,-1,axis=0)
    top = np.roll(input_arr,1,axis=0)
    right = np.roll(input_arr,-1,axis=1)
    left = np.roll(input_arr,1,axis=1)
    upper_left = np.roll(left,1,axis=0)
    upper_right = np.roll(right,1,axis=0)
    bottom_left = np.roll(left,-1,axis=0)
    bottom_right = np.roll(right,-1,axis=0)

    neumann_sum = bottom + top + right + left
    symm_diff_neumann_moore_sum = (upper_left+upper_right+bottom_left+bottom_right)/2

    return((neumann_sum + symm_diff_neumann_moore_sum + input_arr)/G)

@jit
def optimum_wave(input_arr,change,G):
    """The 'wave equation' version. Note that this one also takes the 'change
    map' as an argument."""

    bottom = np.roll(input_arr,-1,axis=0)
    top = np.roll(input_arr,1,axis=0)
    right = np.roll(input_arr,-1,axis=1)
    left = np.roll(input_arr,1,axis=1)

    neumann_sum = bottom + top + right + left

    return(change + (neumann_sum-4*input_arr)/G)

@jit
def optimum_3D(input_arr,G):
    """3D version. Takes a 3D input array and iterates it (memory intensive,
    can easily be improved)."""

    forward = np.roll(input_arr,-1,axis=0)
    backward = np.roll(input_arr,1,axis=0)
    bottom = np.roll(input_arr,-1,axis=1)
    top = np.roll(input_arr,1,axis=1)
    right = np.roll(input_arr,-1,axis=2)
    left = np.roll(input_arr,1,axis=2)

    c1 = np.roll(forward,1,axis=1)
    c2 = np.roll(forward,-1,axis=1)
    c3 = np.roll(forward,1,axis=2)
    c4 = np.roll(forward,-1,axis=2)
    c5 = np.roll(backward,1,axis=1)
    c6 = np.roll(backward,-1,axis=1)
    c7 = np.roll(backward,1,axis=2)
    c8 = np.roll(backward,-1,axis=2)
    c9 = np.roll(top,1,axis=2)
    c10 = np.roll(top,-1,axis=2)
    c11 = np.roll(bottom,1,axis=2)
    c12 = np.roll(bottom,-1,axis=2)

    e1 = np.roll(c3,-1,axis=1)
    e2 = np.roll(c3,1,axis=1)
    e3 = np.roll(c4,-1,axis=1)
    e4 = np.roll(c4,1,axis=1)
    e5 = np.roll(c7,-1,axis=1)
    e6 = np.roll(c7,1,axis=1)
    e7 = np.roll(c8,-1,axis=1)
    e8 = np.roll(c8,1,axis=1)

    msum = forward+backward+bottom+top+right+left
    csum = (c1+c2+c3+c4+c5+c6+c7+c8+c9+c10+c11+c12)/2
    esum = (e1+e2+e3+e4+e5+e6+e7+e8)/3

    rand_offset = np.random.randint(-1000,1000,size=input_arr.shape)
    delta = input_arr+rand_offset

    return((msum + csum + esum + delta)/G)

@jit
def vec_map(arr):
    """Takes a 2D array and returns the (absolute) 'vector map'."""

    y_s,x_s = arr.shape
    h = np.zeros_like(arr)
    for y in range(y_s):
        for x in range(x_s):
            vals = np.zeros(8)
            vals[0] = np.abs(arr[(y+1)%y_s][(x-1)%x_s])
            vals[1] = np.abs(arr[(y+1)%y_s][(x)%x_s])
            vals[2] = np.abs(arr[(y+1)%y_s][(x+1)%x_s])
            vals[3] = np.abs(arr[(y)%y_s][(x+1)%x_s])
            vals[4] = np.abs(arr[(y)%y_s][(x-1)%x_s])
            vals[5] = np.abs(arr[(y-1)%y_s][(x-1)%x_s])
            vals[6] = np.abs(arr[(y-1)%y_s][(x)%x_s])
            vals[7] = np.abs(arr[(y-1)%y_s][(x+1)%x_s])
            mi = np.argmax(np.abs(vals))
            h[y][x] = mi
    return h


#EXAMPLE

#define main grid
grid = np.zeros((128,128))
prev_grid = np.copy(grid)
change = np.zeros_like(grid)

#matplotlib setup
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)

#iterates the main grid when called. Needs to be its own function for mpl animation.
def animate(i):
    global grid
    global prev_grid
    global change
    ax.clear()
    prev_grid = np.copy(grid)
    grid = optimum(grid,7.125)
    change = grid-prev_grid
    ax.imshow(vec_map(change))

#animate
anim = animation.FuncAnimation(fig,animate,interval=10)
plt.show()

# #save the first 512 frames as images (use this INSTEAD of FuncAnimation & plt.show())
# for i in range(512):
#     animate(0)
#     plt.savefig('frame_%05d' % i)
