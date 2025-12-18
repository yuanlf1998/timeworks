import matplotlib.pyplot as plt
import numpy as np 


def plot_x_and_y(x, y, ax=None):
    # If no axes provided, create a new figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        show_plot = True
    else:
        show_plot = False

    L = len(x)

    # 蓝色：原样
    ax.plot(range(L), x, color='blue', label='x')

    # 红色：在 y 前面插入 x[-1]
    y_ext = np.concatenate([[x[-1]], y])
    y_x = range(L - 1, L - 1 + len(y_ext))
    ax.plot(y_x, y_ext, color='red', label='y')
    
    ax.axvline(
        L - 1,
        color='black',
        linestyle='-.',
        linewidth=2.0,
        alpha=0.8,
        label='obs → pred'
    )

    # Add legend and grid
    ax.legend()
    ax.grid()

    # Show plot only if we created the figure
    if show_plot:
        plt.show()

    return ax

