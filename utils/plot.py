from matplotlib import pyplot as plt
import matplotlib.lines as mlines

def draw_line(p1, p2, color='r', segment=True):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if (p2[0] == p1[0]):
        if segment:
            xmin = xmax = p1[0]
            ymin, ymax = p1[1], p2[1]
        else:
            xmin = xmax = p1[0]
            ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xmax - p1[0])
        ymin = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xmin - p1[0])

    l = mlines.Line2D([xmin, xmax], [ymin, ymax], color=color)
    ax.add_line(l)
    return l

if __name__ == "__main__":
    draw_line([1,2], [2,4])
    plt.show()