from typing import List

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from commons.datatype import Datapoint


def visualize_data_point(
        datapoint: Datapoint,
        # There are four options: (0) Current frame, (1) Template, (2) First frame and (3) Second frame.
        # If input option differs from 0 -> 3, then default value 0 is set. One can input a list of options
        # for multiple plots in one figure
        opts: List[int] | int = 0,
):
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(projection='3d')

    opts = [opts] if isinstance(opts, int) else opts
    for opt in opts:
        if opt == 1:
            x = datapoint["template.vertices.positions"][..., 0]
            y = datapoint["template.vertices.positions"][..., 1]
            z = datapoint["template.vertices.positions"][..., 2]
        elif opt == 2:
            x = datapoint["1st_frame.vertices.positions"][..., 0]
            y = datapoint["1st_frame.vertices.positions"][..., 1]
            z = datapoint["1st_frame.vertices.positions"][..., 2]
        elif opt == 3:
            x = datapoint["2nd_frame.vertices.positions"][..., 0]
            y = datapoint["2nd_frame.vertices.positions"][..., 1]
            z = datapoint["2nd_frame.vertices.positions"][..., 2]
        else:
            x = datapoint["vertices.positions"][..., 0]
            y = datapoint["vertices.positions"][..., 1]
            z = datapoint["vertices.positions"][..., 2]

        ax.scatter(x, y, z, c=datapoint["nodes.types"], cmap="rainbow")

    ax.axis('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=30, azim=45)
    plt.show()
