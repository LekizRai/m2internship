import torch

from typing import List

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from commons.datatype import (
    Datapoint,
    NodeType
)


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
            c = datapoint["nodes.types"]
        elif opt == 2:
            x = datapoint["1st_frame.vertices.positions"][..., 0]
            y = datapoint["1st_frame.vertices.positions"][..., 1]
            z = datapoint["1st_frame.vertices.positions"][..., 2]
            c = datapoint["nodes.types"]
        elif opt == 3:
            x = datapoint["2nd_frame.vertices.positions"][..., 0]
            y = datapoint["2nd_frame.vertices.positions"][..., 1]
            z = datapoint["2nd_frame.vertices.positions"][..., 2]
            c = datapoint["nodes.types"]
        else:
            # Set indices
            node_types = datapoint["nodes.types"]
            n_obj_nodes = int((node_types == NodeType.OBJECT).sum().item())
            n_ts_nodes = len(node_types) - n_obj_nodes
            n_ts_comp_nodes = n_ts_nodes // 2
            left_ts_node_idx = list(range(n_ts_comp_nodes))
            right_ts_node_idx = list(range(n_ts_comp_nodes, n_ts_comp_nodes + n_ts_comp_nodes))
            obj_node_idx = list(range(n_ts_comp_nodes + n_ts_comp_nodes, n_ts_comp_nodes + n_ts_comp_nodes + n_obj_nodes))
            idx = right_ts_node_idx + obj_node_idx

            # Compute coordinates and colors
            x = datapoint["vertices.positions"][idx, 0]
            y = datapoint["vertices.positions"][idx, 1]
            z = datapoint["vertices.positions"][idx, 2]
            c = node_types[idx]

            # Compute gripper closing direction (left or right)
            arrow_root = torch.mean(datapoint["vertices.positions"][right_ts_node_idx], dim=-2)
            arrow_dir = datapoint["gripper.closing_directions"][0][1]
            ax.quiver(
                arrow_root[0].item(), arrow_root[1].item(), arrow_root[2].item(),
                arrow_dir[0].item(), arrow_dir[1].item(), arrow_dir[2].item(),
                color="red",
                arrow_length_ratio=0.2
            )

        ax.scatter(x, y, z, c=c, cmap="rainbow")

    ax.axis('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=30, azim=45)
    plt.show()
