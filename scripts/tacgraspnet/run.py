import argparse

from models.tacgraspnet.tacgraspnet_config import TacGraspNetConfig
from data.dgs_dataset.dgs_dataset_config import DGSDatasetConfig
from train import train


if __name__ == "__main__":
    # Initialize model configuration
    model_config = TacGraspNetConfig()

    # Initialize data configuration
    data_config = DGSDatasetConfig()

    # Initialize argument parser for scripts
    arg_parser = argparse.ArgumentParser(
        prog="tacgraspnet_run",
        description="Train or evaluate TacGraspNet on DGS dataset",
    )

    # TODO
    # Meta paths
    # parser.add_argument(
    #     "-mp", "--model_path", help="Path to dump or load trained model at", type=Path
    # )
    # parser.add_argument(
    #     "--sim_output_path",
    #     help="Path to the folder `sim_output` (DefGraspSim results)",
    #     default="/workspace/data",
    #     type=Path,
    # )
    # parser.add_argument(
    #     "--sim_input_path",
    #     help="Path to the folder `sim_input` (DefGraspNets inputs)",
    #     default="/workspace/data",
    #     type=Path,
    # )

    # Running mode (training or evaluation)
    arg_parser.add_argument(
        "-m",
        "--mode",
        choices=[
            "training",
            "evaluation"
        ],
        required=True,
        help="Choose training or evaluation mode",
        default=model_config.mode,
    )

    ########################################
    ## Running
    ########################################
    # Data retrieval strategy for training
    arg_parser.add_argument(
        "-ds",
        "--data_strategy",
        choices=[
            # Train on single object with fixed grasping poses (trajectories), test on the others
            "single_obj",
            # Train on multiple objects with the same fixed grasping poses (trajectories), test on the others
            "multiple_objs_1",
            # Train on subset of objects with the same fixed grasping poses (trajectories),
            # test on the complement of both objects and grasping poses
            "multiple_objs_2"
        ],
        required=True,
        help="Choose data retrieval strategy",
        default=model_config.data_strategy,
    )

    # Which data to train on (also to evaluate on)
    arg_parser.add_argument(
        "-o",
        "--objs",
        type=str,
        nargs="+",
        help="List of objects to run on (training or evaluation). If single object data strategy is on, the first object in the list is chosen",
        default=model_config.objs,
    )

    # Which data to validate model during training
    arg_parser.add_argument(
        "-vo",
        "--validation_objs",
        type=str,
        nargs="+",
        help="Objects to validate the model during training. Only effective in training mode multiple_objs_2. If the list is empty, then random choosing is used",
        default=model_config.validation_objs,
    )

    # Validation size (note that applied for trajectories only)
    arg_parser.add_argument(
        "-vs",
        "--validation_ratio",
        type=float,
        help="Ratio (of trajectories) of the dataset is used during training. Only effective in training mode",
        default=model_config.validation_ratio,
    )

    # Batch size
    arg_parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        help="Batch size for training",
        default=model_config.batch_size,
    )

    # Number of epochs
    arg_parser.add_argument(
        "-ne",
        "--n_epochs",
        type=int,
        help="Number of training epochs",
        default=model_config.n_epochs,
    )

    ########################################
    ## Optimizer
    ########################################
    arg_parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        help="Learning rate",
        default=model_config.optimizer_params["lr"],
    )
    # arg_parser.add_argument(
    #     "--reduce_lr_on_plateau",
    #     action="store_true",
    #     help="Reduce learning rate when loss does not decline after several epochs",
    #)

    ########################################
    ## TacGraspNet
    ########################################
    # Important flags
    arg_parser.add_argument(
        "-td",
        "--use-template-data",
        type=bool,
        help="Indicate whether template data (e.g. vertice positions, ...) are used instead of first frame data or not",
        default=model_config.use_template_data,
    )
    arg_parser.add_argument(
        "-fln",
        "--use-final-layer-norm",
        type=bool,
        help="Indicate whether layer normalization is used for all MLP final layers (except decoder) or not",
        default=model_config.use_final_layer_norm,
    )
    arg_parser.add_argument(
        "-nf",
        "--normalize_features",
        type=bool,
        help="Indicate whether node and edge features are normalized or not",
        default=model_config.normalize_features,
    )
    arg_parser.add_argument(
        "-no",
        "--normalize_outputs",
        type=bool,
        help="Indicate whether the predicted outputs are normalized or not (for loss calculation). \
              It is also used to indicate whether the layer normalization is used for decoder final layer or not",
        default=model_config.normalize_outputs,
    )
    arg_parser.add_argument(
        "-ntsd",
        "--use_node_tetra_separate_decoders",
        type=bool,
        help="Indicate whether we use two separate or only one (combining) decoder for node and tetrahedral features",
        default=model_config.use_node_tetra_separate_decoders,
    )
    arg_parser.add_argument(
        "-sem",
        "--use_separate_edge_mlps",
        type=bool,
        help="Indicate whether we use separate MLPs for different edge types in message passing or not",
        default=model_config.use_separate_edge_mlps,
    )
    arg_parser.add_argument(
        "-mpsm",
        "--use_message_passing_separate_mlps",
        type=bool,
        help="Indicate whether each message passing step has its own set of MLPs or not",
        default=model_config.use_message_passing_separate_mlps,
    )
    arg_parser.add_argument(
        "-tib",
        "--use_translation_inductive_bias",
        type=bool,
        help="Indicate whether translation inductive bias is used for training and evaluation or not",
        default=model_config.use_translation_inductive_bias,
    )

    # Modeling configuration
    arg_parser.add_argument(
        "-nhl",
        "--n_hidden_layers",
        type=int,
        help="Number of hidden layers of all MLPs (e.g. encoders, decoders, updating MLPs, ...) in TacGraspNet",
        default=model_config.n_hidden_layers,
    )
    arg_parser.add_argument(
        "-ld",
        "--latent_dim",
        type=int,
        help="Latent dimension of all features (e.g. node, edge, tetrahedron, ...) to encode them into",
        default=model_config.latent_dim,
    )
    arg_parser.add_argument(
        "-mps",
        "--message_passing_steps",
        type=int,
        help="Number of message passing steps (conducted by GraphNetBlock)",
        default=model_config.message_passing_steps,
    )
    arg_parser.add_argument(
        "-r",
        "--radius",
        type=float,
        help="Radius of the ball (neighborhood) for construction of contact edges (world edges)",
        default=model_config.radius,
    )

    # Argument parsing
    args = arg_parser.parse_args()

    # if args.seed is None:
    #     args.seed = np.random.randint(99999)
    # if args.model_path is None:
    #     args.model_path = (
    #         "/lustre/fswork/projects/rech/tya/ubn15wo/Tactile_Danylo/torchgraspnet/data/runs/"
    #         + datetime.now().strftime("%b%d-%H:%M:%S.%f")
    #         + "-"
    #         + str(args.seed)
    #         + "-"
    #         + args.mode
    #         + "-"
    #         + str(args.tet_stress)
    #         + "-"
    #         + str(args.move_gripper)
    #     )
    #
    # args_path = Path(args.model_path) / "args.pth"
    # # If args dump is present, load these args
    # if args_path.is_file():
    #     # Save epochs to continue training from finished training checkpoint
    #     wished_epochs = args.epochs
    #     args = torch.load(args_path)
    #     args.epochs = wished_epochs

    # experiment_fn = {
    #     "train_single_frame": training.train_single_frame,
    #     "train_single_traj": training.train_single_traj,
    #     "train_single_obj": training.train_single_obj,
    #     "train": training.train,
    # }
    model_config.update(args)
    if args.mode == "training":
        train(model_config)
    else:
        pass
