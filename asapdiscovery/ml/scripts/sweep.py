"""
Script for a Weights & Biases hyperparameter sweep.
"""


def build_model_2d(config=None):
    """
    Build appropriate 2D graph model.

    Parameters
    ----------
    config : Union[str, dict], optional
        Either a dict or JSON file with model config options. If not passed,
        `config` will be taken from `wandb`.

    Returns
    -------
    asapdiscovery.ml.models.GAT
        GAT graph model
    """

    if type(config) is str:
        config = json.load(open(config_fn))
    elif config is None:
        config = wandb.config
    elif type(config) != dict:
        raise ValueError(f"Unknown type of config: {type(config)}")

    config.update({"in_node_feats": CanonicalAtomFeaturizer().feat_size()})

    model = GAT(
        in_feats=config["in_node_feats"],
        hidden_feats=[config["gnn_hidden_feats"]] * config["num_gnn_layers"],
        num_heads=[config["num_heads"]] * config["num_gnn_layers"],
        feat_drops=[config["dropout"]] * config["num_gnn_layers"],
        attn_drops=[config["dropout"]] * config["num_gnn_layers"],
        alphas=[config["alpha"]] * config["num_gnn_layers"],
        residuals=[config["residual"]] * config["num_gnn_layers"],
    )

    return model, config


################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    ## Input arguments
    parser.add_argument(
        "-i", required=True, help="Input directory/glob for docked PDB files."
    )
    parser.add_argument(
        "-exp", required=True, help="JSON file giving experimental results."
    )
    parser.add_argument("-model_params", help="e3nn model parameters.")
    parser.add_argument("-qm9", help="QM9 directory for pretrained model.")
    parser.add_argument(
        "-qm9_target", type=int, default=10, help="QM9 pretrained target."
    )
    parser.add_argument(
        "-cont",
        action="store_true",
        help="Whether to restore training with most recent model weights.",
    )
    parser.add_argument(
        "-achiral", action="store_true", help="Keep only achiral molecules."
    )
    parser.add_argument("-n", default="LIG", help="Ligand residue name.")
    parser.add_argument(
        "-w",
        type=int,
        default=1,
        help="Number of workers to use for dataset loading.",
    )

    ## Output arguments
    parser.add_argument("-model_o", help="Where to save model weights.")
    parser.add_argument("-cache", help="Cache directory for dataset.")

    ## Model parameters
    parser.add_argument(
        "-model",
        required=True,
        help="Which type of model to use (e3nn or schnet).",
    )
    parser.add_argument(
        "-lig",
        action="store_true",
        help="Whether to add e3nn node attributes for ligand atoms.",
    )
    parser.add_argument(
        "-rm_atomref",
        action="store_true",
        help="Remove atomref embedding in QM9 pretrained SchNet.",
    )
    parser.add_argument(
        "-n_dist",
        type=float,
        default=5.0,
        help="Cutoff distance for node neighbors.",
    )
    parser.add_argument("-irr", help="Hidden irreps for e3nn model.")
    parser.add_argument(
        "-config", help="Model config JSON file for graph 2D model."
    )

    ## Training arguments
    parser.add_argument(
        "-n_epochs",
        type=int,
        default=1000,
        help="Number of epochs to train for (defaults to 1000).",
    )
    parser.add_argument(
        "-device",
        default="cuda",
        help="Device to use for training (defaults to GPU).",
    )
    parser.add_argument(
        "--grouped",
        action="store_true",
        help="Group poses for the same compound into one prediction.",
    )

    ## WandB arguments
    parser.add_argument("-proj", help="WandB project name.")
    parser.add_argument("-name", help="WandB run name.")
    parser.add_argument(
        "-e",
        "--extra_config",
        nargs="+",
        help=(
            "Any extra config options to log to WandB. Can provide any "
            "number of comma-separated key-value pairs "
            "(eg --extra_config key1,val1 key2,val2 key3,val3)."
        ),
    )

    return parser.parse_args()

def main():
    args = get_args()

    ds = build_dataset(args)