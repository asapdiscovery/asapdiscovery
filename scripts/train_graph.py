"""
Attempt to train a ligand-only graph network using the same functions as the
structure-based models. Use a bunch of stuff from dgl-lifesci.
"""
import argparse
import dgl
import sys

## Import dgl-lifesci
sys.path.append(
    (
        "/lila/data/chodera/kaminowb/moonshot_ml_dev/dgl-lifesci/examples/"
        "property_prediction/csv_data_configuration/"
    )
)
from utils import get_configure, load_model, predict, init_featurizer


def init_dgl_args():
    dgl_args = {}

    # Choice of graph model to use
    # choices are ['GCN', 'GAT', 'Weave', 'MPNN', 'AttentiveFP', 'gin_supervised_contextpred', 'gin_supervised_infomax', 'gin_supervised_edgepred', 'gin_supervised_masking', 'NF']
    dgl_args["model"] = "GCN"

    # Define split ration for training,validation,test set partitioning of data
    dgl_args["split_ratio"] = "0.8,0.1,0.1"

    # Splitting strategy
    # choices are ['scaffold_decompose', 'scaffold_smiles', 'random']
    dgl_args["split"] = "random"

    # Metric for evaluation
    # choices are ['r2', 'mae', 'rmse']
    dgl_args["metric"] = "mae"

    # Atom featurizer type
    # choices are ['canonical', 'attentivefp']
    dgl_args["atom_featurizer_type"] = "canonical"

    # Bond featurizer type
    # choices are ['canonical', 'attentivefp']
    dgl_args["bond_featurizer_type"] = "canonical"

    # Number of epochs allowed for training
    dgl_args["num_epochs"] = 1000

    # Number of processes for data loading
    dgl_args["num_workers"] = 1

    # Path to save training results
    dgl_args["result_path"] = "regression_results"

    # Number of trials for hyperparameter search
    dgl_args["num_evals"] = None

    # Name of SMILES column
    dgl_args["smiles_column"] = "smiles"

    # Task names to learn
    dgl_args["task_names"] = ["pIC50"]

    return dgl_args


################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    ## Input arguments
    parser.add_argument(
        "-i", required=True, help="Input directory containing docked PDB files."
    )
    parser.add_argument(
        "-exp", required=True, help="JSON file giving experimental results."
    )

    ## Output arguments
    parser.add_argument("-model_o", help="Where to save model weights.")
    parser.add_argument("-plot_o", help="Where to save training loss plot.")
    parser.add_argument("-cache", help="Cache directory.")

    return parser.parse_args()


def main():
    args = get_args()
    dgl_args = init_dgl_args()

    ## Build dataframe for constructing dataset
    ## Get all docked structures
    all_fns = glob(f"{args.i}/*complex.pdb")
    ## Extract crystal structure and compound id from file name
    re_pat = (
        r"(Mpro-.*?_[0-9][A-Z]).*?([A-Z]{3}-[A-Z]{3}-[0-9a-z]{8}-[0-9]+)"
        "_complex.pdb"
    )
    matches = [re.search(re_pat, fn) for fn in all_fns]
    xtal_compounds = [m.groups() for m in matches if m]
    num_found = len(xtal_compounds)

    ## Load the experimental compounds
    exp_compounds = ExperimentalCompoundDataUpdate(
        **json.load(open(args.exp, "r"))
    ).compounds
    ## Filter out achiral molecules and molecules with no pIC50 values
    exp_compounds = [
        c
        for c in exp_compounds
        if (
            ((not achiral) or (c.achiral and achiral))
            and "pIC50" in c.experimental_data
        )
    ]

    ## Get compounds that have both structure and experimental data (this step
    ##  isn't actually necessary for performance, but allows a more fair
    ##  comparison between 2D and 3D models)
    xtal_compound_ids = {c[1] for c in xtal_compounds}

    ## Trim exp_compounds
    exp_compounds = [
        c for c in exp_compounds if c.compound_id in xtal_compound_ids
    ]

    ## Build dataframe
    all_compound_ids, all_smiles, all_pic50 = zip(
        *[
            (c.compound_id, c.smiles, c.experimental_data["pIC50"])
            for c in exp_compounds
        ]
    )
    df = pandas.DataFrame(
        {
            "compound_id": all_compound_ids,
            "smiles": all_smiles,
            "pic50": all_pic50,
        }
    )

    ## Initialize graph featurizer
    dgl_args = init_featurizer(dgl_args)

    ## Make cache directory as necessary
    if args.cache is None:
        cache_dir = f"{args.model_o}/.cache/"
    else:
        cache_dir = args.cache
    os.makedirs(cache_dir, exist_ok=True)

    ## Build dataset
    smiles_to_g = dgl.utils.SMILESToBigraph(add_self_loop=True,
        node_featurizer=dgl_args["node_featurizer"],
        edge_featurizer=dgl_args["edge_featurizer"])
    dataset =
