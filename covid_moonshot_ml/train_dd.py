import argparse
from e3nn import o3
from e3nn.nn.models.gate_points_2101 import Network
from glob import glob
import json
import os
import pickle as pkl
import re
import torch
from torch_geometric.nn import SchNet
from torch_geometric.datasets import QM9

from DockedDataset import DockedDataset
from E3NNBind import E3NNBind
from schema import ExperimentalCompoundDataUpdate
from SchNetBind import SchNetBind
from utils import calc_e3nn_model_info, find_most_recent, train, plot_loss

def add_one_hot_encodings(ds):
    ## Add one hot encodings to each entry in ds
    for _, pose in ds:
        ## Use length 100 for one-hot encoding to account for atoms up to element
        ##  number 100
        pose['x'] = torch.nn.functional.one_hot(pose['z']-1, 100).float()

    return(ds)

def add_lig_labels(ds):
    ## Change key values for ligand labels
    for _, pose in ds:
        pose['z'] = pose['lig'].reshape((-1,1)).float()

    return(ds)

def load_affinities(fn, achiral=True):
    ## Load all compounds with experimental data and filter to only achiral
    ##  molecules (to start)
    exp_compounds = ExperimentalCompoundDataUpdate(
        **json.load(open(fn, 'r'))).compounds
    exp_compounds = [c for c in exp_compounds if c.achiral==achiral]

    affinity_dict = {c.compound_id: c.experimental_data['pIC50'] \
        for c in exp_compounds if 'pIC50' in c.experimental_data}

    return(affinity_dict)

def build_model_e3nn(n_atom_types, num_neighbors, num_nodes, node_attr=False,
    dg=False):
    # input is one-hot encoding of atom type => n_atom_types scalars
    # output is scalar valued binding energy/pIC50 value
    # hidden layers taken from e3nn tutorial (should be tuned eventually)
    # same with edge attribute irreps (and all hyperparameters)
    # need to calculate num_neighbors and num_nodes
    # reduce_output because we just want the one binding energy prediction
    #  across the whole graph

    model_kwargs = {
        'irreps_in': f'{n_atom_types}x0e',
        'irreps_hidden': [(mul, (l, p)) \
            for l, mul in enumerate([10,3,2,1]) for p in [-1, 1]],
        'irreps_out': '1x0e',
        'irreps_node_attr': '1x0e' if node_attr else None,
        'irreps_edge_attr': o3.Irreps.spherical_harmonics(3),
        'layers': 3,
        'max_radius': 3.5,
        'number_of_basis': 10,
        'radial_layers': 1,
        'radial_neurons': 128,
        'num_neighbors': num_neighbors,
        'num_nodes': num_nodes,
        'reduce_output': True
    }

    if dg:
        model = E3NNBind(**model_kwargs)
    else:
        model = Network(**model_kwargs)
    return(model)

def build_model_schnet(qm9=None, dg=False, qm9_target=10):
    ## Load pretrained model if requested, otherwise create a new SchNet
    if qm9 is None:
        if dg:
            model = SchNetBind()
        else:
            model = SchNet()
    else:
        qm9_dataset = QM9(qm9)

        # target=10 is free energy (eV)
        model_qm9, _ = SchNet.from_qm9_pretrained(qm9, qm9_dataset, qm9_target)

        if dg:
            model = SchNetBind(model_qm9.hidden_channels, model_qm9.num_filters,
                model_qm9.num_interactions, model_qm9.num_gaussians,
                model_qm9.cutoff, model_qm9.max_num_neighbors, model_qm9.readout,
                model_qm9.dipole, model_qm9.mean, model_qm9.std,
                model_qm9.atomref.weight.detach().clone())
            model.load_state_dict(model_qm9.state_dict())
        else:
            model = model_qm9

    ## Set interatomic cutoff to 3.5A (default of 10) to make the graph smaller
    model.cutoff = 3.5

    return(model)

################################################################################
def get_args():
    parser = argparse.ArgumentParser(description='')

    ## Input arguments
    parser.add_argument('-i', required=True,
        help='Input directory containing docked PDB files.')
    parser.add_argument('-exp', required=True,
        help='JSON file giving experimental results.')
    parser.add_argument('-model_params', help='e3nn model parameters.')
    parser.add_argument('-qm9', help='QM9 directory for pretrained model.')
    parser.add_argument('-qm9_target', type=int, default=10,
        help='QM9 pretrained target.')
    parser.add_argument('-cont', action='store_true',
        help='Whether to restore training with most recent model weights.')

    ## Output arguments
    parser.add_argument('-model_o', help='Where to save model weights.')
    parser.add_argument('-plot_o', help='Where to save training loss plot.')

    ## Model parameters
    parser.add_argument('-model', required=True,
        help='Which type of model to use (e3nn or schnet).')
    parser.add_argument('-lig', action='store_true',
        help='Whether to treat the ligand and protein atoms separately.')
    parser.add_argument('-dg', action='store_true',
        help='Whether to predict pIC50 directly or via dG prediction.')

    ## Training arguments
    parser.add_argument('-n_epochs', type=int, default=1000,
        help='Number of epochs to train for (defaults to 1000).')
    parser.add_argument('-device', default='cuda',
        help='Device to use for training (defaults to GPU).')
    parser.add_argument('-lr', type=float, default=1e-4,
        help='Learning rate for Adam optimizer (defaults to 1e-4).')

    return(parser.parse_args())

def init(args, rank=False):
    """
    Initialization steps that are common to all analyses.
    """

    ## Get all docked structures
    all_fns = glob(f'{args.i}/*complex.pdb')
    ## Extract crystal structure and compound id from file name
    re_pat = r'(Mpro-P[0-9]{4}_0[AB]).*?([A-Z]{3}-[A-Z]{3}-.*?)_complex.pdb'
    compounds = [re.search(re_pat, fn).groups() for fn in all_fns]

    if rank:
        exp_affinities = None
    else:
        ## Load the experimental affinities
        exp_affinities = load_affinities(args.exp)

        ## Trim docked structures and filenames to remove compounds that don't have
        ##  experimental data
        all_fns, compounds = zip(*[o for o in zip(all_fns, compounds) \
            if o[1][1] in exp_affinities])

    ## Load the dataset
    ds = DockedDataset(all_fns, compounds)

    ## Split dataset into train/test (80/20 split)
    n_train = int(len(ds) * 0.8)
    n_test = len(ds) - n_train
    print(f'{n_train} training samples, {n_test} testing samples', flush=True)
    # use fixed seed for reproducibility
    ds_train, ds_test = torch.utils.data.random_split(ds, [n_train, n_test],
        torch.Generator().manual_seed(42))

    ## Build the model
    if args.model == 'e3nn':
        ## Need to add one-hot encodings to the dataset
        ds_train = add_one_hot_encodings(ds_train)
        ds_test = add_one_hot_encodings(ds_test)

        ## Load or calculate model parameters
        if args.model_params is None:
            model_params = calc_e3nn_model_info(ds_train, 3.5)
        elif os.path.isfile(args.model_params):
            model_params = pkl.load(open(args.model_params, 'rb'))
        else:
            model_params = calc_e3nn_model_info(ds_train, 3.5)
            pkl.dump(model_params, open(args.model_params, 'wb'))
        model = build_model_e3nn(100, *model_params[1:], node_attr=args.lig,
            dg=args.dg)
        model_call = lambda model, d: model(d)

        ## Add lig labels as node attributes if requested
        if args.lig:
            ds_train = add_lig_labels(ds_train)
            ds_test = add_lig_labels(ds_test)

        for k,v in ds_train[0][1].items():
            print(k, v.shape, flush=True)
    elif args.model == 'schnet':
        model = build_model_schnet(args.qm9, args.dg, args.qm9_target)
        if args.dg:
            model_call = lambda model, d: model(d['z'], d['pos'], d['lig'])
        else:
            model_call = lambda model, d: model(d['z'], d['pos'])
    else:
        raise ValueError(f'Unknown model type {args.model}.')

    return(exp_affinities, ds_train, ds_test, model, model_call)

def main():
    args = get_args()
    exp_affinities, ds_train, ds_test, model, model_call = init(args)

    ## Load model weights as necessary
    if args.cont:
        start_epoch, wts_fn = find_most_recent(args.model_o)
        model.load_state_dict(torch.load(wts_fn))

        ## Load error dicts
        if os.path.isfile(f'{args.model_o}/train_err.pkl'):
            train_loss = pkl.load(open(f'{args.model_o}/train_err.pkl',
                'rb')).tolist()
        else:
            train_loss = []
        if os.path.isfile(f'{args.model_o}/test_err.pkl'):
            test_loss = pkl.load(open(f'{args.model_o}/test_err.pkl',
                'rb')).tolist()
        else:
            test_loss = []

        ## Need to add 1 to start_epoch bc the found idx is the last epoch
        ##  successfully trained, not the one we want to start at
        start_epoch += 1
    else:
        start_epoch = 0
        train_loss = []
        test_loss = []

    ## Train the model
    model, train_loss, test_loss = train(model, ds_train, ds_test,
        exp_affinities, args.n_epochs, torch.device(args.device),
        model_call, args.model_o, args.lr, start_epoch, train_loss, test_loss)

    ## Plot loss
    if args.plot_o is not None:
        plot_loss(train_loss.mean(axis=1), test_loss.mean(axis=1), args.plot_o)

if __name__ == '__main__':
    main()
