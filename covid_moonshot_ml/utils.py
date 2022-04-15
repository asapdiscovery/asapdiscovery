from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas
import pickle as pkl
import re
import seaborn as sns
import torch
from torch_cluster import radius_graph

def calc_e3nn_model_info(ds, r):
    num_neighbors = []
    num_nodes = []
    unique_atom_types = set()
    for _, pose in ds:
        edge_src, edge_dst = radius_graph(x=pose['pos'], r=r)
        num_neighbors.extend(Counter(edge_src.numpy()).values())
        num_nodes.append(pose['pos'].shape[0])
        unique_atom_types.update(pose['z'].tolist())

    return(len(unique_atom_types), np.mean(num_neighbors),
        round(np.mean(num_nodes)))

def evaluate(model, ds_train, ds_test, target_dict, model_base, device,
    model_call=lambda model, d: model(d), plot_o=None, pkl_o=None):
    ## Set up loss function
    loss_fn = torch.nn.MSELoss()

    ## Get models and throw an error if no models found
    models = find_all_models(model_base)
    if len(models) == 0:
        raise ValueError(f'No models found in {model_base}')
    if type(models[0]) is int:
        if os.path.isdir(model_base):
            model_fns = [f'{model_base}/{m}.th' for m in models]
        else:
            model_fns = [model_base.format(m) for m in models]
    else:
        model_fns = models[:]
        models = list(range(len(model_fns)))

    ## Evaluate the models
    train_loss = []
    test_loss = []
    with torch.no_grad():
        for wts_fn in model_fns:
            print(wts_fn, flush=True)
            ## Load model weights
            model.load_state_dict(torch.load(wts_fn))
            ## Send model to desired device if it's not there already
            model.to(device)

            ## Evaluate training loss
            tmp_loss = []
            for s, pose in ds_train:
                for k, v in pose.items():
                    pose[k] = v.to(device)
                pred = model_call(model, pose)
                for k, v in pose.items():
                    pose[k] = v.to('cpu')
                # convert to float to match other types
                target = torch.tensor([[target_dict[s]]], device=device).float()
                loss = loss_fn(pred, target)
                tmp_loss.append(loss.item())
            train_loss.append(np.asarray(tmp_loss))
            print(f'Training error: {np.mean(tmp_loss):0.5f}', flush=True)

            tmp_loss = []
            for s, pose in ds_test:
                for k, v in pose.items():
                    pose[k] = v.to(device)
                pred = model_call(model, pose)
                for k, v in pose.items():
                    pose[k] = v.to('cpu')
                # convert to float to match other types
                target = torch.tensor([[target_dict[s]]], device=device).float()
                loss = loss_fn(pred, target)
                tmp_loss.append(loss.item())
            test_loss.append(np.asarray(tmp_loss))
            print(f'Test error: {np.mean(tmp_loss):0.5f}', flush=True)

            if plot_o is not None:
                print('Plotting', flush=True)
                plot_loss(np.mean(train_loss, axis=1),
                    np.mean(test_loss, axis=1), plot_o)
            if pkl_o is not None:
                print('Saving', flush=True)
                pkl.dump([models, np.vstack(train_loss), np.vstack(test_loss)],
                    open(pkl_o, 'wb'))

    return(models, np.vstack(train_loss), np.vstack(test_loss))

def find_all_models(model_base):
    if model_base is None:
        return([])
    elif os.path.isdir(model_base):
        models = [int(fn.split('.')[0]) for fn in os.listdir(model_base) \
            if re.match(r'[0-9]+\.th', fn)]
    elif '{}' in model_base:
        re_match = re.sub(r'{}', r'([0-9]+)', os.path.basename(model_base))
        models = [re.match(re_match, fn) \
            for fn in os.listdir(os.path.dirname(model_base))]
        models = [int(m.group(1)) for m in models if m is not None]
    elif os.path.isfile(model_base):
        return([model_base])
    else:
        return([])

    return(sorted(models))

def find_most_recent(model_wts):
    if model_wts is None:
        return(None)
    elif os.path.isdir(model_wts):
        models = find_all_models(model_wts)
    elif '{}' in model_wts:
        models = find_all_models(model_wts)
        model_wts = os.path.dirname(model_wts)
    elif os.path.isfile(model_wts):
        return(0, model_wts)
    else:
        return(None)

    epoch_use = models[-1]
    return(epoch_use, f'{model_wts}/{epoch_use}.th')

def one_hot_from_atom_types(atom_types, all_atom_types=None):
    ## First get all atom types in a usable format
    ## Just take all unique atom types in atom_types if nothing is passed
    if all_atom_types is None:
        all_atom_types = np.unique(atom_types)
    else:
        all_atom_types = np.asarray(all_atom_types)

    ## Construct map from atom type to new label
    at_map = dict(zip(all_atom_types, np.arange(len(all_atom_types))))
    ## Loop through each atom's type and assign it a new label (needs to be
    ##  0-indexed and consecutive for torch one-hot)
    at_labels = [at_map[at.item()] for at in atom_types]
    return(torch.nn.functional.one_hot(torch.tensor(at_labels),
        len(all_atom_types)))

def plot_loss(train_loss, test_loss, out_fn):
    fig, axes = plt.subplots(nrows=2, figsize=(12,8), sharex=True)
    sns.lineplot(x=range(len(train_loss)), y=train_loss, ax=axes[0])
    sns.lineplot(x=range(len(test_loss)), y=test_loss, ax=axes[1])

    for (ax, loss_type) in zip(axes, ('Training', 'Test')):
        ax.set_ylabel(f'MSE {loss_type} Loss')
        ax.set_xlabel('Epoch')
        ax.set_title(f'MSE {loss_type} Loss')

    fig.savefig(out_fn, dpi=200, bbox_inches='tight')

def train(model, ds_train, ds_test, target_dict, n_epochs, device,
    model_call=lambda model, d: model(d), save_file=None, lr=1e-4,
    start_epoch=0, train_loss=[], test_loss=[]):
    ## Send model to desired device if it's not there already
    model.to(device)

    ## Set up optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr)
    loss_fn = torch.nn.MSELoss()

    ## Train for n epochs
    for epoch_idx in range(start_epoch, n_epochs):
        print(f'Epoch {epoch_idx}/{n_epochs}', flush=True)
        if epoch_idx % 10 == 0 and epoch_idx > 0:
            print(f'Training error: {np.mean(train_loss[-1]):0.5f}')
            print(f'Testing error: {np.mean(test_loss[-1]):0.5f}', flush=True)
        tmp_loss = []
        for (_, compound_id), pose in ds_train:
            optimizer.zero_grad()
            for k, v in pose.items():
                try:
                    pose[k] = v.to(device)
                except AttributeError:
                    pass
            pred = model_call(model, pose)
            for k, v in pose.items():
                try:
                    pose[k] = v.to('cpu')
                except AttributeError:
                    pass
            # convert to float to match other types
            target = torch.tensor([[target_dict[compound_id]]],
                device=device).float()
            loss = loss_fn(pred, target)
            tmp_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        train_loss.append(np.asarray(tmp_loss))


        with torch.no_grad():
            tmp_loss = []
            for (_, compound_id), pose in ds_test:
                for k, v in pose.items():
                    try:
                        pose[k] = v.to(device)
                    except AttributeError:
                        pass
                pred = model_call(model, pose)
                for k, v in pose.items():
                    try:
                        pose[k] = v.to('cpu')
                    except AttributeError:
                        pass
                # convert to float to match other types
                target = torch.tensor([[target_dict[compound_id]]],
                    device=device).float()
                loss = loss_fn(pred, target)
                tmp_loss.append(loss.item())
            test_loss.append(np.asarray(tmp_loss))

        if save_file is None:
            continue
        elif os.path.isdir(save_file):
            torch.save(model.state_dict(), f'{save_file}/{epoch_idx}.th')
            pkl.dump(np.vstack(train_loss),
                open(f'{save_file}/train_err.pkl', 'wb'))
            pkl.dump(np.vstack(test_loss),
                open(f'{save_file}/test_err.pkl', 'wb'))
        elif '{}' in save_file:
            torch.save(model.state_dict(), save_file.format(epoch_idx))
        else:
            torch.save(model.state_dict(), save_file)

    return(model, np.vstack(train_loss), np.vstack(test_loss))