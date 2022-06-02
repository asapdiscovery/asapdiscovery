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