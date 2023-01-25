import numpy as np
import os


def calc_e3nn_model_info(ds, r):
    """
    Calculate parameters to use in creation of an e3nn model.

    Parameters
    ----------
    ds : data.dataset.DockedDataset
        Dataset of structures to use to calculate the parameters
    r : float
        Cutoff to use for neighbor atom calculations

    Returns
    -------
    int
        Number of unique atom types found in `ds`
    float
        Average number of neighbors for each node across all of `ds`
    int
        Rounded average number of nodes per structure in `ds`
    """
    from collections import Counter
    from torch_cluster import radius_graph

    num_neighbors = []
    num_nodes = []
    unique_atom_types = set()
    for _, pose in ds:
        edge_src, edge_dst = radius_graph(x=pose["pos"], r=r)
        num_neighbors.extend(Counter(edge_src.numpy()).values())
        num_nodes.append(pose["pos"].shape[0])
        unique_atom_types.update(pose["z"].tolist())

    return (
        len(unique_atom_types),
        np.mean(num_neighbors),
        round(np.mean(num_nodes)),
    )


def find_all_models(model_base):
    """
    Helper script to find all existing models in the given directory/file base
    name. If the given path is a directory, assume the files are in the format
    {epoch_index}.th. If the given path contains {}, assume that is a
    placeholder for the epoch index. If the given path is an existing file,
    just return that file.
    TODO : Rework this function to make the output more consistent

    Parameters
    ----------
    model_base : str
        Where to look for trained models

    Returns
    -------
    list[str/int]
        Returns sorted list of found epoch indexes if a directory or filename
        with placeholder is given, otherwise returns the given weights file.
    """
    import re

    if model_base is None:
        return []
    elif os.path.isdir(model_base):
        models = [
            int(fn.split(".")[0])
            for fn in os.listdir(model_base)
            if re.match(r"[0-9]+\.th", fn)
        ]
    elif "{}" in model_base:
        re_match = re.sub(r"{}", r"([0-9]+)", os.path.basename(model_base))
        models = [
            re.match(re_match, fn)
            for fn in os.listdir(os.path.dirname(model_base))
        ]
        models = [int(m.group(1)) for m in models if m is not None]
    elif os.path.isfile(model_base):
        return [model_base]
    else:
        return []

    return sorted(models)


def find_most_recent(model_wts):
    """
    Helper script to find the most recent of all found model weight files and
    determine the last epoch trained.

    Parameters
    ----------
    model_wts : str
        Where to look for trained models

    Returns
    -------
    int
        Which epoch of training the weights being used are from
    str
        Path to model weights file
    """
    if model_wts is None:
        return None
    elif os.path.isdir(model_wts):
        models = find_all_models(model_wts)
    elif "{}" in model_wts:
        models = find_all_models(model_wts)
        model_wts = os.path.dirname(model_wts)
    elif os.path.isfile(model_wts):
        return (0, model_wts)
    else:
        return None

    epoch_use = models[-1]
    return (epoch_use, f"{model_wts}/{epoch_use}.th")


def load_weights(model, wts_fn):
    """
    Load weights for an MTENN model, initializing internal layers as necessary.

    Parameters
    ----------
    model: mtenn.Model
        Model to load weights into
    wts_fn: str
        Weights file to load from

    Returns
    -------
    mtenn.Model
        Model with loaded weights
    """
    import torch

    ## Load weights
    try:
        wts_dict = torch.load(wts_fn)
    except RuntimeError:
        wts_dict = torch.load(wts_fn, map_location="cpu")

    ## Initialize linear module in ConcatStrategy
    if "strategy.reduce_nn.weight" in wts_dict:
        model.strategy.reduce_nn = torch.nn.Linear(
            wts_dict["strategy.reduce_nn.weight"].shape[1],
            wts_dict["strategy.reduce_nn.weight"].shape[0],
        )

    loaded_params = set(wts_dict.keys())
    model_params = set(model.state_dict().keys())
    print("extra parameters:", loaded_params - model_params)
    print("missing parameters:", model_params - loaded_params)

    ## Get rid of extra params
    for p in loaded_params - model_params:
        del wts_dict[p]

    ## Load model parameters
    model.load_state_dict(wts_dict)
    print(f"Loaded model weights from {wts_fn}", flush=True)

    return model


def plot_loss(train_loss, val_loss, test_loss, out_fn):
    """
    Plot loss for train, val, and test sets.

    Parameters
    ----------
    train_loss : numpy.ndarray
        Loss at each epoch for train set
    val_loss : numpy.ndarray
        Loss at each epoch for validation set
    test_loss : numpy.ndarray
        Loss at each epoch for test set
    out_fn : str
        Path to save plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(nrows=3, figsize=(12, 8), sharex=True)
    sns.lineplot(x=range(len(train_loss)), y=train_loss, ax=axes[0])
    sns.lineplot(x=range(len(val_loss)), y=val_loss, ax=axes[1])
    sns.lineplot(x=range(len(test_loss)), y=test_loss, ax=axes[2])

    for (ax, loss_type) in zip(axes, ("Training", "Validation", "Test")):
        ax.set_ylabel(f"MSE {loss_type} Loss")
        ax.set_xlabel("Epoch")
        ax.set_title(f"MSE {loss_type} Loss")

    fig.savefig(out_fn, dpi=200, bbox_inches="tight")


def split_molecules(ds, split_fracs, generator=None):
    """
    Split a dataset while keeping different poses of the same molecule in the
    same split. Naively splits based on compound_id, so if some compounds has a
    disproportionate number of poses your splits may be imbalanced.

    Parameters
    ----------
    ds : Union[cml.data.DockedDataset, cml.data.GraphDataset]
        Molecular dataset to split
    split_fracs : List[float]
        List of fraction of compounds to put in each split
    generator : torch.Generator, optional
        Torch Generator object to use for randomness. If none is supplied, use
        torch.default_generator

    Returns
    -------
    List[torch.utils.data.Subset]
        List of Subsets of original dataset
    """
    import torch

    ### TODO: make this whole process more compact

    ## First get all the unique compound_ids
    compound_ids_dict = {}
    for c in ds.compounds.keys():
        try:
            compound_ids_dict[c[1]].append(c)
        except KeyError:
            compound_ids_dict[c[1]] = [c]
    all_compound_ids = list(compound_ids_dict.keys())

    ## Set up generator
    if generator is None:
        generator = torch.default_generator

    ## Shuffle the indices
    indices = torch.randperm(len(all_compound_ids), generator=generator)

    ## For each Subset, grab all molecules with the included compound_ids
    all_subsets = []
    offset = 0
    ## Go up to the last split so we can add anything that got left out from
    ##  float rounding
    for frac in split_fracs[:-1]:
        split_len = int(np.floor(frac * len(indices)))
        incl_compounds = all_compound_ids[offset : offset + split_len]
        offset += split_len

        ## Get subset indices
        subset_idx = []
        for compound_id in incl_compounds:
            for compound in compound_ids_dict[compound_id]:
                subset_idx.extend([i for i in ds.compounds[compound]])
        all_subsets.append(torch.utils.data.Subset(ds, subset_idx))

    ## Finish up anything leftover
    incl_compounds = all_compound_ids[offset:]

    ## Get subset indices
    subset_idx = []
    for compound_id in incl_compounds:
        for compound in compound_ids_dict[compound_id]:
            subset_idx.extend([i for i in ds.compounds[compound]])
    all_subsets.append(torch.utils.data.Subset(ds, subset_idx))

    return all_subsets


def train(
    model,
    ds_train,
    ds_val,
    ds_test,
    target_dict,
    n_epochs,
    device,
    model_call=lambda model, d: model(d),
    loss_fn=None,
    save_file=None,
    lr=1e-4,
    start_epoch=0,
    train_loss=None,
    val_loss=None,
    test_loss=None,
    use_wandb=False,
    batch_size=1,
    optimizer=None,
):
    """
    Train a model.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train
    ds_train : data.dataset.DockedDataset
        Train dataset to train on
    ds_val : data.dataset.DockedDataset
        Validation dataset to evaluate on
    ds_test : data.dataset.DockedDataset
        Test dataset to evaluate on
    target_dict : dict[str->float]
        Dictionary mapping from experimental compound_id to measured pIC50 value
    n_epochs : int
        Number of epochs to train for
    device : torch.device
        Where to run the training
    loss_fn : cml.nn.MSELoss
        Loss function that takes pred, target, in_range, and uncertainty values
        as inputs
    model_call : function(model, dict), default=lambda model, d: model(d)
        Function for calling the model. This is present to account for
        differences in calling the SchNet and e3nn models
    save_file : str, optional
        Where to save model weights and errors at each epoch. If a directory is
        passed, the weights will be saved as {epoch_idx}.th and the
        train/val/test losses will be saved as train_err.pkl, val_err.pkl, and
        test_err.pkl. If a string is passed containing {}, it will be formatted
        with the epoch number. Otherwise, the weights will be saved as the
        passed string
    lr : float, default=1e-4
        Learning rate
    start_epoch : int, default=0
        Which epoch the training is starting on. This is used when restarting
        training to ensure the appropriate number of epochs is reached
    train_loss : list[float], default=None
        List of train losses from previous epochs. Used when restarting training
    val_loss : list[float], default=None
        List of val losses from previous epochs. Used when restarting training
    test_loss : list[float], default=None
        List of test losses from previous epochs. Used when restarting training
    use_wandb : bool, default=False
        Log results with WandB
    batch_size : int, default=1
        Number of samples to predict on before performing backprop
    optimizer : torch.optim.Optimizer, optional
        Optimizer to use for model training. If not used, defaults to Adam
        optimizer

    Returns
    -------
    torch.nn.Module
        Trained model
    numpy.ndarray
        Loss for each structure in `ds_train` from each epoch of training, with
        shape (`n_epochs`, `len(ds_train)`)
    numpy.ndarray
        Loss for each structure in `ds_val` from each epoch of training, with
        shape (`n_epochs`, `len(ds_val)`)
    numpy.ndarray
        Loss for each structure in `ds_test` from each epoch of training, with
        shape (`n_epochs`, `len(ds_test)`)
    """
    import pickle as pkl
    from time import time
    import torch
    from . import MSELoss

    if use_wandb:
        import wandb

    if train_loss is None:
        train_loss = []
    if val_loss is None:
        val_loss = []
    if test_loss is None:
        test_loss = []

    ## Send model to desired device if it's not there already
    model = model.to(device)

    ## Save initial model weights for debugging
    if os.path.isdir(save_file):
        torch.save(model.state_dict(), os.path.join(save_file, "init.th"))

    ## Set up optimizer and loss function
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr)
    print("Using optimizer", optimizer, flush=True)

    if loss_fn is None:
        loss_fn = MSELoss()

    ## Train for n epochs
    for epoch_idx in range(start_epoch, n_epochs):
        print(f"Epoch {epoch_idx}/{n_epochs}", flush=True)
        if epoch_idx % 10 == 0 and epoch_idx > 0:
            print(f"Training error: {np.mean(train_loss[-1]):0.5f}")
            print(f"Validation error: {np.mean(val_loss[-1]):0.5f}")
            print(f"Testing error: {np.mean(test_loss[-1]):0.5f}", flush=True)
        tmp_loss = []

        ## Initialize batch
        batch_counter = 0
        optimizer.zero_grad()
        batch_loss = None
        start_time = time()
        for compound, pose in ds_train:
            if type(compound) is tuple:
                compound_id = compound[1]
            else:
                compound_id = compound
            pred = model_call(model, pose)

            # convert to float to match other types
            target = torch.tensor(
                [[target_dict[compound_id]["pIC50"]]], device=device
            ).float()
            in_range = torch.tensor(
                [[target_dict[compound_id]["pIC50_range"]]], device=device
            ).float()
            uncertainty = torch.tensor(
                [[target_dict[compound_id]["pIC50_stderr"]]], device=device
            ).float()
            loss = loss_fn(pred, target, in_range, uncertainty)

            ## Keep track of loss for each sample
            tmp_loss.append(loss.item())

            ## Update batch_loss
            if batch_loss is None:
                batch_loss = loss
            else:
                batch_loss += loss
            batch_counter += 1

            ## Perform backprop if we've done all the preds for this batch
            if batch_counter == batch_size:
                ## Backprop
                batch_loss.backward()
                optimizer.step()

                ## Reset batch tracking
                batch_counter = 0
                optimizer.zero_grad()
                batch_loss = None

        if batch_counter > 0:
            ## Backprop for final incomplete batch
            batch_loss.backward()
            optimizer.step()
        end_time = time()

        train_loss.append(np.asarray(tmp_loss))
        epoch_train_loss = np.mean(tmp_loss)

        with torch.no_grad():
            tmp_loss = []
            for compound, pose in ds_val:
                if type(compound) is tuple:
                    compound_id = compound[1]
                else:
                    compound_id = compound
                pred = model_call(model, pose)

                # convert to float to match other types
                target = torch.tensor(
                    [[target_dict[compound_id]["pIC50"]]], device=device
                ).float()
                in_range = torch.tensor(
                    [[target_dict[compound_id]["pIC50_range"]]], device=device
                ).float()
                uncertainty = torch.tensor(
                    [[target_dict[compound_id]["pIC50_stderr"]]], device=device
                ).float()
                loss = loss_fn(pred, target, in_range, uncertainty)
                tmp_loss.append(loss.item())
            val_loss.append(np.asarray(tmp_loss))
            epoch_val_loss = np.mean(tmp_loss)

            tmp_loss = []
            for compound, pose in ds_test:
                if type(compound) is tuple:
                    compound_id = compound[1]
                else:
                    compound_id = compound
                pred = model_call(model, pose)

                # convert to float to match other types
                target = torch.tensor(
                    [[target_dict[compound_id]["pIC50"]]], device=device
                ).float()
                in_range = torch.tensor(
                    [[target_dict[compound_id]["pIC50_range"]]], device=device
                ).float()
                uncertainty = torch.tensor(
                    [[target_dict[compound_id]["pIC50_stderr"]]], device=device
                ).float()
                loss = loss_fn(pred, target, in_range, uncertainty)
                tmp_loss.append(loss.item())
            test_loss.append(np.asarray(tmp_loss))
            epoch_test_loss = np.mean(tmp_loss)

        if use_wandb:
            wandb.log(
                {
                    "train_loss": epoch_train_loss,
                    "val_loss": epoch_val_loss,
                    "test_loss": epoch_test_loss,
                    "epoch": epoch_idx,
                    "epoch_time": end_time - start_time,
                }
            )
        if save_file is None:
            continue
        elif os.path.isdir(save_file):
            torch.save(model.state_dict(), f"{save_file}/{epoch_idx}.th")
            pkl.dump(
                np.vstack(train_loss), open(f"{save_file}/train_err.pkl", "wb")
            )
            pkl.dump(
                np.vstack(val_loss), open(f"{save_file}/val_err.pkl", "wb")
            )
            pkl.dump(
                np.vstack(test_loss), open(f"{save_file}/test_err.pkl", "wb")
            )
        elif "{}" in save_file:
            torch.save(model.state_dict(), save_file.format(epoch_idx))
        else:
            torch.save(model.state_dict(), save_file)

        ## Stop if loss has gone to infinity or is NaN
        if (
            np.isnan(epoch_val_loss)
            or (epoch_val_loss == np.inf)
            or (epoch_val_loss == -np.inf)
        ):
            if os.path.isdir(save_file):
                pkl.dump(
                    ds_train,
                    open(os.path.join(save_file, "ds_train.pkl"), "wb"),
                )
                pkl.dump(
                    ds_val,
                    open(os.path.join(save_file, "ds_val.pkl"), "wb"),
                )
                pkl.dump(
                    ds_test,
                    open(os.path.join(save_file, "ds_test.pkl"), "wb"),
                )
            raise ValueError("Unrecoverable loss value reached.")

    return (
        model,
        np.vstack(train_loss),
        np.vstack(val_loss),
        np.vstack(test_loss),
    )
