Using the `ASAP-Alchemy` CLI
=============================

The `ASAP-Alchemy` CLI provides a series of automated workflows and convince functions that when combined create and
end-to-end pipeline enabling the routine running of state-of-the-art alchemical free energy calculations at (Alchemi)scale!
The CLI is designed to get you up and running as quickly as possible and has tried and tested defaults, but also allows you to
customise every part of the workflow if required. To build custom workflows see the Alchemy API tutorial which explains
the API in detail including the customisation options available. Here we will give a very quick over view of the CLI and
how they should be used in production.

## ASAP-Alchemy Pipeline
The below figure shows the overall structure of the `ASAP-Alchemy` pipeline and how it can fit into the design-make-test-analyse
cycle of a drug discovery campaign. Each of the blue dashed boxes shows the command used to run that part of the pipeline. The commands
can be viewed at any time by running:
```shell
asap-alchemy --help
```

Now lets walk through a typical application starting with `prep`.

![alchemy-fig.png](alchemy-fig.png)

## ASAP-Alchemy Prep

`Prep` offers a pipeline of tools to prepare our ligand series for binding free energy calculations including state enumeration,
constrained pose and partial charge generation. To view the default prep workflow we can use the following command to write workflow to file
where it can be edited although this is much easier using the API:

```shell
asap-alchemy prep create -f "prep-workflow.json"
```

The prep workflow can then be executed on a set of ligands (in a local file smi/sdf) using the following command:
```shell
asap-alchemy prep run --factory-file "prep-workflow.json"  \
                      --dataset-name "example-dataset"     \
                      --ligands "ligand_file.sdf"          \
                      --receptor-complex "receptor.json"   \
                      --processors 4
```

or if you use postera you can provide the name of the molecule set to pull the ligands from provided your `POSTERA_API_KEY` is exported
as an environment variable:

```shell
asap-alchemy prep run --factory-file "prep-workflow.json"   \
                      --dataset-name "example-dataset"      \
                      --postera-molset-name "ligand-series" \
                      --receptor-complex "receptor.json"    \
                      --processors 4
```

```{eval-rst}
.. warning::
    This feature is highly experimental and it is recommended that you check the reference structure carefully
```

If you are not sure which reference crystal you would like to use when generating the poses for the  ligands you can
provide a directory of prepared structures using the `asap-prep` CLI and one will be selected for you.

```shell
asap-alchemy prep run --factory-file "prep-workflow.json"   \
                      --dataset-name "example-dataset"      \
                      --postera-molset-name "ligand-series" \
                      --structure-dir "receptor-cache"      \
                      --processors 4
```

```{eval-rst}
.. warning::
    This feature is highly experimental and it is recommended that you check the injected experimental compounds carefully
```

```{eval-rst}
.. note::
    You must export the ``CDD_API_KEY`` and ``CDD_VAULT_NUMBER`` as environment varibales to enable the CDD interface.
```

Experimentally measured ligands can also be injected into the series at this stage via an interface to the CDD vault. By
providing a protocol name the prep workflow will automatically download all ligands screened as part of this protocol and filter
for ligands with an activity within the assay sensitivity range, fully defined stereochemistry and no covalent warhead. These
will then be posed using the same protocol as the target ligands and marked as experimental via an SD tag.

```shell
asap-alchemy prep run --factory-file "prep-workflow.json"   \
                      --dataset-name "example-dataset"      \
                      --postera-molset-name "ligand-series" \
                      --structure-dir "receptor-cache"      \
                      --processors 4                        \
                      --experimental-protocol "assay-1"
```


Once the prep workflow has finished you will find a new directory has been created named after the `--dataset-name` argument.
Within this you will find a PDB file of the receptor along with an SDF of ligands in their constrained pose along with a csv
detailing any ligand for which a pose could not be generated and the reason why. An `prepared_alchemy_dataset.json` file
will also be present which can be used in the next stage of the workflow.

## ASAP-Alchemy Plan

We are now ready to plan an alchemical free energy network using a state-of-the-art workflow built on the [OpenFE](https://docs.openfree.energy/en/stable/)
infrastructure. Our default workflow plans a minimal spanning tree network with redundancy to ensure each ligand is connected to
at least two other ligands in the network, using the Lomap atom mapping and scoring function. Again this can be configured
via the API or via manually editing the workflow file which can be generated using:

```shell
asap-alchemy create "alchemy-factory.json"
```

We can now plan our network using the default workflow and the ligands we have just posed using the `prep` pipeline from
the previous stage. The `prepared_alchemy_dataset.json` file contains everything needed for this next stage including the
ligands, a dataset name and the receptor. The network is then generated by running:

```shell
asap-alchemy plan --alchemy-dataset "prepared_alchemy_dataset.json"
```

Or if you have posed the ligands using some other pipeline you can provide them as an SDF file and the receptor can be
provided as a PDB and should already be protonated:

```shell
asap-alchemy plan --name "my-network"      \
                  --ligands "ligands.sdf"  \
                  --receptor "protein.pdb"
```

If you use the CDD vault to store experimental data and wish to upload your results to postera later you can also set
the name of the assay protocol and biological target which should be associated with this network to save having to supply
them each time you make a prediction later in the workflow:

```shell
asap-alchemy plan --name "my-network"                \
                  --ligands "ligands.sdf"            \
                  --receptor "protein.pdb"           \
                  --experimental-protocol "assay-2"  \
                  --target "SARS-CoV-2-Mac1"
```

After running the `plan` workflow you will find another new directory has been created named after the `--name` argument
which contains a free energy calculation network in a file named `planned_network.json` and an `ligand_network.graphml`
file which can be viewed as an interactive network using the `OpenFE` CLI:

```shell
openfe view-ligand-network ligand_network.graphml
```

## ASAP-Alchemy Bespoke

Before submitting our alchemical free energy network for simulation we can optionally generate molecule specific dihedral
force field parameters using an interface to [OpenFF-BespokeFit](https://github.com/openforcefield/openff-bespokefit).
 BespokeFit is an automated solution for creating bespoke force field parameters for small molecules which are compatible
with OpenFF general force fields such as Parsley and Sage at scale.

```{eval-rst}
.. note::
    Make sure to have your ``BEFLOW_GATEWAY_ADDRESS`` and ``BEFLOW_GATEWAY_PORT`` exported as environment variables
```

Here we assume you already have a running [BespokeFit executor](https://docs.openforcefield.org/projects/bespokefit/en/latest/getting-started/quick-start.html#production-fits) instance running which you can query via the
[BespokeFit CLI](https://docs.openforcefield.org/projects/bespokefit/en/latest/getting-started/bespoke-cli.html#openff-bespoke-executor).

### ASAP-Alchemy Bespoke Submit
We can now submit the ligands in our planned network for bespoke parameterization using one of our pre-defined
bespoke workflows such as the `aimnet2` workflow which offers a good balance between speed and accuracy:

```shell
asap-alchemy bespoke submit --network "planned_network.json"    \
                            --protocol "aimnet2"
```

This command submits each of the ligands to the BespokeFit executor and stores their bespoke task `ID` back into the
provided network file, these `IDs` are then used later to retrieve results and check the status of the parameterization.

For users who want more fine-grained control over the BespokeFit fitting workflow they can provide a JSON file of the
workflow via:

```shell
asap-alchemy bespoke submit --network "planned_network.json"               \
                            --factory-file "my_bespokefit_workflow.json"
```

### ASAP-Alchemy Bespoke Status
To track the progress of the BespokeFit jobs you can use the following command:
```shell
asap-alchemy bespoke status
```

### ASAP-Alchemy Bespoke Gather
Once all the BespokeFit jobs have finished you can gather the results using:

```shell
asap-alchemy bespoke gather
```

This will save the bespoke parameters for each ligand into the `planned_network.json` file which can be used with the
rest of the workflow and the bespoke parameters will be used automatically.

If we some incomplete jobs in this network or some consistent failures this command will fail, you can however
bypass this check using:

```shell
asap-alchemy bespoke gather --allow-missing
```

## ASAP-Alchemy Submit

```{eval-rst}
.. note::
    The commands ``submit``, ``status``, ``restart``, ``stop``, ``gather`` and ``predict`` assume the network file is in the working
    directory allowing you to avoid passing the argument explicitly.
```

At ASAP we make extensive use of the fantastic [Alchemiscale](https://github.com/openforcefield/alchemiscale):
> a high-throughput alchemical free energy execution system for use with HPC, cloud, bare metal, and Folding@Home

This allows us to plan and execute thousands of `OpenFE` based calculations on distributed compute simultaneously, and
provides a convent API to track and manage calculations rather than having to manually sort though hundreds of local files.

```{eval-rst}
.. note::
    Make sure to have your ``ALCHEMISCALE_ID`` and ``ALCHEMISCALE_KEY`` exported as environment variables
```


We can now submit our `planned_network.json` and execute the tasks on Alchemiscale using:

```shell
asap-alchemy submit --network "planned_network.json"    \
                    --organization "my_org"             \
                    --campaign "testing_asap_alchemy"   \
                    --project "target_1"
```

This command has created the network on Alchemiscale under a Scope defined by the combination of the organization, campaign
and project, then created tasks for each transformation and submitted them to be executed! A unique network key is generated
during this process which allows you to quickly look up the network on Alchemiscale and is stored in the `planned_network.json` file.


## ASAP-Alchemy Status


To track to progress of the alchemical network on Alchemiscale you can use the following command:

```shell
asap-alchemy status
```

If your network has some errored tasks we can also retrieve the errors and tracebacks using:

```shell
asap-alchemy status --errors --with-traceback
```

or if you would like to view the status of all currently actioned networks on Alchemiscale under your scope you can use:

```shell
asap-alchemy status --all-networks
```

## ASAP-Alchemy Restart

Sometimes calculations can fail due to a verity of reasons, some of which can be cleared by simply restarting the tasks.
Until automatic restarting is built into Alchemiscale we provide a command which allows you to restart all the
errored tasks in a network:

```shell
asap-alchemy restart
```

## ASAP-Alchemy Stop

If for any reason you want to stop a network, which removes all currently actioned tasks, you will need the network key which
can be found in the `status` command:

```shell
asap-alchemy stop --network-key "network-key"
```

## ASAP-Alchemy Gather

Once our network has completed all its tasks we can gather the results and store them locally for analysis using:

```shell
asap-alchemy gather
```

if the network has some incomplete edges this command will fail, you can however bypass this check using:

```shell
asap-alchemy gather --allow-missing
```

This will create a new copy of the network with the results called `result_network.json`.

## ASAP-Alchemy Predict

Finally, with our local results we can now estimate the binding affinity of our ligands using:

```shell
asap-alchemy predict
```

This will produce two `CSV` files one containing the relative and the other the absolute binding affinity predictions.


If you provided the `experimental-protocol` during the plan stage, experimental data will be extracted from the named
protocol in the CDD vault and automatically used to assess the accuracy of the calculations. The absolute estimates will also
be shifted to be centred around the mean of the experimental values and interactive `HTML` reports will be generated to
help analyse the results in more detail.

If you did not provide the protocol earlier you can provide it as an argument to the prediction command:

```shell
asap-alchemy predict --experimental-protocol "assay-1"
```

or if you keep you experimental data in a different source you can provide it as a formated csv file which matches the CDD
data:

```shell
asap-alchemy predict  --reference-dataset "assay_data.csv" --reference-units "pIC50"
```

If you use postera and would like to upload the results you can provide the molecule set name and a biological target if
not provided earlier:

```shell
asap-alchemy predict --target "SARS-CoV-2-Mac1" --postera-molset-name "alchemy-ligands-1"
```
