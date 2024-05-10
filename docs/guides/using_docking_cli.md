Using the `docking` CLI
======================

## Large scale docking workflow

The large scale docking workflow is intended to run large numbers of virtual designs and prioritise hits based on docking scores. This is useful for sifting through a large number of virtual designs to find promising hits.


**Inputs:**

The large scale docking workflow will dock the following:

- Ligands from Postera (`--postera` `--postera-molset-name` )
- Ligands from a molecule file (`--ligands`)

To a set of liganded complexes from:

- fragalysis (`--fragalysis-dir`)
- Arbitrary set of structures in a directory (`--structure-dir`)
- A single pdb file (`--pdb-file`)

Given a:

- valid target for ASAP (`--target`)

**The workflow does:**

- Protein prep for structures
- Sorting by MCS it will then dock each compound against `n-select`  partner structures.
- Filtering will be done based on the scoring (POSIT confidence > 0.7)  (tunable with `--posit-confidence-cutoff`)
- Best score for each ligand then chosen
- Additional scorers applied if requested
- HTML generated for pose viewer
- if the target has fitness data HTML will be generated for the fitness viewer
- the top N (`--top-n`  default 500) structures will be marked as hits

You can then update or make a new Postera molecule set with the data (`--postera-upload` ) updates existing if you pulled down from one or one with the specified name already exists. If a new name not already present in postera is given with `--postera-molset-name` then a new molecule set will be created. 
****


**Using a cache**

You can use a cache to speed up preparation of the structures, this is especially important when working with Fragalysis as there are lots of structures to prep and this can take a while, but will work with any arbitrary set of structures. To save structures to a cache specify `--cache-dir` (must be an existing directory) and `--save-to-cache` . If you just want to pull from the cache but not save to it, just specify `--cache-dir` .

**ML scorers**

For targets that have ML models deployed for them (currently SARS-CoV-2-Mpro, MERS-CoV-Mpro and SARS-CoV-2-Mac1) you can specify ML scorers on the command line. Do this with something like `--ml-scorer GAT` or `--ml-scorer schnet` . If you want to specify more than one model just specify the flag twice. `--ml-scorer GAT --ml-scorer schnet`.

**Running from a JSON file**

There is also the  capacity to run directly from a JSON file with the appropriate inputs. To do this use `--input-json <file.json>` This is very useful when re-running a previously run job as these JSON files are generated in the output folder of each job.

**Parallelisation**

You can run the workflow in serial (not recommended) or using [Dask](https://www.dask.org/) (`--use-dask`). Dask should be run in  `local` mode. 

Use of these is given in the examples below. It is recommended to have a look at the dask dashboard, whose url is printed in the logfile of the docking job. 

**Postera**

If you wish to use the Postera  platform, contact the ASAP team. There are sevrr

**Examples**

Running on your own computer locally using dask, using a **molecule file**  and docking against **Fragalysis,** using a **cache**  ****and **saving any structures** that are not already prepped to the cache 

```bash
asap-cli docking large-scale 
 --target SARS-CoV-2-Mpro
 --ligands moonshot_subseries.smi
 --fragalysis-dir mpro_fragalysis-24-07-23
 --use-dask
 --cache-dir ./mpro_fragalysis-24-07-23_CACHE
 --save-to-cache
```

Running on your own computer locally using a **molecule file,**  docking against **a pdb file** and uploading to postera. 

```bash
asap-cli docking large-scale 
 --target SARS-CoV-2-Mpro
 --ligands moonshot_subseries.smi 
 --pdb-file my_structure.pdb
 --use-dask
 --postera-upload
 --postera-molset-name MY_POSTERA_MOLSET
```


Running on your own computer locally,  pulling and pushing to **Postera, using** a **set of custom structures**, using GAT and Schnet **ML scorers** and adjusting the **acceptable POSIT cutoff,** how many **MCS matched docking partners,**  and how many **top structures** to return.

```bash
asap-cli docking large-scale 
 --target SARS-CoV-2-Mac1
 --postera
 --postera-molset-name MY_POSTERA_MOLSET
 --postera-upload
 --structure-dir structures_dir # contains PDBs at top level
 --use-dask
 --ml-scorer GAT   # specify each scorer
 --ml-scorer schnet
 --posit-confidence-cutoff 0.9 # only really well docked structures
 --top-n 1000 # return the 1000 best compounds (default 500)
 --n-select 5 # only dock each compound against the top 5 MCS matched structures (default 10)

```

## Small scale docking workflow 

The small scale docking workflow is designed for more detailed investigation of a series of ligands. It has a similar API to the `large-scale` workflows but can also run MD simulations to investigate the stability and dynamics of docked poses. GIFs are then generated from each simulation.  


Request MD with `--md` and a default timestep of `4.0 fs` is used with  and a default number of steps of  `2500000` and reporting interval of 1250 steps. You can change the number of steps with `--md-steps` . I often use `--md-steps 1250` to give one full reporting interval when testing. You can also change the openmm platform with `--md-openmm-platform` . Switching the platform to OpenCL with `--md-openmm-platform` is often useful to fix errors like `Error: Cannot initialize FFT: 5`.


**Examples** 

Running on your own computer locally using dask, using a **molecule file**  and docking against **Fragalysis,** using a **cache**  ****and **saving any structures** that are not already prepped to the cache then **running MD** using the **OpenMM OpenCL platform**

```bash
asap-cli docking small-scale 
 --target SARS-CoV-2-Mpro
 --ligands moonshot_subseries.smi
 --fragalysis-dir mpro_fragalysis-24-07-23
 --use-dask
 --cache-dir ./mpro_fragalysis-24-07-23_CACHE
 --save-to-cache
 --md
 --md-openmm-platform CUDA
```