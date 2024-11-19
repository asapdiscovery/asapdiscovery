Ecosystem
==========

## Key upstream packages

`asapdiscovery` relies extensively on several key packages within the molecular simulation community.

* `asap-alchemy` relies heavily on [Open Free Energy](https://openfree.energy/) and [alchemiscale](https://github.com/openforcefield/alchemiscale) to perform alchemical free energy calculations
* `asap-ml` relies on [MTENN](https://github.com/choderalab/mtenn) to modularize training and evaluation of structure based ML
* `asap-simulation` relies on [OpenMM](https://github.com/openmm/openmm) and [OpenForceField](https://openforcefield.org/) for running and parameterizing MD simulations respectively.
* `asap-spectrum` uses [Colabfold](https://github.com/sokrypton/ColabFold) for protein structure prediction.

Thanks also to the whole FOSS community and the tireless work done to make all of this possible.

## Downstream packages

ASAP maintains several satellite packages that depend on the functionality in the `asapdiscovery` repo.

Each of these help us fulfil our purpose as an open-science drug discovery enterprises.

### Argos

[Argos](https://github.com/asapdiscovery/argos) is an online Django based viewer for 3D renders of fitness data on protein-ligand complexes that you can find [here](https://argos.asapdata.org/accounts/login/?next=/argos_viewer) . Currently restricted to ASAP collaborators, we are aiming to make this open to the public as a webserver in the future.

### Hindsight-public

[Hindsight-public](https://github.com/asapdiscovery/hindsight-public) is an automatically updated repository that tracks the computational predictions made by ASAP that have experimental equivalents and tracks their accuracy versus those experimental outcomes. This is automatically updated from our systems of record and allows us to keep track of our performance in real time.

### FALCBot

[FALCBot](https://github.com/asapdiscovery/FALCBot) is a slackbot to run free energy calculations and ML predictions using the slack API, enabling easy access to predictions for non-experts.


## Spinouts

We are aiming to productize some aspects of the `asapdiscovery` stack into their own standalone products. See `future` for more information.

### Choppa

[Choppa](https://github.com/asapdiscovery/choppa) is a spinout of our fitness PyMOL and HTML viewer for arbitrary fitness data in the form of for example DMS or phylospectrum experiments.

### ASAP-Alchemy

ASAP-Alchemy is a planned spinout of our free energy calculation tooling into a standalone product. Watch this space!
