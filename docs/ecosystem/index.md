Ecosystem
==========

# Key upstream packages 

`asapdiscovery` relies extensively on several key packages within the molecular simulation community. 

* `asap-alchemy` relies heavily on [Open Free Energy](https://openfree.energy/) and [alchemiscale](https://github.com/openforcefield/alchemiscale) to perform alchemical free energy calculations
* `asap-ml` relies on [MTENN](https://github.com/choderalab/mtenn) to modularize training and evaluation of structure based ML
* `asap-simulation` relies on [OpenMM] and [OpenForceField] for running MD simulations
* `asap-genetics` uses [Colabfold](https://github.com/sokrypton/ColabFold) for protein structure prediction. 

Thanks also to the whole FOSS community and the tireless work done to make all of this possible.  

# Downstream packages

ASAP maintains several satellite packages that depend on the functionality in the `asapdiscovery` repo.

Each of these help us fufil our purpose as an open-science drug discovery enterpries. 

## Argos 

[Argos](https://github.com/asapdiscovery/argos) is an online Django based viewer for 3D renders of fitness data on protein-ligand complexes that you can find [here](https://argos.asapdata.org/accounts/login/?next=/argos_viewer) . Currently restricted to ASAP collaborators, we are aiming to make this open to the public as a webserver in the future. 

## Hindsight-public 

[Hindsight-public](https://github.com/asapdiscovery/hindsight-public) is an automatically updated repository that tracks the predictions made for ASAP that have experimental equivalents and tracks their accuracy. This is automatically updated from our systems of record and allows us to 

## FALCBot 

[FALCBot](https://github.com/asapdiscovery/FALCBot) is a slackbot to run free energy calculations and ML predictions using the slack API, enabling easy access to predictions for non-experts. 
