.. asapdiscovery documentation master file, created by
   sphinx-quickstart on Thu Mar 15 13:55:56 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to asapdiscovery's documentation!
=========================================

All pandemics are global health threats. Our best defense is a healthy global antiviral discovery community with a robust pipeline of open discovery tools.
The AI-driven Structure-enabled Antiviral Platform (`ASAP`_) is making this a reality!

The toolkit in this repo is a batteries-included drug discovery pipeline being actively developed in a transparent open-source way, with a focus on computational chemistry and informatics support for medicinal chemistry. 
Coupled with ASAP's `active data disclosures <https://asapdiscovery.org/outputs/>`_ our campaign to develop a new series of antivirals can provide insight into the drug discovery process that is normally conducted behind closed doors.

The `asapdiscovery` toolkit is focused around the following core competencies, organised into submodules:

 - `asapdiscovery-alchemy`: Free energy calculations using OpenFE and Alchemiscale

 - `asapdiscovery-cli`: Command line tools uniting the whole repo.

 - `asapdiscovery-data`: Core data models and integrations with services such as `postera.ai` , `CDD`, and Diamond Light Source's `Fragalysis` database

 - `asapdiscovery-dataviz`: Data and structure visualization using 3DMol and PyMOL

 - `asapdiscovery-docking`: Docking and compound screening with the OpenEye toolkit

 - `asapdiscovery-genetics`: Working with sequence and fitness information

 - `asapdiscovery-ml`: Structure and graph based ML models for predicting compound activity

 - `asapdiscovery-modelling`: Structure prep and standardisation

 - `asapdiscovery-simulation`: MD simulations and analysis using OpenMM

 - `asapdiscovery-workflows`: Workflows that combine components to enable end to end project support




The `asapdiscovery` toolkit is currently focused on ASAP's targets. See `future` for more information on our roadmap and future plans for the toolkit.


Disclaimer
----------
asapdiscovery is pre-alpha and is under very active development, we make no guarantees around correctness and the API is liable to change rapidly at any time.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   tutorials/index
   guides/index
   modules/index
   ecosystem/index
   future/index
 
   


.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   asapdiscovery



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _ASAP: https://asapdiscovery.org