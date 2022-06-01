"""Scripts and models for ML with COVID Moonshot data."""

# Add imports here
# from .covid_moonshot_ml import *
from . import data
from . import docking
from . import nn
from . import schema

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
