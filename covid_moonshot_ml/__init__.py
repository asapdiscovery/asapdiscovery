"""Scripts and models for ML with COVID Moonshot data."""

# Add imports here
from . import data
from . import datasets
from . import docking
from . import nn
from . import schema
from . import utils

# Handle versioneer
from ._version import get_versions

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
