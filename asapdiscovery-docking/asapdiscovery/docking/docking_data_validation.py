# Backwards-compatible re-export.
#
# ``DockingResultCols`` now lives in the ``asapdiscovery.data`` package so that
# ``asapdiscovery.data`` no longer imports from ``asapdiscovery.docking`` at
# runtime (``data`` is the dependency-free foundation package).
# See https://github.com/asapdiscovery/asapdiscovery/issues/1292
from asapdiscovery.data.schema.docking_data_validation import (  # noqa: F401
    DockingResultCols,
)
