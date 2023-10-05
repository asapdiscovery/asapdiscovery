import click
from pathlib import Path
from typing import Optional

from asapdiscovery.docking.workflows import large_scale_docking


@click.group()
def cli():
    pass


def large_scale(
    filename: Optional[str | Path],
    frag_dir: Optional[str | Path],
    postera: bool,
    postera_upload: bool,
    postera_molset_name: Optional[str],
    du_cache: Optional[str | Path],
    target: TargetTags,
    write_final_sdf: bool,
    dask_client: Optional[str],
):
    """
    Plan a FreeEnergyCalculationNetwork using the given factory and inputs. The planned network will be written to file
    in a folder named after the dataset.
    """
    large_scale_docking(
        filename=None,
        frag_dir=None,
        postera=False,
        postera_upload=False,
        postera_molset_name=None,
        du_cache=None,
        target=None,
        write_final_sdf=False,
        dask_client=None,
    )


if __name__ == "__main__":
    cli()
