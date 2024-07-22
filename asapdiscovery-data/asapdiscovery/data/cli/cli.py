import click
import copy
from typing import Optional 

from asapdiscovery.data.services.fragalysis.fragalysis_download import (  # noqa: E402
    API_CALL_BASE_LEGACY,
    BASE_URL_LEGACY,
    download,
    FragalysisTargets,
)

def target(func):
    return click.option(
        "-t",
        "--fragalysis-target",
        type=click.Choice(FragalysisTargets.get_values(), case_sensitive=True),
        help="The target for the workflow",
        required=True,
    )(func)



@click.group()
def data():
    """Do data processing tasks"""
    pass


@data.command()
@target
@click.option("-o", "--output", required=True, help="Output file name.")
@click.option(
    "-x",
    "--extract", 
    is_flag=True, 
    help="Extract file after downloading it."
)
def download_fragalysis(fragalysis_target: Optional[str] = "Mpro", 
                        output: Optional[str] = "output.zip", 
                        extract: Optional[bool] = False):
    
    # NOTE currently most of the targets we care about in fragalysis have been shifted to the "legacy" stack
    # hence the use of the legacy base url and api call, this may change in the future

    # Copy the base call and update the base target with the cli-specified target
    api_call = copy.deepcopy(API_CALL_BASE_LEGACY)
    api_call["target_name"] = fragalysis_target

    download(output, api_call, extract=extract, base_url=BASE_URL_LEGACY)
