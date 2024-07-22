import click
import copy
from typing import Optional 

from asapdiscovery.data.services.fragalysis.fragalysis_download import (  # noqa: E402
    API_CALL_BASE_LEGACY,
    BASE_URL_LEGACY,
    download,
)



@click.group()
def data():
    """Do data processing tasks"""
    pass


@data.command()
@click.option(
    "-t",
    "--target",
    required=True,
    help="Which target to download. Options are [mpro, mac1].",
    type=str.lower,
    default="mpro",
)
@click.option("-o", "--output", required=True, help="Output file name.")
@click.option(
    "-x",
    "--extract", 
    is_flag=True, 
    help="Extract file after downloading it."
)
def download_fragalysis(target: Optional[str] = "mpro", 
                        output: Optional[str] = "output.zip", 
                        extract: Optional[bool] = False):
    # Copy the base call and update the base target with the cli-specified target
    api_call = copy.deepcopy(API_CALL_BASE_LEGACY)
    target = target.lower()
    api_call["target_name"] = target.capitalize()

    download(output, api_call, extract=extract, base_url=BASE_URL_LEGACY)
