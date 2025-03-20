import copy
import logging
import os
from pathlib import Path
from typing import Optional

import click
from asapdiscovery.data.services.cdd.cdd_download import download_molecules
from asapdiscovery.data.services.fragalysis.fragalysis_download import (  # noqa: E402
    API_CALL_BASE_LEGACY,
    BASE_URL_LEGACY,
    FragalysisTargets,
    download,
)
from asapdiscovery.data.services.fragalysis.fragalysis_reader import FragalysisFactory
from asapdiscovery.data.util.utils import cdd_to_schema, cdd_to_schema_pair


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
    "-x", "--extract", is_flag=True, help="Extract file after downloading it."
)
def download_fragalysis(
    fragalysis_target: Optional[str] = "Mpro",
    output: Optional[str] = "output.zip",
    extract: Optional[bool] = False,
):
    # NOTE currently most of the targets we care about in fragalysis have been shifted to the "legacy" stack
    # hence the use of the legacy base url and api call, this may change in the future

    # Copy the base call and update the base target with the cli-specified target
    api_call = copy.deepcopy(API_CALL_BASE_LEGACY)
    api_call["target_name"] = fragalysis_target

    download(output, api_call, extract=extract, base_url=BASE_URL_LEGACY)


@data.command()
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    help="Output CSV file.",
)
@click.option(
    "-tok",
    "--token",
    help=(
        "File containing CDD token. Not used if the CDDTOKEN "
        "environment variable is set."
    ),
)
@click.option(
    "--cache",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    help="Cache CSV file.",
)
@click.option(
    "-v",
    "--vault",
    help="Which CDD vault to download from (defaults to Moonshot vault).",
)
@click.option(
    "-s",
    "--search",
    default="sars_fluorescence_noncovalent_w_dates",
    help=(
        "Either a search id or entry in MOONSHOT_SEARCH_DICT "
        "(see asapdiscovery.data.cdd for more details). Defaults to search "
        "with all noncovalent molecules in the SARS-CoV-2 dose response assay."
    ),
)
@click.option(
    "-smi",
    "--smiles-fieldname",
    default="suspected_SMILES",
    help="Which column in the downloaded CSV file to use as SMILES.",
)
@click.option(
    "-id",
    "--id-fieldname",
    default="Canonical PostEra ID",
    help="Which column in the downloaded CSV file to use as the molecule id.",
)
@click.option("--retain-achiral", is_flag=True, help="Keep achiral molecules.")
@click.option("--retain-racemic", is_flag=True, help="Keep racemic molecules.")
@click.option(
    "--retain-enantiopure", is_flag=True, help="Keep chirally resolved molecules."
)
@click.option(
    "--retain-semiquant",
    is_flag=True,
    help="Keep molecules whose IC50 values are out of range.",
)
@click.option(
    "--retain-all",
    is_flag=True,
    help=(
        "Automatically sets retain_achiral, retain_racemic, retain_enantiopure, "
        "and retain_semiquant."
    ),
)
@click.option(
    "-an",
    "--assay-name",
    default="ProteaseAssay_Fluorescence_Dose-Response_Weizmann",
    help="Assay name to parse as IC50.",
)
@click.option(
    "-T",
    "--temp",
    type=float,
    default=298.0,
    help="Temperature in K to use for delta G conversion.",
)
@click.option(
    "-cp",
    "--cheng-prusoff",
    default="0.375,9.5",
    help=(
        "Comma separated values for [S] and Km to use in the Cheng-Prusoff equation "
        "(assumed to be in the same units). Default values are those used in the "
        "SARS-CoV-2 fluorescence experiments from the COVID Moonshot project (in uM "
        "here). Pass 0 for both values to disable and use the pIC50 approximation."
    ),
)
def download_cdd(
    output: Path,
    token: str | None = None,
    cache: Path | None = None,
    vault: str | None = None,
    search: str = "sars_fluorescence_noncovalent_w_dates",
    smiles_fieldname: str = "suspected_SMILES",
    id_fieldname: str = "Canonical PostEra ID",
    retain_achiral: bool = False,
    retain_racemic: bool = False,
    retain_enantiopure: bool = False,
    retain_semiquant: bool = False,
    retain_all: bool = False,
    assay_name: str = "ProteaseAssay_Fluorescence_Dose-Response_Weizmann",
    temp: float = 298.0,
    cheng_prusoff: str = "0.375,9.5",
):
    # Check retain-all shortcut
    if retain_all:
        retain_achiral = True
        retain_racemic = True
        retain_enantiopure = True
        retain_semiquant = True

    # Parse Cheng-Prusoff args
    cheng_prusoff = list(map(float, cheng_prusoff.split(",")))
    if (len(cheng_prusoff) != 2) or (cheng_prusoff == [0, 0]):
        print(
            "No Cheng-Prusoff parameters passed, using pIC50=pKi approximation.",
            flush=True,
        )
        cheng_prusoff = None
    else:
        print(
            f"Using Cheng-Prusoff values of [S]={cheng_prusoff[0]} and",
            f"Km={cheng_prusoff[1]}",
            flush=True,
        )

    # Set up logging
    logging.basicConfig(level=logging.DEBUG)

    # Set up CDD token
    if "CDDTOKEN" in os.environ:
        header = {"X-CDD-token": os.environ["CDDTOKEN"]}
    elif token and token.exists():
        header = {"X-CDD-token": "".join(token.open().readlines()).strip()}
    else:
        raise ValueError(
            "Must pass a file for --token if the CDDTOKEN environment "
            "variable is not set."
        )

    # Get vault number from environment if not given
    if not vault:
        try:
            vault = os.environ["MOONSHOT_CDD_VAULT_NUMBER"]
        except KeyError:
            raise ValueError("No value specified for vault.")

    _ = download_molecules(
        header,
        vault=vault,
        search=search,
        fn_out=output,
        fn_cache=cache,
        id_fieldname=id_fieldname,
        smiles_fieldname=smiles_fieldname,
        retain_achiral=retain_achiral,
        retain_racemic=retain_racemic,
        retain_enantiopure=retain_enantiopure,
        retain_semiquantitative_data=retain_semiquant,
        assay_name=assay_name,
        dG_T=temp,
        cp_values=cheng_prusoff,
    )


@data.command(name="cdd-to-schema")
@click.option(
    "-i",
    "--in-file",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="CSV file input from CDD.",
)
@click.option(
    "-json",
    "--out-json",
    required=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    help="Output JSON file.",
)
@click.option(
    "-csv",
    "--out-csv",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    help="Output CSV file.",
)
@click.option(
    "-smi",
    "--out-smi-csv",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    help="Output CSV file containing the SMILES and compound id of each compound.",
)
@click.option(
    "-type",
    "--data-type",
    default="std",
    type=click.Choice(["std", "ep"], case_sensitive=False),
    help="What type of data is being loaded (std: standard, ep: enantiomer pairs).",
)
@click.option(
    "--frag-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help=(
        "Fragalysis directory. If passed, any compounds that are found to have a "
        "crystal structure will have a Ligand object attached to their exp_data."
    ),
)
def run_cdd_to_schema(
    in_file: Path,
    out_json: Path,
    out_csv: Path | None = None,
    out_smi_csv: Path | None = None,
    data_type: str = "std",
    frag_dir: Path | None = None,
):
    """
    Script to convert a CSV file downloaded (and filtered) from CDD into Schema
    objects that can be used with the rest of the asapdiscovery pipeline. At a
    minimum, the CSV file must have the following columns:
     * "smiles" or "suspected_SMILES"
     * "Canonical PostEra ID"
     * "pIC50" or "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Avg pIC50"

    Minimal example usage:
    python cdd_to_schema.py \
    -i cdd_downloaded_filtered.csv \
    -json cdd_downloaded_filtered.json
    """
    if data_type.lower() == "std":
        compounds = cdd_to_schema(in_file, out_json, out_csv)
    elif data_type.lower() == "ep":
        _ = cdd_to_schema_pair(in_file, out_json, out_csv)
        return
    else:
        raise ValueError(f"Unknown value for --data-type: {data_type}.")

    # Add in Ligand objects from Fragalysis
    if frag_dir and frag_dir.exists():
        print("Loading data from Fragalysis", flush=True)
        ff = FragalysisFactory(parent_dir=frag_dir)
        complexes = ff.load()
        lig_dict = {c.ligand.compound_name: c.ligand for c in complexes}

        for c in compounds:
            try:
                lig = lig_dict[c.compound_id]
            except KeyError:
                continue

            c.experimental_data["xtal_ligand"] = lig

        n_added = sum(["xtal_ligand" in c.experimental_data for c in compounds])
        print(f"Added {n_added} Ligands", flush=True)
        out_json.write_text("[" + ", ".join([c.json() for c in compounds]) + "]")

    # Write out SMILES
    if out_smi_csv:
        out_smi_csv.write_text(
            "\n".join([f"{c.smiles},{c.compound_id}" for c in compounds])
        )
