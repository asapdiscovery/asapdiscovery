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

import click
from pathlib import Path

from asapdiscovery.data.util.utils import cdd_to_schema, cdd_to_schema_pair


@click.command()
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
    required=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    help="Output CSV file.",
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
def main(
    in_file: Path,
    out_json: Path,
    out_csv: Path | None = None,
    data_type: str = "std",
    frag_dir: Path | None = None,
):
    if data_type.lower() == "std":
        _ = cdd_to_schema(in_file, out_json, out_csv)
    elif data_type.lower() == "ep":
        _ = cdd_to_schema_pair(in_file, out_json, out_csv)
    else:
        raise ValueError(f"Unknown value for --data-type: {data_type}.")


if __name__ == "__main__":
    main()
