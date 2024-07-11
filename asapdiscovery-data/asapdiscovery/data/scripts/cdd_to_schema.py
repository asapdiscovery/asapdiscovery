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

from pathlib import Path

import click
from asapdiscovery.data.services.fragalysis.fragalysis_reader import FragalysisFactory
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


if __name__ == "__main__":
    main()
