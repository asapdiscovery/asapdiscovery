from .postera.molecule_set import MoleculeSetAPI, MoleculeList, MoleculeUpdateList

import pandas as pd

import click


@click.group()
def cli():
    ...


def molecule_set_id_arg(func):
    molecule_set_id = click.argument(
        "molecule_set_id",
        help="Name of the MoleculeSet")

    return molecule_set_id(func)


def input_file_arg(func):
    input_file = click.argument(
        "input_file",
        help="Path to CSV file containing molecule data\n"
        "If using create, the file must include a field containing SMILES.\n"
        "If using update, the file must include a field containing postera molecule ids\n",
    )

    return input_file(func)


def smiles_field_param(func):
    smiles_field = click.argument(
        "--smiles-field",
        help="Input csv field that stores the SMILES",
        default='smiles'
    )
    return smiles_field(func)


def id_field_param(func):
    id_field = click.argument(
        "--id-field",
        help="Input csv field that stores the PostEra Molecule IDs",
        default='id'
    )

    return id_field(func)


@cli.group(help="Subcommands for interacting with the PostEra API")
@click.option(
        "--api-url",
        help="PostEra API url",
        type=str,
        envvar="POSTERA_API_URL",
        show_envvar=True
)
@click.option(
        "--api-token",
        help="PostEra API token",
        type=str,
        envvar="POSTERA_API_TOKEN",
        show_envvar=True
)
@click.option(
        "--api-version",
        help="PostEra API version",
        type=str,
        default='v1',
        envvar="POSTERA_API_VERSION",
        show_envvar=True
)
@click.pass_context
def postera(ctx, api_url, api_token, api_version):
    ctx.ensure_object(dict)

    ctx.obj['api_url'] = api_url
    ctx.obj['api_token'] = api_token
    ctx.obj['api_version'] = api_version


@postera.group()
@click.pass_context
def moleculeset(ctx):

    ctx.obj['moleculesetapi'] = MoleculeSetAPI(
            ctx.obj['api_url'],
            ctx.obj['api_version'],
            ctx.obj['api_token'])


@postera.command(help="")
@smiles_field_param
@click.argument("molecule_set_name")
@input_file_arg
@click.pass_context
def create(
        smiles_field,
        molecule_set_name,
        input_file,
        ctx
    ):

    msa = ctx.obj['moleculesetapi']
    df = pd.read_csv(input_file)

    if smiles_field not in df.columns:
        raise ValueError(
            f"SMILES field '{smiles_field}' not found in input file"
        )

    molecule_list = MoleculeList.from_pandas_df(df, smiles_field=smiles_field)

    molecule_set_id = msa.create(molecule_list, molecule_set_name)
    click.echo(f"Created molecule set with id {molecule_set_id}")


@postera.command(help="")
def list():
    ...


@postera.command(help="")
@molecule_set_id_arg
def get(
    molecule_set_id,
    ctx
    ):
    ...


@postera.command(help="")
@molecule_set_id_arg
def get_molecules(
    molecule_set_id,
    ctx
    ):
    msa = ctx.obj['moleculesetapi']
    df = msa.get_molecules(molecule_set_id)

    outfile_name = f"MoleculeSet_{molecule_set_id}.csv"
    df.to_csv(outfile_name, index=False)
    click.echo(f"Wrote molecule set to {outfile_name}")


@postera.command(help="")
@id_field_param
@molecule_set_id_arg
@input_file_arg
def add_molecules():
    ...


@postera.command(help="")
@id_field_param
@molecule_set_id_arg
@input_file_arg
def update_molecules(
        id_field,
        molecule_set_id,
        input_file,
        ctx
    ):

    msa = ctx.obj['moleculesetapi']

    df = pd.read_csv(input_file)

    if id_field not in df.columns:
        raise ValueError(
            f"PostEra molecule id field '{id_field}' not found in input file"
        )

    update_molecule_list = MoleculeUpdateList.from_pandas_df(df, id_field=id_field)

    molecules_updated = msa.update_custom_data(
        molecule_set_id, update_molecule_list
    )
    click.echo(f"Updated molecules {molecules_updated} in set {molecule_set_id}")
