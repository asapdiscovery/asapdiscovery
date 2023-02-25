import click

from ..cli import cli


def molecule_set_id_arg(func):
    molecule_set_id = click.argument(
        "molecule_set_id",
    )

    return molecule_set_id(func)


def input_file_arg(func):
    input_file = click.argument(
        "input_file",
    )

    return input_file(func)


def smiles_field_param(func):
    smiles_field = click.option(
        "--smiles-field",
        help="Input csv field that stores the SMILES",
        default="smiles",
    )
    return smiles_field(func)


def id_field_param(func):
    id_field = click.option(
        "--id-field",
        help="Input csv field that stores the PostEra Molecule IDs",
        default="id",
    )

    return id_field(func)


@cli.group(help="Commands for interacting with the PostEra API")
@click.option(
    "--api-url",
    help="PostEra API url",
    type=str,
    envvar="POSTERA_API_URL",
    show_envvar=True,
)
@click.option(
    "--api-token",
    help="PostEra API token",
    type=str,
    envvar="POSTERA_API_TOKEN",
    show_envvar=True,
)
@click.option(
    "--api-version",
    help="PostEra API version",
    type=str,
    default="v1",
    envvar="POSTERA_API_VERSION",
    show_envvar=True,
)
@click.pass_context
def postera(ctx, api_url, api_token, api_version):
    ctx.ensure_object(dict)

    ctx.obj["api_url"] = api_url
    ctx.obj["api_token"] = api_token
    ctx.obj["api_version"] = api_version


@postera.group()
@click.pass_context
def moleculeset(ctx):
    from .postera.molecule_set import MoleculeSetAPI

    ctx.obj["moleculesetapi"] = MoleculeSetAPI(
        ctx.obj["api_url"], ctx.obj["api_version"], ctx.obj["api_token"]
    )


@moleculeset.command(help="")
@smiles_field_param
@click.argument("molecule_set_name")
@input_file_arg
@click.pass_context
def create(
    ctx,
    smiles_field,
    molecule_set_name,
    input_file,
):
    import pandas as pd

    from .postera.molecule_set import MoleculeList

    msa = ctx.obj["moleculesetapi"]
    df = pd.read_csv(input_file)

    if smiles_field not in df.columns:
        raise ValueError(f"SMILES field '{smiles_field}' not found in input file")

    molecule_list = MoleculeList.from_pandas_df(df, smiles_field=smiles_field)

    molecule_set_id = msa.create(molecule_set_name, molecule_list)
    click.echo(f"Created molecule set with id {molecule_set_id}")


@moleculeset.command(help="")
@click.pass_context
def list(ctx):
    msa = ctx.obj["moleculesetapi"]

    click.echo(msa.list())


@moleculeset.command(help="")
@molecule_set_id_arg
@click.pass_context
def get(
    ctx,
    molecule_set_id,
):
    ...


@moleculeset.command(help="")
@molecule_set_id_arg
@click.pass_context
def get_molecules(
    ctx,
    molecule_set_id,
):
    msa = ctx.obj["moleculesetapi"]
    df = msa.get_molecules(molecule_set_id)

    outfile_name = f"MoleculeSet_{molecule_set_id}.csv"
    df.to_csv(outfile_name, index=False)
    click.echo(f"Wrote molecule set to {outfile_name}")


@moleculeset.command(help="")
@id_field_param
@molecule_set_id_arg
@input_file_arg
@click.pass_context
def add_molecules():
    ...


@moleculeset.command(help="")
@id_field_param
@molecule_set_id_arg
@input_file_arg
@click.pass_context
def update_molecules(
    ctx,
    id_field,
    molecule_set_id,
    input_file,
):
    import pandas as pd

    from .postera.molecule_set import MoleculeUpdateList

    msa = ctx.obj["moleculesetapi"]

    df = pd.read_csv(input_file)

    if id_field not in df.columns:
        raise ValueError(
            f"PostEra molecule id field '{id_field}' not found in input file"
        )

    update_molecule_list = MoleculeUpdateList.from_pandas_df(df, id_field=id_field)

    molecules_updated = msa.update_molecules(molecule_set_id, update_molecule_list)
    click.echo(f"Updated molecules {molecules_updated} in set {molecule_set_id}")
