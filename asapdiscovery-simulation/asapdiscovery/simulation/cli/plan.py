import click
from typing import Optional


@click.command()
@click.option("-f", "--factory-file", type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False), help="The name of the JSON file containing the FEC factory, if not supplied the default will be used.")
@click.option("-n", "--name", type=click.STRING, help="The name which should be given to this dataset.")
@click.option("-r", "--receptor", type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False), help="The name of the file which contains the prepared receptor.")
@click.option("-l", "--ligands", type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False), help="The file which contains the ligands to use in the planned network.")
@click.option("-c", "--center-ligand", type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False), help="The file which contains the center ligand if required by the network.")
def plan(name: str, receptor: str, ligands: str, center_ligand: Optional[str] = None, factory_file: Optional[str] = None):
    """
    Plan a FreeEnergyCalculationNetwork using the given factory and inputs. The planned network will be written to file
    in a folder named after the dataset.
    """
    from asapdiscovery.simulation.schema.fec import FreeEnergyCalculationFactory
    import os
    from rdkit import Chem
    import openfe

    click.echo(f"Loading FreeEnergyCalculationFactory ...")
    # parse the factory is supplied else get the default
    if factory_file is not None:
        factory = FreeEnergyCalculationFactory.from_file(factory_file)

    else:
        factory = FreeEnergyCalculationFactory()

    click.echo(f"Loading Ligands from {ligands}")
    # parse all required data/ assume sdf currently
    supplier = Chem.SDMolSupplier(ligands, removeHs=False)
    input_ligands = [openfe.SmallMoleculeComponent.from_rdkit(mol) for mol in supplier]
    if center_ligand is not None:
        supplier = Chem.SDMolSupplier(center_ligand, removeHs=False)
        center_ligand = [openfe.SmallMoleculeComponent.from_rdkit(mol) for mol in supplier][0]

    click.echo(f"Loading protein from {receptor}")
    receptor = openfe.ProteinComponent.from_pdb_file(receptor)

    click.echo("Creating FEC network ...")
    planned_network = factory.create_fec_dataset(dataset_name=name, receptor=receptor, ligands=input_ligands,
                                                 central_ligand=center_ligand)
    click.echo(f"Writing results to {name}")
    # output the data to a folder named after the dataset
    os.makedirs(name, exist_ok=True)
    planned_network.to_file(os.path.join(name, "planned_network.json"))
    with open(os.path.join(name, "ligand_network.graphml"), "w") as output:
        output.write(planned_network.network.graphml)
