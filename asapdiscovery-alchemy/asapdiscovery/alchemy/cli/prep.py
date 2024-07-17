import shutil
from typing import Optional

import click
from asapdiscovery.alchemy.cli.utils import SpecialHelpOrder


@click.group(
    short_help="Tools to prepare ligands for Alchemy networks via state expansion and constrained pose generation.",
    cls=SpecialHelpOrder,
    context_settings={"max_content_width": shutil.get_terminal_size().columns - 20},
)
def prep():
    """Tools to prepare ligands for Alchemy networks via state expansion and constrained pose generation."""


@prep.command(
    short_help="Create a new AlchemyPrepWorkflow with default settings and save it to JSON file."
)
@click.option(
    "-f",
    "--filename",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True),
    help="The name of the JSON file the workflow should be saved to.",
    required=True,
)
@click.option(
    "-cs",
    "--core-smarts",
    type=click.STRING,
    help="The SMARTS which should be used to select which atoms to constrain to the reference structure.",
)
def create(filename: str, core_smarts: str):
    """
    Create a new AlchemyPrepWorkflow with default settings and save it to JSON file.
    """
    import rich
    from asapdiscovery.alchemy.cli.utils import print_header
    from asapdiscovery.alchemy.schema.prep_workflow import AlchemyPrepWorkflow
    from rich import pretty
    from rich.padding import Padding

    pretty.install()
    console = rich.get_console()
    print_header(console)
    factory = AlchemyPrepWorkflow(core_smarts=core_smarts)
    factory.to_file(filename=filename)
    message = Padding(
        f"The AlchemyPrepWorkflow has been saved to [repr.filename]{filename}[/repr.filename]",
        (1, 0, 1, 0),
    )
    console.print(message)


@prep.command(
    short_help="From a set of compounds with diverse chemistry, create individual clusters of compounds suitable for ASAP-Alchemy."
)
@click.option(
    "-l",
    "--ligands",
    type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
    help="The file which contains the ligands to use in the planned network.",
)
@click.option(
    "-pm",
    "--postera-molset-name",
    type=click.STRING,
    default=None,
    show_default=True,
    help="The name of the Postera molecule set to pull the input ligands from.",
)
@click.option(
    "-p",
    "--processors",
    default=1,
    show_default=True,
    help="The number of processors which can be used to run the workflow in parallel. `auto` will use (all_cpus -1), "
    "`all` will use all or the exact number of cpus to use can be provided.",
)
@click.option(
    "-mt",
    "--max-transform",
    default=8,
    show_default=True,
    help="The maximum number of allowed heavy atoms changed compared to the MCS during outsider rescue. Increasing "
    "this number will increase the chemical diversity in each cluster, but also decrease the number of compounds in "
    "outsiders/singletons.",
)
@click.option(
    "-onu",
    "--outsider-number",
    default=8,
    show_default=True,
    help="The number of compounds at which a cluster is considered an outsider (at this value or below it, a cluster "
    "will not be processed into an AlchemicalNetwork). Decreasing this number will increase the amount of compounds "
    "generated for an AlchemicalNetwork, but can result in small AlchemicalNetworks.",
)
@click.option(
    "-n",
    "--clusterfiles-prefix",
    type=click.STRING,
    required=True,
    help="The prefix to use as filename for the output cluster(s) SDF file(s).",
)
def alchemize(
    ligands: Optional[str] = None,
    postera_molset_name: Optional[str] = None,
    processors: int = 1,
    max_transform: int = 8,
    outsider_number: int = 8,
    clusterfiles_prefix: str = None,
):
    """
    Split a dataset of ligands into individual clusters that are individually suitable for AlchemicalNetworks because
    they have minimal transformations between all members in the cluster. Clustering is done using Bajorath-Murcko
    scaffolds and subsequent attempts at rescuing outsiders (singletons with size up to `outsider_number`) using
    RDKit's FindMCS implementation.

    Parameters
    ----------
    ligands: The name of the local file which contains the input ligands to be prepared in the workflow.
    postera_molset_name: The name of the postera molecule set we should pull the data from instead of a local file.
    processors: The number of processors which can be used to run the workflow in parallel. `auto` will use all
        cpus -1, `all` will use all or the exact number of cpus to use can be provided.
    max_transform: The maximum number of allowed heavy atoms changed compared to the MCS during outsider rescue
    outsider_number: The number of compounds at which a cluster is considered an outsider (at this value or below it,
        a cluster will not be processed into an AlchemicalNetwork)
    clusterfiles_prefix: The prefix to use as filename for the output cluster(s) SDF file(s).
    """
    from multiprocessing import cpu_count

    import rich
    from asapdiscovery.alchemy.cli.utils import (
        print_header,
        pull_from_postera,
        report_alchemize_clusters,
    )
    from asapdiscovery.data.readers.molfile import MolFileFactory
    from asapdiscovery.alchemy.alchemize import (
        compute_clusters,
        rescue_outsiders,
        write_clusters,
    )
    from rich import pretty
    from rich.padding import Padding

    pretty.install()
    console = rich.get_console()
    print_header(console)

    # workout where the molecules are coming from
    if postera_molset_name is not None:
        postera_download = console.status(
            f"Downloading molecules from Postera molecule set:{postera_molset_name}"
        )
        postera_download.start()
        asap_ligands = pull_from_postera(molecule_set_name=postera_molset_name)
        postera_download.stop()

    else:
        # load the molecules
        asap_ligands = MolFileFactory(filename=ligands).load()

    message = Padding(
        f"Loaded {len(asap_ligands)} ligands from [repr.filename]{postera_molset_name or ligands}[/repr.filename]",
        (1, 0, 1, 0),
    )
    console.print(message)

    # workout the number of processes to use if auto or all
    all_cpus = cpu_count()
    if processors == "all":
        processors = all_cpus
    elif processors == "auto":
        processors = all_cpus - 1
    else:
        # can be a string from click
        processors = int(processors)

    if processors > 1:  # TODO
        raise NotImplementedError(
            "Multiprocessing for rescuing outsiders is not yet implemented."
        )

    # step 1: attempt to create alchemical clusters. Some outliers/singletons ("outsiders") are likely
    outsiders, alchemical_clusters = compute_clusters(
        asap_ligands, outsider_number, console
    )

    # step 2: rescue outsiders by attempting to place them into clusters given some lenience level
    outsiders, alchemical_clusters = rescue_outsiders(
        outsiders, alchemical_clusters, max_transform, processors, console
    )

    message = Padding(
        f"After rescuing outsiders, a total of {report_alchemize_clusters(alchemical_clusters, outsiders)[-1]}"
        f" compounds are in an alchemical cluster. Summary (clustersize/number of clusters): "
        f"{report_alchemize_clusters(alchemical_clusters, outsiders)[0]}. Writing to SDF file(s):",
        (1, 0, 1, 0),
    )
    console.print(message)

    # now write to file
    write_clusters(alchemical_clusters, clusterfiles_prefix, outsiders)

    return alchemical_clusters, clusterfiles_prefix, outsiders


@prep.command(
    short_help="Create an AlchemyDataset by running the given AlchemyPrepWorkflow which will expand the ligand states and generate constrained poses suitable for ASAP-Alchemy."
)
@click.option(
    "-f",
    "--factory-file",
    type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
    help="The name of the JSON file containing the AlchemyPrepWorkflow factory, if not supplied the default will be used.",
)
@click.option(
    "-n",
    "--dataset-name",
    type=click.STRING,
    help="The name of the AlchemyDataset this will also be the name of the folder created.",
)
@click.option(
    "-l",
    "--ligands",
    type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
    help="The file which contains the ligands to use in the planned network.",
)
@click.option(
    "-r",
    "--receptor-complex",
    type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
    help="The name of the JSON file which contains the prepared receptor complex including the crystal ligand.",
)
@click.option(
    "-sd",
    "--structure-dir",
    type=click.Path(resolve_path=True, exists=True, file_okay=False, dir_okay=True),
    help="The name of the folder file which contains the prepared receptor complexs including the crystal ligands from which we should select the best complex.",
)
@click.option(
    "-cs",
    "--core-smarts",
    type=click.STRING,
    help="The SMARTS which should be used to select which atoms to constrain to the reference structure.",
)
@click.option(
    "-p",
    "--processors",
    default="auto",
    show_default=True,
    help="The number of processors which can be used to run the workflow in parallel. `auto` will use (all_cpus -1), "
    "`all` will use all or the exact number of cpus to use can be provided.",
)
@click.option(
    "-pm",
    "--postera-molset-name",
    type=click.STRING,
    default=None,
    show_default=True,
    help="The name of the Postera molecule set to pull the input ligands from.",
)
@click.option(
    "-ep",
    "--experimental-protocol",
    help="The name of the experimental protocol in the CDD vault that should be associated with this Alchemy network.",
    type=click.STRING,
    default=None,
    show_default=True,
)
def run(
    dataset_name: str,
    ligands: Optional[str] = None,
    receptor_complex: Optional[str] = None,
    structure_dir: Optional[str] = None,
    factory_file: Optional[str] = None,
    core_smarts: Optional[str] = None,
    processors: int = 1,
    postera_molset_name: Optional[str] = None,
    experimental_protocol: Optional[str] = None,
):
    """
    Create an AlchemyDataset by running the given AlchemyPrepWorkflow which will expand the ligand states and generate
    constrained poses suitable for ASAP-Alchemy.

    Parameters
    ----------
    dataset_name: The name which should be given to the AlchemyDataset all results will be saved in a folder with the same name.
    ligands: The name of the local file which contains the input ligands to be prepared in the workflow.
    receptor_complex: The name of the local file which contains the prepared complex including the crystal ligand.
    structure_dir: The name of the folder which contains the prepared complexs that we should select the best reference from.
    factory_file: The name of the JSON file with the configured AlchemyPrepWorkflow, if not supplied the default will be
        used.
    core_smarts: The SMARTS string used to identify the atoms in each ligand to be constrained.
    processors: The number of processors which can be used to run the workflow in parallel. `auto` will use all
        cpus -1, `all` will use all or the exact number of cpus to use can be provided.
    postera_molset_name: The name of the postera molecule set we should pull the data from instead of a local file.
    """
    import pathlib
    from multiprocessing import cpu_count

    import pandas
    import rich
    from asapdiscovery.alchemy.cli.utils import print_header, pull_from_postera
    from asapdiscovery.alchemy.schema.prep_workflow import AlchemyPrepWorkflow
    from asapdiscovery.data.readers.molfile import MolFileFactory
    from asapdiscovery.data.schema.complex import PreppedComplex
    from rich import pretty
    from rich.padding import Padding

    pretty.install()
    console = rich.get_console()
    print_header(console)

    # load the factory and set the core smarts if supplied
    if factory_file is not None:
        factory = AlchemyPrepWorkflow.parse_file(factory_file)
        if core_smarts is not None:
            factory.core_smarts = core_smarts
    else:
        factory = AlchemyPrepWorkflow(core_smarts=core_smarts)

    # workout where the molecules are coming from
    if postera_molset_name is not None:
        postera_download = console.status(
            f"Downloading molecules from Postera molecule set:{postera_molset_name}"
        )
        postera_download.start()
        asap_ligands = pull_from_postera(molecule_set_name=postera_molset_name)
        postera_download.stop()

    else:
        # load the molecules
        asap_ligands = MolFileFactory(filename=ligands).load()

    message = Padding(
        f"Loaded {len(asap_ligands)} ligands from [repr.filename]{postera_molset_name or ligands}[/repr.filename]",
        (1, 0, 1, 0),
    )
    console.print(message)

    if receptor_complex is None and structure_dir is not None:
        ref_select_status = console.status(
            f"Selecting best reference complex form {structure_dir}"
        )
        ref_select_status.start()

        from asapdiscovery.alchemy.utils import (
            get_similarity,
            select_reference_for_compounds,
        )
        from asapdiscovery.modeling.protein_prep import ProteinPrepperBase

        reference_complex = ProteinPrepperBase.load_cache(cache_dir=structure_dir)
        ref_complex, largest_ligand = select_reference_for_compounds(
            ligands=asap_ligands, references=reference_complex, check_openmm=True
        )
        ref_select_status.stop()
        # check the similarity of the ligands
        sim = get_similarity(ref_complex.ligand, largest_ligand)
        message = Padding(
            f"Selected {ref_complex.target.target_name} as the best reference structure. Largest ligand in the set: "
            f"{largest_ligand.smiles} reference ligand: {ref_complex.ligand.smiles} have similarity: {sim}",
            (1, 0, 1, 0),
        )
        console.print(message)

    else:
        # always expect the JSON file
        ref_complex = PreppedComplex.parse_file(receptor_complex)

        message = Padding(
            f"Loaded a prepared complex from [repr.filename]{receptor_complex}[/repr.filename]",
            (1, 0, 1, 0),
        )
        console.print(message)

    # workout the number of processes to use if auto or all
    all_cpus = cpu_count()
    if processors == "all":
        processors = all_cpus
    elif processors == "auto":
        processors = all_cpus - 1
    else:
        # can be a string from click
        processors = int(processors)

    # check if we need to add experimental ligands
    if experimental_protocol is not None and factory.n_references > 0:
        from asapdiscovery.alchemy.cli.utils import get_cdd_molecules

        message = Padding(
            f"Requested injection of {factory.n_references} experimental references into the network",
            (1, 0, 1, 0),
        )
        console.print(message)

        cdd_status = console.status(
            f"Downloading experimental ligands from CDD protocol {experimental_protocol}"
        )
        cdd_status.start()
        # get all molecules with data for the given protocol, removing stereo issues and possible warheads
        ref_ligands = get_cdd_molecules(
            protocol_name=experimental_protocol,
            defined_stereo_only=True,
            remove_covalent=True,
        )
        cdd_status.stop()

        message = Padding(
            f"Extracted {len(ref_ligands)} ligands from the CDD protocol {experimental_protocol}",
            (1, 0, 1, 0),
        )
        console.print(message)
    else:
        ref_ligands = None

    message = Padding(
        f"Starting Alchemy-Prep workflow with {processors} processors", (1, 0, 1, 0)
    )
    console.print(message)

    alchemy_dataset = factory.create_alchemy_dataset(
        dataset_name=dataset_name,
        ligands=asap_ligands,
        reference_complex=ref_complex,
        processors=processors,
        reference_ligands=ref_ligands,
    )
    output_folder = pathlib.Path(dataset_name)
    output_folder.mkdir(parents=True, exist_ok=True)

    dataset_file = output_folder.joinpath("prepared_alchemy_dataset.json")
    alchemy_dataset.to_file(dataset_file)
    message = Padding(
        f"Saved AlchemyDataset to [repr.filename]{dataset_file}[/repr.filename]",
        (1, 0, 1, 0),
    )
    console.print(message)

    posed_ligand_file = output_folder.joinpath("posed_ligands.sdf")
    alchemy_dataset.save_posed_ligands(posed_ligand_file)
    message = Padding(
        f"Saved posed ligands to [repr.filename]{posed_ligand_file}[/repr.filename]",
        (1, 0, 1, 0),
    )
    console.print(message)
    receptor_file = output_folder.joinpath(
        f"{alchemy_dataset.reference_complex.target.target_name}.pdb"
    )
    alchemy_dataset.reference_complex.target.to_pdb_file(receptor_file)
    message = Padding(
        f"Saved receptor to [repr.filename]{receptor_file}[/repr.filename]",
        (1, 0, 1, 0),
    )
    console.print(message)

    # create a csv of the failed ligands with the failure reason
    if alchemy_dataset.failed_ligands:
        rows = []
        failed_ligands_file = output_folder.joinpath("failed_ligands.csv")
        # sum all the failed ligands across all stages
        fails = sum([len(values) for values in alchemy_dataset.failed_ligands.values()])
        message = Padding(
            f"[yellow]WARNING {fails} ligands failed to have poses generated see [repr.filename]{failed_ligands_file}[/repr.filename][/yellow]",
            (1, 0, 1, 0),
        )
        console.print(message)
        for fail_type, ligands in alchemy_dataset.failed_ligands.items():
            for ligand in ligands:
                rows.append(
                    {
                        "smiles": ligand.provenance.isomeric_smiles,
                        "name": ligand.compound_name,
                        "failure type": fail_type,
                        # if it was an omega fail print the return code
                        "failure info": ligand.tags.get("omega_return_code", ""),
                    }
                )

        # write to csv
        df = pandas.DataFrame(rows)
        failed_ligands_file = output_folder.joinpath("failed_ligands.csv")
        df.to_csv(failed_ligands_file, index=False)
