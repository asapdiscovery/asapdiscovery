"""
Analyze perses calculations for a benchmark set of ligands annotated with experimental data

"""

# SDF filename containing all ligands annotated with experimental data
# Ligands should be annotated with experimental data using the SD tags
# EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL
# EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL_STDERR

import argparse
import numpy as np
import glob
import os
import sys
import shutil
import pandas as pd
import csv
from rdkit import Chem


def get_perses_realtime_statistics(basepath, trajectory_prefix="out"):
    """
    Retrieve contents of perses realtime analysis YAML files for a given perses output directory.

    Parameters
    ----------
    basepath : str
        Filepath pointing to the output of a single perses transformation
    trajectory_prefix : str, optional, default='out'
        trajectory_prefix used for output files in setup YAML file

    Returns
    -------
    statistics : dict
        statistics[phase] is the contents of the analysis YAML

    """

    statistics = dict()

    import yaml
    import os

    # TODO: Auto-detect phase names from filenames that are present
    for phase in ["vacuum", "complex", "solvent"]:
        filename = f"{basepath}/{trajectory_prefix}-{phase}_real_time_analysis.yaml"
        if os.path.exists(filename):
            with open(filename, "rt") as infile:
                statistics[phase] = yaml.safe_load(infile)

    return statistics


def get_molecule_titles(filename, add_warts=False):
    """
    Get the list of molecule titles (names) from the specified SDF file.

    Parameters
    ----------
    filename : str
        The filename to read molecules from
    add_warts : bool, optional, default=False
        If True, if multiple molecules with the same title are read, a wart (_0, _1, ...) is appended.

    Returns
    -------
    titles : list of str
        List of molecule titles from the provided SDF file
    """
    # Read titles
    titles = list()
    from openeye import oechem

    with oechem.oemolistream(filename) as ifs:
        oemol = oechem.OEGraphMol()
        while oechem.OEReadMolecule(ifs, oemol):
            title = oemol.GetTitle()
            titles.append(title)

    if not add_warts:
        return titles

    #
    # Add warts to duplicate titles
    #

    # Count titles
    from collections import defaultdict

    title_counts = defaultdict(int)
    for title in titles:
        title_counts[title] += 1

    # Add warts to titles that appear multiple times
    wart_index = defaultdict(int)
    for index, title in enumerate(titles):
        if title_counts[title] > 1:
            wart = f"_{wart_index[title]}"
            wart_index[title] += 1
            titles[index] = title + wart

    return titles


def get_molecule_experimental_data(filename):
    """
    Get experimental data for all molecules in an SDF file.
    If the same title appears multiple times, warts will be added

    The following SD tags will be examined
    * `EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL`
    * `EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL_STDERR`


    Parameters
    ----------
    filename : str
        The filename to read molecules from

    Returns
    -------
    graph : networkx.DiGraph
        graph.nodes[title] contains the following attributes for the molecule with title 'title':
            exp_g_i : the experimental free energy of binding in kT
            exp_dg_i : standard error in exp_g_i

        300 K is assumed for the experimental measurements in converting from kcal/mol to kT

    """
    import networkx as nx

    graph = nx.DiGraph()

    from openmm import unit
    from openmmtools.constants import kB

    kT = kB * 300 * unit.kelvin  # thermal energy at 300 K

    # Get titles with warts
    titles = get_molecule_titles(filename, add_warts=False)

    molecule_index = 0
    from openeye import oechem

    with oechem.oemolistream(filename) as ifs:
        oemol = oechem.OEGraphMol()
        while oechem.OEReadMolecule(ifs, oemol):
            # title = oemol.GetTitle()
            title = titles[molecule_index]  # use title with warts added
            tagname = "EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL"
            if oechem.OEHasSDData(oemol, tagname):
                node_data = {
                    "exp_g_i": float(oechem.OEGetSDData(oemol, tagname))
                    * unit.kilocalories_per_mole
                    / kT,
                    "exp_dg_i": float(oechem.OEGetSDData(oemol, tagname + "_STDERR"))
                    * unit.kilocalories_per_mole
                    / kT,
                }

                graph.add_node(title, **node_data)
            molecule_index += 1

    return graph


# Compute 95%CI width
import numpy as np


def pIC50(IC50_series):
    return -np.log10(IC50_series.astype(float) * 1e-6)


def DeltaG(pIC50):
    kT = 0.593  # kcal/mol for 298 K (25C)
    return -kT * np.log(10.0) * pIC50


def try_with_warts(query_nodename, query_data_name, query_data, graph):
    """
    Try to add data to a networkx graph where node names have warts
    """
    success = False
    for i in range(0, 10):
        print(success)
        if not success:
            try:
                graph.nodes[f"{query_nodename}_{i}"][query_data_name] = query_data
                success = True
            except KeyError:
                pass  # try the next wart int
        else:
            break
    return success, graph


def augment_with_expt_from_manual_insert(filename, graph):
    """
    Get experimental data for all molecules in a CSV file.

    The following SD tags will be examined
    * `sars mpro pic50`

    Parameters
    ----------
    filename : str
        The filename to read molecules from

    Returns
    -------
    graph : networkx.DiGraph
        graph.nodes[title] contains the following attributes for the molecule with title 'title':
            exp_g_i : the experimental free energy of binding in kT
            exp_dg_i : standard error in exp_g_i

        300 K is assumed for the experimental measurements in converting from kcal/mol to kT

    """
    from openmm import unit
    from openmmtools.constants import kB

    kT = kB * 300 * unit.kelvin  # thermal energy at 300 K

    ## load all docked ligands that contain experimental data.
    expt_mols = [
        mol
        for mol in Chem.SDMolSupplier(filename)
        if "sars mpro pic50" in list(mol.GetPropNames())
    ]
    expt_dict = {
        mol.GetProp("_Name"): mol.GetProp("sars mpro pic50") for mol in expt_mols
    }

    ## for every node in the graph, add the experimental data if it exists.
    # first get dG + error.
    # currently: pic50 value w/o error. Convert to dG and assume 0.2 kcal/mol error.
    # for node in graph.nodes(data=True):
    #     print(node)

    for name, pic50 in expt_dict.items():
        dg = DeltaG(float(pic50))
        err = 0.2  # TODO implement error parsing here.

        # now add to graph. This is annoying to do because of warts - should try to remove these soon.
        try:
            graph.nodes[name]["exp_DG"] = dg
            graph.nodes[name]["exp_dDG"] = err
        except KeyError:
            success, graph = try_with_warts(name, "exp_DG", dg, graph)
            _, graph = try_with_warts(name, "exp_dDG", err, graph)
            if not success:
                print(
                    f"Attempted insert of {name} experimental data ({pic50}) but node not found in graph. Ignoring."
                )

    return graph


def collapse_states(perses_graph, filename):
    """
    Read epik state penalties from SDF and collapse protonation/tautomeric states

    The following SD tags will be examined
    * `r_epik_State_Penalty`


    Parameters
    ----------
    filename : str
        The SDF filename to read state penalties from
    graph : networkx.DiGraph
        The graph to annotate

    """
    # Create a copy of the graph
    import copy

    perses_graph = copy.deepcopy(perses_graph)

    from openmm import unit
    from openmmtools.constants import kB

    kT = kB * 300 * unit.kelvin  # thermal energy at 300 K

    # Get titles with warts
    titles = get_molecule_titles(filename, add_warts=True)

    # Read state penalties into perses network
    molecule_index = 0
    from openeye import oechem

    with oechem.oemolistream(filename) as ifs:
        oemol = oechem.OEGraphMol()
        while oechem.OEReadMolecule(ifs, oemol):
            title = titles[molecule_index]  # use title with warts added
            try:
                state_penalty_in_kT = (
                    float(oechem.OEGetSDData(oemol, "r_epik_State_Penalty"))
                    * unit.kilocalories_per_mole
                    / kT
                )  # protonation/tautomeric state penalty
            except ValueError:
                state_penalty_in_kT = 1
            try:
                perses_graph.nodes[title]["state_penalty_dg_i"] = state_penalty_in_kT
            except KeyError:
                molecule_index += 1
                continue
            molecule_index += 1

    # Create new graph
    import networkx as nx
    import numpy as np

    collapsed_graph = nx.DiGraph()  # graph of free energy estimates
    unique_titles = set(get_molecule_titles(filename, add_warts=False))
    for ligand_title in unique_titles:
        # Retrieve all absolute free energy estimates
        ligand_title = ligand_title.split("_")[0]
        state_penalty_i = list()
        g_i = list()
        dg_i = list()
        microstates = list()
        for microstate in perses_graph.nodes:
            if ligand_title in perses_graph.nodes[microstate]["ligand_title"]:
                microstates.append(microstate)
                state_penalty_i.append(
                    perses_graph.nodes[microstate]["state_penalty_dg_i"]
                )
                g_i.append(perses_graph.nodes[microstate]["mle_g_i"])
                dg_i.append(perses_graph.nodes[microstate]["mle_dg_i"])

        state_penalty_i = np.array(state_penalty_i)
        g_i = np.array(g_i)
        dg_i = np.array(dg_i)

        if len(state_penalty_i) == 0 or len(g_i) == 0:
            print(f"Warning: g_i or state penalty data missing for {ligand_title}")
            continue

        # print(ligand_title)
        # print(microstates)
        # print(state_penalty_i)
        # print(state_penalty_i + g_i)

        from scipy.special import logsumexp

        g = -logsumexp(-(state_penalty_i + g_i))
        w_i = np.exp(-(state_penalty_i + g_i) + g)
        dg = np.sqrt(np.sum((w_i * dg_i) ** 2))

        collapsed_graph.add_node(ligand_title, mle_g_i=g, mle_dg_i=dg)

    # Compute edges
    for i in collapsed_graph.nodes:
        for j in collapsed_graph.nodes:
            if i != j:
                g_ij = (
                    collapsed_graph.nodes[j]["mle_g_i"]
                    - collapsed_graph.nodes[i]["mle_g_i"]
                )
                dg_ij = np.sqrt(
                    collapsed_graph.nodes[j]["mle_dg_i"] ** 2
                    + collapsed_graph.nodes[i]["mle_dg_i"] ** 2
                )
                collapsed_graph.add_edge(i, j, g_ij=g_ij, g_dij=dg_ij)

    return collapsed_graph


def get_perses_network_results(basepath, trajectory_prefix="out"):
    """
    Read real-time statistics for all perses transformations in 'basepath' launched via the perses CLI and build a network of estimated free energies.

    .. todo ::

    * Enable user to specify one or more experimental measurements that can be passed to DiffNet to improve the resulting estimate

    Parameters
    ----------
    basepath : str
        Filepath pointing to the output of a single perses transformation
    trajectory_prefix : str, optional, default='out'
        trajectory_prefix used for output files in setup YAML file

    Returns
    -------
    graph : networkx.DiGraph()
        NetworkX graph containing the estimated free energies of all edges and overall MLE free energy estimate solution.
        graph.edges(data=True) will return a list of (i, j, properties) directed i -> j edges with these properties:
           'g_ij' : MBAR free energy estimate for i -> j transformation (in units of kT); negative indicates j should bind more tightly than i
           'g_dij' : standard error uncertainty estimate for g_ij (also in units of kT)

    """
    # Get list of all YAML files generated by the CLI
    yaml_filenames = glob.glob(f"{basepath}/*/perses-*.yaml")

    # Read each transformation summary and assemble the statistics into a graph
    import networkx as nx

    graph = nx.DiGraph()  # graph of free energy estimates

    import yaml
    import numpy as np
    from rich.progress import track

    for filename in track(
        yaml_filenames, description="[blue]Retrieving results of perses calculations..."
    ):
        try:
            with open(filename, "rt") as infile:
                perses_input = yaml.safe_load(infile)
        except yaml.scanner.ScannerError as e:
            # Some files may become corrupted for unknown reasons
            print(e)
            continue

        # Extract initial and final ligand indices
        old_ligand_index = perses_input["old_ligand_index"]
        new_ligand_index = perses_input["new_ligand_index"]
        path = perses_input["trajectory_directory"]

        if not basepath == ".":  # need to correct the relative path in the YAML.
            ligand_file = "/".join(basepath.split("/")[:-2]) + perses_input[
                "ligand_file"
            ].replace("..", "")
        else:
            ligand_file = perses_input["ligand_file"]

        # Extract names of molecules from the input SDF file used to launch simulations
        # NOTE: This requires the ligand SDF file to be present and have the same path name
        # TODO: We don't need to do this over and over again if the ligand file is the same
        ligand_titles_with_warts = get_molecule_titles(ligand_file, add_warts=True)
        ligand_titles = get_molecule_titles(ligand_file, add_warts=False)
        old_ligand_title = ligand_titles_with_warts[old_ligand_index]
        new_ligand_title = ligand_titles_with_warts[new_ligand_index]

        # Retrieve realtime statistics for this edge
        statistics = get_perses_realtime_statistics(
            basepath + path, trajectory_prefix=trajectory_prefix
        )
        # Include this edge if both complex and solvent have useful data
        if ("solvent" in statistics) and ("complex" in statistics):
            # TODO: Extract more statistics about run completion
            # NOTE: We will provide an API for making it easier to gather information about overall binding free energy statistics

            # Package up edge attributes
            edge_fe = (
                statistics["complex"][-1]["mbar_analysis"]["free_energy_in_kT"]
                - statistics["solvent"][-1]["mbar_analysis"]["free_energy_in_kT"]
            )
            edge_dfe = np.sqrt(
                statistics["complex"][-1]["mbar_analysis"]["standard_error_in_kT"] ** 2
                + statistics["solvent"][-1]["mbar_analysis"]["standard_error_in_kT"]
                ** 2
            )
            if edge_dfe == 0:
                edge_dfe += 0.0000001  # prevents division by 0.
            edge_attributes = {
                "g_ij": edge_dfe,
                "g_dij": edge_dfe,
            }

            graph.add_edge(old_ligand_title, new_ligand_title, **edge_attributes)
            # Make sure we add the old ligand titles as well
            graph.nodes[old_ligand_title]["ligand_title"] = ligand_titles[
                old_ligand_index
            ]
            graph.nodes[new_ligand_title]["ligand_title"] = ligand_titles[
                new_ligand_index
            ]
    print(f"Read {len(graph.edges)} perses transformations")

    return graph


def mle(graph):
    """
    Use DiffNet maximum likelihood estimator (MLE) to estimate overall absolute free energies of each ligand
    omitting any experimental measurements

    https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00528

    Parameters:

    graph : networkx.DiGraph()
        NetworkX graph containing the estimated free energies of all edges and overall MLE free energy estimate solution.
        graph.edges(data=True) will return a list of (i, j, properties) directed i -> j edges with these properties:
           'g_ij' : MBAR free energy estimate for i -> j transformation (in units of kT); negative indicates j should bind more tightly than i
           'g_dij' : standard error uncertainty estimate for g_ij (also in units of kT)
    """

    from openff.arsenic import stats

    g_i, C_ij = stats.mle(graph, factor="g_ij", node_factor="exp_DG")

    # Populate graph with MLE estimates
    dg_i = np.sqrt(np.diag(C_ij))
    for node, g, dg in zip(graph.nodes, g_i, dg_i):
        graph.nodes[node]["mle_g_i"] = g
        graph.nodes[node]["mle_dg_i"] = dg

    return graph


def generate_arsenic_plots(
    experimental_data_graph,
    perses_graph,
    arsenic_csv_filename="benchmark.csv",
    target="benchmark",
    relative_plot_filename="relative.pdf",
    absolute_plot_filename="absolute.pdf",
):
    """
    Generate an arsenic CSV file and arsenic plots

    .. warning:: The CSV file will be deprecated once arsenic object model is improved.

    Parameters
    ----------
    experimental_data_graph : networkx.DiGraph
        graph.nodes[title] contains the following attributes for the molecule with title 'title':
            exp_g_i : the experimental free energy of binding in kT
            exp_dg_i : standard error in exp_g_i
    perses_graph : networkx.DiGraph()
        NetworkX graph containing the estimated free energies of all edges and overall MLE free energy estimate solution.
        graph.edges(data=True) will return a list of (i, j, properties) directed i -> j edges with these properties:
           'g_ij' : MBAR free energy estimate for i -> j transformation (in units of kT); negative indicates j should bind more tightly than i
           'g_dij' : standard error uncertainty estimate for g_ij (also in units of kT)
    arsenic_csv_filename : str, optional, default='arsenic.csv'
        Path to arsenic CSV input file to be generated
    target : str, optional, default='target'
        Target name to use in plots
    relative_plot_filename : str, optional, default='relative.pdf'
        Relative free energy comparison with experiment plot
        This plot compares the direct computed edges (without MLE corrections) with experimental free energy differences
    absolute_plot_filename : str, optional, default='absolute.pdf'
        Absolute free energy comparison with experiment plot
        This plot compares the MLE-derived absolute comptued free energies with experimental free energies
        with the computed free energies shifted to the experimental mean
    """
    from openmm import unit
    from openmmtools.constants import kB

    kT = kB * 300 * unit.kelvin  # thermal energy at 300 K

    # Write arsenic CSV file
    with open(arsenic_csv_filename, "w") as csv_file:
        # Experimental block
        # print header for block
        csv_file.write("# Experimental block\n")
        csv_file.write("# Ligand, expt_DG, expt_dDG\n")
        # Extract ligand name, expt_DG and expt_dDG from ligands dictionary
        for ligand_name, data in experimental_data_graph.nodes(data=True):
            csv_file.write(
                f"{ligand_name}, {data['exp_g_i'] * kT/unit.kilocalories_per_mole}, {data['exp_dg_i'] * kT/unit.kilocalories_per_mole}\n"
            )

        # Calculated block
        # print header for block
        csv_file.write("# Calculated block\n")
        csv_file.write(
            "# Ligand1,Ligand2, calc_DDG, calc_dDDG(MBAR), calc_dDDG(additional)\n"
        )
        # Loop through simulation, extract ligand1 and ligand2 indices, convert to names, create string with
        # ligand1, ligand2, calc_DDG, calc_dDDG(MBAR), calc_dDDG(additional)
        # write string in csv file
        for ligand1, ligand2, data in perses_graph.edges(data=True):
            csv_file.write(
                f"{ligand1}, {ligand2}, {data['g_ij'] * kT/unit.kilocalories_per_mole}, {data['g_dij'] * kT/unit.kilocalories_per_mole}, 0.0\n"
            )  # hardcoding additional error as 0.0

    # Generate comparison plots
    from openff.arsenic import plotting, wrangle

    # Generate arsenic plots comparing experimental and calculated free energies
    fe = wrangle.FEMap(arsenic_csv_filename)

    # Generate relative plot
    print(f"Generating {relative_plot_filename}...")
    plotting.plot_DDGs(
        fe.graph,
        target_name=f"{target}",
        title=f"Relative binding energies - {target}",
        figsize=5,
        units="kcal/mol",
        filename=relative_plot_filename,
    )

    # Generate absolute plot, with experimental data shifted to correct mean
    print(f"Generating {absolute_plot_filename}...")
    # experimental_mean_dg = np.asarray([node[1]["exp_DG"] for node in fe.graph.nodes(data=True)]).mean()
    experimental_mean_dg = np.asarray(
        [
            data["exp_g_i"] * kT / unit.kilocalories_per_mole
            for node, data in experimental_data_graph.nodes(data=True)
        ]
    ).mean()
    plotting.plot_DGs(
        fe.graph,
        target_name=f"{target}",
        title=f"Absolute binding energies - {target}",
        figsize=5,
        units="kcal/mol",
        filename=absolute_plot_filename,
        shift=experimental_mean_dg,
    )


def display_predictions(graph):
    """
    Display the predicted free energies in a table.
    """
    from rich.console import Console
    from rich.table import Table

    table = Table(title="perses free energy estimates (up to additive constant)")

    table.add_column("ligand", justify="left", style="cyan", no_wrap=True)
    table.add_column("perses ΔG / kT", justify="centered", style="magenta")

    # Sort ligands
    sorted_ligands = list(graph.nodes)
    sorted_ligands.sort(key=lambda ligand_name: graph.nodes[ligand_name]["mle_g_i"])

    ligand_dgs_sorted = []
    ligand_uncs_sorted = []
    for ligand_name in sorted_ligands:
        data = graph.nodes[ligand_name]
        table.add_row(ligand_name, f"{data['mle_g_i']:6.1f} ± {data['mle_dg_i']:5.1f}")

        ligand_dgs_sorted.append(data["mle_g_i"])
        ligand_uncs_sorted.append(data["mle_dg_i"])

    console = Console()
    console.print(table)

    return sorted_ligands, ligand_dgs_sorted, ligand_uncs_sorted


def gather_additional_data(sorted_ligands, path_to_postera_csv):
    """
    For a list of InChI tags, gather more meaningful data on these ligands based on the supplied CSV.
    Currently grabs all columns but this can be reduced as they may not all be needed/ postera will probably
    try to regenerate these when uploading this file back up to postera.

    Parameters
    ----------
    sorted_ligands : list
        list containing strings of InChI tags
    path_to_postera_csv : str
        CSV downloaded from postera moleculeset containing smiles etc that was used for docking prior to perses.
    """
    # can't use pandas because the inchi messes with regex matching. Use csv module instead.
    if ".csv" in path_to_postera_csv:
        with open(path_to_postera_csv, "r") as readfile:
            reader = csv.reader(readfile)
            postera_csv = [row for row in reader]
    elif ".smi" in path_to_postera_csv:
        with open(path_to_postera_csv, "r") as readfile:
            reader = csv.reader(readfile, delimiter="\t")
            postera_csv = [["SMILES", "UID"]]
            [postera_csv.append(row) for row in reader]

    # now we can just parse the loaded csv file.
    postera_rows = []

    for ligand_tag in sorted_ligands:
        nowart_ligand_tag = ligand_tag.split("_")[
            0
        ]  # JS -> needed to match with non-warted ASAP MOL NAME
        found = False
        for row in postera_csv:
            if (
                ligand_tag in row[1]
                or nowart_ligand_tag in row[0]
                or nowart_ligand_tag in row[1]
            ):  # InChIs have been truncated but still specific enough.
                postera_rows.append(row)
                found = True
        if (
            not found
        ):  # if ligand is not in input postera data it must be a reference crystal.
            postera_rows.append(["reference"])

    return postera_rows, postera_csv[0]


def get_reference_data(receptor_path, basepath):
    """
    Grabs the docking reference ligand as an RDKit object and the CSV data from docking outputs.

    Parameters:
    -----------
    receptor_path : str
        path to receptor file that was used for simulations

    basepath : str
        path that contains perses simulations.
    """
    ref_lig = Chem.SDMolSupplier(receptor_path.replace("protein.pdb", "ligand.sdf"))[0]

    with open(f"{'/'.join(basepath.split('/')[:-2])}/test.smi", "r") as smiles_file:
        reader = csv.reader(smiles_file, delimiter=" ")
        for row in reader:
            if not "InChI" in row[1]:  # this is the reference ligand from fragalysis.
                ref_lig_smiles, ref_lig_name = row

    return ref_lig_name, ref_lig


def write_predictions(
    original_postera_data_sorted,
    ligand_dgs_sorted,
    ligand_uncs_sorted,
    out_file,
    headers,
    ref_lig_name,
    ref_lig,
):
    """
    Writes a CSV with per-ligand dG predictions.

    Parameters
    ----------
    original_postera_data_sorted, ligand_dgs_sorted, ligand_uncs_sorted : list, list, list
        lists containing the ligands' original postera data(see gather_additional_data()),
        their dG values and their uncertainties.
    out_file : str
        Path to write output csv to.
    headers : list
        list of strings containing headers to use for the output CSV.
    ref_lig_name : str
        name of reference ligand, typically the fragalysis ID
    ref_lig : RDKit Mol
        mol object of reference ligand

    """
    # add FECs headers to headers.
    headers.append("perses_dG")
    headers.append("perses_unc")

    # print(original_postera_data_sorted, ligand_dgs_sorted, ligand_uncs_sorted)
    # all lists are already sorted so can just zip and write.
    with open(f"{out_file}.csv", "w") as writefile:
        writer = csv.writer(writefile)
        writer.writerow(headers)
        for row, dg, unc in zip(
            original_postera_data_sorted, ligand_dgs_sorted, ligand_uncs_sorted
        ):
            if row[0] == "reference":
                # need to manually reconstruct column values.
                if ref_lig:
                    row = [Chem.MolToSmiles(ref_lig), ref_lig_name]
                else:
                    row = ["FAILED SMILES", ref_lig_name]
                    print(f"Warning: reference mol {ref_lig_name} failed.")
                [row.append(na) for na in ["N/A"] * (len(headers) - 4)]
            row.append(f"{dg:6.1f}")
            row.append(f"{unc:5.1f}")
            writer.writerow(row)
    print(f"Done --> wrote output to {out_file}.csv.")


def write_structures(receptor_path, ref_lig, ref_lig_name, ligands_file, out_file):
    """
    Writes the PDB of the protein used for the simulations as well as the multi-SDF containing the reference and docked poses.

    Parameters:
    -----------
    receptor_path : str
        Path to receptor that was used in simulations
    ref_lig : RDKit Mol
        mol object of reference ligand
    ref_lig_name : str
        name of reference ligand, typically the fragalysis ID
    ligands_file : str
        path to docked ligands that were used in simulations
    out_file : str
        Path to write output PDB to.
    """
    # first just copy over the protein.
    shutil.copy(receptor_path, f"{out_file}_protein.pdb")
    print(f"Wrote reference protein to {out_file}_protein.pdb.")

    # docked ligands is a bit more involved. We'll use the previously written output CSV to get a nicely ordered multi-SDF
    # with dG predictions as SD tags.
    # first load all docked ligands. We can take any protonation state.
    mols = [mol for mol in Chem.SDMolSupplier(ligands_file)]
    mol_dict = {}
    for (
        mol
    ) in (
        mols
    ):  # these will be inchis for prospective and a fragalysis ID for reference.
        mol_dict[mol.GetProp("_Name")] = mol

    # now load the previously written CSV output.
    with open(f"{out_file}.csv", "r") as readfile, Chem.SDWriter(
        f"{out_file}_ligands.sdf"
    ) as output_sdf:
        reader = csv.reader(readfile)
        next(reader)  # skip header
        for row in reader:
            # first find the molecule object to write to the multi-SDF.
            mol = None
            for k, v in mol_dict.items():
                if k in row[1]:
                    mol = mol_dict[k]  # found based on inchi.
                elif row[1] == ref_lig_name:
                    mol = ref_lig
            if not mol:
                raise ValueError(
                    f"Unable to match {out_file}.csv entry {row[1]} to docked ligands {ligands_file}."
                )

            # we have the molecule, now we can write out the multi-SDF while adding the predicted FE SD tags.
            mol.SetProp("_Name", row[1])
            mol.SetProp("perses_dG", row[-2])
            mol.SetProp("perses_unc", row[-1])

            output_sdf.write(mol)
    print(f"Wrote docked ligands to {out_file}_ligands.sdf.")


def check_args(basepath, ligands_file, receptor_file):
    """
    Checks that the supplied paths exist and contain meaningful information.

    Parameters
    ----------
    basepath : str
        Location for perses simulations, should contain folders named *-His163(*)-*-*.
    ligands_file : str
        Path to SDF containing docked structures
    receptor_file : str
        Path to PDB protein used in simulations

    """
    if not glob.glob(f"{basepath}/*-His163(*)-*-*"):
        raise ValueError(
            f"No perses simulation folders called *-His163(*)-*-* found in {basepath}."
        )

    if not os.path.exists(ligands_file):
        raise ValueError(f"No file found for supplied path {ligands_file}.")

    if not os.path.exists(receptor_file):
        raise ValueError(f"No file found for supplied path {receptor_file}.")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arg_parser.add_argument(
        "--dir", type=str, help="Base directory where results live", required=True
    )
    arg_parser.add_argument(
        "--docked",
        type=str,
        help="Name of the ligands sdf file in the base directory.",
        required=True,
    )
    arg_parser.add_argument(
        "--receptor",
        type=str,
        help="Path to receptor that was used in simulations. Also uses reference ligand.sdf from this path.",
        required=True,
    )
    arg_parser.add_argument(
        "--input_csv",
        type=str,
        help="Name of the postera ligands csv file in the base directory.",
        required=True,
    )
    arg_parser.add_argument(
        "--out", type=str, help="Name of output files containing data.", required=True
    )

    args = arg_parser.parse_args()
    basepath = args.dir
    ligands_file = args.docked
    receptor_path = args.receptor
    input_csv = args.input_csv
    out_file = args.out

    # do some quick checks.
    check_args(basepath, ligands_file, receptor_path)

    # Get perses free energy estimates
    perses_graph = get_perses_network_results(basepath)

    # augment with experimental data from the docked file, if available.
    perses_graph = augment_with_expt_from_manual_insert(ligands_file, perses_graph)

    # get MLE estimates of FE using DiffNet
    perses_graph = mle(perses_graph)

    # Check that we have sufficient data to analyze the graph
    if len(perses_graph.nodes) == 0:
        raise Exception(
            "No edges have generated sufficient data to compare with experiment yet. Both solvent and complex phases must have provided data to analyze."
        )

    collapsed_graph = collapse_states(perses_graph, ligands_file)

    # Show the predictions
    sorted_ligands, ligand_dgs_sorted, ligand_uncs_sorted = display_predictions(
        collapsed_graph
    )

    ref_lig_name, ref_lig = get_reference_data(receptor_path, basepath)

    original_postera_data_sorted, headers = gather_additional_data(
        sorted_ligands, input_csv
    )

    written_df = write_predictions(
        original_postera_data_sorted,
        ligand_dgs_sorted,
        ligand_uncs_sorted,
        out_file,
        headers,
        ref_lig_name,
        ref_lig,
    )

    write_structures(receptor_path, ref_lig, ref_lig_name, ligands_file, out_file)

    # TODO: add simulations in a format that makes it easy to validate edges.
