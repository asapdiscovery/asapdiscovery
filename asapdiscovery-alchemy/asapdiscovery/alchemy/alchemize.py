from asapdiscovery.alchemy.cli.utils import report_alchemize_clusters
from asapdiscovery.data.schema.ligand import Ligand, write_ligands_to_multi_sdf
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.Scaffolds import MurckoScaffold
from rich.padding import Padding
from tqdm import tqdm


def compute_clusters(ligands: list[Ligand], outsider_number: int, console=None):
    """Clusters ligands into Bajorath-Murcko scaffolds

    Args:
        ligands (list[Ligand]): Ligand objects to cluster
        outsider_number (int): Number of ligands to consider as outsiders
        console: Rich console object for logging

    Returns:
        tuple[dict[str, list[Ligand]], dict[str, list[Ligand]]]: Outsiders and clusters
    """
    # STEP 1: cluster by Bajorath-Murcko scaffold (fast)
    PATT = Chem.MolFromSmarts(
        "[$([D1]=[*])]"
    )  # selects any atoms that are not connected to another heavy atom
    bm_scaffs = []  # first get regular Bemis-Murcko scaffolds

    for lig in tqdm(ligands, desc="Computing Murcko scaffolds"):
        bm_scaffs.append(MurckoScaffold.GetScaffoldForMol(lig.to_rdkit()))

    bbm_scaffs_smi = [
        Chem.MolToSmiles(AllChem.DeleteSubstructs(scaff, PATT)) for scaff in bm_scaffs
    ]  # make them into Bajorath-Bemis-Murcko scaffolds

    # now embed back with original molecules
    mols_with_bbm = [[mol, bbm] for mol, bbm in zip(ligands, bbm_scaffs_smi)]

    # sort so that we make sure that BBM scaffolds get grouped together nicely
    mols_with_bbm = sorted(mols_with_bbm, key=lambda x: x[1], reverse=False)

    # now group them together using the BBM scaffold as dict keys.
    bbm_groups = {}
    for mol, bbm_scaff_smi in mols_with_bbm:
        mol.set_SD_data({"bajorath-bemis-murcko-scaffold": bbm_scaff_smi})
        bbm_groups.setdefault(bbm_scaff_smi, []).append(mol)

    outsiders = {}
    alchemical_clusters = {}
    for bbm_scaff, mols in bbm_groups.items():
        if len(mols) > outsider_number:
            alchemical_clusters[bbm_scaff] = mols
        else:
            outsiders[bbm_scaff] = mols

    message = Padding(
        f"Placed {report_alchemize_clusters(alchemical_clusters, outsiders)[-1]} compounds into "
        "alchemical clusters",
        (1, 0, 1, 0),
    )
    if console:
        console.print(message)

    return outsiders, alchemical_clusters


def partial_sanitize(mol: Mol) -> Mol:
    """Does the minimal number of steps for a molecule object to be workable by rdkit;
    won't throw errors if the mol is funky.

    Args:
        mol (Mol): RDKit molecule object

    Returns:
        Mol: Sanitized RDKit molecule object
    """
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(
        mol,
        Chem.SanitizeFlags.SANITIZE_FINDRADICALS
        | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
        | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
        | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
        | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
        catchErrors=True,
    )
    return mol


def calc_mcs_residuals(mol1: Mol, mol2: Mol) -> tuple[int, int]:
    """Subtract the MCS from two molecules and return the number of heavy atoms remaining after removing the MCS from both

    Args:
        mol1 (Mol): RDKit molecule object
        mol2 (Mol): RDKit molecule object

    Returns:
        tuple[int, int]: Number of heavy atoms remaining in mol1 and mol2 after removing MCS
    """
    mcs = Chem.MolFromSmarts(
        rdFMCS.FindMCS(
            [mol1, mol2],
            matchValences=False,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            matchChiralTag=False,
        ).smartsString
    )

    return Chem.rdMolDescriptors.CalcNumHeavyAtoms(
        AllChem.DeleteSubstructs(mol1, mcs)
    ), Chem.rdMolDescriptors.CalcNumHeavyAtoms(AllChem.DeleteSubstructs(mol2, mcs))


def rescue_outsiders(
    outsiders, alchemical_clusters, max_transform, processors: int, console=None
) -> tuple[dict[str, list[Ligand]], dict[str, list[Ligand]]]:
    """
    STEP 2: rescue outsiders by attempting to place them into Alchemical clusters (slow)
    now for every singleton  try to find a suitable cluster to add it into

    Args:
        outsiders: dict of str: list of Ligands
        alchemical_clusters: dict of str: list of RDKit mols
        max_transform: int
        processors: int
        console: Rich console object for logging

    Returns:
        tuple[dict[str, list[Ligand]], dict[str, list[Ligand]]]: Outsiders and clusters
    """

    message = Padding(
        f"Working to add outsiders into alchemical clusters using {processors} processor(s)",
        (1, 0, 1, 0),
    )
    if console:
        console.print(message)
    singletons_to_move = {}
    for singleton_bbm_scaff, _ in tqdm(outsiders.items(), desc="Rescuing outsiders"):
        singleton_bbm_scaff_ps = partial_sanitize(
            Chem.MolFromSmiles(singleton_bbm_scaff, sanitize=False)
        )

        best_match_total_ha = max_transform * 2
        best_match_cluster_bbm_scaff = None

        # check every BBM scaffold in the clusters we've already found
        for cluster_bbm_scaff, cluster_mols in alchemical_clusters.items():
            cluster_bbm_scaff_ps = partial_sanitize(
                Chem.MolFromSmiles(cluster_bbm_scaff, sanitize=False)
            )

            cluster_mol_n_residual, singleton_mol_n_residual = calc_mcs_residuals(
                cluster_bbm_scaff_ps, singleton_bbm_scaff_ps
            )

            # first discard this match if any of the residuals are higher than the defined cutoff
            if (
                cluster_mol_n_residual > max_transform
                or singleton_mol_n_residual > max_transform
            ):
                continue
            elif (
                cluster_mol_n_residual + singleton_mol_n_residual < best_match_total_ha
            ):
                best_match_total_ha = cluster_mol_n_residual + singleton_mol_n_residual
                best_match_cluster_bbm_scaff = cluster_bbm_scaff

        if best_match_cluster_bbm_scaff:
            # we can add it to the right cluster and remove it from the singletons.
            # need to do this outside loop to not break the dict iterator though
            singletons_to_move[singleton_bbm_scaff] = best_match_cluster_bbm_scaff

    for singleton_bbm_scaff, cluster_to_move_to in singletons_to_move.items():
        # first copy the mols from the singleton over to the intended cluster
        for singleton_mol in outsiders[singleton_bbm_scaff]:
            alchemical_clusters[cluster_to_move_to].append(singleton_mol)

        # then purge the singleton from the dict of singletons
        outsiders.pop(singleton_bbm_scaff)

    return outsiders, alchemical_clusters


def write_clusters(alchemical_clusters, clusterfiles_prefix, outsiders, console=None):
    """Stores clusters to individual SDF files using the clusterfiles prefix variable. Useful for CLI."""
    for i, (bbm_cluster_smiles, ligands) in enumerate(alchemical_clusters.items()):
        output_filename = f"{clusterfiles_prefix}_{i}.sdf"
        # add bbm_cluster_smiles to each ligand as an SD tag. This way we can see both the ligand's BBM scaffold
        # AND the BBM scaffold cluster the ligand has been assigned to.
        [
            lig.set_SD_data(
                {"bajorath-bemis-murcko-scaffold-cluster": bbm_cluster_smiles}
            )
            for lig in ligands
        ]

        # write sdf
        write_ligands_to_multi_sdf(output_filename, ligands, overwrite=True)

        message = Padding(
            f"Wrote cluster {i} ({len(ligands)} assigned to Bajorath-Bemis-Murcko scaffold {bbm_cluster_smiles}) "
            f"to {output_filename}",
            (1, 0, 1, 0),
        )
        if console:
            console.print(message)

    # also write all outsiders to a single SDF
    outsider_ligands_nested = [lig for _, lig in outsiders.items()]
    outsider_ligands = [lig for ligs in outsider_ligands_nested for lig in ligs]

    output_filename_outsiders = f"{clusterfiles_prefix}_outsiders.sdf"
    write_ligands_to_multi_sdf(
        output_filename_outsiders, outsider_ligands, overwrite=True
    )
    message = Padding(
        f"Wrote {len(outsider_ligands)} outsiders to {output_filename_outsiders}.",
        (1, 0, 1, 0),
    )
    if console:
        console.print(message)
