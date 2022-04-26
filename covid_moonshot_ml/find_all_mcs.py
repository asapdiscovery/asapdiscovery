import argparse
import json
import multiprocessing as mp
import numpy as np
from openeye import oechem, oedepict
import pickle as pkl
from rdkit import Chem
from rdkit.Chem import rdFMCS, Draw
from rdkit.Chem.Draw import rdMolDraw2D
import time

from run_docking import parse_xtal
from schema import ExperimentalCompoundDataUpdate, CrystalCompoundData, \
    EnantiomerPairList

def rank_structures_openeye(exp_smi, exp_id, search_smis, search_ids,
    smi_conv, str_based=False, out_fn=None, n_draw=0):
    """
    Rank all molecules in search_mols based on their MCS with exp_mol.

    Parameters
    ----------
    exp_smi : str
        SMILES string of the experimental compound
    exp_id : str
        CDD compound ID of the experimental compound
    search_smis : List[str]
        List of SMILES of the ligands in the crystal compounds
    search_ids : List[str]
        List of IDs of crystal compounds
    smi_conv : function
        Function to convert a SMILES string to oechem.OEGraphMol
    str_based : bool
        Whether to use a structure-based search (True) or a more strict
        element-based search (False).
    out_fn : str
        If not None, the prefix to save overlap molecule structure drawings.
    n_draw : int
        Draw top n_draw matched molecules
    """

    if str_based:
        """
        For structure based matching
        Options for atom matching:
          * Aromaticity
          * HvyDegree - # heavy atoms bonded to
          * RingMember
        Options for bond matching:
          * Aromaticity
          * BondOrder
          * RingMember
        """
        atomexpr = oechem.OEExprOpts_Aromaticity | oechem.OEExprOpts_HvyDegree | \
            oechem.OEExprOpts_RingMember
        bondexpr = oechem.OEExprOpts_Aromaticity | oechem.OEExprOpts_BondOrder | \
            oechem.OEExprOpts_RingMember
    else:
        """
        For atom based matching
        Options for atom matching (predefined AutomorphAtoms):
          * AtomicNumber
          * Aromaticity
          * RingMember
          * HvyDegree - # heavy atoms bonded to
        Options for bond matching:
          * Aromaticity
          * BondOrder
          * RingMember
        """
        atomexpr = oechem.OEExprOpts_AutomorphAtoms
        bondexpr = oechem.OEExprOpts_Aromaticity | oechem.OEExprOpts_BondOrder | \
            oechem.OEExprOpts_RingMember

    ## Set up the search pattern and MCS objects
    exp_mol = smi_conv(exp_smi)
    pattern_query = oechem.OEQMol(exp_mol)
    pattern_query.BuildExpressions(atomexpr, bondexpr)
    mcss = oechem.OEMCSSearch(pattern_query)
    mcss.SetMCSFunc(oechem.OEMCSMaxAtomsCompleteCycles())

    ## Prepare exp_mol for drawing
    oedepict.OEPrepareDepiction(exp_mol)

    sort_args = []
    for smi in search_smis:
        mol = smi_conv(smi)

        ## MCS search
        mcs = next(iter(mcss.Match(mol, True)))
        sort_args.append((mcs.NumBonds(), mcs.NumAtoms()))

    sort_args = np.asarray(sort_args)
    sort_idx = np.lexsort(-sort_args.T)

    ## Find all substructure matching atoms and draw the molecule with those
    ##  atoms highlighted
    if out_fn is not None:
        for i in range(min(n_draw, len(search_smis))):
            mol_idx = sort_idx[i]
            smi = search_smis[mol_idx]
            mol = smi_conv(smi)

            ## Set up xtal mol for drawing
            oedepict.OEPrepareDepiction(mol)

            ## Set up aligned image
            alignres = oedepict.OEPrepareAlignedDepiction(mol, mcss)
            image = oedepict.OEImage(400, 200)
            grid = oedepict.OEImageGrid(image, 1, 2)
            opts = oedepict.OE2DMolDisplayOptions(grid.GetCellWidth(),
                grid.GetCellHeight(), oedepict.OEScale_AutoScale)
            opts.SetTitleLocation(oedepict.OETitleLocation_Hidden)
            exp_scale = oedepict.OEGetMoleculeScale(exp_mol, opts)
            search_scale = oedepict.OEGetMoleculeScale(mol, opts)
            opts.SetScale(min(exp_scale, search_scale))
            exp_disp = oedepict.OE2DMolDisplay(mcss.GetPattern(), opts)
            search_disp = oedepict.OE2DMolDisplay(mol, opts)

            if alignres.IsValid():
                exp_abset = oechem.OEAtomBondSet(alignres.GetPatternAtoms(),
                    alignres.GetPatternBonds())
                oedepict.OEAddHighlighting(exp_disp, oechem.OEBlueTint,
                    oedepict.OEHighlightStyle_BallAndStick, exp_abset)

                search_abset = oechem.OEAtomBondSet(alignres.GetTargetAtoms(),
                    alignres.GetTargetBonds())
                oedepict.OEAddHighlighting(search_disp, oechem.OEBlueTint,
                    oedepict.OEHighlightStyle_BallAndStick, search_abset)

            exp_cell = grid.GetCell(1, 1)
            oedepict.OERenderMolecule(exp_cell, exp_disp)

            search_cell = grid.GetCell(1, 2)
            oedepict.OERenderMolecule(search_cell, search_disp)

            oedepict.OEWriteImage(f'{out_fn}_{search_ids[mol_idx]}_{i}.png',
                image)

    """
    https://docs.eyesopen.com/toolkits/python/depicttk/molalign.html
    https://docs.eyesopen.com/toolkits/python/oechemtk/patternmatch.html#section-patternmatch-mcss
    https://docs.eyesopen.com/toolkits/python/oechemtk/OEChemConstants/OEExprOpts.html#OEChem::OEExprOpts::HvyDegree
    https://docs.eyesopen.com/toolkits/python/oechemtk/OEChemClasses/OEMCSSearch.html
    """

    return(sort_idx)


def rank_structures_rdkit(exp_smi, exp_id, search_smis, search_ids,
    smi_conv, str_based=False, out_fn=None, n_draw=0):
    """
    Rank all molecules in search_mols based on their MCS with exp_mol.

    Parameters
    ----------
    exp_smi : str
        SMILES string of the experimental compound
    exp_id : str
        CDD compound ID of the experimental compound
    search_smis : List[str]
        List of SMILES of the ligands in the crystal compounds
    search_ids : List[str]
        List of IDs of crystal compounds
    smi_conv : function
        Function to convert a SMILES string to rdkit.Molecule
    str_based : bool
        Whether to use a structure-based search (True) or a more strict
        element-based search (False)
    out_fn : str
        If not None, the prefix to save overlap molecule structure drawings.
    n_draw : int
        Draw top n_draw matched molecules
    """

    if str_based:
        atom_compare = rdFMCS.AtomCompare.CompareAny
    else:
        atom_compare = rdFMCS.AtomCompare.CompareElements

    print(f'Starting {exp_id}', flush=True)
    start_time = time.time()

    exp_mol = smi_conv(exp_smi)

    sort_args = []
    mcs_smarts = []
    for smi in search_smis:
        ## Convert SMILES to molecule
        mol = smi_conv(smi)

        ## Perform MCS search for each search molecule
        # maximize atoms first and then bonds
        # ensure that all ring bonds match other ring bonds and that all rings
        #  must be complete (allowing for incomplete rings causes problems
        #  for some reason)
        mcs = rdFMCS.FindMCS([exp_mol, mol], maximizeBonds=False,
            ringMatchesRingOnly=True, completeRingsOnly=True,
            atomCompare=atom_compare)
        # put bonds before atoms because lexsort works backwards
        sort_args.append((mcs.numBonds, mcs.numAtoms))
        mcs_smarts.append(mcs.smartsString)

    sort_args = np.asarray(sort_args)
    sort_idx = np.lexsort(-sort_args.T)

    ## Find all substructure matching atoms and draw the molecule with those
    ##  atoms highlighted
    if out_fn is not None:
        for i in range(min(n_draw, len(search_smis))):
            mol_idx = sort_idx[i]
            smi = search_smis[mol_idx]
            mol = smi_conv(smi)

            patt = Chem.MolFromSmarts(mcs_smarts[mol_idx])
            hit_ats = list(mol.GetSubstructMatch(patt))
            hit_bonds = []
            for bond in patt.GetBonds():
                try:
                    aid1 = hit_ats[bond.GetBeginAtomIdx()]
                    aid2 = hit_ats[bond.GetEndAtomIdx()]
                except IndexError as e:
                    print(len(hit_ats), bond.GetBeginAtomIdx(),
                        bond.GetEndAtomIdx(), flush=True)
                    print(sort_args[mol_idx], flush=True)
                    print(search_ids[mol_idx], exp_id, flush=True)
                    print(Chem.MolToSmiles(exp_mol), Chem.MolToSmiles(mol),
                        flush=True)
                    print(i, mcs_smarts[mol_idx], flush=True)
                    raise e
                hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())

            d = rdMolDraw2D.MolDraw2DCairo(500, 500)
            rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=hit_ats,
                highlightBonds=hit_bonds)
            d.FinishDrawing()
            d.WriteDrawingText(f'{out_fn}_{search_ids[mol_idx]}_{i}.png')

    Draw.MolToFile(exp_mol, f'{out_fn}.png')

    end_time = time.time()
    print(f'Finished {exp_id} ({end_time-start_time} s)', flush=True)
    return(sort_idx)

def smi_conv_rdkit(s):
    return(Chem.MolFromSmiles(Chem.CanonSmiles(s)))

def smi_conv_oe(s):
    mol = oechem.OEGraphMol()
    oechem.OESmilesToMol(mol, s)
    return(mol)

################################################################################
def get_args():
    parser = argparse.ArgumentParser(description='')

    ## Input arguments
    parser.add_argument('-exp', required=True,
        help='JSON file giving experimental results.')
    parser.add_argument('-x', required=True,
        help='CSV file with crystal compound information.')
    parser.add_argument('-x_dir', required=True,
        help='Directory with crystal structures.')

    ## Output arguments
    parser.add_argument('-o', required=True, help='Main output directory.')

    ## Performance arguments
    parser.add_argument('-n', default=1, type=int,
        help='Number of processors to use.')
    parser.add_argument('-sys', default='rdkit',
        help='Which package to use for MCS search [rdkit, oe].')
    parser.add_argument('-str', action='store_true', help=('Use '
        'structure-based matching instead of element-based matching for MCS.'))
    parser.add_argument('-ep', action='store_true',
        help='Input data is in EnantiomerPairList format.')

    return(parser.parse_args())

def main():
    args = get_args()

    ## Load all compounds with experimental data and filter to only achiral
    ##  molecules (to start)
    if args.ep:
        exp_compounds = [c for ep in EnantiomerPairList(
            **json.load(open(args.exp, 'r'))).pairs \
            for c in (ep.active, ep.inactive)]
    else:
        exp_compounds = [c for c in ExperimentalCompoundDataUpdate(
            **json.load(open(args.exp, 'r'))).compounds if c.smiles is not None]
        exp_compounds = np.asarray([c for c in exp_compounds if c.achiral])

    ## Find relevant crystal structures
    xtal_compounds = parse_xtal(args.x, args.x_dir)

    ## See if I can add a title to the MCS plots for the xtal id
    compound_ids = [c.compound_id for c in exp_compounds]
    xtal_ids = [x.dataset for x in xtal_compounds]
    xtal_smiles = [x.smiles for x in xtal_compounds]

    if args.sys.lower() == 'rdkit':
        ## Convert SMILES to RDKit mol objects for MCS
        ## Need to canonicalize SMILES first because RDKit MCS seems to have
        ##  trouble with non-canon SMILES
        smi_conv = smi_conv_rdkit
        rank_fn = rank_structures_rdkit
    elif args.sys.lower() == 'oe':
        smi_conv = smi_conv_oe
        rank_fn = rank_structures_openeye

    print(f'{len(exp_compounds)} experimental compounds')
    print(f'{len(xtal_compounds)} crystal structures')
    print('Finding best docking structures', flush=True)
    ## Prepare the arguments to pass to starmap
    mp_args = [(c.smiles, c.compound_id, xtal_smiles, xtal_ids, smi_conv,
        args.str, f'{args.o}/{c.compound_id}', 10) for c in exp_compounds]
    n_procs = min(args.n, mp.cpu_count(), len(exp_compounds))
    with mp.Pool(n_procs) as pool:
        res = pool.starmap(rank_fn, mp_args)

    pkl.dump([compound_ids, xtal_ids, res],
        open(f'{args.o}/mcs_sort_index.pkl', 'wb'))

if __name__ == '__main__':
    main()
