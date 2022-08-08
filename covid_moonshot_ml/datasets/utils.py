import json
import pandas
from rdkit.Chem import CanonSmiles, FindMolChiralCenters, MolFromSmiles
import re
import logging

from ..schema import ExperimentalCompoundData, ExperimentalCompoundDataUpdate, \
    EnantiomerPair, EnantiomerPairList

MPRO_SEQRES = """\
SEQRES   1 A  306  SER GLY PHE ARG LYS MET ALA PHE PRO SER GLY LYS VAL
SEQRES   2 A  306  GLU GLY CYS MET VAL GLN VAL THR CYS GLY THR THR THR
SEQRES   3 A  306  LEU ASN GLY LEU TRP LEU ASP ASP VAL VAL TYR CYS PRO
SEQRES   4 A  306  ARG HIS VAL ILE CYS THR SER GLU ASP MET LEU ASN PRO
SEQRES   5 A  306  ASN TYR GLU ASP LEU LEU ILE ARG LYS SER ASN HIS ASN
SEQRES   6 A  306  PHE LEU VAL GLN ALA GLY ASN VAL GLN LEU ARG VAL ILE
SEQRES   7 A  306  GLY HIS SER MET GLN ASN CYS VAL LEU LYS LEU LYS VAL
SEQRES   8 A  306  ASP THR ALA ASN PRO LYS THR PRO LYS TYR LYS PHE VAL
SEQRES   9 A  306  ARG ILE GLN PRO GLY GLN THR PHE SER VAL LEU ALA CYS
SEQRES  10 A  306  TYR ASN GLY SER PRO SER GLY VAL TYR GLN CYS ALA MET
SEQRES  11 A  306  ARG PRO ASN PHE THR ILE LYS GLY SER PHE LEU ASN GLY
SEQRES  12 A  306  SER CYS GLY SER VAL GLY PHE ASN ILE ASP TYR ASP CYS
SEQRES  13 A  306  VAL SER PHE CYS TYR MET HIS HIS MET GLU LEU PRO THR
SEQRES  14 A  306  GLY VAL HIS ALA GLY THR ASP LEU GLU GLY ASN PHE TYR
SEQRES  15 A  306  GLY PRO PHE VAL ASP ARG GLN THR ALA GLN ALA ALA GLY
SEQRES  16 A  306  THR ASP THR THR ILE THR VAL ASN VAL LEU ALA TRP LEU
SEQRES  17 A  306  TYR ALA ALA VAL ILE ASN GLY ASP ARG TRP PHE LEU ASN
SEQRES  18 A  306  ARG PHE THR THR THR LEU ASN ASP PHE ASN LEU VAL ALA
SEQRES  19 A  306  MET LYS TYR ASN TYR GLU PRO LEU THR GLN ASP HIS VAL
SEQRES  20 A  306  ASP ILE LEU GLY PRO LEU SER ALA GLN THR GLY ILE ALA
SEQRES  21 A  306  VAL LEU ASP MET CYS ALA SER LEU LYS GLU LEU LEU GLN
SEQRES  22 A  306  ASN GLY MET ASN GLY ARG THR ILE LEU GLY SER ALA LEU
SEQRES  23 A  306  LEU GLU ASP GLU PHE THR PRO PHE ASP VAL VAL ARG GLN
SEQRES  24 A  306  CYS SER GLY VAL THR PHE GLN
SEQRES   1 B  306  SER GLY PHE ARG LYS MET ALA PHE PRO SER GLY LYS VAL
SEQRES   2 B  306  GLU GLY CYS MET VAL GLN VAL THR CYS GLY THR THR THR
SEQRES   3 B  306  LEU ASN GLY LEU TRP LEU ASP ASP VAL VAL TYR CYS PRO
SEQRES   4 B  306  ARG HIS VAL ILE CYS THR SER GLU ASP MET LEU ASN PRO
SEQRES   5 B  306  ASN TYR GLU ASP LEU LEU ILE ARG LYS SER ASN HIS ASN
SEQRES   6 B  306  PHE LEU VAL GLN ALA GLY ASN VAL GLN LEU ARG VAL ILE
SEQRES   7 B  306  GLY HIS SER MET GLN ASN CYS VAL LEU LYS LEU LYS VAL
SEQRES   8 B  306  ASP THR ALA ASN PRO LYS THR PRO LYS TYR LYS PHE VAL
SEQRES   9 B  306  ARG ILE GLN PRO GLY GLN THR PHE SER VAL LEU ALA CYS
SEQRES  10 B  306  TYR ASN GLY SER PRO SER GLY VAL TYR GLN CYS ALA MET
SEQRES  11 B  306  ARG PRO ASN PHE THR ILE LYS GLY SER PHE LEU ASN GLY
SEQRES  12 B  306  SER CYS GLY SER VAL GLY PHE ASN ILE ASP TYR ASP CYS
SEQRES  13 B  306  VAL SER PHE CYS TYR MET HIS HIS MET GLU LEU PRO THR
SEQRES  14 B  306  GLY VAL HIS ALA GLY THR ASP LEU GLU GLY ASN PHE TYR
SEQRES  15 B  306  GLY PRO PHE VAL ASP ARG GLN THR ALA GLN ALA ALA GLY
SEQRES  16 B  306  THR ASP THR THR ILE THR VAL ASN VAL LEU ALA TRP LEU
SEQRES  17 B  306  TYR ALA ALA VAL ILE ASN GLY ASP ARG TRP PHE LEU ASN
SEQRES  18 B  306  ARG PHE THR THR THR LEU ASN ASP PHE ASN LEU VAL ALA
SEQRES  19 B  306  MET LYS TYR ASN TYR GLU PRO LEU THR GLN ASP HIS VAL
SEQRES  20 B  306  ASP ILE LEU GLY PRO LEU SER ALA GLN THR GLY ILE ALA
SEQRES  21 B  306  VAL LEU ASP MET CYS ALA SER LEU LYS GLU LEU LEU GLN
SEQRES  22 B  306  ASN GLY MET ASN GLY ARG THR ILE LEU GLY SER ALA LEU
SEQRES  23 B  306  LEU GLU ASP GLU PHE THR PRO PHE ASP VAL VAL ARG GLN
SEQRES  24 B  306  CYS SER GLY VAL THR PHE GLN
"""

def add_seqres(pdb_in, pdb_out=None):
    """
    Add SARS-CoV2 MPRO residue sequence to PDB header.

    Parameters
    ----------
    pdb_in : str
        Input PDB file.
    pdb_out : str, optional
        Output PDB file. If not given, appends _seqres to the input file.
    """

    pdbfile_lines = [line for line in open(pdb_in, 'r') if 'UNK' not in line]
    pdbfile_lines = [line for line in pdbfile_lines if 'LINK' not in line]
    ## Fix bad CL atom names
    pdbfile_lines = [re.sub('CL', 'Cl', l) for l in pdbfile_lines]
    # # remove ligand hetatoms
    # pdbfile_lines = [ line for line in pdbfile_lines if 'LIG' not in line ]
    pdbfile_contents = ''.join(pdbfile_lines)
    if not 'SEQRES' in pdbfile_contents:
        pdbfile_contents = MPRO_SEQRES + pdbfile_contents

    if pdb_out is None:
        pdb_out = f'{pdb_in[:-4]}_seqres.pdb'
    with open(pdb_out, 'w') as fp:
        fp.write(pdbfile_contents)

    print(f'Wrote {pdb_out}', flush=True)

def cdd_to_schema(cdd_csv, out_json, achiral=False):
    """
    Convert a CDD-downloaded CSV file into a JSON file containing an
    ExperimentalCompoundDataUpdate. CSV file must contain the following headers:
        * suspected_SMILES
        * Canonical PostEra ID
        * ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Avg pIC50

    Parameters
    ----------
    cdd_csv : str
        CSV file downloaded from CDD.
    out_json : str
        JSON file to save to.
    achiral : bool, default=False
        Only keep achiral molecules

    Returns
    -------
    ExperimentalCompoundDataUpdate
        The parsed ExperimentalCompoundDataUpdate.
    """

    ## Load and remove any straggling compounds w/o SMILES data
    df = pandas.read_csv(cdd_csv)
    df = df.loc[~df['suspected_SMILES'].isna(),:]

    ## Filter out chiral molecules if requested
    achiral_df = get_achiral_molecules(df)
    if achiral:
        df = achiral_df.copy()
    achiral_label = [compound_id in achiral_df['Canonical PostEra ID'].values
        for compound_id in df['Canonical PostEra ID']]

    ## Get rid of the </> signs, since we really only need the values to sort
    ##  enantiomer pairs
    pic50_key = 'ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Avg pIC50'
    df = df.loc[~df[pic50_key].isna(),:]
    pic50_range = [-1 if '<' in c else (1 if '>' in c else 0) \
        for c in df[pic50_key]]
    pic50_vals = [float(c.strip('<> ')) for c in df[pic50_key]]
    df['pIC50'] = pic50_vals
    df['pIC50_range'] = pic50_range

    compounds = []
    for i, (_, c) in enumerate(df.iterrows()):
        compound_id = c['Canonical PostEra ID']
        smiles = c['suspected_SMILES']
        experimental_data = {
            'pIC50': c['pIC50'],
            'pIC50_range': c['pIC50_range']
        }

        compounds.append(ExperimentalCompoundData(
            compound_id=compound_id,
            smiles=smiles,
            racemic=False,
            achiral=achiral_label[i],
            absolute_stereochemistry_enantiomerically_pure=True,
            relative_stereochemistry_enantiomerically_pure=True,
            experimental_data=experimental_data
        ))
    compounds = ExperimentalCompoundDataUpdate(compounds=compounds)

    with open(out_json, 'w') as fp:
        fp.write(compounds.json())
    print(f'Wrote {out_json}', flush=True)

    return(compounds)

def cdd_to_schema_pair(cdd_csv, out_json):
    """
    Convert a CDD-downloaded CSV file into a JSON file containing an
    EnantiomerPairList. CSV file must contain the following headers:
        * suspected_SMILES
        * Canonical PostEra ID
        * ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Avg pIC50

    Parameters
    ----------
    cdd_csv : str
        CSV file downloaded from CDD.
    out_json : str
        JSON file to save to.

    Returns
    -------
    EnantiomerPairList
        The parsed EnantiomerPairList.
    """

    ## Load and remove any straggling compounds w/o SMILES data
    df = pandas.read_csv(cdd_csv)
    df = df.loc[~df['suspected_SMILES'].isna(),:]

    ## Remove stereochemistry tags and get canonical SMILES values (to help
    ##  group stereoisomers)
    smi_nostereo = [CanonSmiles(s, useChiral=False) \
        for s in df['suspected_SMILES']]
    df['suspected_SMILES_nostereo'] = smi_nostereo

    ## Sort by non-stereo SMILES to put the enantiomer pairs together
    df = df.sort_values('suspected_SMILES_nostereo')

    ## Get rid of the </> signs, since we really only need the values to sort
    ##  enantiomer pairs
    pic50_key = 'ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Avg pIC50'
    pic50_range = [-1 if '<' in c else (1 if '>' in c else 0) \
        for c in df[pic50_key]]
    pic50_vals = [float(c[pic50_key].strip('<> ')) for _, c in df.iterrows()]
    df['pIC50_range'] = pic50_range
    df['pIC50'] = pic50_vals

    enant_pairs = []
    ## Loop through the enantiomer pairs and rank them
    for ep in df.groupby('suspected_SMILES_nostereo'):
        ## Make sure there aren't any singletons
        if ep[1].shape[0] != 2:
            print(f'{ep[1].shape[0]} mols for {ep[0]}', flush=True)
            continue

        p = []
        ## Sort by pIC50 value, higher to lower
        ep = ep[1].sort_values('pIC50', ascending=False)
        for _, c in ep.iterrows():
            compound_id = c['Canonical PostEra ID']
            smiles = c['suspected_SMILES']
            experimental_data = {
                'pIC50': c['pIC50'],
                'pIC50_range': c['pIC50_range']
            }

            p.append(ExperimentalCompoundData(
                compound_id=compound_id,
                smiles=smiles,
                racemic=False,
                achiral=False,
                absolute_stereochemistry_enantiomerically_pure=True,
                relative_stereochemistry_enantiomerically_pure=True,
                experimental_data=experimental_data
            ))

        enant_pairs.append(EnantiomerPair(active=p[0], inactive=p[1]))

    ep_list = EnantiomerPairList(pairs=enant_pairs)

    with open(out_json, 'w') as fp:
        fp.write(ep_list.json())
    print(f'Wrote {out_json}', flush=True)

    return(ep_list)

def get_achiral_molecules(mol_df):
    """
    Remove chiral molecules.

    Parameters
    ----------
    mol_df : pandas.DataFrame
        DataFrame containing compound information

    Returns
    -------
    pandas.DataFrame
        DataFrame containing compound information for all achiral molecules

    """
    ## Check whether a SMILES is chiral or not
    check_achiral = lambda smi: len(FindMolChiralCenters(MolFromSmiles(smi),
        includeUnassigned=True, includeCIP=False,
        useLegacyImplementation=False)) == 0
    ## Check each molecule, first looking at suspected_SMILES, then
    ##  shipment_SMILES if not present
    achiral_idx = []
    for _, r in mol_df.iterrows():
        if ('suspected_SMILES' in r) and \
            (not pandas.isna(r['suspected_SMILES'])):
            achiral_idx.append(check_achiral(r['suspected_SMILES']))
        elif ('shipment_SMILES' in r) and \
            (not pandas.isna(r['shipment_SMILES'])):
            achiral_idx.append(check_achiral(r['shipment_SMILES']))
        else:
            raise ValueError(f'No SMILES found for {r["Canonical PostEra ID"]}')

    return(mol_df.loc[achiral_idx,:])

def filter_molecules_dataframe(
        mol_df,
        smiles_fieldname='suspected_SMILES',
        retain_achiral=False,
        retain_racemic=False,
        retain_enantiopure=False,
        retain_semiquantitative_data=False,
    ):
    """
    Filter a dataframe of molecules to retain those specified.

    Parameters
    ----------
    mol_df : pandas.DataFrame
        DataFrame containing compound information
    smiles_fieldname : str, optional, default='suspected_SMILES'
        Field name to use for reference SMILES
    retain_achrial : bool, optional, default=False
        If True, retain achiral measurements
    retain_racemic : bool, optional, default=False
        If True, retain racemic measurements
    retain_enantiopure : bool, optional, default=False
        If True, retain chrially resolved measurements
    retain_out_of_range_data : bool, optional, default=False
        If True, retain semiquantitative data (data outside assay dynamic range)
    Returns
    -------
    pandas.DataFrame
        DataFrame containing compound information for all achiral molecules

    """
    # Define functions to evaluate whether molecule is achiral, racemic, or resolved
    is_achiral = lambda smi: len(FindMolChiralCenters(MolFromSmiles(smi),
        includeUnassigned=True, includeCIP=False,
        useLegacyImplementation=False)) == 0
    is_racemic = lambda smi: (len(FindMolChiralCenters(MolFromSmiles(smi),
        includeUnassigned=True, includeCIP=False,
        useLegacyImplementation=False)) - len(FindMolChiralCenters(MolFromSmiles(smi),
            includeUnassigned=False, includeCIP=False,
            useLegacyImplementation=False))) > 0
    is_enantiopure = lambda smi: (not is_achiral(smi)) and (not is_racemic(smi))

    # Re-label SMILES field and change name of PostEra ID field
    mol_df = mol_df.rename(columns={smiles_fieldname : 'smiles', 'Canonical PostEra ID' : 'name'})

    logging.debug('Filtering molecules dataframe')
    # Get rid of any molecules that snuck through without SMILES field specified
    logging.debug(f'  dataframe contains {mol_df.shape} entries')
    idx = mol_df.loc[:,'smiles'].isna()
    mol_df = mol_df.loc[~idx,:].copy()
    logging.debug(f'  dataframe contains {mol_df.shape} entries after removing molecules with unspecified {smiles_fieldname} field')
    # Convert CXSMILES to SMILES by removing extra info
    mol_df.loc[:,'smiles'] = [s.strip('|').split()[0] for s in mol_df.loc[:,'smiles']]

    # Determine which molecules will be retained
    include_flags = []
    for _, row in mol_df.iterrows():
        smiles = row['smiles']
        include_this_molecule = (retain_achiral and is_achiral(smiles)) or (retain_racemic and is_racemic(smiles)) or (retain_enantiopure and is_enantiopure(smiles))
        include_flags.append(include_this_molecule)
    mol_df = mol_df.loc[include_flags,:]

    # Filter out semiquantitative data, if requested
    if not retain_semiquantitative_data:
        include_flags = []
        for _, row in mol_df.iterrows():
            try:
                _ = float(row['ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 (µM)'])
                include_flags.append(True)
            except ValueError as e:
                include_flags.append(False)
        mol_df = mol_df.loc[include_flags,:]
        logging.debug(f'  dataframe contains {mol_df.shape} entries after removing semiquantitative data')

    # Compute pIC50s and uncertainties from 95% CIs
    # TODO: In future, we can provide CIs as well
    import numpy as np
    import sigfig
    pIC50_series = []
    pIC50_stderr_series = []
    for _, row in mol_df.iterrows():
        pIC50 = row['ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Avg pIC50'] # string
        try:
            IC50 = float(row['ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 (µM)']) * 1e-6 # molar
            IC50_lower = float(row['ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 CI (Lower) (µM)']) * 1e-6 # molar
            IC50_upper = float(row['ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 CI (Upper) (µM)']) * 1e-6 # molar
            
            pIC50 = -np.log10(IC50)
            pIC50_stderr = np.abs(-np.log10(IC50_lower) + np.log10(IC50_upper)) / 4.0 # assume normal distribution

            # Render into string with appropriate sig figs
            pIC50, pIC50_stderr = sigfig.round(pIC50, uncertainty=pIC50_stderr, sep=tuple) # strings

        except ValueError:
            # Could not convert to string because value was semiquantitative
            if row['ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 (µM)'] == '(IC50 could not be calculated)':
                pIC50 = '< 4.0' # lower limit of detection

            # Keep pIC50 string
            # Use default pIC50 error
            print(row)
            pIC50, pIC50_stderr = pIC50, '0.5' # strings

        pIC50_series.append(pIC50)
        pIC50_stderr_series.append(pIC50_stderr)
    
    mol_df['pIC50'] = pIC50_series
    mol_df['pIC50_stderr'] = pIC50_stderr_series

    # Retain only fields we need
    mol_df = mol_df.filter(['smiles', 'name', 'pIC50', 'pIC50_stderr'])
    logging.debug(f'\n{mol_df}')
    
    return mol_df
