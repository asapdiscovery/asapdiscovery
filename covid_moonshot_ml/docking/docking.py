from kinoml.core.components import BaseProtein
from kinoml.core.ligands import RDKitLigand
from kinoml.core.systems import ProteinLigandComplex
from kinoml.features.complexes import OEPositDockingFeaturizer
import pandas

from ..schema import CrystalCompoundData

def build_docking_systems(exp_compounds, xtal_compounds, compound_idxs):
    systems = []
    for (c, idx) in zip(exp_compounds, compound_idxs):
        x = xtal_compounds[idx][0]
        protein = BaseProtein(name='MPRO')
        protein.path = x.str_fn
        protein.chain_id = x.str_fn.split('_')[-2][-1]
        protein.expo_id = 'LIG'
        ligand = RDKitLigand.from_smiles(smiles=c.smiles, name=c.compound_id)
        systems.append(ProteinLigandComplex(components=[protein, ligand]))

    return(systems)

def parse_xtal(x_fn, x_dir):
    df = pandas.read_csv(x_fn)

    ## Find all P-files
    idx = [(type(d) is str) and ('-P' in d) for d in df['Dataset']]

    ## Build argument dicts for the CrystalCompoundData objects
    xtal_dicts = [dict(zip(('smiles', 'dataset'), r[1].values)) \
        for r in df.loc[idx,['SMILES', 'Dataset']].iterrows()]

    ## Add structure filename information
    for d in xtal_dicts:
        fn_base = (f'{x_dir}/{d["dataset"]}_0{{}}/{d["dataset"]}_0{{}}_'
            'seqres.pdb')
        fn = fn_base.format('A', 'A')
        if os.path.isfile(fn):
            d['str_fn'] = fn
        else:
            fn = fn_base.format('B', 'B')
            assert os.path.isfile(fn), f'No structure found for {d["dataset"]}.'
            d['str_fn'] = fn

    ## Build CrystalCompoundData objects for each row
    xtal_compounds = [CrystalCompoundData(**d) for d in xtal_dicts]

    return(xtal_compounds)

def run_docking(cache_dir, output_dir, loop_db, n_procs, docking_systems):
    featurizer = OEPositDockingFeaturizer(cache_dir=cache_dir,
        output_dir=output_dir, loop_db=loop_db, n_processes=n_procs)
    docking_systems = featurizer.featurize(docking_systems)

    return(docking_systems)
