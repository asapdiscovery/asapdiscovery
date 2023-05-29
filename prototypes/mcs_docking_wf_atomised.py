from dask import delayed 

# inputs to the delayed functions are 
# probably length 1 lists of the objects

@delayed
def prep_targets(target: List[Target]):
    pp: ProteinPrepper = ProteinPrepper(target)
    prepped_targets: List[Target] = pp.prep_all()

    return prepped_targets

@delayed
def prep_ligands(ligand: List[Ligand]):
    lp: LigandPrepper = LigandPrepper(ligand)
    prepped_ligands: List[Ligand] = lp.prep_all()
    return prepped_ligands


@delayed
def docking_wf(docking_inputs: List[DockingInput]):

   docker: Docker_OE = Docker_OE(docking_inputs)
   docking_results: List[DockingResults] = docker.dock_all()
   
   return docking_results



# main 

def main():
    ligand_fr = FileReader(ligand_fn, type="ligand", format='sdf')
    target_fr = FileReader(target_fn, type="target", format='pdb')

    prepped_targets = []
    for target in target_fr.load():
        prep_t = prep_target(ligand: Ligand)
        prepped_targets.append(prep_t)

    prepped_ligands = []
    for ligand in ligand_fr.load():
        prep_l = prep_ligand(ligand: Ligand)
        prepped_ligands.append(prep_l)

    # true barrier as need to have all ligands and all targets prepped to choose pairs
    # actually is this even true? We only need the list of TARGETS to know which are the best
    ligand_selector: LigandSelector = MCSLigandSelector(prep_l, prep_t)
    docking_inputs: List[DockingInput] = ligand_selector.select()


    docking_results = []
    for docking_input in docking_inputs:
        docking_wf(docking_input)
        docking_results.append(dr)

    # blah blah blah


if __name__ == "__main__":
    main()


# equivalent serial code

ligand_fr = FileReader(ligand_fn, type="ligand", format='sdf')
target_fr = FileReader(target_fn, type="target", format='pdb')

ligands = ligand_fr.load()
targets = target_fr.load()

lp: LigandPrepper = LigandPrepper(ligands)
prepped_ligands: List[Ligand] = lp.prep_all()

pp: ProteinPrepper = ProteinPrepper(targets)
prepped_targets: List[Target] = pp.prep_all()

ligand_selector: LigandSelector = MCSLigandSelector(prepped_ligand, prepped_target)
docking_inputs: List[DockingInput] = ligand_selector.select()

docker: Docker_OE = Docker_OE(docking_inputs)
docking_results: List[DockingResults] = docker.dock_all()