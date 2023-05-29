from dask import delayed 

    
@delayed
def docking_wf(ligand, target, ..):
    # all these lists have length 1 in this case 

    lp: LigandPrepper = LigandPrepper(ligand)
    prepped_ligand: List[Ligand] = lp.prep_all()

    pp: ProteinPrepper = ProteinPrepper(targets)
    prepped_target: List[Target] = pp.prep_all()

    ligand_selector: LigandSelector = PairwiseLigandSelector(prepped_ligand, prepped_target)
    docking_inputs: List[DockingInput] = ligand_selector.select()

    docker: Docker_OE = Docker_OE(docking_inputs)
    docking_results: List[DockingResult] = docker.dock_all()

    return docking_results


# main 

def main():
    ligand_fr = FileReader(ligand_fn, type="ligand", format='sdf')
    target_fr = FileReader(target_fn, type="target", format='pdb')

    for target in target_fr.load():
        for ligand in ligand_fr.load():
            docking_wf(ligand: Ligand, target: Target)


if __name__ == "__main__":
    main()


## equivalent serial code

ligand_fr = FileReader(ligand_fn, type="ligand", format='sdf')
target_fr = FileReader(target_fn, type="target", format='pdb')

ligands = ligand_fr.load()
targets = target_fr.load()

lp: LigandPrepper = LigandPrepper(ligands)
prepped_ligands: List[Ligand] = lp.prep_all()

pp: ProteinPrepper = ProteinPrepper(targets)
prepped_targets: List[Target] = pp.prep_all()

ligand_selector: LigandSelector = PairwiseLigandSelector(prepped_ligand, prepped_target)
docking_inputs: List[DockingInput] = ligand_selector.select()

docker: Docker_OE = Docker_OE(docking_inputs)
docking_results: List[DockingResults] = docker.dock_all()



