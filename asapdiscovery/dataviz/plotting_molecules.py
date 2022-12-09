def display_openeye_ligand(mol, out_fn="test.png", aligned=False):
    from openeye import oedepict

    if not aligned:
        oedepict.OEPrepareDepiction(mol)
    disp = oedepict.OE2DMolDisplay(mol)
    clearbackground = False
    oedepict.OERenderMolecule(out_fn, disp, clearbackground)
