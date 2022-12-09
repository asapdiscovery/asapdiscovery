def display_openeye_ligand(mol, out_fn="test.png"):
    from openeye import oedepict

    oedepict.OEPrepareDepiction(mol)
    disp = oedepict.OE2DMolDisplay(mol)
    clearbackground = False
    oedepict.OERenderMolecule(out_fn, disp, clearbackground)
