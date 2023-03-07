Running `test-forcefield_generation.py` is expected to generate the following log file in outputs/log.yaml
```
01_Mpro-P2660_0A_EDG-MED-b1ef7fe3-1_prepped_receptor_0.pdb: No template found for
  residue 307 (LIG).  This might mean your input topology is missing some atoms or
  bonds, or possibly that you are using the wrong force field.
02_Mpro-P2660_0A_EDG-MED-b1ef7fe3-1_4RSP_fauxalysis.pdb: No template found for residue
  145 (CSO).  The set of atoms matches CCYS, but the bonds are different.
03_Mpro-P2660_0A_EDG-MED-b1ef7fe3-1_4RSP_fauxalysis_protonated.pdb: No template found
  for residue 8 (HIS).  The set of atoms matches HIP, but the bonds are different.
```