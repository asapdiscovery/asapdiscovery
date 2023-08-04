import logging
from pathlib import Path
from typing import Optional, Union

from asapdiscovery.data.logging import FileLogger
from asapdiscovery.data.openeye import oechem, oeomega
from pydantic import BaseModel, Field


class StereoExpanderOptions(BaseModel):
    warts: bool = Field(False, description="Add warts to the output file")
    force_flip: bool = Field(
        False, description="Force enumeration of stereo centers even if defined"
    )
    postera_names: bool = Field(
        True, description="Use PostEra names for output if present"
    )
    debug: bool = Field(False, description="Print debug messages")


class StereoExpander:
    def __init__(
        self, options: StereoExpanderOptions, logger: Optional[logging.Logger] = None
    ):
        self.options = options

        if logger is None:
            logger_cls = FileLogger(
                "stereo_enumeration",
                path="./",
                stdout=True,
                level=logging.INFO if not self.options.debug else logging.DEBUG,
            )
            self.logger = logger_cls.getLogger()
        else:
            self.logger = logger

        self.logger.info("StereoExpander initialized")
        self.logger.info(f"Options: {self.options}")

        self.flipperOpts = oeomega.OEFlipperOptions()
        self.flipperOpts.SetWarts(self.options.warts)
        self.flipperOpts.SetEnumSpecifiedStereo(self.options.force_flip)

    def _expand_mol(self, mol: oechem.OEMol) -> list[oechem.OEMol]:
        """
        Expand a single molecule to stereoisomers

        Parameters
        ----------
        mol : oechem.OEMol
            The molecule to expand

        Returns
        -------
        expanded_mols : List[oechem.OEMol]
            The expanded molecules
        """
        self.logger.debug(f"Molecule Title: {mol.GetTitle()}")
        # set title to molecule name from postera if available
        if self.options.postera_names:
            mol_name_sd = oechem.OEGetSDData(mol, "Molecule Name")
            if mol_name_sd:
                self.logger.debug(f"Molecule Name: {mol_name_sd}")
                mol.SetTitle(mol_name_sd)

        expanded_mols = []
        for enantiomer in oeomega.OEFlipper(mol, self.flipperOpts):
            fmol = oechem.OEMol(enantiomer)
            if self.options.debug:
                smiles = oechem.OEMolToSmiles(fmol)
                self.logger.debug(f"SMILES: {smiles}")
            expanded_mols.append(fmol)
        return expanded_mols

    def expand_mol(self, mol: oechem.OEMol) -> list[oechem.OEMol]:
        return self._expand_mol(mol)

    def expand_structure_file(
        self, infile: Union[str, Path], outfile: Optional[Union[str, Path]] = None
    ) -> list[oechem.OEMol]:
        """
        Expand a structure file to stereoisomers

        Parameters
        ----------
        infile : Union[str, Path]
            The input file
        outfile : Optional[Union[str, Path]], optional
            The output file, by default None

        Returns
        -------
        all_expanded_mols : List[oechem.OEMol]
        """
        _infile = str(infile)
        ifs = oechem.oemolistream()
        if not ifs.open(_infile):
            oechem.OEThrow.Fatal(f"Unable to open {_infile} for reading")

        if outfile:
            _outfile = str(outfile)
            ofs = oechem.oemolostream()
            if not ofs.open(_outfile):
                oechem.OEThrow.Fatal(f"Unable to open {_outfile} for writing")

        all_expanded_mols = []
        for mol in ifs.GetOEMols():
            expanded_mols = self._expand_mol(mol)
            all_expanded_mols.extend(expanded_mols)

        if outfile:
            for mol in all_expanded_mols:
                oechem.OEWriteMolecule(ofs, mol)
            ofs.close()

        ifs.close()
        return all_expanded_mols
