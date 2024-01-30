"""
HTML blocks for visualising poses
"""


class HTMLBlockData:
    @classmethod
    def get_color_method(cls, method: str) -> str:
        """
        get the coloring method block for the protein viz (subpocket or b-factor)
        """
        return getattr(cls, f"color_method_{method}")

    @classmethod
    def get_missing_residues(cls, missing_residues: set[int] = None) -> str:
        """
        Get the missing residues block for a target.
        """
        if missing_residues is None:
            return ""
        else:
            missing_res_formatted = " or ".join([str(i) for i in missing_residues])
            return f"""
        protein.addRepresentation( 'surface', {{color: 'blue', sele: '{missing_res_formatted}', opacity: 1, side: 'front', surfaceType: 'av', probeRadius: 4.0, scaleFactor: 2.0}} );
        """

    @classmethod
    def get_orient(cls) -> str:
        """
        Get the master orient array.
        """
        return getattr(cls, "HTML_orient_master")

    HTML_orient_master = """\
    [-20.57886926575674, 1.4522825208738426, -17.79941670319449, 77.15778084734526, -0.5549623159570264, -0.11539982591793649, 0.8088242618222696, -0.15653441006175986]
    """

    color_method_subpockets = """\
        protein.removeAllRepresentations();
        protein.addRepresentation( 'surface', {color: pocket_scheme, sele: 'not ligand', opacity: 0.8, side: 'front', surfaceType: 'av'} );
        protein.addRepresentation( 'ball+stick', {sele: 'ligand', opacity: 1, multipleBond: 'symmetric'} );
"""
    color_method_fitness = """\
        protein.removeAllRepresentations();
        protein.addRepresentation( 'surface', {color: pocket_scheme, sele: 'not ligand', opacity: 1, side: 'front', surfaceType: 'av'} );
        protein.addRepresentation( 'ball+stick', {sele: 'ligand', opacity: 1, multipleBond: 'symmetric'} );
"""
