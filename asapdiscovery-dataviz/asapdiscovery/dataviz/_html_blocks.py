from .viz_targets import VizTargets

"""
HTML blocks for visualising poses.
Pose orientation per target can be found by 
"""

class HTMLBlockData:
    @classmethod
    def get_pocket_color(cls, target: str) -> str:
        """
        Get the color for the pocket of a target.
        """
        # need protein name
        protein_name = VizTargets.get_target_name(target, underscore=True)
        return getattr(cls, f"colour_{protein_name}")

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
    def get_orient(cls, target: str) -> str:
        """
        Get the orient array for a target.
        """
        target_ = VizTargets.get_name_underscore(target)
        return getattr(cls, f"HTML_orient_{target_}")

    HTML_orient_SARS_CoV_2_Mpro = """\
    [19.43872410226519, 11.81781888855718, 13.596857815858694, 71.59651614705442, 0.8265984029795093, 0.2984942958173893, 0.4405725231221419, 0.18311768736524137]
    """

    HTML_orient_SARS_CoV_2_Mac1 = """\
    [
    -3.5200155623997147,
    -5.050560643099713,
    -12.108040862949323,
    71.69598666461106,
    0.3887976484684321,
    0.42408332663180826,
    -0.5833075994459622,
    -0.5733602402041021
    ]
    """

    HTML_orient_MERS_CoV_Mpro = """\
    [-40.052249328550225, -96.01719571090237, 2.0532316220967157, 54.161214447021486, 0.10313246368166844, 0.25413742037735987, 0.9514159603894136, 0.13994833623584255]
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