def spruce_protein(
    initial_prot,
    return_du=False,
    seqres=None,
    loop_db=None,
    site_residue="HIS:41: :A:0: ",
):
    import oechem, oespruce
    from asapdiscovery.data.openeye import openeye_perceive_residues

    ## Add Hs to prep protein and ligand
    oechem.OEAddExplicitHydrogens(initial_prot)

    ## Set up DU building options
    opts = oespruce.OEMakeDesignUnitOptions()
    opts.SetSuperpose(False)
    ## Options set from John's function ########################################
    ## (https://github.com/FoldingAtHome/covid-moonshot/blob/454098f4255467f4655102e0330ebf9da0d09ccb/synthetic-enumeration/sprint-14-quinolones/00-prep-receptor.py)
    opts.GetPrepOptions().SetStrictProtonationMode(True)
    # set minimal number of ligand atoms to 5, e.g. a 5-membered ring fragment\
    opts.GetSplitOptions().SetMinLigAtoms(5)

    # also consider alternate locations outside binding pocket, important for later filtering
    opts.GetPrepOptions().GetEnumerateSitesOptions().SetCollapseNonSiteAlts(
        True
    )

    # alignment options, only matches are important
    opts.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetSeqAlignMethod(
        oechem.OESeqAlignmentMethod_Identity
    )
    opts.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetSeqAlignGapPenalty(
        -1
    )
    opts.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetSeqAlignExtendPenalty(
        0
    )

    # Both N- and C-termini should be zwitterionic
    # Mpro cleaves its own N- and C-termini
    # See https://www.pnas.org/content/113/46/12997
    opts.GetPrepOptions().GetBuildOptions().SetCapNTermini(False)
    opts.GetPrepOptions().GetBuildOptions().SetCapCTermini(False)
    # Don't allow truncation of termini, since force fields don't have
    #  parameters for this
    opts.GetPrepOptions().GetBuildOptions().GetCapBuilderOptions().SetAllowTruncate(
        False
    )
    # Set Build Loop and Sidechain Opts
    sc_opts = oespruce.OESidechainBuilderOptions()

    loop_opts = oespruce.OELoopBuilderOptions()
    loop_opts.SetSeqAlignMethod(oechem.OESeqAlignmentMethod_Identity)
    loop_opts.SetSeqAlignGapPenalty(-1)
    loop_opts.SetSeqAlignExtendPenalty(0)
    loop_opts.SetLoopDBFilename(loop_db)
    loop_opts.SetBuildTails(True)

    ## Allow for adding residues at the beginning/end if they're missing
    opts.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetBuildTails(
        True
    )

    if loop_db is not None:
        print("Adding loop")
        opts.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetLoopDBFilename(
            loop_db
        )

    ## Structure metadata object
    metadata = oespruce.OEStructureMetadata()

    ## Add SEQRES metadata
    if seqres:
        print("adding seqres")
        all_prot_chains = {
            res.GetExtChainID()
            for res in oechem.OEGetResidues(initial_prot)
            if (res.GetName() != "LIG") and (res.GetName() != "HOH")
        }
        for chain in all_prot_chains:
            seq_metadata = oespruce.OESequenceMetadata()
            seq_metadata.SetChainID(chain)
            seq_metadata.SetSequence(seqres)
            metadata.AddSequenceMetadata(seq_metadata)

    ## Construct spruce filter
    spruce_opts = oespruce.OESpruceFilterOptions()
    spruce = oespruce.OESpruceFilter(spruce_opts, opts)

    ## Spruce!
    from openeye import oegrid

    grid = oegrid.OESkewGrid()

    oespruce.OEBuildLoops(initial_prot, metadata, sc_opts, loop_opts)
    oespruce.OEBuildSidechains(initial_prot, sc_opts)
    oechem.OEPlaceHydrogens(initial_prot)
    spruce.StandardizeAndFilter(initial_prot, grid, metadata)

    ## Re-percieve residues so that atom number and connect records dont get screwed up
    openeye_perceive_residues(initial_prot)

    if return_du:
        dus = list(
            oespruce.OEMakeDesignUnits(
                initial_prot, metadata, opts, site_residue
            )
        )
        try:
            return dus[0]
        except IndexError:
            return initial_prot

    return initial_prot
