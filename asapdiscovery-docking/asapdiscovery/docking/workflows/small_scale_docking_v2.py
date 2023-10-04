

def small_scale_docking(filename=None, receptor=None, postera=False, postera_upload=False, postera_molset_name=None, du_cache=None, target=None):

    inputs = SmallScaleDockingInputs( # TODO: make this class that validates inputs
        filename=filename,
        receptor=receptor,
        postera=postera,
        postera_upload=postera_upload,
        postera_molset_name=postera_molset_name,
        target=target,
    ) 


    if postera:
        # load postera
        postera_settings = PosteraSettings()
        postera = PosteraFactory(settings=postera_settings, molecule_set_name=inputs.postera_molset_name)
        query_ligands = postera.pull()
    else:
        # load from file
        molfile = MolFileFactory.from_file(inputs.filename)
        query_ligands = molfile.ligands


    if inputs.receptor_is_dir:
        prepper = DirPrepper()
        prepped_complexes = prepper.prep(inputs.receptor, use_dask=True)

    else:
        complex = Complex.from_pdb(inputs.receptor)
        prepped_complex = PreppedComplex.from_complex(complex)
        prepped_complexes = [prepped_complex]

    # select the best receptor FOR EACH LIGAND, in case of only one prepped_complexes this is trivial
    selector = MCSSelector()
    pairs = selector.select(query_ligands, prepped_complexes, n_select=1, use_dask=True) # TODO: add dask parallelism to selector


    # dock pairs
    docker = POSITDocker()
    results = docker.dock(pairs, use_dask=True)

    # write results to dataframe
    result_df = make_df_from_docking_results(results)
    result_df = rename_output_columns_for_manifold(result_df, target, [DockingResultCols], manifold_validate=True, drop_non_output=True) # TODO:  we can make this nicer for sure, this function is ugly AF
    result_df.to_csv("docking_results.csv", index=False)

    pose_viz = PoseVisualiser(type="pose") # TODO: make class or change old one 
    pose_viz = pose_viz.visualise(results, inputs.target, use_dask=True)
    ... # combine with dataframe 

    fitness_viz = PoseVisualiser(type="fitness") # TODO: make class or change old one 
    fitness_viz = fitness_viz.visualise(results, inputs.target, use_dask=True)
    ... # combine with dataframe

    md_factory = MDFactory() # TODO: make class or change old one
    md_results = md_factory.run(results, use_dask=True)
    ... # combine with dataframe


    if postera_upload:
        postera_uploader = PosteraUploader(settings=settings, molecule_set_name=inputs.postera_molset_name, artifact_cols=[...]) #TODO: make this more compact wrapper for postera uploader
        postera_uploader.upload(df)
    else:
        pass # :)




