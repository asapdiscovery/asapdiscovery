

def large_scale_docking(filename=None, frag_dir=None, postera=False, postera_upload=False, postera_molset_name=None, du_cache=None, target=None):

    inputs = LargeScaleDockingInputs( # TODO: make this class that validates inputs
        filename=filename,
        frag_dir=frag_dir,
        postera=postera,
        postera_upload=postera_upload,
        postera_molset_name=postera_molset_name,
        du_cache=du_cache,
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


    # load fragalysis and ligands
    fragalysis = FragalysisFactory.from_dir(inputs.frag_dir, du_cache=inputs.du_cache) # TODO: needs caching added
    prepped_complexes = fragalysis.prepped_complexes

    # define selector and select pairs
    selector = MCSSelector()
    pairs = selector.select(query_ligands, prepped_complexes, n_select=10, use_dask=True) # TODO: add dask parallelism to selector

    # dock pairs
    docker = POSITDocker()
    results = docker.dock(pairs, use_dask=True)

    # write results to dataframe
    result_df = make_df_from_docking_results(results)
    result_df = rename_output_columns_for_manifold(result_df, target, [DockingResultCols], manifold_validate=True, drop_non_output=True) # TODO:  we can make this nicer for sure, this function is ugly AF
    result_df.to_csv("docking_results.csv", index=False)

    if postera_upload:
        postera_uploader = PosteraUploader(settings=settings, molecule_set_name=inputs.postera_molset_name) #TODO: make this more compact wrapper for postera uploader
        postera_uploader.upload(df)
    else:
        pass # :)




