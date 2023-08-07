import pkg_resources

# import each file
MERS_CoV_Mpro_SEQRES = pkg_resources.resource_filename(__name__, "mpro_mers_seqres.yaml")
SARS_CoV_2_Mpro_SEQRES = pkg_resources.resource_filename(__name__, "mpro_sars2_seqres.yaml")
SARS_CoV_2_Mac1_SEQRES = pkg_resources.resource_filename(__name__, "mac1_sars2_seqres.yaml")
