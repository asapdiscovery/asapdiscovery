import pkg_resources

# import each file
MERS_CoV_Mpro_SEQRES = pkg_resources.resource_filename(
    __name__, "mpro_mers_seqres.yaml"
)
SARS_CoV_2_Mpro_SEQRES = pkg_resources.resource_filename(
    __name__, "mpro_sars2_seqres.yaml"
)
SARS_CoV_2_Mac1_SEQRES = pkg_resources.resource_filename(
    __name__, "mac1_sars2_seqres.yaml"
)

SARS_CoV_2_fitness_data = pkg_resources.resource_filename(
    __name__, "aa_fitness_sars_cov_2.json"
)
master_structures = {
    "SARS-CoV-2-Mpro": pkg_resources.resource_filename(
        __name__, "master_sars_cov_2_mpro.pdb"
    ),
    "SARS-CoV-2-Mac1": pkg_resources.resource_filename(
        __name__, "master_sars_cov_2_mac1.pdb"
    ),
    "SARS-CoV-2-Mac1-monomer": pkg_resources.resource_filename(
        __name__, "master_sars_cov_2_mac1.pdb"
    ),
    "MERS-CoV-Mpro": pkg_resources.resource_filename(
        __name__, "master_mers_cov_mpro.pdb"
    ),
}
