import pkg_resources

# SEQRES in YAML format
MERS_CoV_Mpro_SEQRES = pkg_resources.resource_filename(
    __name__, "master_seqres/mers_cov_mpro_seqres.yaml"
)
SARS_CoV_2_Mpro_SEQRES = pkg_resources.resource_filename(
    __name__, "master_seqres/sars_cov_2_mpro_seqres.yaml"
)
SARS_CoV_2_Mac1_SEQRES = pkg_resources.resource_filename(
    __name__, "master_seqres/sars_cov_2_mac1_seqres.yaml"
)

# Fitness data in JSON format
SARS_CoV_2_fitness_data = pkg_resources.resource_filename(
    __name__, "aa_fitness_sars_cov_2.json"
)

# Reference PDB files to align targets to for consistent dataviz
master_structures = {
    "SARS-CoV-2-Mpro": pkg_resources.resource_filename(
        __name__, "master_structures/sars_cov_2_mpro.pdb"
    ),
    "SARS-CoV-2-Mac1": pkg_resources.resource_filename(
        __name__, "master_structures/sars_cov_2_mac1.pdb"
    ),
    "SARS-CoV-2-Mac1-monomer": pkg_resources.resource_filename(
        __name__, "master_structures/sars_cov_2_mac1.pdb"
    ),
    "MERS-CoV-Mpro": pkg_resources.resource_filename(
        __name__, "master_structures/mers_cov_mpro.pdb"
    ),
    "EV-D68-3Cpro": pkg_resources.resource_filename(
        __name__, "master_structures/ev_d68_3cpro.pdb"
    ),
    "EV-A71-3Cpro": pkg_resources.resource_filename(
        __name__, "master_structures/ev_a71_3cpro.pdb"
    ),
    "ZIKV-NS2B-NS3pro": pkg_resources.resource_filename(
        __name__, "master_structures/zikv_ns2b_ns3pro.pdb"
    ),
    "DENV-NS2B-NS3pro": pkg_resources.resource_filename(
        __name__, "master_structures/denv_ns2b_ns3pro.pdb"
    ),
}
