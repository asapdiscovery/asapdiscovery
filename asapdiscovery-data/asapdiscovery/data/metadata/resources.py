import pkg_resources

from asapdiscovery.data.postera.manifold_data_validation import TargetTags


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
EV_D68_3Cpro_SEQRES = pkg_resources.resource_filename(
    __name__, "master_seqres/ev_d68_3cpro.yaml"
)
EV_A71_3Cpro_SEQRES = pkg_resources.resource_filename(
    __name__, "master_seqres/ev_a71_3cpro.yaml"
)
ZIKV_NS2B_NS3pro_SEQRES = pkg_resources.resource_filename(
    __name__, "master_seqres/zikv_ns2b_ns3pro.yaml"
)
DENV_NS2B_NS3pro_SEQRES = pkg_resources.resource_filename(
    __name__, "master_seqres/denv_ns2b_ns3pro.yaml"
)

seqres_data = {
    TargetTags("MERS-CoV-Mpro").value: MERS_CoV_Mpro_SEQRES,
    TargetTags("SARS-CoV-2-Mpro").value: SARS_CoV_2_Mpro_SEQRES,
    TargetTags("SARS-CoV-2-Mac1").value: SARS_CoV_2_Mac1_SEQRES,
    TargetTags("EV-D68-3Cpro").value: EV_D68_3Cpro_SEQRES,
    TargetTags("EV-A71-3Cpro").value: EV_A71_3Cpro_SEQRES,
    TargetTags("ZIKV-NS2B-NS3pro").value: ZIKV_NS2B_NS3pro_SEQRES,
    TargetTags("DENV-NS2B-NS3pro").value: DENV_NS2B_NS3pro_SEQRES,
}

# Fitness data in JSON format
SARS_CoV_2_fitness_data = pkg_resources.resource_filename(
    __name__, "aa_fitness_sars_cov_2.json"
)


targets_with_fitness_data = [
    TargetTags("SARS-CoV-2-Mpro"),
    TargetTags("SARS-CoV-2-Mac1"),
]

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
