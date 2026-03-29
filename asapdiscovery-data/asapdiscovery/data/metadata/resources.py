from importlib import resources

from asapdiscovery.data.services.postera.manifold_data_validation import TargetTags

RESOURCE_BASE_PATH = resources.files("asapdiscovery.data.metadata")

# SEQRES in YAML format
MERS_CoV_Mpro_SEQRES = RESOURCE_BASE_PATH / "master_seqres/mers_cov_mpro_seqres.yaml"
SARS_CoV_2_Mpro_SEQRES = (
    RESOURCE_BASE_PATH / "master_seqres/sars_cov_2_mpro_seqres.yaml"
)
SARS_CoV_2_Mac1_SEQRES = (
    RESOURCE_BASE_PATH / "master_seqres/sars_cov_2_mac1_seqres.yaml"
)
SARS_CoV_2_N_protein_SEQRES = (
    RESOURCE_BASE_PATH / "master_seqres/sars_cov_2_n_protein_seqres.yaml"
)
EV_D68_3Cpro_SEQRES = RESOURCE_BASE_PATH / "master_seqres/ev_d68_3cpro.yaml"
EV_A71_3Cpro_SEQRES = RESOURCE_BASE_PATH / "master_seqres/ev_a71_3cpro.yaml"
EV_A71_2Apro_SEQRES = RESOURCE_BASE_PATH / "master_seqres/ev_a71_2apro.yaml"
ZIKV_NS2B_NS3pro_SEQRES = RESOURCE_BASE_PATH / "master_seqres/zikv_ns2b_ns3pro.yaml"
ZIKV_RdRppro_SEQRES = RESOURCE_BASE_PATH / "master_seqres/zikv_rdrppro.yaml"
DENV_NS2B_NS3pro_SEQRES = RESOURCE_BASE_PATH / "master_seqres/denv_ns2b_ns3pro.yaml"
EV_A71_Capsid_SEQRES = RESOURCE_BASE_PATH / "master_seqres/ev_a71_capsid.yaml"

EV_D68_Capsid_SEQRES = RESOURCE_BASE_PATH / "master_seqres/ev_d68_capsid.yaml"


seqres_data = {
    TargetTags("MERS-CoV-Mpro").value: MERS_CoV_Mpro_SEQRES,
    TargetTags("SARS-CoV-2-Mpro").value: SARS_CoV_2_Mpro_SEQRES,
    TargetTags("SARS-CoV-2-Mac1").value: SARS_CoV_2_Mac1_SEQRES,
    TargetTags("SARS-CoV-2-N-protein").value: SARS_CoV_2_N_protein_SEQRES,
    TargetTags("EV-D68-3Cpro").value: EV_D68_3Cpro_SEQRES,
    TargetTags("EV-A71-3Cpro").value: EV_A71_3Cpro_SEQRES,
    TargetTags("EV-A71-2Apro").value: EV_A71_2Apro_SEQRES,
    TargetTags("ZIKV-NS2B-NS3pro").value: ZIKV_NS2B_NS3pro_SEQRES,
    TargetTags("ZIKV-RdRppro").value: ZIKV_RdRppro_SEQRES,
    TargetTags("DENV-NS2B-NS3pro").value: DENV_NS2B_NS3pro_SEQRES,
    TargetTags("EV-A71-Capsid").value: EV_A71_Capsid_SEQRES,
    TargetTags("EV-D68-Capsid").value: EV_D68_Capsid_SEQRES,
}

# Fitness data in JSON format
SARS_CoV_2_fitness_data = RESOURCE_BASE_PATH / "aa_fitness_sars_cov_2.json"

ZIKV_NS2B_NS3pro_fitness_data = RESOURCE_BASE_PATH / "aa_fitness_zikv_ns2b3.json"

ZIKV_RdRppro_fitness_data = RESOURCE_BASE_PATH / "aa_fitness_zikv_rdrppro.json"

targets_with_fitness_data = [
    TargetTags("SARS-CoV-2-Mpro"),
    TargetTags("SARS-CoV-2-Mac1"),
    TargetTags("SARS-CoV-2-N-protein"),
    TargetTags("ZIKV-NS2B-NS3pro"),
    TargetTags("ZIKV-RdRppro"),
]

# Reference PDB files to align targets to for consistent dataviz
master_structures = {
    "SARS-CoV-2-Mpro": RESOURCE_BASE_PATH / "master_structures/sars_cov_2_mpro.pdb",
    "SARS-CoV-2-Mac1": RESOURCE_BASE_PATH / "master_structures/sars_cov_2_mac1.pdb",
    "SARS-CoV-2-N-protein": RESOURCE_BASE_PATH
    / "master_structures/sars_cov_2_n_protein.pdb",
    "SARS-CoV-2-Mac1-monomer": RESOURCE_BASE_PATH
    / "master_structures/sars_cov_2_mac1.pdb",
    "MERS-CoV-Mpro": RESOURCE_BASE_PATH / "master_structures/mers_cov_mpro.pdb",
    "EV-D68-3Cpro": RESOURCE_BASE_PATH / "master_structures/ev_d68_3cpro.pdb",
    "EV-A71-3Cpro": RESOURCE_BASE_PATH / "master_structures/ev_a71_3cpro.pdb",
    "EV-A71-2Apro": RESOURCE_BASE_PATH / "master_structures/ev_a71_2apro.pdb",
    "ZIKV-NS2B-NS3pro": RESOURCE_BASE_PATH / "master_structures/zikv_ns2b_ns3pro.pdb",
    "ZIKV-RdRppro": RESOURCE_BASE_PATH / "master_structures/zikv_rdrppro.pdb",
    "DENV-NS2B-NS3pro": RESOURCE_BASE_PATH / "master_structures/denv_ns2b_ns3pro.pdb",
    "EV-A71-Capsid": RESOURCE_BASE_PATH / "master_structures/ev_a71_capsid.pdb",
    "EV-D68-Capsid": RESOURCE_BASE_PATH / "master_structures/ev_d68_capsid.pdb",
}


active_site_chains = {
    "SARS-CoV-2-Mpro": "A",
    "SARS-CoV-2-Mac1": "A",
    "SARS-CoV-2-N-protein": "A",
    "SARS-CoV-2-Mac1-monomer": "A",
    "MERS-CoV-Mpro": "A",
    "EV-D68-3Cpro": "A",
    "EV-A71-3Cpro": "A",
    "EV-A71-2Apro": "A",
    "ZIKV-NS2B-NS3pro": "B",
    "ZIKV-RdRppro": "A",
    "DENV-NS2B-NS3pro": "B",
    "EV-A71-Capsid": "A",
    "EV-D68-Capsid": "A",
}


FINTSCORE_PARAMETERS = RESOURCE_BASE_PATH / "fintscore_parameters.yaml"
