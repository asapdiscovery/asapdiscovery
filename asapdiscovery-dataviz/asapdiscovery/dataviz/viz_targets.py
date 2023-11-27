from enum import Enum

# PUT ALL THIS INTO YAML, READ IN HERE ON IMPORT

PROTEIN_MAPPING = {
    "SARS-CoV-2-Mpro": "Mpro",
    "MERS-CoV-Mpro": "Mpro",
    "SARS-CoV-2-Mac1": "Mac1",
    "SARS-CoV-2-Mac1-monomer": "Mac1",
    "EV-D68-3Cpro": "3Cpro",
    "EV-A71-3Cpro": "3Cpro",
    "ZIKV-NS2B-NS3pro": "NS3pro",
    "DENV-NS2B-NS3pro": "NS3pro",
}

# needs underscores to match attr names
TARGET_MAPPING = {
    "SARS-CoV-2-Mpro": "SARS-CoV-2-Mpro",
    "MERS-CoV-Mpro": "MERS-CoV-Mpro",
    "SARS-CoV-2-Mac1": "SARS-CoV-2-Mac1",
    "SARS-CoV-2-Mac1-monomer": "SARS-CoV-2-Mac1",
    "EV-D68-3Cpro": "EV-D68-3Cpro",
    "EV-A71-3Cpro": "EV-A71-3Cpro",
    "ZIKV-NS2B-NS3pro": "ZIKV-NS2B-NS3pro",
    "DENV-NS2B-NS3pro": "DENV-NS2B-NS3pro",
}


# enum for allowed targets
# TODO make configurable from YAML file
class VizTargets(str, Enum):
    viz_SARS_CoV_2_Mpro = "SARS-CoV-2-Mpro"
    viz_MERS_CoV_Mpro = "MERS-CoV-Mpro"
    viz_SARS_CoV_2_Mac1 = "SARS-CoV-2-Mac1"
    viz_SARS_CoV_2_Mac1_monomer = "SARS-CoV-2-Mac1-monomer"
    viz_EV_D68_3Cpro = "EV-D68-3Cpro"
    viz_EV_A71_3Cpro = "EV-A71-3Cpro"
    viz_ZIKV_NS2B_NS3pro = "ZIKV-NS2B-NS3pro"
    viz_DENV_NS2B_NS3pro = "DENV-NS2B-NS3pro"

    @classmethod
    def get_allowed_targets(cls) -> list[str]:
        return [t.value for t in cls]

    @classmethod
    def get_name_underscore(cls, target: str) -> str:
        t = target.replace("-", "_")
        if t not in [v.replace("-", "_") for v in cls.get_allowed_targets()]:
            raise ValueError(
                f"Target {target} not in allowed targets: {cls.get_allowed_targets()}"
            )
        return t

    @classmethod
    def get_protein_name(cls, target: str, underscore: bool = True) -> str:
        p = PROTEIN_MAPPING[target]
        if underscore:
            p = p.replace("-", "_")
        return p

    @classmethod
    def get_target_name(cls, target: str, underscore: bool = True) -> str:
        t = TARGET_MAPPING[target]
        if underscore:
            t = t.replace("-", "_")
        return t
