from enum import Enum

PROTEIN_MAPPING = {
    "SARS-CoV-2-Mpro": "Mpro",
    "MERS-CoV-Mpro": "Mpro",
    "MERS-CoV-Mpro-7ene": "Mpro",
    "MERS-CoV-Mpro-272": "Mpro",
    "SARS-CoV-2-Mac1": "Mac1",
    "SARS-CoV-2-Mac1-monomer": "Mac1",
}

# needs underscores to match attr names
TARGET_MAPPING = {
    "SARS-CoV-2-Mpro": "SARS-CoV-2-Mpro",
    "MERS-CoV-Mpro": "MERS-CoV-Mpro",
    "MERS-CoV-Mpro-7ene": "MERS-CoV-Mpro",
    "MERS-CoV-Mpro-272": "MERS-CoV-Mpro",
    "SARS-CoV-2-Mac1": "SARS-CoV-2-Mac1",
    "SARS-CoV-2-Mac1-monomer": "SARS-CoV-2-Mac1",
}


# enum for allowed targets
# TODO make configurable from YAML file
class VizTargets(Enum):
    viz_SARS_CoV_2_Mpro = "SARS-CoV-2-Mpro"
    viz_MERS_CoV_Mpro = "MERS-CoV-Mpro"
    viz_MERS_CoV_Mpro_7ene = "MERS-CoV-Mpro-7ene"
    viz_MERS_CoV_Mpro_272 = "MERS-CoV-Mpro-272"
    viz_SARS_CoV_2_Mac1 = "SARS-CoV-2-Mac1"
    viz_SARS_CoV_2_Mac1_monomer = "SARS-CoV-2-Mac1-monomer"

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
