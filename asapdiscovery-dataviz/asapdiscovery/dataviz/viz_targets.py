from enum import Enum

# enum for allowed targets
# TODO make configurable from YAML file
class VizTargets(Enum):
    viz_sars2_mpro = "sars2_mpro"
    viz_mers_mpro = "mers_mpro"
    viz_7ene_mpro = "7ene_mpro"
    viz_272_mpro = "272_mpro"
    viz_sars2_mac1 = "sars2_mac1"

    @staticmethod
    def get_allowed_targets() -> list[str]:
        return [t.value for t in VizTargets]