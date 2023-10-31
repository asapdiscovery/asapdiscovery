from asapdiscovery.data.postera.manifold_data_validation import TargetTags
from asapdiscovery.data.metadata.resources import seqres_data


def seqres_by_target(target: TargetTags):
    res_prefix = target.value
    resource = f"{res_prefix}_SEQRES.yaml"
    try:
        res = seqres_data.get(resource)
    except KeyError:
        raise ValueError(f"Resource {resource} not found.")
    return res
