from collections import defaultdict


def get_first_value_of_dict_of_lists(d: dict[str, list]) -> dict[str, str]:
    """
    Get the first value of a dictionary of lists
    """
    return {k: v[0] for k, v in d.items()}


def get_list_of_dicts_from_dict_of_lists(d: dict[str, list]) -> list[dict[str, str]]:
    """
    Get the first value of a dictionary of lists
    """
    return [{k: value} for k, v in d.items() for value in v]


def get_dict_of_lists_from_dict_of_str(d: dict[str, str], length=1) -> dict[str, list]:
    """
    Convert a dictionary of strings to a dictionary of lists with length `length`.
    """
    new_dict = defaultdict(list)
    _ = [new_dict[k].append(v) for k, v in d.items() for _ in range(length)]
    return dict(new_dict)


def get_dict_of_lists_from_list_of_dicts(d: list[dict[str, str]]) -> dict[str, list]:
    """
    Convert a list of dictionaries to a dictionary of lists.
    """
    new_dict = defaultdict(list)
    _ = [new_dict[k].append(v) for d_ in d for k, v in d_.items()]
    return dict(new_dict)
