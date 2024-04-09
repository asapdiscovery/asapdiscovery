from collections import defaultdict


def get_first_value_of_dict_of_lists(d: dict[str, list]) -> dict[str, str]:
    """
    Get the first value of a dictionary of lists

    Example:
    >>> get_first_value_of_dict_of_lists({'a': [1, 2], 'b': [3, 4]})
    {'a': 1, 'b': 3}
    """
    return {k: v[0] for k, v in d.items()}


def get_dict_of_lists_from_dict_of_str(d: dict[str, str], length=1) -> dict[str, list]:
    """
    Convert a dictionary of strings to a dictionary of lists with length `length`.

    Example:
    >>> get_dict_of_lists_from_dict_of_str({'a': '1', 'b': '2'}, length=2)
    {'a': ['1', '1'], 'b': ['2', '2']}
    """
    new_dict = defaultdict(list)
    _ = [new_dict[k].append(v) for k, v in d.items() for _ in range(length)]
    return dict(new_dict)


def get_dict_of_lists_from_list_of_dicts(d: list[dict[str, str]]) -> dict[str, list]:
    """
    Convert a list of dictionaries to a dictionary of lists.

    Example:
    >>> get_dict_of_lists_from_list_of_dicts([{'a': '1', 'b': '2'}, {'a': '3', 'b': '4'}])
    {'a': ['1', '3'], 'b': ['2', '4']}
    """
    new_dict = defaultdict(list)
    _ = [new_dict[k].append(v) for d_ in d for k, v in d_.items()]
    return dict(new_dict)
