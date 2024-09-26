import os

def strip_empty_str_in_list(input_list: list) -> list:
    """cleanup empty strings among the list"""
    return [l for l in input_list if l != "" and l]

def str_to_list(path: str) -> list:
    return strip_empty_str_in_list(path.split(os.path.sep))