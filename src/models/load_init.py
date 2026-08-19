import sys
sys.path.insert(0, "../..")

import json
from src.models.functiondef import FunctionCatalog

def take_functions_definition(path: str) -> dict:
    """
    Load the function definitions from a JSON file.
    
    Args:
        path (str): The path to the JSON file containing the function definitions.
    Returns:
        dict: A dictionary containing the function definitions loaded from the JSON file.
    """
    with open(path, "r", encoding='utf-8') as f:
        load = json.load(f)
    return load

def load_function_catalog(path: str) -> FunctionCatalog:
    """
    Load the function catalog from a JSON file.
    
    Args:
        path (str): The path to the JSON file containing the function definitions.
    Returns:
        FunctionCatalog: An instance of the FunctionCatalog class containing the loaded functions.s
    """
    functions_definition = take_functions_definition(path)
    function_catalog = FunctionCatalog.model_validate({ "functions": functions_definition })
    return function_catalog

def load_vocab_inverse(path: str) -> dict:
    """
    Load the inverted vocabulary from a JSON file.

    Args:
        path (str): The path to the JSON file containing the vocabulary.

    Returns:
        dict: An inverted vocabulary dictionary, mapping token IDs to their corresponding text.
    """
    with open(path, "r", encoding='utf-8') as f:
        vocab = json.load(f)
    return {v: k for k, v in vocab.items()}