import sys
sys.path.insert(0, "../..")

import json
from src.models.functiondef import FunctionCatalog, FunctionDefinition

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

def what_function_are_calling(partial_text: str, catalog: FunctionCatalog) -> FunctionDefinition:
    """
    Given a partial text, determine which function is being called from the catalog.
    """
    for function in catalog.functions:
        if function.name == partial_text:
            return function
    raise ValueError("No matching function found.")

def build_function_definition(catalog: FunctionCatalog) -> str:
    """Build a string representation of the function definitions in the catalog."""
    description = [f"- {f.name}: {f.description}" for f in catalog.functions]
    return "\n".join(description)