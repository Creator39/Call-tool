import sys

sys.path.insert(0, "../..")

from llm_sdk import Small_LLM_Model
from src.fsm.name_matcher import name_matcher
from src.fsm.value_matcher import generate_boolean_value, generate_number_value, generate_string_value
from src.models.load_init import what_function_are_calling
from src.models.functiondef import FunctionCatalog

def generate_function_call(
    prompt: str,
    catalog: FunctionCatalog,
    model: Small_LLM_Model,
    initial_context: str,
    noms_valide: list[str],
    plausible_token_name: dict[int, str],
    plausible_number: dict[int, str],
    plausible_string: dict[int, str],
    plausible_boolean: dict[int, str],
) -> str:
    """
    Generate a function call based on the provided prompt and catalog.

    Args:
        prompt (str): The input prompt for the model.
        catalog (FunctionCatalog): The catalog of available functions.
        model (Small_LLM_Model): The language model to use for generation.
        initial_context (str): The initial context to provide to the model.
        noms_valide (list[str]): A list of valid function names.
        plausible_token_name (dict[int, str]): A mapping of token IDs to plausible token names.
        plausible_number (dict[int, str]): A mapping of token IDs to plausible number tokens.
        plausible_string (dict[int, str]): A mapping of token IDs to plausible string tokens.
        plausible_boolean (dict[int, str]): A mapping of token IDs to plausible boolean tokens.
    
    Returns:
        str: The generated function call as a string.
    """
    full_text_json = f'{{"prompt": "{prompt}", "name": "'
    full_text_context = initial_context + full_text_json
    full_text = name_matcher(full_text_context, noms_valide, plausible_token_name, model)
    function_name = full_text[len(initial_context + full_text_json):-1]
    functions = what_function_are_calling(function_name, catalog)
    full_text += ', "parameters": {'
    for index, (key, value) in enumerate(functions.parameters.items()):
        is_last = index == len(functions.parameters) - 1
        full_text += f'"{key}": '
        if value.type == "number":
            full_text += generate_number_value(full_text, plausible_number, model)
        elif value.type == "string":
            full_text += '"'
            result = generate_string_value(full_text, plausible_string, model) + '"'
            full_text += result
        elif value.type == "boolean":
            full_text += generate_boolean_value(full_text, plausible_boolean, model)
        if not is_last:
            full_text += ", "
        else:
            full_text += "}}"
    return full_text[len(initial_context):]