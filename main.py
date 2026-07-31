from llm_sdk import Small_LLM_Model
from src.functiondef import FunctionDefinition
from src.functiondef import FunctionCatalog
import json

def mask_logits(id_autorized: set[int], logits: list[float]) -> list[float]:
    new_logits = [logits[i] if i in id_autorized else float('-inf') for i in range(len(logits))]
    return new_logits


def select_next_token(logits: list[float]) -> int:
    best : int = logits.index(max(logits))
    return best

def take_functions_definition(path: str) -> dict:
    with open(path, "r", encoding='utf-8') as f:
        load = json.load(f)
    return load

def load_function_catalog(path: str) -> FunctionCatalog:
    functions_definition = take_functions_definition(path)
    function_catalog = FunctionCatalog.model_validate({ "functions": functions_definition })
    return function_catalog

def main():
    load = load_function_catalog("data/input/functions_definition.json")
    for i in load.functions:
        print(i)
if __name__ == "__main__":
    main()
