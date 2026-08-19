import sys

sys.path.insert(0, "..")

from src.generation.decoding import mask_logits, select_next_token
from src.models.load_init import load_function_catalog

def main():
    load_function = load_function_catalog("data/input/functions_definition.json")
    for i in load_function.functions:
        print(i.name)
if __name__ == "__main__":
    main()
