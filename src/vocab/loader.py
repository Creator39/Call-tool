import sys

sys.path.insert(0, "../..")

from llm_sdk import Small_LLM_Model
import json

def loader_vocab(llm: Small_LLM_Model) -> dict[int,str]:
    """
    Load the inverted vocabulary from a JSON file using the provided LLM model.

    Args:
        llm (Small_LLM_Model): An instance of the Small_LLM_Model class.
    Returns:
        dict[int,str]: An inverted vocabulary dictionary, mapping token IDs to their corresponding text.
    """
    path = llm.get_path_to_vocab_file()
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return {v: k for k, v in vocab.items()}