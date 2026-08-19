import sys

sys.path.insert(0, "..")

from src.generation.decoding import mask_logits, select_next_token
from llm_sdk import Small_LLM_Model

def filter_names(text_partiel: str, noms_valides: list[str]) -> list[str]:
    """
    Return a list of valid names that start with the given partial text.

    Args:
        text_partiel (str): The partial text to match against.
        noms_valides (list[str]): A list of valid names to filter.
    Returns:
        list[str]: A list of valid names that start with the given partial text.
    """
    return [i for i in noms_valides if i.startswith(text_partiel)]

def next_characters(text_partiel: str, noms_valides: list[str]) -> set[str]:
    """
    Return a set of next characters that can follow the given partial text based on valid names.

    Args:
        text_partiel (str): The partial text to match against.
        noms_valides (list[str]): A list of valid names to filter.
    Returns:
        set[str]: A set of next characters that can follow the given partial text.
    """
    candidats = filter_names(text_partiel, noms_valides)
    return {i[len(text_partiel)] for i in candidats if len(i) > len(text_partiel)}

def allowed_token_ids_for_token(partial_text: str, 
                                name_valids: list[str], 
                                plausible_vocab: dict[int,str]) -> set[int]:
    """
    Return a set of allowed token IDs for completing a partial text.

    Args:
        partial_text (str): The partial text to complete.
        name_valids (list[str]): The list of valid function names.
        vocab_inverse (dict[int,str]): An inverted vocabulary dictionary, mapping token IDs to their corresponding text.
    Returns:
        set[int]: A set of allowed token IDs for completing the partial text.
    """
    max_length = max(len(name) for name in name_valids)
    allowed_set = set()
    for token_id , token_text in plausible_vocab.items():
        if len(partial_text + token_text) <= max_length:
            if filter_names(partial_text + token_text, name_valids):
                allowed_set.add(token_id)
    return allowed_set

def name_matcher(full_prefix: str,
                 noms_valides: list,
                 plausible_vocab: dict,
                 model: Small_LLM_Model) -> str:
    """
    Generate a name that matches the allowed names in the catalog, given a full prefix.

    Args:
        full_prefix (str): The prefix to start the name generation.
        noms_valides (list): A list of valid names to match against.
        plausible_vocab (dict): A dictionary mapping token IDs to their corresponding text.
        model (Small_LLM_Model): The language model used for generating names.
    Returns:
        str: A generated name that matches the allowed names in the catalog.
    """
    partial_text = ""
    while True:
        allowed_token_ids = allowed_token_ids_for_token(partial_text, noms_valides, plausible_vocab)
        
        if not allowed_token_ids:
            break
        full_text = full_prefix + partial_text
        input_ids = model.encode(full_text).tolist()[0]
        logits = model.get_logits_from_input_ids(input_ids)
        masked_logits = mask_logits(allowed_token_ids, logits)
        best_id = select_next_token(masked_logits)
        
        token_text = plausible_vocab.get(best_id,"")
        partial_text += (token_text or "")
    
    return full_prefix + partial_text + '"'