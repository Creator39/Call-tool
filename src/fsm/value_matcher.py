import sys

sys.path.insert(0, "..")

from src.generation.decoding import mask_logits, select_next_token
from llm_sdk import Small_LLM_Model
from src.fsm.name_matcher import name_matcher

def allowed_chars_for_number(partial_value: str) -> set[str]:
    """
    allowed characters for a number token, based on the current partial value.

    Args:
        partial_value (str): The current partial value of the number token.
    
    Returns:
        set[str]: A set of allowed characters for the number token.
    """
    if not partial_value:
        allowed_chars = set("0123456789-")
    else:
        allowed_chars = set("0123456789,.}")
    return allowed_chars

def allowed_token_ids_for_number(partial_value: str, plausible_vocab: dict[int, str]) -> set[int]:
    """
    Allow only the token ids that correspond to plausible number tokens based on the provided plausible vocabulary.

    Args:
        partial_value (str): The partial value to be matched.
        plausible_vocab (dict[int, str]): A dictionary mapping token ids to token texts.
    
    Returns:
        set[int]: A set of allowed token ids that correspond to plausible number tokens.
    """
    allowed_caracters = allowed_chars_for_number(partial_value)
    return {
        token_id
        for token_id, token_text in plausible_vocab.items()
        if all(c in allowed_caracters for c in token_text)
    }

def generate_number_value(
    full_prefix: str,
    plausible_vocab: dict[int, str],
    model: Small_LLM_Model
    )-> str:
    """
    Generate a number value that matches the allowed characters for a number token, given a full prefix.

    Args:
        full_prefix (str): The full prefix to be used for generating the number value.
        plausible_vocab (dict[int, str]): A dictionary mapping token ids to token texts.
        model (Small_LLM_Model): The language model to be used for generating the number value
    
    Returns:
        str: The generated number value that matches the allowed characters for a number token.
    """
    partial_value = ""
    while True:
        allowed_token_ids = allowed_token_ids_for_number(partial_value, plausible_vocab)
        full_text = full_prefix + partial_value
        input_ids = model.encode(full_text).tolist()[0]
        logits = model.get_logits_from_input_ids(input_ids)
        masked_logits = mask_logits(allowed_token_ids, logits)
        best_id = select_next_token(masked_logits)
        token_text = plausible_vocab.get(best_id, "")
        
        if "," in token_text or "}" in token_text :
            break
        
        partial_value += token_text
    return partial_value

def generate_string_value(
        full_prefix: str,
        plausible_vocab: dict[int, str],
        model: Small_LLM_Model,
    ) -> str:
    """
    Generate a string value for parameter of type string json and semantically correct with function definition.
    
    Args:
        full_prefix (str): The prefix string to condition the generation on.
        plausible_vocab (set): A set of plausible tokens for the generation.
        model (Small_LLM_Model): The language model used for generation.
    Returns:
        str: The generated string value.
    """
    partial_value = ""
    allowed_token = set(plausible_vocab)
    while True:
        full_text = full_prefix + partial_value
        input_ids = model.encode(full_text).tolist()[0]
        logits = model.get_logits_from_input_ids(input_ids)
        mask = mask_logits(allowed_token, logits)
        best_id = select_next_token(mask)
        token_text = plausible_vocab.get(best_id, "")
        if '"' in token_text:
            break
        partial_value += (token_text or "")
    return partial_value

def generate_boolean_value(full_prefix: str,
                           plausible_vocab: dict[int, str],
                           model: Small_LLM_Model) -> str:
    """
    Generate a boolean value ("true" or "false") based on the given full_prefix.

    Args:
        full_prefix (str): The prefix string to guide the generation.
        plausible_vocab (dict[int, str]): A dictionary mapping token IDs to plausible boolean tokens.
        model (Small_LLM_Model): The language model used for generation.

    Returns:
        str: The generated boolean value ("true" or "false").
    """
    raw_text = name_matcher(full_prefix,["true", "false"], plausible_vocab, model)
    return raw_text[len(full_prefix):-1]