import sys

sys.path.insert(0, "../..")

from llm_sdk import Small_LLM_Model

def mask_logits(id_autorized: set[int], logits: list[float]) -> list[float]:
    """
    Mask the logits by setting the values of unauthorized token IDs to negative infinity.

    Args:
        id_autorized (set[int]): A set of authorized token IDs.
        logits (list[float]): A list of logits corresponding to token IDs.
    Returns:
        list[float]: A new list of logits where unauthorized token IDs have been set to negative infinity.
    """
    new_logits = [logits[i] if i in id_autorized else float('-inf') for i in range(len(logits))]
    return new_logits


def select_next_token(logits: list[float]) -> int:
    """
    Seletion the next token based on the provided logits.
    
    Args:
        logits (list[float]): A list of logits corresponding to token IDs.
    Returns:
        int: The index of the token with the highest logit value.
    """
    best : int = logits.index(max(logits))
    return best

def decode_vocab(dico_inverse: dict[int, str], model: Small_LLM_Model) -> dict[int, str]:
    """
    Decode the vocabulary using the provided model.

    Args:
        dico_inverse (dict[int, str]): A dictionary mapping token IDs to their corresponding tokens.
        model (Small_LLM_Model): The language model used for decoding.
    
    Returns:
        dict[int, str]: A dictionary mapping token IDs to their decoded string representations.
    """
    return {
        token_id: model.decode([token_id])
        for token_id in dico_inverse
    }