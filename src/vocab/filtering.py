import sys
sys.path.insert(0, "../..")

def is_plausible_token_name(token : str,
                            allowed_caracters : set[str]) -> bool:
    """
    Verified if a token name is plausible based on the allowed characters.
    
    Args:
        token (str): The token name to be checked.
        allowed_caracters (set[str]): A set of allowed characters.
    Returns:
        bool: True if the token name is plausible, False otherwise.
    """
    return all(c in allowed_caracters for c in token)

def build_plausible_vocab(dico_inverse: dict[int,str],
                          allowed_caracters : set[str]) -> dict[int,str]:
    """
    Construct a plausible vocabulary by filtering the inverted vocabulary based on allowed characters.

    Args:
        dico_inverse (dict[int,str]): An inverted vocabulary dictionary mapping token IDs to their corresponding text.
        allowed_caracters (set[str]): A set of allowed characters.
    Returns:
        dict[int,str]: A new dictionary containing only the token IDs and their corresponding text that are plausible based on the allowed characters.
    """
    return {
        token_id : token_text
        for token_id, token_text in dico_inverse.items() 
        if is_plausible_token_name(token_text, allowed_caracters)
    }