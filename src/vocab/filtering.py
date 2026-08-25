import sys
sys.path.insert(0, "../..")

from typing import Callable

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

def is_plausible_number_token(token: str) -> bool:
    """
    Checks if a token is a plausible number token.

    Arguments:
        token (str): The token to check.
    
    Returns:
        bool: True if the token is a plausible number token, False otherwise.
    """
    allowed_chars = set("0123456789.-,}")
    return all(char in allowed_chars for char in token)

def build_plausible_vocab(dico_inverse: dict[int,str],
                          is_plausible_token: Callable[[str], bool]) -> dict[int,str]:
    """
    Construct a plausible vocabulary by filtering the inverted vocabulary based on allowed characters.

    Args:
        dico_inverse (dict[int,str]): An inverted vocabulary dictionary mapping token IDs to their corresponding text.
        is_plausible_token (Callable[[str], bool]): A function that checks if a token is plausible based on allowed characters.
    Returns:
        dict[int,str]: A new dictionary containing only the token IDs and their corresponding text that are plausible based on the allowed characters.
    """
    return {
        token_id : token_text
        for token_id, token_text in dico_inverse.items() 
        if is_plausible_token(token_text)
    }

def is_plausible_string_token(token: str) -> bool:
    """
    Only double quotes are not allowed in string tokens, as they are used to delimit the string.
    
    Args:
        token (str): The token to check.
    Returns:
        bool: True if the token is a plausible string token, False otherwise.
    """
    forbidden_chars = set('\\”')
    return all(char not in forbidden_chars for char in token)