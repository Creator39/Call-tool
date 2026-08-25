import sys

sys.path.insert(0,"..")
from src.vocab.filtering import is_plausible_number_token, is_plausible_string_token
from src.vocab.filtering import build_plausible_vocab
from llm_sdk import Small_LLM_Model
from src.vocab.loader import loader_vocab
from src.fsm.value_matcher import allowed_token_ids_for_number
from src.generation.decoding import mask_logits, select_next_token
from src.generation.decoding import decode_vocab
from src.fsm.value_matcher import generate_string_value

def main():
    model = Small_LLM_Model()
    dico_inverse = loader_vocab(model)
    dico_decode = decode_vocab(dico_inverse, model)
    plausible_vocab = build_plausible_vocab(dico_decode, is_plausible_string_token)
    text = generate_string_value(
        full_prefix='prompt": "Reverse the string \'hello\'", "name": "fn_reverse_string","parameters": {"s": "',
        plausible_vocab=plausible_vocab,
        model=model
    )
    print(f"finale value : {text}")

main()
