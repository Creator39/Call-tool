import sys

sys.path.insert(0,"..")
from src.vocab.filtering import is_plausible_number_token
from src.vocab.filtering import build_plausible_vocab
from llm_sdk import Small_LLM_Model
from src.vocab.loader import loader_vocab
from src.fsm.value_matcher import allowed_token_ids_for_number
from src.generation.decoding import mask_logits, select_next_token

def main():
    model = Small_LLM_Model()
    dico_inverse = loader_vocab(model)
    plausible_vocab = build_plausible_vocab(dico_inverse, is_plausible_number_token)
    print(f"Number of plausible tokens: {len(plausible_vocab)}")
    print(f"Number of total tokens: {len(dico_inverse)}")
if __name__ == "__main__":
    main()
