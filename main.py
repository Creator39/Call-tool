from llm_sdk import Small_LLM_Model
import json

def mask_logits(id_autorized: list[int], logits: list[float]) -> list[float]:
    new_logits = [logits[i] if i in id_autorized else float('-inf') for i in range(len(logits))]
    return new_logits


def main():
    model = Small_LLM_Model()
    text = "What is the sum of 2 and 3?"
    liste = model.encode(text).tolist()[0]

    logist = model.get_logits_from_input_ids(liste)

    best = logist.index(max(logist))

    print(f"le logits le plus elever {best}")
    vocab_path = model.get_path_to_vocab_file()

    with open(vocab_path,'r', encoding='utf-8') as f:
        load_file = json.load(f)

    dico_inverse = {j:i for i, j in load_file.items()}
    candidat = [i for i in range(len(dico_inverse)) if '{' == dico_inverse[i]]
    new_logits = mask_logits(candidat,logist)
    best_match = new_logits.index(max(new_logits))
    print(f"Le meilleur choix {best_match} le texte: {model.decode([best_match])}")

if __name__ == "__main__":
    main()
