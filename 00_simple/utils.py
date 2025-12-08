import numpy as np
import spacy

nlp = spacy.load("en_core_web_lg")

def tokenize(sentence):
    return nlp(sentence)

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [w.lemma_ for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag