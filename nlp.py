from spacy.matcher import Matcher
import spacy
from code.breeds_list import breeds
from thefuzz import process  # Fuzzy matching for mistakes correction

# !pip install https://huggingface.co/spacy/en_core_web_sm/resolve/main/en_core_web_sm-any-py3-none-any.whl


# model spaCy
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

patterns = [[{"LOWER": breed.lower()}] for breed in breeds.values()]
matcher.add("BREED", patterns)

# breed_quess = []

# text = input("Enter a sentence: ")

def text_input(text):
    breed_quess = []
    doc = nlp(text)
    matches = matcher(doc)

    detected_breeds = [doc[start:end].text for match_id, start, end in matches]

    corrected_breeds = []
    for word in doc:
        match, score = process.extractOne(word.text, list(breeds.values()))
        if score > 85:
            corrected_breeds.append(match)

    if detected_breeds or corrected_breeds:
        # print("Detected breeds (including possible corrections):")
        for breed in set(detected_breeds + corrected_breeds):
            # print("-", breed)
            breed_quess.append(breed)
        return(breed_quess)
    else:
        print("No breeds found in the text.")


