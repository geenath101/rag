from nltk.stem import PorterStemmer
import string

def pre_process_str(moviname:str) -> list[str]:
    stemmer = PorterStemmer()
    mapping = str.maketrans("","",string.punctuation)
    trans_str = moviname.lower().translate(mapping)
    tokenized = trans_str.split()
    return[ stemmer.stem(s) for s in tokenized if s not in STOP_WORD_LIST ]
