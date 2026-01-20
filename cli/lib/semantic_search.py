from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self):
        print(f"loading the module")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # print(f"Model loaded: {model}")
        # print(f"Max sequence length: {model.max_seq_length}")
        # self.model.encode(text)

    def generate_embeddings(self,text):
        if text is None or text == "":
            raise ValueError("Text cannot be empty.")
        return self.model.encode(text)
        

def verify_model():
    sm = SemanticSearch()
    print(f"Model loaded: {repr(sm.model)}")
    print(f"Max sequence length: {sm.model.max_seq_length}")

def embed_text(text):
    sm = SemanticSearch()
    embedding = sm.generate_embeddings(text) 
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")