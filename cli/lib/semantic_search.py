from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
       
    def generate_embeddings(self,text):
        if text is None or text == "":
            raise ValueError("Text cannot be empty.")
        return self.model.encode(text)
    
    def build_embeddings(self,documents):
        self.documents = documents
        doc_list_as_str  = []
        for d in self.documents:
            self.document_map[d["id"]] = d
            doc_list_as_str.append(f"{d['title']}: {d['description']}")
        self.embeddings = self.model.encode(sentences=doc_list_as_str,show_progress_bar=True)
        np.save("cache/movie_embeddings.npy",self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self,documents):
        self.documents = documents
        for d in self.documents:
            self.document_map[d['id']] = d
        try:
            if os.path.exists("cache/movie_embeddings.npy"):
                self.embeddings = np.load("cache/movie_embeddings.npy")
        except Exception as ex:
            print(f"exception occurred while reading file from cahce {ex}")
        if self.embeddings is not None and len(self.embeddings) == len(self.documents):
            return self.embeddings
        else:
            print(f" document sizes are not equal ")
            return self.build_embeddings(documents)
        
    def search(self, query, limit):
        """
        1. Spaceflight IC-1: An Adventure in Space (score: 0.4406)
        The opening narrative is given by a man in a high ranking military uniform. He tells us the film is ...

        2. Adventureland (score: 0.4150)
        In 1987, James Brennan (Jesse Eisenberg) has two plans. The first plan is to have a summer vacation ...

        3. Odyssey 5 (score: 0.4038)
        The story follows six people on a routine flight of the space shuttle Odyssey, on August 7, 2007: fo...
        """
        search_temp = []
        if self.embeddings is not None:
            emb_query = self.generate_embeddings(query)
            for i in range(len(self.documents)):       
                cos_sim = cosine_similarity(emb_query,self.embeddings[i])
                _doc = self.documents[i]
                _temp = (_doc["id"],cos_sim)
                search_temp.append(_temp)
        else:
            raise ValueError(" Load embeddings first ")
        sorted_list = sorted(search_temp,key=lambda item : item[1],reverse=True)
        for m in range(5):
            _item = sorted_list[m]
            _doc = self.document_map[_item[0]]
            _title = _doc["title"]
            _description = _doc["description"]
            print(f"{m+1}. {_title}")
            print(f"{_description}")
            print(f"")
                  


def search_query(query,limit):
    sm = verify_embeddings()
    return sm.search(query,limit)
        
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)
     
        
def embed_query_text(query):
    sm = SemanticSearch()
    embedding = sm.generate_embeddings(query) 
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def verify_embeddings():
    sm = SemanticSearch()
    with open("data/movies.json") as file:
        documents = json.load(file)
        embeddings = sm.load_or_create_embeddings(documents["movies"])
        print(f"Number of docs:   {len(documents["movies"])}")
        print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
    return sm

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