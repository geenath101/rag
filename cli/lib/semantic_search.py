from sentence_transformers import SentenceTransformer
import numpy as np
import os
import re


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
    
    def build_embeddings(self):
        doc_list_as_str  = []
        for d in self.documents["movies"]:
            self.document_map[d["id"]] = d
            doc_list_as_str.append(f"{d['title']}: {d['description']}")
        self.embeddings = self.model.encode(sentences=doc_list_as_str,show_progress_bar=True)
        np.save("cache/movie_embeddings.npy",self.embeddings)
        return self.embeddings
    
    def load_doc_and_doc_map(self,documents):
        self.documents = documents
        for d in self.documents:
            self.document_map[d['d']] = d
    
    def load_or_create_embeddings(self):
        try:
            if os.path.exists("cache/movie_embeddings.npy"):
                self.embeddings = np.load("cache/movie_embeddings.npy")
        except Exception as ex:
            print(f"exception occurred while reading file from cahce {ex}")
        if self.embeddings is not None and len(self.embeddings) == len(self.documents):
            return self.embeddings
        else:
            print(f" document sizes are not equal ")
            return self.build_embeddings()
        
    def search(self, query, limit):
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
                  

def do_semantic_chunking(query,chunk_size,overlap_value):
    sentence_list = re.split(r"(?<=[.!?])\s+",query)
    final_list = []
    do_chunking_and_overlap(final_list,sentence_list,chunk_size,overlap_value)
    return final_list

def do_chunking_and_overlap(final_list,sentence_list,chunk_size,overlap_value):
    group_list = []
    for itemList in chunk_list(sentence_list,chunk_size):
        group_list.append(itemList)
    overlaped_list = do_overlap(group_list,overlap_value)
    for item in overlaped_list:
        if len(item) > chunk_size:
            do_chunking_and_overlap(final_list,item,chunk_size,overlap_value)
        else:
            final_list.append(item)
   
            

def do_random_chunking(query,chunk_size,overlap_value):
    qury_list = query.split(" ")
    """
        imprative approch to grouping
    """
    # f, h = divmod(len(qury_list),chunk_size)
    # group_count = f + 1 if h > 0 else f
    # group_list = []
    # print(f"group count is ... {group_count}")
    # for gnumber in range(group_count):
    #     g = []
    #     starting_index = gnumber * chunk_size
    #     for i in range(chunk_size):
    #         access_index = starting_index + i
    #         if access_index < len(qury_list):
    #             g.append(qury_list[access_index])
    #     group_list.append(g)
    # print(f" group list ......... {group_list}")
    group_list = []
    for c in chunk_list(qury_list,chunk_size):
        group_list.append(c)
    over_lapped_items = do_overlap(group_list,overlap_value)
    return over_lapped_items
    

def chunk_list(data_list,chunk_size):
    for i in range(0,len(data_list),chunk_size):
        yield data_list[i: i + chunk_size]

def do_overlap(result,overlap_value):
    complete_list = []
    for i,r in enumerate(result):
        overlap_items = []
        if overlap_value is not None and overlap_value > 0 and i > 0:
            _p = result[i-1]
            overlap_items = _p[-overlap_value:]    
        overlap_items.extend(r)
        complete_list.append(overlap_items)
        
    return complete_list


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

def verify_embeddings(documents):
    sm = SemanticSearch()
    sm.load_doc_and_doc_map(documents)
    embeddings = sm.load_or_create_embeddings()
    print(f"Number of docs:   {len(documents["movies"])}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
    

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