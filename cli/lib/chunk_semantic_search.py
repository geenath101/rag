from semantic_search import SemanticSearch

class ChunkedSemanticSearch(SemanticSearch):

    def __init__(self,model_name="all-miniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings =  None
        self.chunk_metadata = None

    def build_chunk_embeddings(self,documents):
        self.load_doc_and_doc_map(documents)
        chunk_list = []
        chunk_meta_data = {}
        for d in self.documents:





# def main() -> None:
#     ch = ChunkedSemanticSearch()
#     ch.build_chunk_embeddings

# if __name__ == "__main__" :
#     main()
