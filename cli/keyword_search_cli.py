#!/usr/bin/env python3

import argparse
import json
import string
import pickle
import math
from nltk.stem import PorterStemmer
import collections
from concurrent.futures import ProcessPoolExecutor
from collections import Counter

STOP_WORD_LIST=[]
MOVIE_DATABASE=""
BM25_K1 = 1.5
BM25_B = 0.75


def pre_process_str(moviname:str) -> list[str]:
        stemmer = PorterStemmer()
        mapping = str.maketrans("","",string.punctuation)
        trans_str = moviname.lower().translate(mapping)
        tokenized = trans_str.split()
        return[ stemmer.stem(s) for s in tokenized if s not in STOP_WORD_LIST ]

def check_for_match(query:list[str],title:list[str])->bool:
    for q in query:
        for k in title:
            if q in k:
                return True
    return False

def init_database():
    json_file = open("data/movies.json")
    val = json.load(json_file)
    return val["movies"]


class InvertedIndex:
    def __init__(self):
        self.index = collections.defaultdict(set)
        self.docmap = {}
        self.tern_frequencies = {}
        self.doc_length = {}
               
    def __get_documents(self,term):
        print(f"size of the index is {len(self.index)}")
        searched_item=[]
        for token,doc_list in self.index.items():
            if term.lower() == token:
                print(f"found the token {token} and list of tokens {doc_list}" )
                searched_item = list(doc_list)
        return sorted(searched_item)
    
    def get_bm2f_tf(self,doc_id,term,k1=BM25_K1):
        tf = self.get_tf(doc_id,term)
        avg_doc_length = self.get_avg_doc_length()
        length_norm = 1 - BM25_B + BM25_B * (self.doc_length[doc_id] / avg_doc_length)
        tfs = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        #tfs = (tf * (k1 + 1)) / (tf + k1)
        return tfs
    
    def load(self,fileName):
        try:
            with open(fileName,'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            print(f"File not found")

    def save(self):
        print(f"saving documents in cache")
        with open('cache/docmap.pkl','wb') as file:
            pickle.dump(self.docmap,file)
        with open('cache/index.pkl','wb') as file:
            pickle.dump(self.index,file)
        with open('cache/term_frequencies.pkl','wb') as file:
            pickle.dump(self.tern_frequencies,file)
        with open('cache/doc_length.pkl','wb') as file:
            pickle.dump(self.doc_length,file)

    def get_tf(self,doc_id,term):
        #print(f"items in tern_frequency {self.tern_frequencies.get(doc_id)}")
        return self.tern_frequencies.get(doc_id)[term]
    
    def get_bm25_idf(self, q: str) -> float:
        # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
        n = len(self.docmap)
        df = len(self.index.get(q))
        #print(f"total documents {n}")
        #print(f"document frquency {df}")
        return math.log(((n - df + 0.5) / (df + 0.5)) + 1)
    

    def get_avg_doc_length(self) -> float:
        total = sum(self.doc_length.values())
        #for key,val in self.doc_length:
        #    total = sum (val)
        return total/len(self.doc_length)


    # def get_bm2f(self,doc_id,term):
    #     tfs = self.get_bm2f_tf(doc_id,term)
    #     # no need to calculate idf for each doc
    #     idf = self.get_bm25_idf(term)
    #     return tfs * idf
    
    def bm25_search(self, query, limit):
        tq = pre_process_str(query)
        scores = {}
        for q in tq:
            _idf = self.get_bm25_idf(q)
            #print(f"inverse document frequency ====> {_idf}")
            for d in self.index[q]:
                tf = self.get_bm2f_tf(d,q)
                #print(f" term frequency ===> {tf}")
                sc = _idf * tf
                if d not in scores:
                    scores[d] = sc
                else:
                    #print(f"accumilating already existing id ")
                    scores[d] = scores.get(d) + sc
        sorted_scores = sorted(scores.items(),key=lambda item:item[1],reverse=True)
        return sorted_scores[:4]
        

    
    def build(self):
        for m in MOVIE_DATABASE:
            self.docmap[m["id"]] = m
        print(f"preparing tasks")
        tasks =[(m["id"],f"{m['title']} {m['description']}") for m in MOVIE_DATABASE]
        print(f"size of the task list {len(tasks)}")
        with ProcessPoolExecutor() as executor:
            results = executor.map(InvertedIndex._worker_process,tasks)
        for doc_id,tokens in results:
            self.doc_length[doc_id] = len(tokens)
            self.tern_frequencies[doc_id] = Counter(tokens)
            for token in tokens:
                self.index[token].add(doc_id)
        

    @staticmethod
    def _worker_process(task):
        doc_id, text = task
        tokens = pre_process_str(text)
        return doc_id, tokens
    

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI run simpel search by uv run cli/keyword_search_cli.py search 'movie name'")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("build",help="build the local index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="return term frequency")
    tf_parser.add_argument("doc_id",type=int, help="document id")
    tf_parser.add_argument("term", type=str, help="term")

    idf_parser = subparsers.add_parser("idf", help="inverse document frequency")
    idf_parser.add_argument("term", type=str, help="term")

    tfidf_parser = subparsers.add_parser("tfidf", help="term frequency and idf")
    tfidf_parser.add_argument("id",type=int,help="id")
    tfidf_parser.add_argument("term", type=str, help="term")

    bm25_tf_parser = subparsers.add_parser(
    "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using bm25search")
    bm25search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "build":
            invered_index = InvertedIndex()
            invered_index.build()
            invered_index.save()
        case "search":
            invered = InvertedIndex()
            invered.index = invered.load("cache/index.pkl")
            invered.docmap = invered.load("cache/docmap.pkl")
            invered.tern_frequencies = invered.load("cache/term_frequencies.pkl")
            query_list = pre_process_str(args.query)
            #print(f"query index {query_list}")
            total_document_list = []
            for q in query_list:
                 if invered.index.get(q) is not None:
                    for i in invered.index.get(q):
                         total_document_list.append(i)
            total_document_list.sort()
            for m in total_document_list[:5]:
                print(f"{m}. {invered.docmap.get(m)['title']}")
        case "tf":
            inverted_in = InvertedIndex()
            inverted_in.tern_frequencies = inverted_in.load("cache/term_frequencies.pkl")
            print(f"{inverted_in.get_tf(args.doc_id,args.term)}")
        case "idf":
            print(f"IDF term {args.term}")
            inverted_in = InvertedIndex()
            inverted_in.index = inverted_in.load("cache/index.pkl")
            inverted_in.docmap = inverted_in.load("cache/docmap.pkl")
            q = pre_process_str(args.term)
            #print(f"after pre processing {q}")
            document_count = len(inverted_in.index.get(q[0]))
            idf = math.log((len(inverted_in.docmap) + 1) / (document_count + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            i = InvertedIndex()
            i.index = i.load("cache/index.pkl")
            i.docmap = i.load("cache/docmap.pkl")
            i.tern_frequencies = i.load("cache/term_frequencies.pkl")
            q = pre_process_str(args.term)
            id = args.id
            term_freq = i.get_tf(id,q[0])
            #print(f"after pre processing {q}")
            document_count = len(i.index.get(q[0]))
            idf = math.log((len(i.docmap) + 1) / (document_count + 1))
            tf_idf = idf * term_freq
            print(f"TF-IDF score of '{args.term}' in document '{args.id}': {tf_idf:.2f}")
        case "bm25idf":
            i = InvertedIndex()
            i.index = i.load("cache/index.pkl")
            i.doc_length = i.load("cache/doc_length.pkl")
            i.docmap = i.load("cache/docmap.pkl")
            q = pre_process_str(args.term)
            bm25idf = i.get_bm25_idf(q[0])
            print(f"BM25 IDF score of '{q[0]}': {bm25idf:.2f}")
        case "bm25tf":
            i = InvertedIndex()
            i.index = i.load("cache/index.pkl")
            i.tern_frequencies = i.load("cache/term_frequencies.pkl")
            i.doc_length = i.load("cache/doc_length.pkl")
            #print(f"term frequency {i.tern_frequencies}")
            term = pre_process_str(args.term)
            bm25tf = i.get_bm2f_tf(args.doc_id,term[0])
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "bm25search":
            i = InvertedIndex()
            i.index = i.load("cache/index.pkl")
            i.tern_frequencies = i.load("cache/term_frequencies.pkl")
            i.doc_length = i.load("cache/doc_length.pkl")
            i.docmap = i.load("cache/docmap.pkl")
            sorted_scores = i.bm25_search(args.query,5)
            #(15) The Adventures of Mowgli - Score: 7.79
            for item in sorted_scores:
             print(f"- {i.docmap.get(item[0])['title']}")
             print(f"- {item[1]:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    STOP_WORD_LIST = open("data/stopwords.txt").read().splitlines()
    MOVIE_DATABASE = init_database()
    main()