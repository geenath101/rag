#!/usr/bin/env python3

import argparse
import json
import string
import pickle
from nltk.stem import PorterStemmer
import collections
from concurrent.futures import ProcessPoolExecutor

STOP_WORD_LIST=[]
MOVIE_DATABASE=""

class InvertedIndex:
    def __init__(self):
        self.index = collections.defaultdict(set)
        self.docmap={}
               
    def __get_documents(self,term):
        print(f"size of the index is {len(self.index)}")
        searched_item=[]
        for token,doc_list in self.index.items():
            if term.lower() == token:
                print(f"found the token {token} and list of tokens {doc_list}" )
                searched_item = list(doc_list)
        return sorted(searched_item)
    
    def load(self,fileName):
        try:
            with open(fileName,'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            print(f"File not found")
    
    
    def build(self):
        for m in MOVIE_DATABASE:
            self.docmap[m["id"]] = m['title']
        print(f"preparing tasks")
        tasks =[(m["id"],f"{m['title']} {m['description']}") for m in MOVIE_DATABASE]
        print(f"size of the task list {len(tasks)}")
        with ProcessPoolExecutor() as executor:
            results = executor.map(InvertedIndex._worker_process,tasks)
        for doc_id,unique_tokens in results:
            print(f"adding document {doc_id}")
            for token in unique_tokens:
                self.index[token].add(doc_id)
        print(f"saving documents in cahce")
        with open('cache/docmap.pkl','wb') as file:
            pickle.dump(self.docmap,file)
        with open('cache/index.pkl','wb') as file:
            pickle.dump(self.index,file)
        #print(f"First document for token 'merida' = {self.__get_documents("merida")[0]}")

    @staticmethod
    def _worker_process(task):
        doc_id, text = task
        tokens = pre_process_str(text)
        return doc_id, set(tokens)
    

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

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI run simpel search by uv run cli/keyword_search_cli.py search 'movie name'")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    subparsers.add_parser("build",help="build the local index")
    search_parser.add_argument("query", type=str, help="Search query")
    args = parser.parse_args()

    match args.command:
        case "build":
            invered_index = InvertedIndex()
            invered_index.build()
        case "search":
            #print(f"Searching for: {args.query}")
            #json_file = open("data/movies.json") 
            #val = json.load(json_file)
            #movie_list = val["movies"]
            #for e in MOVIE_DATABASE:
                #if check_for_match(pre_process_str(args.query),pre_process_str(e["title"])):
                    #search_items.append(e)
            invered = InvertedIndex()
            invered.index = invered.load("cache/index.pkl")
            invered.docmap = invered.load("cache/docmap.pkl")
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
        case _:
            parser.print_help()


if __name__ == "__main__":
    STOP_WORD_LIST = open("data/stopwords.txt").read().splitlines()
    MOVIE_DATABASE = init_database()
    main()