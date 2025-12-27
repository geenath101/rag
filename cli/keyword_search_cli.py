#!/usr/bin/env python3

import argparse
import json
import string
import pickle
from nltk.stem import PorterStemmer


STOP_WORD_LIST=[]
MOVIE_DATABASE=""

class InvertedIndex:
    index={}
    docmap={}
    #def __init__(self,index,docmap):
        #self.index = index
        #self.docmap = docmap

    #improve this more    
    def __add_document(self,doc_id,text):
        print(f"adding documents {doc_id}")
        torkenized_list = pre_process_str(text)
        found = False
        for t in torkenized_list:
            if len(self.index) > 0:
                #print(f"size of the index is {len(self.index)}")
                #print(f"idex values are {self.index}")
                for token , doc_list in self.index.items():
                    if t == token:
                        #print(f"match found with token {token}")
                        found = True
                        doc_list.add(doc_id)
                        #print(f"added document to the list {doc_list}")
            if not found or len(self.index) == 0:
                self.index[t] = set()
                self.index[t].add(doc_id)
                found=False

    def __get_documents(self,term):
        print(f"size of the index is {len(self.index)}")
        for token,doc_list in self.index.items():
            if term.lower() == token:
                return doc_list.sort()
        return []
    
    def build(self):
        for m in MOVIE_DATABASE:
            self.docmap[m["id"]] = m
            contat_string = f"{m['title']} {m['description']}"
            self.__add_document(m["id"],contat_string)
        print(f"saving documents in cahce")
        with open('cache/docmap.pkl','wb') as file:
            pickle.dump(self.docmap,file)
        with open('cache/index.pkl','wb') as file:
            pickle.dump(self.index,file)
        print(self.index)


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
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
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
            print(f"Searching for: {args.query}")
            search_items=[]
            #json_file = open("data/movies.json") 
            #val = json.load(json_file)
            #movie_list = val["movies"]
            for e in MOVIE_DATABASE:
                if check_for_match(pre_process_str(args.query),pre_process_str(e["title"])):
                    search_items.append(e)
            for m in search_items[:5]:
                print(f"{m["id"]}. {m["title"]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    STOP_WORD_LIST = open("data/stopwords.txt").read().splitlines()
    MOVIE_DATABASE = init_database()
    main()