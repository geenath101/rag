#!/usr/bin/env python3

import argparse
import os
import json
from lib.semantic_search import SemanticSearch
from lib.semantic_search import verify_model
from lib.semantic_search import embed_text
from lib.semantic_search import verify_embeddings
from lib.semantic_search import embed_query_text
from lib.semantic_search import search_query
from lib.semantic_search import do_random_chunking
from lib.semantic_search import do_semantic_chunking

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify",help="verify the model")
    subparsers.add_parser("verify_embeddings",help="verify embeddings")

    embed = subparsers.add_parser("embed_text", help="embed text")
    embed.add_argument("query",type=str,help="term")

    embed_q = subparsers.add_parser("embedquery", help="embed text")
    embed_q.add_argument("query",type=str,help="term")

    search = subparsers.add_parser("search", help="search")
    search.add_argument("query",type=str,help="query")
    search.add_argument( "--limit", type=int, default=5, help="Maximum number of results (default: 5)")
    
    chunk = subparsers.add_parser("chunk",help="chunk")
    chunk.add_argument("query",type=str,help="text need to be chunked")
    chunk.add_argument("--chunk-size",type=int,default=200)
    chunk.add_argument("--overlap",type=int,default=200)

    sementic_search = subparsers.add_parser("semantic_chunk",help="sementic chunking")
    sementic_search.add_argument("query",type=str,help="query")
    sementic_search.add_argument("--max-chunk-size",type=int,help="max chunk size")
    sementic_search.add_argument("--overlap",type=int,help="overlap")
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.query)
        case "verify_embeddings":
            with open('data/movies.json') as file:
                document = json.load(file)
                verify_embeddings(document)
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search_query(args.query,5)
        case "chunk":
            result = do_random_chunking(args.query,args.chunk_size,args.overlap)
            print(f"Chunking {len(args.query)} characters")
            printResults(result)
        case"semantic_chunk":
            print(f"Semantically chunking {len(args.query)} characters")
            result = do_semantic_chunking(args.query,args.max_chunk_size,args.overlap)
            printResults(result)
        case _:
            parser.print_help()

def printResults(result):
    for i,r in enumerate(result):
        print(f"{i+1}. {" ".join(r)}")   

if __name__ == "__main__":
    main()