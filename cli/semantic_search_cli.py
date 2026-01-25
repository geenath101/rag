#!/usr/bin/env python3

import argparse

from lib.semantic_search import SemanticSearch
from lib.semantic_search import verify_model
from lib.semantic_search import embed_text
from lib.semantic_search import verify_embeddings
from lib.semantic_search import embed_query_text
from lib.semantic_search import search_query


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
    search.add_argument( "--limit", type=int, default=5, 
        help="Maximum number of results (default: 5)"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.query)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search_query(args.query,5)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()