#!/usr/bin/env python3

import argparse

from lib.semantic_search import SemanticSearch
from lib.semantic_search import verify_model
from lib.semantic_search import embed_text


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify",help="verify the model")
    
    embed = subparsers.add_parser("embed_text", help="embed text")
    embed.add_argument("query",type=str,help="term")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.query)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()