#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def get_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("Error: OPENAI_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)
    return OpenAI(api_key=key)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pdf-qa",
        description="Ask questions about a PDF using OpenAI RAG",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest a PDF into vectors.json")
    p_ingest.add_argument("pdf", help="Path to the PDF file")

    p_ask = sub.add_parser("ask", help="Ask a question about the ingested PDF")
    p_ask.add_argument("question", help="Your question in quotes")

    args = parser.parse_args()

    if args.command == "ingest":
        from ingest import ingest
        client = get_client()
        try:
            ingest(args.pdf, client)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "ask":
        from query import ask
        client = get_client()
        try:
            answer = ask(args.question, client)
            print(answer)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
