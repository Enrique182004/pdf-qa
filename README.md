# pdf-qa

AI-powered PDF document Q&A using OpenAI embeddings and GPT-4o-mini RAG pipeline.

## How it works

1. **Ingest** — splits the PDF into text chunks, embeds each one with `text-embedding-3-small`, saves to `vectors.json`
2. **Ask** — embeds your question, finds the top-5 most relevant chunks by cosine similarity, sends them as context to `gpt-4o-mini`

## Setup

```bash
git clone https://github.com/Enrique182004/pdf-qa.git
cd pdf-qa
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add your OpenAI API key
```

## Usage

```bash
# Ingest a PDF
python main.py ingest resume.pdf

# Ask questions
python main.py ask "What programming languages does this person know?"
python main.py ask "Summarize the work experience."
```

## Stack

- `pypdf` — PDF text extraction
- `openai` — `text-embedding-3-small` embeddings + `gpt-4o-mini` completions
- `numpy` — cosine similarity

## Requirements

- Python 3.10+
- OpenAI API key
