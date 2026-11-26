.PHONY: install ingest api test lint eval-retrieval eval-ragas eval-ragas-vector baseline

-include .env
export

install:
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -r requirements-dev.txt

ingest:              ## download SciFact (5,183 abstracts) and build the FAISS index
	python -m ragqa.ingest scifact

api:
	uvicorn ragqa.api:app --reload --port 8000

test:
	pytest

lint:
	ruff check .

eval-retrieval:      ## BM25 vs vector vs hybrid on 300 test queries (no LLM)
	python -m eval.retrieval_eval

eval-ragas:          ## RAGAS faithfulness / relevancy on 300 questions
	python -m eval.ragas_eval

eval-ragas-vector:   ## same, with vector-only retrieval for comparison
	python -m eval.ragas_eval --mode vector

baseline:            ## accept current numbers as the CI baseline (commit eval/baselines/*.json)
	python -m eval.retrieval_eval --update-baseline
	python -m eval.ragas_eval --update-baseline
