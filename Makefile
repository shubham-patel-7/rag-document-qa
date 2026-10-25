.PHONY: install docs ingest eval test app all
install:
	pip install -r requirements.txt && pip install -e .
docs:
	python scripts/make_sample_docs.py
ingest:
	docqa ingest --docs data/sample_docs --index index/sample
eval:
	docqa eval --docs data/sample_docs --questions data/eval/questions.json --embedders tfidf minilm
test:
	pytest -q
app:
	streamlit run app/streamlit_app.py
all: install docs ingest eval test
