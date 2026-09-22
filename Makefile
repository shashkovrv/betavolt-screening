.PHONY: test app data train db clean

test:
	pytest tests/

app:
	streamlit run app/main.py

db:
	python -m src.database.repository

pareto:
	python -m src.screening.pareto

shap:
	python -m src.models.explainability

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +