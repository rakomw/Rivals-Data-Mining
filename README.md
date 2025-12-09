# Rivals-Data-Mining
Project for Data Mining course at CSUN, scraping and analyzing match records for Marvel Rivals.

## Dataset
- 330,914 matches with 105 features (19 team stats + 86 hero composition)

## Models
- **Tree Methods:** Random Forest, XGBoost, CatBoost (~90% accuracy)
- **SVM:** LinearSVC with RBF kernel approximation
- **Neural Network:** Custom 2-layer feedforward network (NumPy)
- **RAG Pipeline:** Natural language queries via Ollama + ChromaDB

## Usage
```bash
pip install -r requirements.txt
python benchmark.py      # Run tree model comparison
python rag.py            # Start RAG CLI (requires: ollama pull llama3.1 && ollama pull mxbai-embed-large)
```
