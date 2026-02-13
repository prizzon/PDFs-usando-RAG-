# 📄🤖 Chatbot baseado em PDFs usando RAG (Embeddings + Busca Vetorial)

Este projeto implementa um chatbot que responde perguntas **com base no conteúdo de PDFs**.
A solução segue o padrão **RAG (Retrieval-Augmented Generation)**:
1) Carrega PDFs  
2) Divide em chunks  
3) Gera embeddings  
4) Indexa em uma base vetorial (FAISS)  
5) Recupera trechos relevantes e gera respostas fundamentadas

## 🧩 Tecnologias
- Python
- LangChain
- FAISS (Vector Store local)
- OpenAI Embeddings + LLM
- Streamlit (chat web)

## 📁 Estrutura
- `inputs/pdfs/`: coloque seus PDFs aqui
- `src/ingest.py`: indexação (PDF → embeddings → FAISS)
- `src/app_streamlit.py`: interface do chat
- `src/rag.py`: lógica de recuperação e resposta

## ▶️ Como executar
1. Instale dependências:
```bash
pip install -r requirements.txt

