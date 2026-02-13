import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

VECTOR_DIR = "vectorstore"

SYSTEM_PROMPT = """Você é um assistente que responde APENAS usando o conteúdo recuperado dos PDFs.
Se a resposta não estiver nos trechos, diga claramente que não encontrou nos documentos.
Responda em português, de forma objetiva.
"""

def load_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    if not os.path.exists(VECTOR_DIR):
        raise RuntimeError("Vectorstore não encontrado. Rode: python src/ingest.py")
    return FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)

def answer_question(question: str, k: int = 4) -> dict:
    vs = load_vectorstore()
    docs = vs.similarity_search(question, k=k)

    context = "\n\n".join(
        [f"[Fonte: {d.metadata.get('source','PDF')}, pág. {d.metadata.get('page','?')}]\n{d.page_content}"
         for d in docs]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Pergunta: {question}\n\nTrechos recuperados:\n{context}")
    ]

    resp = llm.invoke(messages)

    return {
        "answer": resp.content,
        "sources": [
            {
                "source": d.metadata.get("source", "PDF"),
                "page": d.metadata.get("page", "?")
            } for d in docs
        ]
    }

