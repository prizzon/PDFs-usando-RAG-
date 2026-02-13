import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

PDF_DIR = os.path.join("inputs", "pdfs")
VECTOR_DIR = "vectorstore"

def main():
    if not os.path.exists(PDF_DIR):
        raise RuntimeError(f"Pasta não encontrada: {PDF_DIR}")

    loader = PyPDFDirectoryLoader(PDF_DIR)
    docs = loader.load()

    if not docs:
        raise RuntimeError("Nenhum PDF carregado. Coloque PDFs em inputs/pdfs/")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(VECTOR_DIR)

    print(f"✅ PDFs carregados: {len(docs)}")
    print(f"✅ Chunks gerados: {len(chunks)}")
    print(f"✅ Vectorstore salvo em: {VECTOR_DIR}")

if __name__ == "__main__":
    main()

