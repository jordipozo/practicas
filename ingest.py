from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

DOCS_PATH = "./docs"
DB_PATH = "./chroma_db"

def cargar_documentos(ruta):
    documentos = []
    for fichero in os.listdir(ruta):
        ruta_completa = os.path.join(ruta, fichero)
        if fichero.endswith(".pdf"):
            loader = PyPDFLoader(ruta_completa)
            documentos.extend(loader.load())
        elif fichero.endswith(".txt"):
            loader = TextLoader(ruta_completa, encoding="utf-8")
            documentos.extend(loader.load())
    return documentos

def indexar(documentos):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, # Ajustado para mantener párrafos de manuales enteros
        chunk_overlap=100 # Mayor solapamiento para no cortar frases críticas
    )
    chunks = splitter.split_documents(documentos)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = Chroma.from_documents(chunks, embeddings, persist_directory=DB_PATH)
    print(f"Indexados {len(chunks)} chunks en {DB_PATH}")
    return db

if __name__ == "__main__":
    docs = cargar_documentos(DOCS_PATH)
    indexar(docs)