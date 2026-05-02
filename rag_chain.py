from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from pydantic import SecretStr

DB_PATH = "./chroma_db"
LM_STUDIO_URL = "http://localhost:1234/v1"

PROMPT_TEMPLATE = """
Eres un experto instructor de supervivencia. Usa ÚNICAMENTE el siguiente contexto para responder.
Si la respuesta no está en el contexto, di textualmente: 'No tengo información sobre eso, lo siento.'

Contexto:
{context}

Pregunta: {question}
Respuesta:
"""

def crear_cadena_rag():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 4})
    
    llm = ChatOpenAI(
        base_url=LM_STUDIO_URL,
        api_key=SecretStr("lm-studio"),
        model="local-model",
        temperature=0.1
    )
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE
    )
    
    cadena = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return cadena