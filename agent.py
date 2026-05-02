from langchain.tools import tool
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic import hub
from langchain_openai import ChatOpenAI
from rag_chain import crear_cadena_rag
from datetime import datetime
from langchain_classic.memory import ConversationBufferMemory
from pydantic import SecretStr
import re

LM_STUDIO_URL = "http://localhost:1234/v1"

@tool
def consultar_documentos(pregunta: str) -> str:
    """Consulta la base de conocimiento interna (manuales) para responder preguntas sobre supervivencia, refugios, fuego o nudos."""
    cadena = crear_cadena_rag()
    resultado = cadena.invoke({"query": pregunta})
    return resultado["result"]

@tool
def calcular_agua_necesaria(input_str: str) -> str:
    """Usa esta herramienta cuando el usuario pregunte cuánta agua necesita un grupo.
    Simplemente pásale la cantidad de personas y días."""
    try:
        numeros = re.findall(r'\d+', input_str)
        if len(numeros) >= 2:
            personas = int(numeros[0])
            dias = int(numeros[1])
            litros_totales = personas * dias * 3 # 3 litros por persona al día
            return f"Para {personas} personas durante {dias} días en condiciones normales, se necesitan {litros_totales} litros de agua potable."
        else:
            return "Error: No encontré los dos números (personas y días) en el input."
    except Exception as e:
        return f"Error al calcular: {e}"

def crear_agente():
    llm = ChatOpenAI(
        base_url=LM_STUDIO_URL,
        api_key=SecretStr("lm-studio"),
        model="local-model",
        temperature=0.1
    )
    
    tools = [consultar_documentos, calcular_agua_necesaria]
    prompt = hub.pull("hwchase17/react")
    
    # Creamos el agente
    agente = create_react_agent(llm, tools, prompt)
    
    # Memoria conversacional
    memory = ConversationBufferMemory(memory_key="chat_history")
    
    return AgentExecutor(
        agent=agente, 
        tools=tools, 
        verbose=True, 
        memory=memory,
        handle_parsing_errors=True,
        max_iterations=5 # Previene bucles infinitos
    )

if __name__ == "__main__":
    print("Iniciando Asistente de Supervivencia... Escribe 'salir' para terminar.")
    executor = crear_agente()
    
    while True:
        pregunta = input("\nTú: ")
        if pregunta.lower() in ["salir", "exit"]: 
            break
        try:
            respuesta = executor.invoke({"input": pregunta})
            print(f"\nAsistente: {respuesta['output']}")
        except Exception as e:
            print(f"\nAsistente: Hubo un problema al procesar la respuesta. (Asegúrate de que LM Studio está corriendo)")