# Proyecto Final: RAG Local - Asistente de Supervivencia y Montañismo

Este proyecto implementa un sistema RAG (Retrieval-Augmented Generation) completamente local utilizando LangChain, ChromaDB y un modelo LLM ejecutado a través de LM Studio. El sistema actúa como un agente inteligente capaz de responder preguntas basadas en manuales de supervivencia en formato PDF, además de contar con herramientas propias (Tools) para realizar cálculos específicos.

## Requisitos Previos

* **Python:** 3.10 o superior.
* **LM Studio:** Instalado y configurado en el equipo.

## Configuración de LM Studio

Para que el agente funcione correctamente, es necesario tener el servidor de inferencia local encendido:

1. Abre LM Studio.
2. Descarga el modelo recomendado: **`Mistral-7B-Instruct-v0.3 (Q4_K_M)`**.
3. Ve a la pestaña **Local Server** y carga el modelo.
4. Asegúrate de que el servidor está escuchando en el puerto `1234` (URL: `http://localhost:1234/v1`).
5. Pulsa **Start Server**.

## Instalación de Dependencias

1. Abre una terminal en la carpeta raíz del proyecto (`rag_project`).
2. Activa tu entorno virtual (si no tienes uno, créalo con `python3 -m venv venv` y actívalo con `source venv/bin/activate`).
3. Instala todas las dependencias necesarias ejecutando:
```bash
pip install -r requirements.txt
```

## Orden de Ejecución

El proyecto consta de dos fases principales que deben ejecutarse en orden:

### 1. Ingesta e Indexación de Documentos

Primero, debemos procesar los documentos PDF/TXT que se encuentran en la carpeta `docs/` para crear nuestra base de datos vectorial.

```bash
python ingest.py
```

*Nota: Este script solo necesita ejecutarse una vez, o cada vez que añadas un nuevo documento a la carpeta `docs/`. La base de datos se guarda de forma persistente en `chroma_db/`.*

### 2. Ejecución del Agente

Una vez indexados los documentos (y con LM Studio corriendo), inicia el asistente interactivo:

```bash
python agent.py
```

## 🎯 Ejemplos de Uso

Una vez que el agente esté en marcha, puedes hacerle preguntas en lenguaje natural. Aquí tienes ejemplos de cómo el sistema utiliza sus diferentes herramientas:

**Ejemplo 1: Búsqueda RAG en los manuales (Tool: consultar_documentos)**
> **Tú:** ¿Qué pasos debo seguir para hacer un refugio básico en el desierto?
> **Asistente:** *Para construir un refugio básico en un desierto cálido, sigue estos pasos: 1. Busca una zona protegida con sombra... 2. Entiérrate en la arena...* (Información extraída directamente de los PDFs).

**Ejemplo 2: Uso de Tool personalizada (Tool: calcular_agua_necesaria)**
> **Tú:** ¿Cuánta agua necesitamos 4 personas para 5 días?
> **Asistente:** *Para 4 personas durante 5 días en condiciones normales, se necesitan 60 litros de agua potable.* (El agente detecta la intención matemática, extrae los números usando expresiones regulares y ejecuta el cálculo en Python sin buscar en los PDFs).

