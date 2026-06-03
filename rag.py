import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Modelo de embeddings gratuito de HuggingFace
MODELO_EMBEDDINGS = 'sentence-transformers/all-MiniLM-L6-v2'

# Modelo de lenguaje local con Ollama
MODELO_LLM = 'llama3.2'

# Directorio donde se guardan los indices FAISS
DIRECTORIO_INDICES = 'indices'

def cargar_documento(ruta_pdf):
    # Carga y divide el PDF en fragmentos
    cargador  = PyPDFLoader(ruta_pdf)
    documentos = cargador.load()

    divisor = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    fragmentos = divisor.split_documents(documentos)
    return fragmentos

def crear_base_vectorial(fragmentos, nombre):
    # Crea la base de datos vectorial con los fragmentos del documento
    embeddings    = HuggingFaceEmbeddings(model_name=MODELO_EMBEDDINGS)
    base_vectorial = FAISS.from_documents(fragmentos, embeddings)

    # Guarda el indice localmente
    os.makedirs(DIRECTORIO_INDICES, exist_ok=True)
    base_vectorial.save_local(os.path.join(DIRECTORIO_INDICES, nombre))

    return base_vectorial

def crear_cadena_qa(base_vectorial):
    # Configura el LLM local con Ollama
    llm = OllamaLLM(model=MODELO_LLM)

    # Prompt personalizado para respuestas en español
    plantilla = """Usa el siguiente contexto para responder la pregunta en español.
Si no encuentras la respuesta en el contexto, di que no tienes esa informacion.
Se conciso y claro en tu respuesta.

Contexto: {context}

Pregunta: {question}

Respuesta:"""

    prompt    = PromptTemplate.from_template(plantilla)
    retriever = base_vectorial.as_retriever(search_kwargs={'k': 3})

    # Pipeline RAG con la API moderna de LangChain
    cadena = (
        {'context': retriever, 'question': RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return cadena

def procesar_pdf(ruta_pdf):
    # Procesa un PDF y retorna la cadena QA lista para usar
    nombre = os.path.splitext(os.path.basename(ruta_pdf))[0]

    fragmentos     = cargar_documento(ruta_pdf)
    base_vectorial = crear_base_vectorial(fragmentos, nombre)
    cadena         = crear_cadena_qa(base_vectorial)

    return cadena, len(fragmentos)