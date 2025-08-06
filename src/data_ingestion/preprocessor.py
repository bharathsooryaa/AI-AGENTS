from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
from src.data_ingestion import chunker
load_dotenv()


def split_Document(document: list[Document],type = 'agentic'):
    if type == 'agentic':
        return chunker.chunker_agentic(document)
    if type == 'recursive':
        return chunker.chunker_recursive(document)
def get_embedding_function(mod = 1):
    if mod == 0:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
    elif mod == 1:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            max_retries=3,
            timeout=120,  # Increased timeout to handle potential delays
        )
    else:
        raise ValueError("Invalid embedding mode specified.")
    return embeddings