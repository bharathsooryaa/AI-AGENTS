from lib.pdf_loader import load_pdf, split_Document
from lib.db_manager import Qdrant_manager
from lib.chunker import chunker_recursive

# pip install langchain-openai langchain-pinecone pinecone-client gptcache openai python-dotenv

import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import pinecone

# GPTCache-related imports
from langchain_community.cache import GPTCache
from gptcache import cache as gpcache
from gptcache.manager.factory import manager_factory
from gptcache.processor.pre import get_prompt

# ---------- Semantic LLM Cache Setup ----------

def init_gptcache(cache_obj, llm_id: str = "chat-llm"):
    hashed = llm_id.replace(" ", "_")
    data_manager = manager_factory(manager="sqlite", vector_params={"data_dir": f"gptcache_{hashed}"})
    cache_obj.init(pre_embedding_func=get_prompt, data_manager=data_manager)

from langchain_core.globals import set_llm_cache
set_llm_cache(GPTCache(init_gptcache))

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

# ---------- Placeholders ----------

def load_pdf(filepath: str):
    loader = PDFLoader(filepath)
    return loader.load()

def chunker(doc_pages):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(doc_pages)

# ---------- Pinecone + RAG Setup ----------

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east1-gcp")

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
emb = OpenAIEmbeddings(model="text-embedding-3-large")

# Set index name and ensure it's ready
index_name = "rag-index"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=emb.embed_query("test").shape[0], metric="cosine")
index = pinecone.Index(index_name)

vector_store = PineconeVectorStore(index=index, embedding=emb)

# Load documents and ingest into Pinecone
docs = load_pdf("data/my_doc.pdf")
chunks = chunker(docs)
vector_store.add_documents(documents=chunks)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

def cached_rag_query(query: str):
    return qa_chain({"query": query})

# ---------- Example Usage ----------

if __name__ == "__main__":
    q1 = "What is the main purpose of the document?"
    res1 = cached_rag_query(q1)
    print("Answer:", res1["result"])
    for doc in res1["source_documents"]:
        print("-", doc.page_content[:200], "...")

    q2 = "Tell me the document's main goal"
    res2 = cached_rag_query(q2)
    print("Cached Answer:", res2["result"])
