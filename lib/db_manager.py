from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore as Qdrant
import os
from dotenv import load_dotenv
from lib.pdf_loader import get_embedding_function
from qdrant_client.models import VectorParams, Distance
from qdrant_client import QdrantClient

load_dotenv()

class Qdrant_manager:
    def __init__(self):
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        if not self.qdrant_api_key:
            raise ValueError("QDRANT_API_KEY is not set in the environment variables.")
        
        self.qdrant_client = QdrantClient(
            url="https://d5a5f5ce-ffe6-4b64-b58e-361b6ec60509.us-west-2-0.aws.cloud.qdrant.io",
            api_key=self.qdrant_api_key,
        )

    def add_to_qdrant(self, chunks: list[Document], collection_name="gemini_embeddings"):
        # Create collection if it doesn't exist
        try:
            self.qdrant_client.get_collection(collection_name)
            print(f"Collection '{collection_name}' already exists.")
        except Exception:
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
            print(f"Collection '{collection_name}' created successfully.")

        # Add IDs to chunks
        chunks_with_ids = self.calculate_chunk_ids(chunks)
        
        # Create vector store and add all documents at once
        embeddings = get_embedding_function()
        vector_store = Qdrant(
            client=self.qdrant_client,
            collection_name=collection_name,
            embedding=embeddings,
        )

        print(f"Adding {len(chunks_with_ids)} documents to Qdrant...")
        vector_store.add_documents(chunks_with_ids)
        print(f"✅ Successfully added all documents to collection '{collection_name}'")

    def calculate_chunk_ids(self, chunks):
        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"

            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id
            chunk.metadata["id"] = chunk_id

        return chunks

    def clear_database(self, collection_name="gemini_embeddings"):
        try:
            self.qdrant_client.delete_collection(collection_name=collection_name)
            print(f"✨ Successfully deleted collection '{collection_name}'")
        except Exception as e:
            print(f"❌ Failed to delete collection '{collection_name}': {e}")