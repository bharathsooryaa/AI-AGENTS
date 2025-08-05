import os
import uuid
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# Load environment variables from .env file
load_dotenv()

def get_embedding_function(model_name="BAAI/bge-small-en-v1.5", device="cpu"):
    """
    Initializes and returns a BGE embedding function from HuggingFace.
    """
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}
    return HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

class Pinecone_manager:
    def __init__(self):
        """
        Initializes the Pinecone client and the embedding function.
        """
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is not set in the environment variables.")

        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        
        # Initialize the embedding function
        self.embeddings = get_embedding_function()
        # The BGE-small model has a dimension of 384
        self.vector_dimension = 384 

    def add_to_pinecone(self, chunks: list[Document], index_name="gemini-embeddings"):
        """
        Checks for a Pinecone index, creates one if it doesn't exist,
        and upserts document chunks.
        """
        # Check if the index already exists
        if index_name not in self.pc.list_indexes().names():
            print(f"Index '{index_name}' not found. Creating a new one...")
            self.pc.create_index(
                name=index_name,
                dimension=self.vector_dimension,
                metric="cosine", # Cosine similarity is a common choice
                spec=ServerlessSpec(
                    cloud='aws', 
                    region='us-west-2'
                )
            )
            print(f"Index '{index_name}' created successfully.")
        else:
            print(f"Index '{index_name}' already exists.")

        # Connect to the specific index
        index = self.pc.Index(index_name)

        # Prepare vectors for bulk upload
        print(f"Preparing {len(chunks)} documents for upload...")
        texts = [chunk.page_content for chunk in chunks]
        vectors = self.embeddings.embed_documents(texts)
        
        # Create a list of vectors in the format Pinecone expects: (id, vector, metadata)
        vectors_to_upsert = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            point_id = str(uuid.uuid4())
            metadata = dict(chunk.metadata)
            metadata['content'] = chunk.page_content
            vectors_to_upsert.append((point_id, vector, metadata))

        # Bulk upload all vectors
        print(f"Uploading {len(vectors_to_upsert)} vectors to Pinecone...")
        # Upsert in batches for better performance with large datasets
        batch_size = 100 
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            index.upsert(vectors=batch)
        
        print(f"✅ Successfully uploaded all documents to index '{index_name}'")

    def clear_database(self, index_name="gemini-embeddings"):
        """
        Deletes a specific Pinecone index.
        """
        try:
            if index_name in self.pc.list_indexes().names():
                self.pc.delete_index(name=index_name)
                print(f"✨ Successfully deleted index '{index_name}'")
            else:
                print(f"Index '{index_name}' does not exist.")
        except Exception as e:
            print(f"❌ Failed to delete index '{index_name}': {e}")

    def search_similar(self, query: str, index_name="gemini-embeddings", limit=5):
        """
        Searches for documents similar to the query vector.
        """
        if index_name not in self.pc.list_indexes().names():
            print(f"Index '{index_name}' does not exist. No search can be performed.")
            return []
            
        index = self.pc.Index(index_name)
        query_vector = self.embeddings.embed_query(query)
        
        results = index.query(
            vector=query_vector,
            top_k=limit,
            include_metadata=True
        )
        
        # Format results to match the desired output: [(payload, score)]
        return [(match['metadata'], match['score']) for match in results['matches']]

# --- Sample Usage to Test Functionality ---
if __name__ == "__main__":
    # 1. Create a Pinecone manager instance
    pinecone_manager = Pinecone_manager()
    INDEX_NAME = "my-test-index"

    # 2. Create some sample documents
    sample_docs = [
        Document(page_content="The policy covers knee surgery for individuals over 40.", metadata={"source": "policy_a.pdf"}),
        Document(page_content="Pre-existing conditions have a waiting period of 24 months.", metadata={"source": "policy_a.pdf"}),
        Document(page_content="Policies active for less than 6 months have limited coverage.", metadata={"source": "policy_b.pdf"}),
        Document(page_content="Coverage is provided for surgeries performed in Pune and Mumbai.", metadata={"source": "policy_b.pdf"}),
    ]

    # 3. Add documents to the Pinecone index
    pinecone_manager.add_to_pinecone(sample_docs, index_name=INDEX_NAME)

    # 4. Perform a similarity search
    print("\n--- Searching for 'knee surgery coverage' ---")
    search_query = "What is the coverage for knee surgery?"
    similar_docs = pinecone_manager.search_similar(search_query, index_name=INDEX_NAME, limit=2)

    if similar_docs:
        for doc, score in similar_docs:
            print(f"Score: {score:.4f}")
            print(f"Content: {doc.get('content', 'N/A')}")
            print(f"Source: {doc.get('source', 'N/A')}\n")
    else:
        print("No similar documents found.")

    # 5. Clean up the database by deleting the index
    print(f"\n--- Cleaning up index '{INDEX_NAME}' ---")
    pinecone_manager.clear_database(index_name=INDEX_NAME)