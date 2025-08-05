import types
from lib.pdf_loader import load_pdf, split_Document
from lib.db_manager import Qdrant_manager
from lib.chunker import chunker_recursive
from google import genai
from google.adk.agents import Agent
from dotenv import load_dotenv
load_dotenv()

model = "gemini-2.5-flash-lite"

client = genai.Client()

def preprocess_query(user_query: str, model:str) -> str:
        """
        Preprocess and enhance the user query for better retrieval
        
        Args:
            user_query: Raw user input
            
        Returns:
            Enhanced query for vector search
        """
        enhancement_prompt = f"""
        Analyze this user query and create an optimized search query that would help find relevant documents:
        
        User Query: "{user_query}"
        
        Provide a refined search query that:
        1. Extracts key concepts and keywords
        2. Removes unnecessary words
        3. Includes relevant synonyms or related terms
        4. Maintains the original intent
        
        Return only the optimized search query, nothing else.
        """
        
        try:
            res = response = client.models.generate_content(
            model=model,
            contents=enhancement_prompt,
            config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
            ),
        )
            enhanced_query = response.text.strip()
            return enhanced_query if enhanced_query else user_query
        except Exception as e:
            print(f"Query preprocessing failed: {e}")
            return user_query
