# LLM Document Processing System for Insurance Claims

A RAG-based solution for the Bajaj Hackathon to automate insurance query processing. This system uses LLMs to parse natural language queries, retrieve relevant clauses from policy documents, and generate structured JSON responses with decisions and justifications.

## Core Functionality

1.  **Data Ingestion & Indexing**: Processes PDFs into a searchable vector database.
2.  **Query Parsing**: Extracts key details (age, procedure, etc.) from user queries.
3.  **Retrieval**: Performs semantic search to find relevant policy clauses.
4.  **Generation**: Evaluates retrieved clauses to produce a final, justified decision in JSON format.

## Tech Stack

-   **Backend**: Python, Flask
-   **LLM/RAG**: LangChain, OpenAI
-   **Vector Store**: FAISS
-   **Document Loading**: PyPDF2

## Quick Start

1.  **Prerequisites**: Python 3.9+, OpenAI API Key.
2.  **Setup**:
    ```bash
    # Navigate to project folder
    cd llm-doc-processing

    # Create environment and install dependencies
    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt

    # Add API key to a new .env file
    echo 'OPENAI_API_KEY="your-api-key"' > .env
    ```
3.  **Add Documents**: Place policy PDFs in the `data/raw/pdfs/` directory.
4.  **Run**:
    ```bash
    python src/main.py
    ```
5.  **Access**: Open `http://127.0.0.1:5000` in your browser.

## Future Improvements

-   Support for `.docx`, `.txt`, and `.eml` files.
-   Implement advanced retrieval strategies like re-ranking.
-   Build a more robust frontend with a framework like React.
-   Add a database to log query history.