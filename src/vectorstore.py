"""
vectorstore.py
--------------
Module responsible for:
1. Loading raw text documents.
2. Splitting text into semantically coherent, overlapping chunks.
3. Generating dense embeddings via HuggingFace (all-MiniLM-L6-v2).
4. Storing vectors into a persistent local ChromaDB database.
5. Exposing a clean retriever interface for LangGraph nodes.
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Default Paths and Configurations
DEFAULT_DATA_PATH = os.path.join("data", "sample_docs.txt")
DEFAULT_CHROMA_DIR = os.path.join("data", "chroma_db")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


_CACHED_EMBEDDING_MODEL = None
_CACHED_RETRIEVER = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initializes and returns a cached singleton HuggingFace dense embedding model.
    'all-MiniLM-L6-v2' maps sentences & paragraphs to a 384-dimensional dense vector space.
    """
    global _CACHED_EMBEDDING_MODEL
    if _CACHED_EMBEDDING_MODEL is None:
        _CACHED_EMBEDDING_MODEL = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _CACHED_EMBEDDING_MODEL


def load_and_split_documents(
    file_path: str = DEFAULT_DATA_PATH,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    Loads raw text and splits it into overlapping chunks using RecursiveCharacterTextSplitter.
    
    Args:
        file_path: Path to the source text file.
        chunk_size: Maximum character length per chunk (~100 tokens).
        chunk_overlap: Sliding window overlap to preserve boundary context.
        
    Returns:
        List of Document objects ready for vector embedding.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Knowledge document not found at: {file_path}")

    # Step 1: Load raw document
    loader = TextLoader(file_path, encoding="utf-8")
    raw_documents = loader.load()

    # Step 2: Split text recursively by paragraphs, sentences, and words
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(raw_documents)
    return chunks


def build_or_load_vectorstore(
    persist_directory: str = DEFAULT_CHROMA_DIR,
    force_rebuild: bool = False,
) -> Chroma:
    """
    Creates a new persistent Chroma vectorstore or loads an existing one from disk.
    
    Args:
        persist_directory: Folder path where ChromaDB files are saved.
        force_rebuild: If True, re-chunks and re-embeds the source documents.
        
    Returns:
        Chroma vectorstore instance.
    """
    embedding_model = get_embedding_model()

    # If ChromaDB directory already exists and rebuild is not forced, load from disk
    if os.path.exists(persist_directory) and os.listdir(persist_directory) and not force_rebuild:
        print(f"[INFO] Loading existing ChromaDB vectorstore from: {persist_directory}")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model,
        )
        return vectorstore

    print("[INFO] Building ChromaDB vectorstore from source document...")
    chunks = load_and_split_documents()
    print(f"[INFO] Loaded and split document into {len(chunks)} chunks.")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
    )
    print(f"[SUCCESS] ChromaDB vectorstore successfully built and persisted to: {persist_directory}")
    return vectorstore


def get_retriever(k: int = 3, persist_directory: str = DEFAULT_CHROMA_DIR):
    """
    Provides a standard LangChain retriever interface for the LangGraph agent.
    
    Args:
        k: Number of most similar chunks to retrieve per query.
        persist_directory: Path to persistent vector database.
        
    Returns:
        VectorStoreRetriever instance.
    """
    global _CACHED_RETRIEVER
    if _CACHED_RETRIEVER is None:
        vectorstore = build_or_load_vectorstore(persist_directory=persist_directory)
        _CACHED_RETRIEVER = vectorstore.as_retriever(search_kwargs={"k": k})
    return _CACHED_RETRIEVER


if __name__ == "__main__":
    print("--- Testing Vectorstore Ingestion & Semantic Retrieval ---")
    
    # 1. Build or load the vector store
    vstore = build_or_load_vectorstore(force_rebuild=True)
    
    # 2. Test semantic similarity search with an in-domain question
    test_query = "What is the memory requirement for the Nexus-X Protocol?"
    print(f"\nQuery: '{test_query}'")
    
    results = vstore.similarity_search(test_query, k=2)
    print(f"Retrieved {len(results)} relevant chunks:\n")
    for i, doc in enumerate(results, 1):
        print(f"--- Chunk #{i} ---")
        print(doc.page_content.strip())
        print("-" * 40)
