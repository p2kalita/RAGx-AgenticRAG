"""
Install the required libraries

Load every repository into LangChain.

Load every PDF and CSV into memory.

Split documents into chunks.
"""

###############################################################
# RQ3 - STEP 3, 4, 5 : DOCUMENT INGESTION & CHUNKING
###############################################################

from typing import Dict, List, Tuple
import os
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

DEFAULT_BASE_PATH = "/content/enterprise-presales-knowledge-orchestration/data/enterprise_repository"


def define_repositories(base_path: str = DEFAULT_BASE_PATH) -> Dict[str, str]:
    """
    Construct repository folder paths dictionary.
    """
    repositories = {
        "product": os.path.join(base_path, "product"),
        "architecture": os.path.join(base_path, "architecture"),
        "cloud_adoption": os.path.join(base_path, "cloud_adoption"),
        "governance": os.path.join(base_path, "governance"),
        "commercial": os.path.join(base_path, "commercial"),
        "methodology": os.path.join(base_path, "methodology"),
        "operations": os.path.join(base_path, "operations")
    }
    return repositories


def load_documents_from_repositories(
    repositories: Dict[str, str]
) -> List[Document]:
    """
    Load all PDF and CSV files from each repository and tag metadata with repository name.
    """
    documents = []

    for repo_name, folder in repositories.items():
        if not os.path.exists(folder):
            print(f"Warning: Folder {folder} not found. Skipping.")
            continue

        print(f"\nLoading {repo_name}")
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            if file.endswith(".pdf"):
                loader = PyPDFLoader(path)
                docs = loader.load()
            elif file.endswith(".csv"):
                loader = CSVLoader(path)
                docs = loader.load()
            else:
                continue

            for d in docs:
                d.metadata["repository"] = repo_name

            documents.extend(docs)

    print("Total documents:", len(documents))
    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    print("Chunks created:", len(chunks))
    return chunks


def run_ingestion_and_chunking(
    base_path: str = DEFAULT_BASE_PATH,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Tuple[List[Document], List[Document]]:
    """
    Execute full ingestion and chunking pipeline.
    """
    repos = define_repositories(base_path)
    documents = load_documents_from_repositories(repos)
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    return documents, chunks


if __name__ == "__main__":
    docs, chunks = run_ingestion_and_chunking()

