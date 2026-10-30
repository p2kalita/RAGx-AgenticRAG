"""
Generate embeddings.

Build the vector database, with  one vector database for each enterprise knowledge repository.

Verify that every repository has been successfully indexed before retrieval.
"""

###############################################################
# RQ3 - STEP 6, 7, 8 : EMBEDDINGS & VECTOR DATABASES
###############################################################

from typing import List, Dict, Any, Tuple
from collections import defaultdict
import os
import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def initialize_embedding_model(
    model_name: str = "text-embedding-3-small"
) -> OpenAIEmbeddings:
    """
    Initialize OpenAI embeddings model using OPENAI_API_KEY from environment or Colab userdata.
    """
    try:
        from google.colab import userdata
        api_key = userdata.get('OPENAI_API_KEY')
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    except ImportError:
        pass

    embedding_model = OpenAIEmbeddings(
        model=model_name
    )
    return embedding_model


def build_repository_vector_databases(
    chunks: List[Document],
    embedding_model: OpenAIEmbeddings,
    base_persist_dir: str = "./vectordb"
) -> Dict[str, Chroma]:
    """
    Create one Chroma vector database for each knowledge repository.

    Parameters
    ----------
    chunks : List[Document]
        Document chunks tagged with repository metadata.
    embedding_model : OpenAIEmbeddings
        Initialized embedding model.
    base_persist_dir : str
        Directory to persist Chroma databases.

    Returns
    -------
    Dict[str, Chroma]
        Dictionary mapping repository names to Chroma database instances.
    """
    vector_dbs = {}

    # Group chunks by repository
    repository_chunks = defaultdict(list)
    for chunk in chunks:
        repository = chunk.metadata["repository"]
        repository_chunks[repository].append(chunk)

    print("=" * 80)
    print("Building Repository Vector Databases")
    print("=" * 80)

    for repository, docs in repository_chunks.items():
        persist_dir = f"{base_persist_dir}/{repository}"

        db = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            persist_directory=persist_dir
        )

        vector_dbs[repository] = db
        print(f"{repository:20s} -> {len(docs):5d} chunks")

    print("\nVector databases created successfully.")
    return vector_dbs


def verify_vector_databases(vector_dbs: Dict[str, Chroma]) -> pd.DataFrame:
    """
    Verify indexed chunk count per repository vector database.
    """
    print("=" * 80)
    print("Repository Summary")
    print("=" * 80)

    summary = []
    for repo in vector_dbs:
        count = vector_dbs[repo]._collection.count()
        summary.append([repo, count])

    summary_df = pd.DataFrame(
        summary,
        columns=["Repository", "Indexed Chunks"]
    )

    try:
        display(summary_df)
    except NameError:
        print(summary_df)

    return summary_df


def run_vector_db_pipeline(
    chunks: List[Document],
    base_persist_dir: str = "./vectordb"
) -> Tuple[Dict[str, Chroma], pd.DataFrame]:
    """
    Execute full embeddings and vector database construction pipeline.
    """
    embed_model = initialize_embedding_model()
    vector_dbs = build_repository_vector_databases(chunks, embed_model, base_persist_dir)
    summary_df = verify_vector_databases(vector_dbs)
    return vector_dbs, summary_df


if __name__ == "__main__":
    import importlib.util

    current_dir = os.path.dirname(os.path.abspath(__file__))
    ingest_path = os.path.join(current_dir, "03_document_ingestion_and_chunking.py")

    if os.path.exists(ingest_path):
        spec = importlib.util.spec_from_file_location("step_03_ingest", ingest_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _, chunks = mod.run_ingestion_and_chunking()
        vector_dbs, summary_df = run_vector_db_pipeline(chunks)

