"""
Retrieve only from the repository selected by RQ2.

Verify that the routing mechanism retrieves relevant information from the repository.

This section retrieves knowledge only from the repositories identified by the Knowledge Routing Agent (RQ2).
"""

###############################################################
# RQ3 - STEP 9, 10, 11 : SINGLE & MULTI-REPOSITORY RETRIEVAL
###############################################################

from typing import Dict, List, Tuple, Any, Optional

DEFAULT_REPOSITORY_MAPPING = {
    "public documentation": "product",
    "architecture blueprint": "architecture",
    "methodology": "methodology",
    "governance framework": "governance",
    "commercial model": "commercial",
    "pricing template": "commercial",
    "operations": "operations",
    "cloud adoption": "cloud_adoption",
    "risk register": "governance",
    "sme": "product"
}


def retrieve_context(
    question: str,
    repository: str,
    top_k: int = 5,
    vector_dbs: Optional[Dict[str, Any]] = None
) -> str:
    """
    Retrieve relevant chunks from the selected single repository.
    """
    db = vector_dbs[repository]

    retriever = db.as_retriever(
        search_kwargs={"k": top_k}
    )

    docs = retriever.invoke(question)

    context = "\n\n".join(
        [d.page_content for d in docs]
    )

    return context


def test_single_retrieval(
    question: str,
    repository_full_string: str,
    vector_dbs: Dict[str, Any],
    mapping: Optional[Dict[str, str]] = None
) -> str:
    """
    Test retrieval from the first component of the required repository string.
    """
    repo_map = mapping if mapping is not None else DEFAULT_REPOSITORY_MAPPING
    first_repo_component_raw = repository_full_string.split("|")[0].strip().lower()
    repository = repo_map.get(first_repo_component_raw, "product")

    print("=" * 80)
    print("Question")
    print("=" * 80)
    print(question)
    print("\nRepository (for single-repo test)")
    print(repository)
    print("\nRetrieved Context")

    context = retrieve_context(question, repository, top_k=5, vector_dbs=vector_dbs)
    print(context[:3000])
    return context


def retrieve_multi_repository_context(
    question: str,
    repositories: List[str],
    top_k: int = 3,
    vector_dbs: Optional[Dict[str, Any]] = None
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Retrieve knowledge from one or more repositories selected by Knowledge Routing Agent (RQ2).
    """
    all_context = []
    retrieved_sources = []

    mapping = DEFAULT_REPOSITORY_MAPPING

    for repo in repositories:
        repo = repo.strip().lower()

        if repo not in mapping:
            continue

        repo_name = mapping[repo]

        if repo_name not in vector_dbs:
            continue

        retriever = vector_dbs[repo_name].as_retriever(
            search_kwargs={"k": top_k}
        )

        docs = retriever.invoke(question)

        for d in docs:
            all_context.append(d.page_content)
            retrieved_sources.append({
                "Repository": repo_name,
                "Source": d.metadata.get("source", "Unknown")
            })

    context = "\n\n".join(all_context)
    return context, retrieved_sources


if __name__ == "__main__":
    print("Context retrieval mechanisms defined.")

