"""
Generate enterprise responses using the retrieved multi-repository knowledge.

Validate that repository routing, retrieval and response generation are functioning correctly.

Store generated enterprise responses for reproducibility and to avoid regenerating during iterative analysis.

Baseline GPT (No Enterprise Knowledge): Generate responses using only GPT internal knowledge.

Agentic RAG (Ground Truth Repository): Generate responses using the actual repositories annotated in the corpus.

Agentic RAG (RQ2 Predicted Repository): Use the repository predicted by the Knowledge Routing Agent.

Agentic RAG (All Repositories) - Retrieve from every repository without routing.
"""

###############################################################
# RQ3 - STEP 12 TO 18 : RESPONSE GENERATION AGENT & 4 STRATEGIES
###############################################################

from typing import Dict, Any, Optional
import os
import time
from openai import OpenAI

DEFAULT_MODEL = "gpt-4.1"


def get_openai_client() -> OpenAI:
    """
    Instantiate OpenAI client from environment variable.
    """
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def generate_enterprise_response(
    question: str,
    retrieved_context: str,
    client: Optional[OpenAI] = None,
    model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    Generate structured enterprise response using retrieved knowledge.
    """
    cli = client if client is not None else get_openai_client()

    prompt = f"""

You are a Senior Enterprise Presales Architect.

Answer the question using ONLY the retrieved enterprise knowledge.

If information is unavailable,
say so.

Question

{question}

Enterprise Knowledge

{retrieved_context}

Generate

1. Executive Summary

2. Technical Explanation

3. Architecture or Implementation Recommendations

4. Presales Positioning or Client Advice

"""

    response = cli.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an enterprise AI assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return {
        "response": response.choices[0].message.content,
        "prompt": prompt,
        "model": model
    }


def baseline_response(
    question: str,
    client: Optional[OpenAI] = None,
    model: str = DEFAULT_MODEL
) -> str:
    """
    Strategy 1: Generate response using only GPT internal knowledge without enterprise context.
    """
    cli = client if client is not None else get_openai_client()

    response = cli.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful enterprise presales architect. Answer the question directly."
            },
            {
                "role": "user",
                "content": question
            }
        ],
        temperature=0.2
    )
    return response.choices[0].message.content


def ground_truth_rag(
    question: str,
    required_repo: str,
    vector_dbs: Dict[str, Any],
    client: Optional[OpenAI] = None,
    retrieve_fn=None
) -> str:
    """
    Strategy 2: Retrieve from ground truth repository and generate response.
    """
    repos = required_repo.split("|")
    context, sources = retrieve_fn(question, repos, vector_dbs=vector_dbs)
    result = generate_enterprise_response(question, context, client=client)
    return result["response"]


def predicted_rag(
    question: str,
    predicted_repo: str,
    vector_dbs: Dict[str, Any],
    client: Optional[OpenAI] = None,
    retrieve_fn=None
) -> str:
    """
    Strategy 3: Retrieve from RQ2 predicted repository and generate response.
    """
    repos = predicted_repo.split("|")
    context, sources = retrieve_fn(question, repos, vector_dbs=vector_dbs)
    result = generate_enterprise_response(question, context, client=client)
    return result["response"]


def all_repository_rag(
    question: str,
    vector_dbs: Dict[str, Any],
    client: Optional[OpenAI] = None,
    retrieve_fn=None
) -> str:
    """
    Strategy 4: Retrieve from all repositories without routing and generate response.
    """
    all_repos = list(vector_dbs.keys())
    context, sources = retrieve_fn(question, all_repos, vector_dbs=vector_dbs)
    result = generate_enterprise_response(question, context, client=client)
    return result["response"]


if __name__ == "__main__":
    print("Response generation strategies defined.")

