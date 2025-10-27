"""Query pipeline for Cyberpunk 2077 transcript."""

from kedro.pipeline import Node, Pipeline
from .nodes import find_relevant_contexts, format_prompt_with_context, query_llm


def create_pipeline() -> Pipeline:
    """Create the query pipeline."""
    return Pipeline(
        [
            Node(
                func=find_relevant_contexts,
                inputs=["params:user_query", "transcript_chunks", "wiki_embeddings", "character_list", "params:max_chunks", "params:character_bonus", "params:wiki_weight"],
                outputs="relevant_chunks",
                name="find_relevant_chunks",
            ),
            Node(
                func=format_prompt_with_context,
                inputs=["query_prompt", "params:user_query", "relevant_chunks", "params:max_context_length"],
                outputs="formatted_prompt",
                name="format_prompt_with_context",
            ),
            Node(
                func=query_llm,
                inputs=["transcript_chunks", "wiki_embeddings", "character_list", "params:max_context_length", "query_prompt"],
                outputs="llm_response",
                name="query_llm",
            ),
        ]
    )