"""Query pipeline for Cyberpunk 2077 transcript."""

from kedro.pipeline import Node, Pipeline
from .nodes import find_relevant_chunks, format_prompt_with_context, query_llm


def create_pipeline() -> Pipeline:
    """Create the query pipeline."""
    return Pipeline(
        [
            Node(
                func=find_relevant_chunks,
                inputs=["params:user_query", "transcript_index", "params:max_chunks"],
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
                inputs="formatted_prompt",
                outputs="llm_response",
                name="query_llm",
            ),
        ]
    )
