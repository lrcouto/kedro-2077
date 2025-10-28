"""
This is a boilerplate pipeline 'query_discord'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline
from .nodes import find_relevant_contexts, format_prompt_with_context, query_llm


def create_pipeline() -> Pipeline:
    """Create the query pipeline."""
    return Pipeline(
        [
            Node(
                func=find_relevant_contexts,
                inputs=["params:user_query", "transcript_chunks", "wiki_embeddings", "character_list", "params:max_chunks", "params:character_bonus", "params:wiki_weight"],
                outputs="relevant_contexts_discord",
                name="find_relevant_contexts_node",
            ),
            Node(
                func=format_prompt_with_context,
                inputs=["query_prompt", "params:user_query", "relevant_contexts_discord", "params:max_context_length"],
                outputs="formatted_prompt_discord",
                name="format_prompt_with_context_node",
            ),
            Node(
                func=query_llm,
                inputs="formatted_prompt_discord",
                outputs="llm_response_discord",
                name="query_llm_node",
            ),
        ]
    )