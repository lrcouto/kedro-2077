"""
This is a boilerplate pipeline 'process_transcript'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline
from .nodes import chunk_transcript, extract_characters, partition_transcript_chunks

def create_pipeline(**kwargs) -> Pipeline:
    """Create the process transcript pipeline."""
    return Pipeline(
        [
            Node(
                func=chunk_transcript,
                inputs="cyberpunk_transcript",
                outputs="raw_transcript_chunks",
                name="chunk_transcript",
            ),
            Node(
                func=partition_transcript_chunks,
                inputs="raw_transcript_chunks",
                outputs="transcript_chunks",
                name="partition_transcript_chunks",
            ),
            Node(
                func=extract_characters,
                inputs="cyberpunk_transcript",
                outputs="character_list",
                name="extract_characters",
            ),
        ]
    )
