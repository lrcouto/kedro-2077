"""
This is a boilerplate pipeline 'process_transcript'
generated using Kedro 1.0.0
"""
import re
from typing import Any, Dict, List


def chunk_transcript(transcript: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    """Split the transcript into overlapping chunks for better context."""
    # Clean up whitespaces
    cleaned_transcript = re.sub(r'\n+', '\n', transcript.strip())
    
    # Split into sentences/paragraphs
    sentences = re.split(r'(?<=[.!?])\s+', cleaned_transcript)
    
    chunks = []
    start_idx = 0
    
    while start_idx < len(sentences):
        end_idx = min(start_idx + chunk_size, len(sentences))
        chunk_text = ' '.join(sentences[start_idx:end_idx])
        
        chunks.append({
            'text': chunk_text,
            'chunk_id': len(chunks),
            'start_sentence': start_idx,
            'end_sentence': end_idx - 1,
            'character_count': len(chunk_text)
        })
        
        # Move start_idx forward by chunk_size - overlap
        start_idx = max(start_idx + chunk_size - overlap, start_idx + 1)
    
    return chunks


def extract_characters(transcript: str) -> List[str]:
    """Extract unique character names from the transcript."""
    # Pattern to match character names (usually followed by a colon)
    character_pattern = r'^([A-Za-z\s]+):'
    
    characters = set()
    for line in transcript.split('\n'):
        match = re.match(character_pattern, line.strip())
        if match:
            character_name = match.group(1).strip()
            if character_name and len(character_name) > 1:
                characters.add(character_name)
    
    return sorted(list(characters))


def create_transcript_index(chunks: Dict[str, Any]) -> Dict[str, Any]:
    """Create an index of the transcript for quick searching.

    This function expects a PartitionedDataset read shape (a dict mapping
    partition keys to partition payloads). Partition payloads must be either
    a single chunk dict or a list of chunk dicts.
    """
    combined: List[Dict[str, Any]] = []
    for partition_key in sorted(chunks.keys()):
        value = chunks[partition_key]

        try:
            if hasattr(value, "load") and callable(value.load):
                loaded = value.load()
            elif callable(value):
                loaded = value()
            else:
                loaded = value
        except Exception as exc:
            raise TypeError(f"failed to load partition '{partition_key}': {exc}") from exc

        if isinstance(loaded, list):
            combined.extend(loaded)
        elif isinstance(loaded, dict):
            combined.append(loaded)
        else:
            raise TypeError(
                f"partition '{partition_key}' contains unsupported payload type: {type(loaded)!r}. "
                "Expected dict or list of dicts produced by partition_transcript_chunks."
            )

    index = {
        'total_chunks': len(combined),
        'total_characters': sum(chunk.get('character_count', 0) for chunk in combined),
        'chunks': combined
    }

    return index


def partition_transcript_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Convert a list of chunk dicts into a partition mapping for Kedro's PartitionedDataSet.

    Returns a dict where keys are partition names and values are the chunk payloads.
    Example: {"chunk_0": { ...chunk data... }, "chunk_1": { ... }}
    """
    partitions: Dict[str, Dict[str, Any]] = {}
    for chunk in chunks:
        chunk_id = chunk.get('chunk_id')
        if chunk_id is None:
            # fallback to index position
            chunk_id = len(partitions)

        partition_key = f"chunk_{chunk_id}"
        partitions[partition_key] = chunk

    return partitions