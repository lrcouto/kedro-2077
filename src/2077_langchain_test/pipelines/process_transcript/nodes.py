"""
This is a boilerplate pipeline 'process_transcript'
generated using Kedro 1.0.0
"""
import re
from typing import Any, Dict, List


def chunk_transcript(transcript: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    """Split the transcript into overlapping chunks for better context."""
    # Clean up the transcript
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


def create_transcript_index(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create an index of the transcript for quick searching."""
    index = {
        'total_chunks': len(chunks),
        'total_characters': sum(chunk['character_count'] for chunk in chunks),
        'chunks': chunks
    }
    
    return index