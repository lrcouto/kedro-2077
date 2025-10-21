"""Query pipeline nodes for Cyberpunk 2077 transcript."""

from typing import Any, Dict, List
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer, util

# Load model once so it doesn't reload per node execution
_model = SentenceTransformer("all-MiniLM-L6-v2")


def find_relevant_chunks(
    query: str,
    transcript_chunks: Dict[str, Any],
    character_list: List[str],
    max_chunks: int = 2,
    character_bonus: float = 0.05,  # small bonus for character mentions
) -> List[Dict[str, Any]]:
    """
    Find the most relevant chunks for a given query directly from a PartitionedDataset,
    using embeddings for semantic similarity and with a bonus for matching character names.

    Args:
        query: The user query string.
        transcript_chunks: Dict of partitions (each loaded transcript chunk).
        character_list: Optional list of character names to boost relevance.
        max_chunks: Maximum number of chunks to return.
        character_bonus: Weight added for each matched character in the chunk text.
    """
    # Encode the query
    query_emb = _model.encode(query, convert_to_tensor=True)

    # Identify which characters are in the query (if applicable)
    mentioned_characters = []
    if character_list:
        query_lower = query.lower()
        mentioned_characters = [c for c in character_list if c.lower() in query_lower]

    scored_chunks = []

    for partition_name, chunk_data in transcript_chunks.items():
        if not isinstance(chunk_data, dict) or "text" not in chunk_data:
            continue

        chunk_text = chunk_data["text"]
        chunk_emb = _model.encode(chunk_text, convert_to_tensor=True)

        # Base similarity score (cosine similarity)
        similarity = util.cos_sim(query_emb, chunk_emb).item()

        # Add small bonus if characters mentioned in query appear in this chunk
        if mentioned_characters:
            chunk_lower = chunk_text.lower()
            for char in mentioned_characters:
                if char.lower() in chunk_lower:
                    similarity += character_bonus

        scored_chunks.append((similarity, chunk_data))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    return [chunk for _, chunk in scored_chunks[:max_chunks]]


def format_prompt_with_context(prompt_template: Any, user_query: str, relevant_chunks: List[Dict[str, Any]], max_context_length: int = 2000) -> str:
    """Format the prompt template with user query and transcript context."""
    # Combine relevant chunks into context, but truncate each chunk
    context_parts = []
    total_length = 0
    
    for chunk in relevant_chunks:
        chunk_text = chunk['text']
        
        # Truncate chunk if it's too long
        if len(chunk_text) > max_context_length:
            chunk_text = chunk_text[:max_context_length] + "..."
        
        # Check if adding this chunk would exceed token limit
        if total_length + len(chunk_text) > max_context_length * len(relevant_chunks):
            break
            
        context_parts.append(chunk_text)
        total_length += len(chunk_text)
    
    transcript_context = "\n\n---\n\n".join(context_parts)
    
    # Format the prompt template with the variables
    formatted_prompt = prompt_template.format(
        user_query=user_query,
        transcript_context=transcript_context
    )
    
    return formatted_prompt


def query_llm(formatted_prompt: str) -> str:
    """Query LLM with the formatted prompt."""
    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    
    # Send the prompt and get response
    response = llm.invoke(formatted_prompt)

    print(response.content)
    
    return response.content