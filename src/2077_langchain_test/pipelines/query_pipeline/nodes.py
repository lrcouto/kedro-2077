"""Query pipeline nodes for Cyberpunk 2077 transcript."""

from typing import Any, Dict, List
from langchain_openai import ChatOpenAI


def find_relevant_chunks(
    query: str,
    transcript_chunks: Dict[str, Any],
    max_chunks: int = 2
) -> List[Dict[str, Any]]:
    """
    Find the most relevant chunks for a given query directly from a PartitionedDataset.
    
    Args:
        query: The user query string.
        transcript_chunks: A dict of partitions (each one is a loaded chunk of transcript).
        max_chunks: The maximum number of chunks to return.
    """
    query_lower = query.lower()
    query_words = set(query_lower.split())

    scored_chunks = []

    # Iterate through all partitions
    for partition_name, chunk_data in transcript_chunks.items():
        # Each partition may be a dict (e.g., {"text": "...", ...})
        if not isinstance(chunk_data, dict) or "text" not in chunk_data:
            continue

        chunk_text_lower = chunk_data["text"].lower()
        chunk_words = set(chunk_text_lower.split())

        # Simple overlap scoring
        word_overlap = len(query_words.intersection(chunk_words))
        word_ratio = word_overlap / len(query_words) if query_words else 0

        # Bonus for exact phrase matches
        phrase_bonus = 1 if query_lower in chunk_text_lower else 0

        score = word_ratio + phrase_bonus

        if score > 0:
            scored_chunks.append((score, partition_name, chunk_data))

    # Sort by score (descending)
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    # Return the top chunks (you can also include partition_name if helpful)
    return [chunk for _, _, chunk in scored_chunks[:max_chunks]]


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
    """Query OpenAI LLM with the formatted prompt."""
    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    
    # Send the prompt and get response
    response = llm.invoke(formatted_prompt)

    print(response.content)
    
    return response.content