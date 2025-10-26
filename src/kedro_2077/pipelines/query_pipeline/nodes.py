"""Query pipeline nodes for Cyberpunk 2077 transcript."""

from typing import Any, Dict, List, Tuple
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
import torch

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

# Load model once so it doesn't reload per node execution
_model = SentenceTransformer("all-MiniLM-L6-v2")


def find_relevant_contexts(
    query: str,
    transcript_chunks: Dict[str, Any],
    wiki_embeddings: Dict[str, Dict[str, Any]],
    character_list: List[str],
    max_chunks: int = 5,
    character_bonus: float = 0.05,
    wiki_weight: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Retrieve top relevant contexts from both transcript chunks and wiki embeddings.

    Args:
        query: The user query string.
        transcript_chunks: PartitionedDataset with text chunks.
        wiki_embeddings: Dict with 'page_title' -> {'text': ..., 'embedding': np.ndarray}.
        character_list: Character names list to boost relevance.
        max_chunks: Max number of transcript chunks to return.
        character_bonus: Similarity boost for character matches.
        wiki_weight: Relative weight of wiki similarity when combining results.

    Returns:
        List of the most relevant text contexts (mixed transcript + wiki).
    """
    import torch

    query_emb = _model.encode(query, convert_to_tensor=True)

    # Characters mentioned in the query
    query_lower = query.lower()
    mentioned_characters = [c for c in character_list if c.lower() in query_lower]

    results = []

    # ---- Transcript similarity ----
    for chunk_data in transcript_chunks.values():
        if not isinstance(chunk_data, dict) or "text" not in chunk_data:
            continue
        text = chunk_data["text"]
        emb = _model.encode(text, convert_to_tensor=True)
        sim = util.cos_sim(query_emb, emb).item()

        if mentioned_characters:
            for c in mentioned_characters:
                if c.lower() in text.lower():
                    sim += character_bonus

        results.append((sim, "transcript", text))

    # ---- Wiki similarity ----
    for title, page in wiki_embeddings.items():
        # Get embedding and convert to tensor if it isn't already
        emb = page["embedding"]
        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb)
        text = page["text"]

        sim = util.cos_sim(query_emb, emb).item() * wiki_weight
        results.append((sim, "wiki", f"{title}: {text[:1000]}..."))

    # Sort by score
    results.sort(key=lambda x: x[0], reverse=True)

    # Return top-N contexts
    top_results = [
        {"source": src, "text": txt, "similarity": sim}
        for sim, src, txt in results[:max_chunks]
    ]

    return top_results


def format_prompt_with_context(prompt_template: Any, user_query: str, contexts: List[Dict[str, Any]], max_context_length: int = 2000) -> str:
    """Format the LLM prompt with user query and retrieved contexts."""
    context_blocks = []
    for ctx in contexts:
        src_label = f"[{ctx['source'].upper()}]"
        truncated = ctx["text"][:max_context_length]
        context_blocks.append(f"{src_label}\n{truncated}")

    combined_context = "\n\n---\n\n".join(context_blocks)

    formatted_prompt = prompt_template.format(
        user_query=user_query,
        transcript_context=combined_context
    )

    return formatted_prompt


def query_llm(formatted_prompt: str) -> str:
    """Query LLM with the formatted prompt."""
    # Load  credentials
    conf_path = Path(__file__).resolve().parents[4] / settings.CONF_SOURCE
    conf_loader = OmegaConfigLoader(conf_source=str(conf_path))
    credentials = conf_loader["credentials"]
    openai_api_key = credentials["openai"]["api_key"]

    # Initialize the LLM
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini", temperature=0.2)
    
    # Send the prompt and get response
    response = llm.invoke(formatted_prompt)
    
    return response.content