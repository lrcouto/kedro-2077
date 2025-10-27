"""Query pipeline nodes for Cyberpunk 2077 transcript."""

from typing import Any, Dict, List
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
import torch

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings


# Load model once so it doesn't reload per node execution
_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load credentials and initialize LLM once or they'll reload every time the loop runs
conf_path = Path(__file__).resolve().parents[4] / settings.CONF_SOURCE
conf_loader = OmegaConfigLoader(conf_source=str(conf_path))
credentials = conf_loader["credentials"]
openai_api_key = credentials["openai"]["api_key"]

llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini", temperature=0.2)


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


def format_prompt_with_context(
    prompt_template: ChatPromptTemplate,
    user_query: str,
    contexts: List[Dict[str, Any]],
    max_context_length: int = 2000,
):
    """
    Format a ChatPromptTemplate with the user query and retrieved contexts.
    """

    context_blocks = []
    for ctx in contexts:
        src_label = f"[{ctx['source'].upper()}]"
        truncated = ctx["text"][:max_context_length]
        context_blocks.append(f"{src_label}\n{truncated}")

    combined_context = "\n\n---\n\n".join(context_blocks)

    messages = prompt_template.format_messages(
        user_query=user_query,
        transcript_context=combined_context
    )

    return messages


def query_llm(
    transcript_chunks: Dict[str, Any] = None,
    wiki_embeddings: Dict[str, Dict[str, Any]] = None,
    character_list: List[str] = None,
    max_context_length: int = 2000,
    prompt_template: ChatPromptTemplate = None
) -> None:
    """
    Interactive conversation loop to allow the chat
    to start automatically when executing `kedro run`.
    Maintains conversation history for context.
    """

    print("\nI am a machine that answers questions about Cyberpunk 2077!")
    print("Type your question about the game world or characters.")
    print("Type 'exit' to quit.\n")

    conversation_history: List[Any] = []

    while True:
        user_query = input("🟢 You: ").strip()
        if not user_query:
            continue
        if user_query.lower() in {"exit", "quit"}:
            print("👋 Goodbye, choom!")
            return ""

        # Hacky cursed loop to find relevant contexts and format prompt each turn
        contexts = find_relevant_contexts(
            query=user_query,
            transcript_chunks=transcript_chunks,
            wiki_embeddings=wiki_embeddings,
            character_list=character_list,
        )

        new_messages = format_prompt_with_context(
            prompt_template=prompt_template,
            user_query=user_query,
            contexts=contexts,
            max_context_length=max_context_length
        )

        # Append new messages to conversation history
        conversation_history.extend(new_messages)
        response = llm.invoke(conversation_history)

        print("\n⚪ LLM:", response.content)
        print("\n" + "-" * 80 + "\n")

        # Append LLM response to conversation history for next turn
        conversation_history.append({"role": "ai", "content": response.content})
