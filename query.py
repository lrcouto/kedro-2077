from pathlib import Path
from kedro.framework.startup import bootstrap_project
from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession


def run_query(project_path: Path, query: str) -> str:
    metadata = bootstrap_project(project_path)
    configure_project(metadata.package_name)

    # Each query requires a fresh session with runtime_params
    with KedroSession.create(
        project_path=project_path,
        runtime_params={"user_query": query}
    ) as session:
        result = session.run(pipeline_name="__default__")

    llm_response = result.get("llm_response")
    if hasattr(llm_response, "load"):
        return llm_response.load()
    elif isinstance(llm_response, str):
        return llm_response
    else:
        return "Could not load LLM response."


def main():
    project_path = Path(__file__).resolve().parent
    print("I am a machine that answers questions about Cyberpunk 2077! (type 'exit' or 'quit' to stop)")
    while True:
        query = input("\nEnter your question:\n> ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not query:
            continue

        response = run_query(project_path, query)
        print("\nLLM Response:\n")
        print(response)


if __name__ == "__main__":
    main()
