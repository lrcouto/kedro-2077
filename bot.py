import os
import asyncio
import discord
from discord.ext import commands
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.framework.project import configure_project
from pathlib import Path


# --- Discord setup ---
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix=commands.when_mentioned, intents=intents)

@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user} (ID: {bot.user.id})")

@bot.command(name="hello")
async def hello(ctx):
    """Simple hello command to test the bot."""
    await ctx.send("Hello, I am a bot and I exist! 👋")

# --- Kedro pipeline command ---
@bot.command(name="run-query")
async def run_query(ctx, *, user_query: str):
    """Run the Kedro 'query_llm_discord' pipeline with a user query asynchronously."""

    await ctx.send(f"🚀 Running Kedro pipeline `query_discord` for query: `{user_query}`...")

    project_path = Path(__file__).resolve().parent
    metadata = bootstrap_project(project_path)
    configure_project(metadata.package_name)

    try:
        # Run the blocking Kedro code in a separate thread
        def run_kedro():
            with KedroSession.create(
                project_path=project_path,
                runtime_params={"user_query": user_query}
            ) as session:
                return session.run(pipeline_name="query_discord")

        result = await asyncio.to_thread(run_kedro)

        # Extract LLM node output
        llm_memory_dataset = result.get("llm_response_discord")
        llm_response = result.get("query_llm")
        if llm_memory_dataset:
            llm_response = llm_memory_dataset.load()
            if len(llm_response) > 1900:
                llm_response = llm_response[:1900] + "\n...(truncated)..."
            await ctx.send(f"🤖 {llm_response}")
        else:
            await ctx.send("⚠️ No response returned by the LLM.")

    except Exception as e:
        await ctx.send(f"❌ Error running pipeline: {e}")

# --- Run the bot ---
if __name__ == "__main__":
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise EnvironmentError("DISCORD_TOKEN not set")
    bot.run(token)