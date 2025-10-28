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


# --- Help command ---
@bot.command(name="/help")
async def show_help(ctx):
    """Display all available bot commands."""
    embed = discord.Embed(
        title="🤖 Kedro 2077 Bot — Command Guide",
        description="Here's what I can do, choom:",
        color=0x00FFAA
    )

    embed.add_field(
        name="🧠 `/query <question>`",
        value="Ask me anything about Cyberpunk 2077. I'll run the Kedro LLM pipeline and answer data from the game.",
        inline=False
    )

    embed.add_field(
        name="🧩 `/build`",
        value="Rebuild the transcript partitions and wiki embeddings. This may take a while — I'll let you know when it's done.",
        inline=False
    )

    embed.add_field(
        name="ℹ️ `/help`",
        value="Show this command list.",
        inline=False
    )

    embed.set_footer(text="Powered by Kedro 🔶")

    await ctx.send(embed=embed)


# --- Test if bot exists ---
@bot.command(name="/hello")
async def hello(ctx):
    """Simple hello command to test the bot."""
    await ctx.send("Hello, I am a bot and I exist! 👋")


# --- Build embeddings and partition transcript ---
@bot.command(name="/build")
async def build_embeddings(ctx):
    """Run the data processing pipeline asynchronously."""

    await ctx.send("⏳ Building embeddings from wiki and transcript data, please wait...")

    project_path = Path(__file__).resolve().parent
    metadata = bootstrap_project(project_path)
    configure_project(metadata.package_name)

    try:
        # Run the blocking Kedro code in a separate thread
        def run_kedro():
            with KedroSession.create(project_path=project_path) as session:
                return session.run(pipeline_name="process_transcript")

        await asyncio.to_thread(run_kedro)
        await ctx.send("✅ Embeddings and transcript partitions built successfully!")

    except Exception as e:
        await ctx.send(f"❌ Error running pipeline: {e}")


# --- Query LLM ---
@bot.command(name="/query")
async def run_query(ctx, *, user_query: str):
    """Run the Kedro 'query_discord' pipeline with a user query asynchronously."""

    await ctx.send(f"🚀 Running Kedro pipeline for query: `{user_query}`...\n\n")

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
                return session.run(pipeline_name="query_pipeline", tags=["discord"])

        result = await asyncio.to_thread(run_kedro)

        # Extract LLM node output
        llm_memory_dataset = result.get("llm_response_discord")
        llm_response = result.get("query_llm_discord")
        if llm_memory_dataset:
            llm_response = llm_memory_dataset.load()
            if len(llm_response) > 1900:
                max_len = 2000
                for i in range(0, len(llm_response), max_len):
                    await ctx.send(llm_response[i:i+max_len])
            else:
                await ctx.send(llm_response)
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