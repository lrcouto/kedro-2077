# I am a machine that answers questions about Cyberpunk 2077!

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

This is a small project using Kedro and Langchain to run LLM queries on the full transcript of Cyberpunk 2077, and the content of the Cyberpunk wiki.

## Setup

- Install requirements with `pip install -r requirements.txt`
- Add you OpenAI API key to `conf/local/credentials.yml` like this:

```yml
openai:
  api_key: "your-api-key"
```

- `kedro run --pipeline=process_transcript` will process the raw data and build embeddings.

## Running as a CLI "conversation" bot

Once the data is processed, you can run `kedro run --pipeline=query_pipeline --tags=cli` to start a CLI "chatbot".

## Running as a Discord bot

### Setting up the bot

Log into your Discord account (or [create](https://discord.do/how-to-create-a-discord-account/) one)

Go into the [Discord Developer Portal](https://discord.com/developers/applications) and click "New Application":

- On the "General Information" tab, name and describe your bot.

- On the "Bot" tab, look for the "Message Content Intent" option and make sure it's enabled. Without this, the bot will not be able to respond to written commands in a Discord message.

- Click the "Reset Token" button and save the API token that appears on the screen.

- On the "OAuth2" tab, go to the OAuth2 URL Generator and select "bot". Give your bot the following permissions:
  - View Channels
  - Read Message history
  - Send Messages

- Save the invite URL at the bottom of the screen.

[Create a new Discord](https://discord.com/blog/starting-your-first-discord-server) server or use one where you have admin permissions, and use the invite URL to have your bot join the server.

### Running the bot

Export the API token from the Discord developer panel to your environment.

- `export DISCORD_TOKEN=<your token>`

Run `bot.py` to initialize the bot. It should appear on your server as an online user.

Send a direct message to the bot (`@Bot Name`) followed by one of those commands:

- `/help`: Display all commands

- `/build`: Run the `process_transcript` pipeline and rebuild embeddings from raw data. 

- `/query <your query>`: Ask the bot a question about Cyberpunk 2077

## How does it work?

### Handling the data

The core idea of this project is to enable efficient retrieval of message context so that the bot can generate coherent responses using a large language model, while sticking to the information contained in the data sources and avoiding hallucinations.

There are two files to be used as data sources. One is a 400-page text file that contains the full transcript of a playthrough of Cyberpunk 2077, with all dialogue between characters. The other is a full download of the [Cyberpunk Wiki](https://cyberpunk.fandom.com/wiki/Cyberpunk_Wiki), containing descriptions of missions, characters, items, etc. This second file is in .json format.

For the transcript, I chose to use a `PartitionedDataset` to store intermediate data, such as message embeddings or processed message chunks. This choice allows for looking for specific chunks of the transcript that might contain information relevant to the user query. It also saves a list of character names, to help with this search in case the user asks for information on a specific character.

I chose to use Sentence-Transformers to generate embeddings for textual data. These embeddings capture semantic similarity, enabling the bot to retrieve contextually relevant messages even when users phrase their queries differently. This embedding-based approach significantly improves the bot’s accuracy and coherence compared to simple keyword matching

The embeddings generated from the wiki data were stored in a `PickleDataset`. This provides a convenient way to store Python objects natively (e.g., lists of vectors or fitted models) without additional conversion steps. Although it doesn't have the full benefits of a vector database, it was a quick solution that did not require writing a custom dataset, and the amount of data is small enough that the performance is still adequate.

### Prompting

The reason this project was initially made was to test the experimental `LangChainPromptDataset` during its development with an actual LLM involved.

The prompt  itself is stored as a JSON file. This JSON file defines the prompt structure and placeholders for context variables (like the latest message or previous turns). The file is loaded through a `LangChainPromptDataset`, which uses a `JSONDataset` as its underlying Kedro dataset. When the pipeline runs, this configuration is automatically converted into a `ChatPromptTemplate`, allowing for easy iteration and experimentation on prompt design.

Specifically for the CLI chatbot version of this project, using the ChatPromptTemplate to structure inputs in a consistent and flexible way. This allows the bot to maintain continuity — it can “remember” prior messages in a conversation and respond coherently while the Kedro session runs.

### Integration with Discord

This project integrates Kedro with Discord using the [discord.py](https://discordpy.readthedocs.io/en/stable/) library. This allows users to trigger data pipelines and query the LLM directly from Discord messages.

Since Kedro’s session and pipeline execution are blocking operations, we use Python’s asyncio.to_thread() to offload them into a background thread. This ensures that the Discord bot remains responsive to user input and other commands while Kedro processes data, builds embeddings, or queries the LLM. Each command — such as `/build` or `/query` — bootstraps the Kedro project using `bootstrap_project()` and `configure_project()`, ensuring the full Kedro context is correctly initialized before execution. This also allows multiple users to query the bot simultaneously. 

The `/build` command runs the process_transcript pipeline asynchronously to generate embeddings and partitioned transcript data, while the `/query` command executes the query_pipeline with the user’s message as a runtime parameter. The pipeline’s output (including the model’s response and memory dataset) is then streamed back to the Discord channel, automatically handling message length limits and error reporting.
