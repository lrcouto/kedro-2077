# I am a machine that answers questions about Cyberpunk 2077!

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

This is a small project using Kedro and Langchain to run LLM queries on the full transcript of Cyberpunk 2077.

## How to run

- Install requirements with `pip install -r requirements.txt`
- Add you OpenAI API key to `conf/local/credentials.yml` like this:

```yml
openai:
  api_key: "your-api-key"
```

- Run the `query.py` Python script