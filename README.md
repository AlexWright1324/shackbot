# shack-bot

A Discord bot built with discord.py.

## Setup

Install [uv](https://docs.astral.sh/uv/) if you haven't already:

Create a virtual environment and install dependencies:

```bash
uv sync
```

## Running

Set your Discord token as an environment variable:

```bash
export DISCORD_TOKEN="your-token-here"
```

Or using dotenv, create a `.env` file with the following content:

```bash
DISCORD_TOKEN=your-token-here
```

Run the bot:

```bash
uv run src/main.py
```
