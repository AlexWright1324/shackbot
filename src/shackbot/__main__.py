import os
import asyncio
import discord

from discord import app_commands
from discord.ext import commands

from dotenv import load_dotenv


def create_bot() -> commands.Bot:
    """Create and configure the Discord bot instance."""
    # Create bot instance with intents
    intents = discord.Intents.default()
    intents.guilds = True
    intents.guild_messages = True
    intents.message_content = True

    bot = commands.Bot(command_prefix="!", intents=intents)

    @bot.event
    async def on_ready():
        if bot.user is not None:
            print(f"Logged in as {bot.user.name}#{bot.user.discriminator}")

        plugins = ["face", "nerd"]
        print(f"Loading {len(plugins)} plugins...")

        for plugin in plugins:
            print(f" - Loading {plugin}...")
            try:
                await bot.load_extension(f"plugins.{plugin}")
                print("   Loaded")
            except Exception as e:
                print("   Failed to load:")
                print(e)

        try:
            synced = await bot.tree.sync()
            print(f"Synced {len(synced)} command(s)")
        except Exception as e:
            print(f"Error syncing commands: {e}")

        print("Bot is ready.")

    return bot


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Get token from environment variable
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise ValueError("DISCORD_TOKEN environment variable not set")

    # Create and run the bot
    bot = create_bot()
    bot.run(token)


if __name__ == "__main__":
    main()
