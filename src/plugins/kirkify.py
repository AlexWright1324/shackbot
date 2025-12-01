import discord
from discord import app_commands
from discord.ext import commands

import asyncio
import traceback

from importlib.resources import files

from face2face import Face2Face
from media_toolkit import ImageFile

MAX_WORKERS = 1
MAX_SIZE = 8 * 1024 * 1024  # 8MB


class Kirkify(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.f2f = None
        self.kirk = None
        self.queue: asyncio.Queue[tuple[discord.Interaction, discord.Attachment]] = (
            asyncio.Queue()
        )
        self.workers = [asyncio.create_task(self.worker()) for _ in range(MAX_WORKERS)]

    async def worker(self):
        while True:
            interaction, attachment = await self.queue.get()
            try:
                await self.kirkify(interaction, attachment)
            except Exception as _:
                print(traceback.format_exc())
                content = "An error occurred while processing your image"
                try:
                    await interaction.edit_original_response(content=content)
                except discord.errors.NotFound:
                    await interaction.followup.send(content, ephemeral=True)
            finally:
                self.queue.task_done()

    async def kirkify(
        self, interaction: discord.Interaction, attachment: discord.Attachment
    ):
        if not self.f2f or not self.kirk:
            content = "The kirkify service is not available at the moment\nPlease try again later"
            await interaction.edit_original_response(content=content)
            return

        # TODO: Resize images before processing
        # TODO: Compress images after processing
        bytes = await attachment.read()

        allowed_types = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
        if attachment.content_type in allowed_types:
            media = ImageFile().from_bytes(bytes)
            output_filename = "kirkified.png"
        else:
            content = "Unsupported media type.\nPlease upload a static image (PNG, JPEG, or WebP)"
            await interaction.edit_original_response(content=content)
            return

        # Run face swapping in a separate thread to avoid blocking
        swapped = await asyncio.to_thread(self.f2f.swap, media=media, faces="kirk")

        if isinstance(swapped, ImageFile):
            content = "No face detected in the media"
            await interaction.edit_original_response(content=content)
            return

        swapped = ImageFile().from_np_array(swapped)

        bytes_io = swapped.to_bytes_io()

        if (fileSize := bytes_io.getbuffer().nbytes) > MAX_SIZE:
            content = f"The kirkified image is too large to send: ({fileSize / (1024 * 1024):.1f}MB)"
            await interaction.edit_original_response(content=content)
            return

        discordFile = discord.File(bytes_io, output_filename)

        await interaction.delete_original_response()
        await interaction.followup.send(file=discordFile)

    async def cog_load(self):
        self.f2f = await asyncio.to_thread(Face2Face)

        kirkPath = str(files("shackbot.static").joinpath("kirk.jpg"))
        self.kirk = await asyncio.to_thread(self.f2f.add_face, "kirk", kirkPath)

    @app_commands.command(name="kirkify")
    @app_commands.describe(attachment="Image to kirkify")
    @app_commands.allowed_installs(guilds=True, users=True)
    @app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
    async def command(
        self, interaction: discord.Interaction, attachment: discord.Attachment
    ):
        await interaction.response.defer(ephemeral=True)

        queueSize = self.queue.qsize()
        await self.queue.put((interaction, attachment))

        await interaction.followup.send(
            f"Your image has been queued for kirkification!\nPosition in queue: {queueSize}",
            ephemeral=True,
        )


async def setup(bot: commands.Bot):
    await bot.add_cog(Kirkify(bot))
