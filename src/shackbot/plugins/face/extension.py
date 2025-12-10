import io
import discord
from discord import app_commands
from discord.ext import commands

import asyncio
import traceback

from importlib.resources import files

import numpy as np
from PIL import Image

from .FaceAnalyser import FaceAnalyser
from .FaceSwapper import FaceSwapper
from .FaceEnhancer import FaceEnhancer

from .models import SWAPPER, INSIGHT, FACE_ENHANCER_MODELS

from shackbot.file import download

MAX_WORKERS = 1
MAX_SIZE = 8 * 1024 * 1024  # 8MB


async def setup(bot: commands.Bot):
    await bot.add_cog(Face(bot))


class Face(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
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
        # TODO: Resize images before processing
        # TODO: Compress images after processing
        bytes = await attachment.read()
        try:
            with Image.open(
                io.BytesIO(bytes), formats=["PNG", "JPEG", "WEBP"]
            ) as media:
                media.load()
        except Exception as _:
            print(traceback.format_exc())
            content = "Unsupported media type.\nPlease upload a static image (PNG, JPEG, or WebP)"
            await interaction.edit_original_response(content=content)
            return

        data = np.array(media)

        faces = await asyncio.to_thread(
            self.face_analyser.analyze,
            data,
            extract_embedding=True,
            extract_landmarks=False,
        )
        if len(faces) == 0:
            content = "No face detected in the media"
            await interaction.edit_original_response(content=content)
            return

        swapped = data
        for face in faces:
            swapped = await asyncio.to_thread(
                self.swapper.swap,
                img=swapped,
                target_face=face,
                source_face=self.kirk,
                paste_back=True,
            )
            swapped = await asyncio.to_thread(
                self.face_enhancer.enhance_face,
                target_face=face,
                temp_vision_frame=swapped,
            )

        swapped = Image.fromarray(swapped)
        bytes_io = io.BytesIO()
        swapped.save(bytes_io, "jpeg")
        bytes_io.seek(0)

        if (fileSize := bytes_io.getbuffer().nbytes) > MAX_SIZE:
            content = f"The kirkified image is too large to send: ({fileSize / (1024 * 1024):.1f}MB)"
            await interaction.edit_original_response(content=content)
            return

        discordFile = discord.File(bytes_io, "kirkified.png")

        await interaction.delete_original_response()
        await interaction.followup.send(file=discordFile)

    async def cog_load(self):
        # Ensure models are downloaded
        face_enhancer = "gfpgan_1.4"
        await download(SWAPPER["url"], SWAPPER["path"])
        await download(INSIGHT["url"], INSIGHT["path"], INSIGHT.get("extract"))
        await download(
            FACE_ENHANCER_MODELS[face_enhancer]["url"],
            FACE_ENHANCER_MODELS[face_enhancer]["path"],
        )

        self.swapper = await asyncio.to_thread(FaceSwapper, SWAPPER["path"])
        self.face_analyser = await asyncio.to_thread(FaceAnalyser, INSIGHT["path"])
        self.face_enhancer = await asyncio.to_thread(
            FaceEnhancer, FACE_ENHANCER_MODELS[face_enhancer]
        )

        kirkPath = str(files("shackbot.static").joinpath("kirk.jpg"))
        with Image.open(kirkPath) as kirk:
            kirk.load()
        kirk = np.array(kirk)

        faces = await asyncio.to_thread(
            self.face_analyser.analyze,
            kirk,
            extract_embedding=True,
            extract_landmarks=False,
        )
        if len(faces) == 0:
            raise Exception("No face found in kirk.jpg")

        self.kirk = faces[0]

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
