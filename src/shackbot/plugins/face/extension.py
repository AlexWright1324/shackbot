"""Discord cog for face-swapping images with Kirk's face."""

import asyncio
import io
import logging
from importlib.resources import files
from typing import Self

import discord
import numpy as np
from discord import app_commands
from discord.ext import commands
from PIL import Image, ImageSequence

from shackbot.file import download

from .Face import Face as FaceData
from .FaceAnalyser import FaceAnalyser
from .FaceEnhancer import FaceEnhancer
from .FaceSwapper import FaceSwapper
from .models import FACE_ENHANCER_MODELS, INSIGHT, SWAPPER

logger = logging.getLogger(__name__)

# Configuration
MAX_WORKERS = 1
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB Discord limit
SUPPORTED_FORMATS = ["PNG", "JPEG", "WEBP", "GIF"]
DEFAULT_FACE_ENHANCER = "gfpgan_1.4"

# Error messages
ERR_NO_FACE = "No face detected in the media"
ERR_PROCESSING = "An error occurred while processing your image"
ERR_UNSUPPORTED = (
    "Unsupported media type.\nPlease upload a valid: PNG, JPEG, WebP or animated GIF"
)
ERR_TOO_LARGE = "The kirkified image is too large to send: ({size:.1f}MB)"
ERR_GIF_FAILED = "Could not process GIF frames"


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Face(bot))


class Face(commands.Cog):
    """Cog for face-swapping images with Kirk's face."""

    swapper: FaceSwapper
    face_analyser: FaceAnalyser
    face_enhancer: FaceEnhancer
    kirk: FaceData

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.queue: asyncio.Queue[tuple[discord.Interaction, discord.Attachment]] = (
            asyncio.Queue()
        )
        self.workers: list[asyncio.Task[None]] = []

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    async def cog_load(self) -> None:
        """Initialize models and start worker tasks."""
        await self._download_models()
        await self._initialize_models()
        await self._load_source_face()
        self.workers = [asyncio.create_task(self._worker()) for _ in range(MAX_WORKERS)]

    async def cog_unload(self) -> None:
        """Cancel worker tasks on unload."""
        for worker in self.workers:
            worker.cancel()

    async def _download_models(self) -> None:
        """Download required model files."""
        enhancer_config = FACE_ENHANCER_MODELS[DEFAULT_FACE_ENHANCER]
        await asyncio.gather(
            download(SWAPPER["url"], SWAPPER["path"]),
            download(INSIGHT["url"], INSIGHT["path"], INSIGHT.get("extract")),
            download(enhancer_config["url"], enhancer_config["path"]),
        )

    async def _initialize_models(self) -> None:
        """Initialize face processing models."""
        enhancer_config = FACE_ENHANCER_MODELS[DEFAULT_FACE_ENHANCER]
        self.swapper, self.face_analyser, self.face_enhancer = await asyncio.gather(
            asyncio.to_thread(FaceSwapper, SWAPPER["path"]),
            asyncio.to_thread(FaceAnalyser, INSIGHT["path"]),
            asyncio.to_thread(FaceEnhancer, enhancer_config),
        )

    async def _load_source_face(self) -> None:
        """Load Kirk's face as the source for swapping."""
        kirk_path = str(files("shackbot.static").joinpath("kirk.jpg"))
        with Image.open(kirk_path) as img:
            img.load()
            kirk_data = np.array(img)

        faces = await self._analyze_faces(kirk_data)
        if not faces:
            raise RuntimeError("No face found in kirk.jpg")
        self.kirk = faces[0]

    # ─────────────────────────────────────────────────────────────────────────
    # Worker
    # ─────────────────────────────────────────────────────────────────────────

    async def _worker(self) -> None:
        """Process queued face-swap requests."""
        while True:
            interaction, attachment = await self.queue.get()
            try:
                await self._process_request(interaction, attachment)
            except Exception:
                logger.exception("Error processing face swap request")
                await self._send_error(interaction, ERR_PROCESSING)
            finally:
                self.queue.task_done()

    async def _send_error(self, interaction: discord.Interaction, message: str) -> None:
        """Send an error message to the user."""
        try:
            await interaction.edit_original_response(content=message)
        except discord.errors.NotFound:
            await interaction.followup.send(message, ephemeral=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Face Processing
    # ─────────────────────────────────────────────────────────────────────────

    async def _analyze_faces(self, image: np.ndarray) -> list[FaceData]:
        """Detect and analyze faces in an image."""
        return await asyncio.to_thread(
            self.face_analyser.analyze,
            image,
            extract_embedding=True,
            extract_landmarks=False,
        )

    async def _swap_and_enhance(
        self, image: np.ndarray, faces: list[FaceData]
    ) -> np.ndarray:
        """Swap and enhance all faces in an image."""
        result = image
        for face in faces:
            result = await asyncio.to_thread(
                self.swapper.swap,
                img=result,
                target_face=face,
                source_face=self.kirk,
                paste_back=True,
            )
            result = await asyncio.to_thread(
                self.face_enhancer.enhance_face,
                target_face=face,
                temp_vision_frame=result,
            )
        return result

    async def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame: detect faces and swap if found."""
        faces = await self._analyze_faces(frame)
        if faces:
            return await self._swap_and_enhance(frame, faces)
        return frame

    # ─────────────────────────────────────────────────────────────────────────
    # Image Processing
    # ─────────────────────────────────────────────────────────────────────────

    async def _process_static(
        self, media: Image.Image
    ) -> tuple[io.BytesIO | None, str]:
        """Process a static image (PNG, JPEG, WebP)."""
        media.load()
        data = np.array(media)

        faces = await self._analyze_faces(data)
        if not faces:
            return None, ERR_NO_FACE

        swapped = await self._swap_and_enhance(data, faces)

        buffer = io.BytesIO()
        Image.fromarray(swapped).save(buffer, "JPEG", quality=95)
        buffer.seek(0)
        return buffer, "kirkified.jpg"

    async def _process_gif(self, media: Image.Image) -> tuple[io.BytesIO | None, str]:
        """Process an animated GIF."""
        duration = media.info.get("duration", 100)
        frames: list[Image.Image] = []

        for frame in ImageSequence.Iterator(media):
            rgb_data = np.array(frame.convert("RGB"))
            processed = await self._process_frame(rgb_data)
            frames.append(Image.fromarray(processed))

        if not frames:
            return None, ERR_GIF_FAILED

        buffer = io.BytesIO()
        frames[0].save(
            buffer,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
            optimize=True,
        )
        buffer.seek(0)
        return buffer, "kirkified.gif"

    # ─────────────────────────────────────────────────────────────────────────
    # Request Handling
    # ─────────────────────────────────────────────────────────────────────────

    async def _process_request(
        self, interaction: discord.Interaction, attachment: discord.Attachment
    ) -> None:
        """Process a kirkify request from start to finish."""
        # Notify user that processing has started
        await interaction.edit_original_response(content="Processing your image...")

        # Read and validate media
        data = await attachment.read()
        try:
            with Image.open(io.BytesIO(data), formats=SUPPORTED_FORMATS) as media:
                if media.format == "GIF":
                    result, filename = await self._process_gif(media)
                else:
                    result, filename = await self._process_static(media)
        except Exception:
            await self._send_error(interaction, ERR_UNSUPPORTED)
            return

        # Handle processing failure
        if result is None:
            await self._send_error(interaction, filename)
            return

        # Check file size
        file_size = result.getbuffer().nbytes
        if file_size > MAX_FILE_SIZE:
            size_mb = file_size / (1024 * 1024)
            await self._send_error(interaction, ERR_TOO_LARGE.format(size=size_mb))
            return

        # Send result
        await interaction.delete_original_response()
        await interaction.followup.send(file=discord.File(result, filename))

    # ─────────────────────────────────────────────────────────────────────────
    # Commands
    # ─────────────────────────────────────────────────────────────────────────

    @app_commands.command(name="kirkify")
    @app_commands.describe(attachment="Image to kirkify")
    @app_commands.allowed_installs(guilds=True, users=True)
    @app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
    async def command(
        self: Self, interaction: discord.Interaction, attachment: discord.Attachment
    ) -> None:
        """Replace faces in an image with Kirk's face."""
        await interaction.response.defer(ephemeral=True)

        position = self.queue.qsize()
        await self.queue.put((interaction, attachment))

        await interaction.followup.send(
            f"Your image has been queued for kirkification!\nPosition in queue: {position}",
            ephemeral=True,
        )
