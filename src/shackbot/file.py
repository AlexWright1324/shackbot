import os
import aiofiles
from aiofiles import os as aioos
import aiohttp
from async_unzip import unzipper

CHUNK_SIZE = 64 * 1024


async def download(url: str, path: str, extract: bool | None = None):
    if await aioos.path.exists(path):
        # File already exists, no need to download
        return

    # Create directories if they don't exist
    await aioos.makedirs(os.path.dirname(path), exist_ok=True)

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            # TODO: Fallback mechanism in case of failure
            response.raise_for_status()

            if extract:
                print(f"Extracting model from {url} to {path}...")
                await unzipper.unzip_stream(
                    response.content.iter_chunked(CHUNK_SIZE),
                    path=path,
                    in_memory=True,
                )
                return

            print(f"Downloading model from {url} to {path}...")
            async with aiofiles.open(path, "wb") as f:
                async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                    await f.write(chunk)
