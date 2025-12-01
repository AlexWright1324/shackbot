import discord
from discord.ext import commands


class Nerd(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.target_user_id = 285808510028087297

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.id != self.target_user_id:
            return

        # TODO: end formatting, only add space before emoji if needed
        # rstrip
        await message.reply(f"{message.content.rstrip()} ☝️🤓")


async def setup(bot: commands.Bot):
    await bot.add_cog(Nerd(bot))
