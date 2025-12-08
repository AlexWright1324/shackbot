import discord
from discord.ext import commands


class Nerd(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.target_user_id = 285808510028087297
        self.muted_at = None

    """
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.id != self.target_user_id:
            return

        # TODO: end formatting, only add space before emoji if needed
        # rstrip
        await message.reply(f"{message.content.rstrip()} ☝️🤓")
    """

    @commands.Cog.listener()
    async def on_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ):
        if member.id != self.target_user_id:
            return

        if before.channel is None or after.channel is None:
            # Allow moving between channels without triggering
            self.muted_at = None
            return

        if before.mute == after.mute:
            return

        if after.mute:
            self.muted_at = discord.utils.utcnow()
            return

        if self.muted_at is None:
            return

        muted_duration = discord.utils.utcnow() - self.muted_at
        muted_seconds = muted_duration.total_seconds()
        if muted_seconds < 60 * 5:
            return

        hours = int(muted_seconds // 3600)
        minutes = int((muted_seconds % 3600) // 60)
        seconds = int(muted_seconds % 60)
        time_parts = []
        if hours > 0:
            time_parts.append(f"{hours}h")
        if minutes > 0:
            time_parts.append(f"{minutes}m")
        if seconds > 0:
            time_parts.append(f"{seconds}s")
        time_str = " ".join(time_parts)

        channels = member.guild.text_channels
        if member.guild.system_channel is not None:
            channels.append(member.guild.system_channel)

        # Prioritize channel named "general"
        channels.sort(key=lambda c: c.name != "general")

        for channel in channels:
            if not channel.permissions_for(member.guild.me).send_messages:
                continue

            await channel.send(f"Bro had din dins for {time_str} ☝️🤓")
            break


async def setup(bot: commands.Bot):
    await bot.add_cog(Nerd(bot))
