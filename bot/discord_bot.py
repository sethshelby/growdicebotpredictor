"""
Discord bot implementation for Growdice crash prediction
"""

import discord
from discord.ext import commands, tasks
import logging
from datetime import datetime
import asyncio

from .commands import PredictionCommands
from data.collector import DataCollector
from data.database import DatabaseManager
from analysis.predictor import CrashPredictor

class GrowdiceBot(commands.Bot):
    """Main Discord bot class"""
    
    def __init__(self, settings):
        intents = discord.Intents.default()
        intents.message_content = True
        
        super().__init__(
            command_prefix=settings.command_prefix,
            intents=intents,
            help_command=None
        )
        
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.db_manager = DatabaseManager(settings.database_path)
        self.data_collector = DataCollector(self.db_manager)
        self.predictor = CrashPredictor(self.db_manager)
        
        # Setup commands
        self.add_cog(PredictionCommands(self))
        
    async def setup_hook(self):
        """Called when the bot is starting up"""
        self.logger.info("Setting up bot...")
        
        # Initialize database
        await self.db_manager.initialize()
        
        # Start background tasks
        self.data_update_task.start()
        self.model_update_task.start()
        
    async def on_ready(self):
        """Called when the bot is ready"""
        self.logger.info(f'{self.user} has connected to Discord!')
        self.logger.info(f'Bot is in {len(self.guilds)} guilds')
        
        # Set bot status
        await self.change_presence(
            activity=discord.Game("Analyzing Growdice patterns 📊")
        )
        
    async def on_command_error(self, ctx, error):
        """Handle command errors"""
        if isinstance(error, commands.CommandNotFound):
            await ctx.send("❌ Command not found. Use `!help` for available commands.")
        elif isinstance(error, commands.MissingRequiredArgument):
            await ctx.send("❌ Missing required argument. Use `!help` for command usage.")
        else:
            self.logger.error(f"Command error: {error}")
            await ctx.send("❌ An error occurred while processing the command.")
            
    @tasks.loop(seconds=300)  # Run every 5 minutes
    async def data_update_task(self):
        """Background task to collect new data"""
        try:
            self.logger.info("Starting data collection...")
            await self.data_collector.collect_latest_data()
            self.logger.info("Data collection completed")
        except Exception as e:
            self.logger.error(f"Data collection error: {e}")
            
    @tasks.loop(hours=1)  # Run every hour
    async def model_update_task(self):
        """Background task to update prediction models"""
        try:
            self.logger.info("Updating prediction models...")
            await self.predictor.update_models()
            self.logger.info("Model update completed")
        except Exception as e:
            self.logger.error(f"Model update error: {e}")
            
    async def start(self):
        """Start the bot"""
        await super().start(self.settings.discord_token)
        
    async def close(self):
        """Close the bot and cleanup"""
        self.logger.info("Shutting down bot...")
        
        # Cancel background tasks
        self.data_update_task.cancel()
        self.model_update_task.cancel()
        
        # Close database connection
        await self.db_manager.close()
        
        await super().close()