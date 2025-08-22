"""
Growdice Crash Predictor Discord Bot
Main entry point for the bot application
"""

import os
import asyncio
import logging
from dotenv import load_dotenv

from bot.discord_bot import GrowdiceBot
from config.settings import Settings

def setup_logging():
    """Configure logging for the application"""
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('bot.log'),
            logging.StreamHandler()
        ]
    )

async def main():
    """Main application entry point"""
    # Load environment variables
    load_dotenv()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load configuration
    settings = Settings()
    
    # Validate required environment variables
    if not settings.discord_token:
        logger.error("DISCORD_TOKEN not found in environment variables")
        return
    
    logger.info("Starting Growdice Crash Predictor Bot...")
    
    # Initialize and run the bot
    bot = GrowdiceBot(settings)
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot error: {e}")
    finally:
        await bot.close()

if __name__ == "__main__":
    asyncio.run(main())