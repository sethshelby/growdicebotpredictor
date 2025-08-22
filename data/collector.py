"""
Data collection module for Growdice.net game data
"""

import aiohttp
import asyncio
import logging
from datetime import datetime, timedelta
import json
import random
from bs4 import BeautifulSoup

class DataCollector:
    """Collects game data from Growdice.net"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self.session = None
        self.last_collection = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            
    async def collect_latest_data(self):
        """Collect the latest game data"""
        async with self:
            try:
                # Get recent games data
                games_data = await self._fetch_recent_games()
                
                if games_data:
                    # Store in database
                    await self._store_game_data(games_data)
                    self.last_collection = datetime.utcnow()
                    self.logger.info(f"Collected {len(games_data)} game records")
                else:
                    self.logger.warning("No new game data collected")
                    
            except Exception as e:
                self.logger.error(f"Data collection error: {e}")
                
    async def _fetch_recent_games(self):
        """Fetch recent game data from Growdice.net"""
        try:
            # NOTE: This is a mock implementation
            # Replace with actual Growdice.net API endpoints or scraping logic
            
            # Simulate API endpoint (replace with real URL)
            url = "https://growdice.net/api/recent-games"  # Mock URL
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_game_data(data)
                else:
                    self.logger.error(f"HTTP {response.status} from Growdice API")
                    return None
                    
        except aiohttp.ClientError as e:
            self.logger.error(f"Network error fetching game data: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error parsing game data: {e}")
            return None
            
    def _parse_game_data(self, raw_data):
        """Parse raw game data into structured format"""
        games = []
        
        try:
            # Mock data parser - replace with actual parsing logic
            for game in raw_data.get('games', []):
                parsed_game = {
                    'timestamp': datetime.fromisoformat(game['timestamp']),
                    'multiplier': float(game['multiplier']),
                    'crashed': bool(game['crashed']),
                    'crash_point': float(game.get('crash_point', 0))
                }
                games.append(parsed_game)
                
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error(f"Error parsing game data: {e}")
            
        return games
        
    async def _store_game_data(self, games_data):
        """Store game data in database"""
        try:
            for game in games_data:
                await self.db_manager.insert_game_data(
                    timestamp=game['timestamp'],
                    multiplier=game['multiplier'],
                    crashed=game['crashed'],
                    crash_point=game['crash_point']
                )
        except Exception as e:
            self.logger.error(f"Error storing game data: {e}")
            
    async def get_collection_stats(self):
        """Get data collection statistics"""
        try:
            stats = await self.db_manager.get_collection_stats()
            
            # Calculate uptime
            if self.last_collection:
                uptime = datetime.utcnow() - self.last_collection
                uptime_str = f"{uptime.days}d {uptime.seconds//3600}h"
            else:
                uptime_str = "Unknown"
                
            return {
                'total_games': stats.get('total_games', 0),
                'total_crashes': stats.get('total_crashes', 0),
                'crash_rate': stats.get('crash_rate', 0),
                'date_range': stats.get('date_range', 'No data'),
                'uptime': uptime_str,
                'last_sync': self.last_collection.strftime('%H:%M:%S') if self.last_collection else 'Never',
                'healthy': stats.get('total_games', 0) > 0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            return {
                'total_games': 0,
                'total_crashes': 0,
                'crash_rate': 0,
                'date_range': 'Error',
                'uptime': 'Unknown',
                'last_sync': 'Error',
                'healthy': False
            }

    async def simulate_data_collection(self):
        """Simulate data collection for testing purposes"""
        self.logger.info("Simulating data collection...")
        
        # Generate mock game data
        mock_games = []
        base_time = datetime.utcnow() - timedelta(hours=1)
        
        for i in range(50):
            # Simulate realistic game patterns
            crashed = random.random() < 0.3  # 30% crash rate
            multiplier = random.uniform(1.0, 10.0) if not crashed else random.uniform(1.0, 3.0)
            crash_point = multiplier if crashed else 0
            
            game = {
                'timestamp': base_time + timedelta(minutes=i),
                'multiplier': multiplier,
                'crashed': crashed,
                'crash_point': crash_point
            }
            mock_games.append(game)
            
        await self._store_game_data(mock_games)
        self.last_collection = datetime.utcnow()
        self.logger.info(f"Simulated {len(mock_games)} game records")