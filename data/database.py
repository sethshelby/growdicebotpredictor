"""
Database management for game data storage
"""

import aiosqlite
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

class DatabaseManager:
    """Manages SQLite database operations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize database and create tables"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Create game_data table
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS game_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        multiplier REAL NOT NULL,
                        crashed BOOLEAN NOT NULL,
                        crash_point REAL DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create predictions table
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        predicted_crash_prob REAL NOT NULL,
                        predicted_rounds INTEGER NOT NULL,
                        actual_outcome BOOLEAN,
                        confidence_score REAL NOT NULL,
                        model_version TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for better performance
                await db.execute('''
                    CREATE INDEX IF NOT EXISTS idx_game_timestamp 
                    ON game_data(timestamp)
                ''')
                
                await db.execute('''
                    CREATE INDEX IF NOT EXISTS idx_prediction_timestamp 
                    ON predictions(timestamp)
                ''')
                
                await db.commit()
                self.logger.info("Database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Database initialization error: {e}")
            raise
            
    async def insert_game_data(self, timestamp: datetime, multiplier: float, 
                             crashed: bool, crash_point: float = 0):
        """Insert new game data"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute('''
                    INSERT INTO game_data (timestamp, multiplier, crashed, crash_point)
                    VALUES (?, ?, ?, ?)
                ''', (timestamp, multiplier, crashed, crash_point))
                await db.commit()
        except Exception as e:
            self.logger.error(f"Error inserting game data: {e}")
            
    async def insert_prediction(self, timestamp: datetime, crash_prob: float,
                              rounds: int, confidence: float, model_version: str = "v1.0"):
        """Insert new prediction"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute('''
                    INSERT INTO predictions (timestamp, predicted_crash_prob, 
                                           predicted_rounds, confidence_score, model_version)
                    VALUES (?, ?, ?, ?, ?)
                ''', (timestamp, crash_prob, rounds, confidence, model_version))
                await db.commit()
        except Exception as e:
            self.logger.error(f"Error inserting prediction: {e}")
            
    async def get_recent_games(self, limit: int = 1000) -> List[Dict]:
        """Get recent game data"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute('''
                    SELECT * FROM game_data 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            self.logger.error(f"Error getting recent games: {e}")
            return []
            
    async def get_crash_patterns(self, hours_back: int = 24) -> List[Dict]:
        """Get crash patterns from specified time period"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
            
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute('''
                    SELECT * FROM game_data 
                    WHERE timestamp >= ? AND crashed = 1
                    ORDER BY timestamp DESC
                ''', (cutoff_time,))
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            self.logger.error(f"Error getting crash patterns: {e}")
            return []
            
    async def get_prediction_history(self, limit: int = 50) -> List[Dict]:
        """Get prediction history"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute('''
                    SELECT * FROM predictions 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            self.logger.error(f"Error getting prediction history: {e}")
            return []
            
    async def update_prediction_outcome(self, prediction_id: int, actual_outcome: bool):
        """Update prediction with actual outcome"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute('''
                    UPDATE predictions 
                    SET actual_outcome = ? 
                    WHERE id = ?
                ''', (actual_outcome, prediction_id))
                await db.commit()
        except Exception as e:
            self.logger.error(f"Error updating prediction outcome: {e}")
            
    async def get_collection_stats(self) -> Dict:
        """Get data collection statistics"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Total games
                cursor = await db.execute('SELECT COUNT(*) FROM game_data')
                total_games = (await cursor.fetchone())[0]
                
                # Total crashes
                cursor = await db.execute('SELECT COUNT(*) FROM game_data WHERE crashed = 1')
                total_crashes = (await cursor.fetchone())[0]
                
                # Date range
                cursor = await db.execute('''
                    SELECT MIN(timestamp), MAX(timestamp) FROM game_data
                ''')
                date_range = await cursor.fetchone()
                
                crash_rate = total_crashes / total_games if total_games > 0 else 0
                
                date_range_str = "No data"
                if date_range[0] and date_range[1]:
                    start_date = datetime.fromisoformat(date_range[0]).strftime('%Y-%m-%d')
                    end_date = datetime.fromisoformat(date_range[1]).strftime('%Y-%m-%d')
                    date_range_str = f"{start_date} to {end_date}"
                
                return {
                    'total_games': total_games,
                    'total_crashes': total_crashes,
                    'crash_rate': crash_rate,
                    'date_range': date_range_str
                }
                
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            return {
                'total_games': 0,
                'total_crashes': 0,
                'crash_rate': 0,
                'date_range': 'Error'
            }
            
    async def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data to manage database size"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            async with aiosqlite.connect(self.db_path) as db:
                # Delete old game data
                cursor = await db.execute('''
                    DELETE FROM game_data 
                    WHERE timestamp < ?
                ''', (cutoff_date,))
                
                # Delete old predictions
                await db.execute('''
                    DELETE FROM predictions 
                    WHERE timestamp < ?
                ''', (cutoff_date,))
                
                await db.commit()
                self.logger.info(f"Cleaned up data older than {cutoff_date}")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
            
    async def close(self):
        """Close database connections"""
        # SQLite connections are closed automatically with async context managers
        self.logger.info("Database manager closed")