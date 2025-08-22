"""
Configuration management for the Discord bot
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Settings:
    """Bot configuration settings"""
    
    # Discord settings
    discord_token: str = os.getenv('DISCORD_TOKEN', '')
    command_prefix: str = os.getenv('COMMAND_PREFIX', '!')
    
    # Database settings
    database_path: str = os.getenv('DATABASE_PATH', 'data/growdice.db')
    
    # Data collection settings
    data_update_interval: int = int(os.getenv('DATA_UPDATE_INTERVAL', '300'))  # 5 minutes
    
    # Logging settings
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    log_file: str = os.getenv('LOG_FILE', 'bot.log')
    
    # Prediction settings
    min_data_points: int = int(os.getenv('MIN_DATA_POINTS', '50'))
    prediction_confidence_threshold: float = float(os.getenv('PREDICTION_CONFIDENCE_THRESHOLD', '0.5'))
    
    # Model settings
    model_update_interval: int = int(os.getenv('MODEL_UPDATE_INTERVAL', '3600'))  # 1 hour
    max_training_samples: int = int(os.getenv('MAX_TRAINING_SAMPLES', '5000'))
    
    # API settings (for future use)
    growdice_api_key: Optional[str] = os.getenv('GROWDICE_API_KEY')
    api_rate_limit: int = int(os.getenv('API_RATE_LIMIT', '60'))  # requests per minute
    
    def validate(self) -> bool:
        """Validate required settings"""
        if not self.discord_token:
            return False
        return True