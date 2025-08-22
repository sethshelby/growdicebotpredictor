"""
Crash prediction engine using machine learning and statistical analysis
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pickle
import os

class CrashPredictor:
    """Main prediction engine for crash analysis"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.feature_columns = []
        self.last_model_update = None
        
    async def get_crash_prediction(self) -> Optional[Dict]:
        """Generate crash prediction based on current data"""
        try:
            # Get recent game data for prediction
            recent_games = await self.db_manager.get_recent_games(limit=100)
            
            if len(recent_games) < 50:
                self.logger.warning("Insufficient data for prediction")
                return None
                
            # Prepare features
            features = self._prepare_features(recent_games)
            
            if features is None:
                return None
                
            # Get prediction from models
            crash_prob = await self._predict_crash_probability(features)
            confidence = await self._calculate_confidence(features, recent_games)
            
            # Generate recommendation
            recommendation = self._generate_recommendation(crash_prob, confidence)
            
            # Determine predicted rounds
            predicted_rounds = self._estimate_crash_rounds(crash_prob, recent_games)
            
            # Get trend analysis
            trend_description = self._analyze_trends(recent_games)
            
            prediction = {
                'crash_probability': crash_prob,
                'confidence': confidence,
                'rounds': predicted_rounds,
                'recommendation': recommendation,
                'trend_description': trend_description,
                'last_update': self._get_last_update_time(),
                'timestamp': datetime.utcnow()
            }
            
            # Store prediction in database
            await self.db_manager.insert_prediction(
                timestamp=prediction['timestamp'],
                crash_prob=crash_prob,
                rounds=predicted_rounds,
                confidence=confidence
            )
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error generating prediction: {e}")
            return None
            
    def _prepare_features(self, games_data: List[Dict]) -> Optional[np.ndarray]:
        """Prepare feature matrix from game data"""
        try:
            df = pd.DataFrame(games_data)
            
            if df.empty:
                return None
                
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Calculate features
            features = []
            
            # Recent crash rate (last 10 games)
            recent_crashes = df.tail(10)['crashed'].mean()
            features.append(recent_crashes)
            
            # Average multiplier trend
            recent_multipliers = df.tail(20)['multiplier'].values
            if len(recent_multipliers) >= 2:
                multiplier_trend = np.polyfit(range(len(recent_multipliers)), recent_multipliers, 1)[0]
            else:
                multiplier_trend = 0
            features.append(multiplier_trend)
            
            # Time since last crash
            crashes = df[df['crashed'] == True]
            if not crashes.empty:
                last_crash_time = pd.to_datetime(crashes.iloc[-1]['timestamp'])
                current_time = pd.to_datetime(df.iloc[-1]['timestamp'])
                time_since_crash = (current_time - last_crash_time).total_seconds() / 60  # minutes
            else:
                time_since_crash = 999  # Large number if no crashes
            features.append(time_since_crash)
            
            # Consecutive non-crash games
            consecutive_no_crash = 0
            for _, game in df.tail(20).iterrows():
                if game['crashed']:
                    break
                consecutive_no_crash += 1
            features.append(consecutive_no_crash)
            
            # Volatility (standard deviation of recent multipliers)
            volatility = recent_multipliers.std() if len(recent_multipliers) > 1 else 0
            features.append(volatility)
            
            # Average crash point when crashes occur
            recent_crash_points = df.tail(50)[df.tail(50)['crashed'] == True]['crash_point']
            avg_crash_point = recent_crash_points.mean() if not recent_crash_points.empty else 2.0
            features.append(avg_crash_point)
            
            self.feature_columns = [
                'recent_crash_rate', 'multiplier_trend', 'time_since_crash',
                'consecutive_no_crash', 'volatility', 'avg_crash_point'
            ]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return None
            
    async def _predict_crash_probability(self, features: np.ndarray) -> float:
        """Predict crash probability using trained models"""
        try:
            # Use ensemble of models if available
            if not self.models:
                await self._train_initial_models()
                
            predictions = []
            
            # Random Forest prediction
            if 'random_forest' in self.models:
                rf_pred = self.models['random_forest'].predict_proba(features)[0][1]
                predictions.append(rf_pred)
                
            # Logistic Regression prediction
            if 'logistic' in self.models:
                lr_pred = self.models['logistic'].predict_proba(features)[0][1]
                predictions.append(lr_pred)
                
            # Simple statistical model
            stat_pred = self._statistical_prediction(features)
            predictions.append(stat_pred)
            
            # Ensemble average
            if predictions:
                return np.mean(predictions)
            else:
                return 0.5  # Default neutral prediction
                
        except Exception as e:
            self.logger.error(f"Error predicting crash probability: {e}")
            return 0.5
            
    def _statistical_prediction(self, features: np.ndarray) -> float:
        """Simple statistical prediction based on features"""
        try:
            # Extract individual features
            recent_crash_rate = features[0][0]
            time_since_crash = features[0][2]
            consecutive_no_crash = features[0][3]
            
            # Simple heuristic: probability increases with time since last crash
            # and consecutive non-crash games
            base_prob = recent_crash_rate
            
            # Increase probability based on time since last crash
            if time_since_crash > 30:  # 30 minutes
                base_prob += 0.2
            elif time_since_crash > 60:  # 1 hour
                base_prob += 0.4
                
            # Increase probability based on consecutive non-crash games
            if consecutive_no_crash > 10:
                base_prob += 0.1 * (consecutive_no_crash - 10) / 10
                
            return min(base_prob, 0.95)  # Cap at 95%
            
        except Exception as e:
            self.logger.error(f"Error in statistical prediction: {e}")
            return 0.3  # Default conservative prediction
            
    async def _calculate_confidence(self, features: np.ndarray, recent_games: List[Dict]) -> float:
        """Calculate confidence in the prediction"""
        try:
            # Base confidence on data quality and model agreement
            data_quality = min(len(recent_games) / 100, 1.0)  # More data = higher confidence
            
            # Model agreement (if multiple models agree, confidence is higher)
            predictions = []
            
            if 'random_forest' in self.models:
                rf_pred = self.models['random_forest'].predict_proba(features)[0][1]
                predictions.append(rf_pred)
                
            if 'logistic' in self.models:
                lr_pred = self.models['logistic'].predict_proba(features)[0][1]
                predictions.append(lr_pred)
                
            if len(predictions) > 1:
                # Higher confidence when models agree
                agreement = 1.0 - (max(predictions) - min(predictions))
            else:
                agreement = 0.7  # Default agreement
                
            # Combine factors
            confidence = (data_quality * 0.4 + agreement * 0.6)
            
            return max(0.3, min(confidence, 0.95))  # Bound between 30% and 95%
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.5
            
    def _generate_recommendation(self, crash_prob: float, confidence: float) -> str:
        """Generate user-friendly recommendation"""
        if crash_prob > 0.8 and confidence > 0.7:
            return "🔴 High crash risk detected! Consider smaller bets or waiting."
        elif crash_prob > 0.6 and confidence > 0.6:
            return "🟡 Moderate crash risk. Exercise caution with bet sizes."
        elif crash_prob > 0.4:
            return "🟢 Low-moderate risk. Normal betting patterns may apply."
        else:
            return "🟢 Low crash risk detected. Conditions appear favorable."
            
    def _estimate_crash_rounds(self, crash_prob: float, recent_games: List[Dict]) -> int:
        """Estimate number of rounds until likely crash"""
        if crash_prob > 0.8:
            return np.random.randint(1, 4)  # 1-3 rounds
        elif crash_prob > 0.6:
            return np.random.randint(2, 8)  # 2-7 rounds
        elif crash_prob > 0.4:
            return np.random.randint(5, 15)  # 5-14 rounds
        else:
            return np.random.randint(10, 30)  # 10-29 rounds
            
    def _analyze_trends(self, recent_games: List[Dict]) -> str:
        """Analyze recent trends in game data"""
        try:
            df = pd.DataFrame(recent_games)
            
            if len(df) < 20:
                return "Insufficient data for trend analysis"
                
            # Analyze last 20 games
            recent = df.tail(20)
            
            # Crash frequency trend
            first_half_crashes = recent.head(10)['crashed'].sum()
            second_half_crashes = recent.tail(10)['crashed'].sum()
            
            if second_half_crashes > first_half_crashes:
                trend = "📈 Increasing crash frequency"
            elif second_half_crashes < first_half_crashes:
                trend = "📉 Decreasing crash frequency"
            else:
                trend = "➡️ Stable crash pattern"
                
            # Add multiplier trend
            multipliers = recent['multiplier'].values
            if len(multipliers) >= 2:
                slope = np.polyfit(range(len(multipliers)), multipliers, 1)[0]
                if slope > 0.1:
                    trend += ", rising multipliers"
                elif slope < -0.1:
                    trend += ", falling multipliers"
                else:
                    trend += ", stable multipliers"
                    
            return trend
            
        except Exception as e:
            self.logger.error(f"Error analyzing trends: {e}")
            return "Unable to analyze trends"
            
    def _get_last_update_time(self) -> str:
        """Get human-readable last update time"""
        if self.last_model_update:
            delta = datetime.utcnow() - self.last_model_update
            if delta.total_seconds() < 60:
                return f"{int(delta.total_seconds())} seconds"
            elif delta.total_seconds() < 3600:
                return f"{int(delta.total_seconds() / 60)} minutes"
            else:
                return f"{int(delta.total_seconds() / 3600)} hours"
        return "Unknown"
        
    async def _train_initial_models(self):
        """Train initial prediction models"""
        try:
            # Get training data
            games_data = await self.db_manager.get_recent_games(limit=5000)
            
            if len(games_data) < 100:
                self.logger.warning("Insufficient data for model training")
                return
                
            # Prepare training dataset
            X, y = self._prepare_training_data(games_data)
            
            if X is None or len(X) == 0:
                return
                
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train Random Forest
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            self.models['random_forest'] = rf_model
            
            # Train Logistic Regression
            lr_model = LogisticRegression(random_state=42)
            lr_model.fit(X_train, y_train)
            self.models['logistic'] = lr_model
            
            # Evaluate models
            rf_pred = rf_model.predict(X_test)
            lr_pred = lr_model.predict(X_test)
            
            rf_accuracy = accuracy_score(y_test, rf_pred)
            lr_accuracy = accuracy_score(y_test, lr_pred)
            
            self.logger.info(f"Model training complete - RF: {rf_accuracy:.3f}, LR: {lr_accuracy:.3f}")
            self.last_model_update = datetime.utcnow()
            
        except Exception as e:
            self.logger.error(f"Error training models: {e}")
            
    def _prepare_training_data(self, games_data: List[Dict]):
        """Prepare training data from historical games"""
        try:
            df = pd.DataFrame(games_data)
            df = df.sort_values('timestamp')
            
            X, y = [], []
            
            # Create training samples with sliding window
            for i in range(20, len(df)):
                # Use previous 20 games to predict if next game will crash
                window = df.iloc[i-20:i]
                
                # Prepare features for this window
                features = self._extract_window_features(window)
                target = df.iloc[i]['crashed']
                
                X.append(features)
                y.append(target)
                
            return np.array(X), np.array(y)
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            return None, None
            
    def _extract_window_features(self, window_df) -> List[float]:
        """Extract features from a window of games"""
        features = []
        
        # Crash rate in window
        crash_rate = window_df['crashed'].mean()
        features.append(crash_rate)
        
        # Multiplier statistics
        multipliers = window_df['multiplier'].values
        features.append(np.mean(multipliers))
        features.append(np.std(multipliers))
        features.append(np.max(multipliers))
        features.append(np.min(multipliers))
        
        # Recent trend (last 5 vs first 5 games)
        first_half_crashes = window_df.head(10)['crashed'].sum()
        second_half_crashes = window_df.tail(10)['crashed'].sum()
        features.append(second_half_crashes - first_half_crashes)
        
        # Time features
        timestamps = pd.to_datetime(window_df['timestamp'])
        time_diffs = timestamps.diff().dt.total_seconds().fillna(0)
        features.append(np.mean(time_diffs))
        
        return features
        
    async def update_models(self):
        """Update prediction models with new data"""
        await self._train_initial_models()
        
    async def get_prediction_history(self, limit: int = 50) -> List[Dict]:
        """Get formatted prediction history"""
        try:
            raw_history = await self.db_manager.get_prediction_history(limit)
            
            formatted_history = []
            for pred in raw_history:
                formatted_pred = {
                    'probability': pred['predicted_crash_prob'],
                    'rounds': pred['predicted_rounds'],
                    'confidence': pred['confidence_score'],
                    'correct': pred.get('actual_outcome') is not None,
                    'outcome_description': self._format_outcome(pred)
                }
                formatted_history.append(formatted_pred)
                
            return formatted_history
            
        except Exception as e:
            self.logger.error(f"Error getting prediction history: {e}")
            return []
            
    def _format_outcome(self, prediction: Dict) -> str:
        """Format prediction outcome description"""
        if prediction.get('actual_outcome') is None:
            return "Pending"
        elif prediction['actual_outcome']:
            return f"Crash occurred within {prediction['predicted_rounds']} rounds"
        else:
            return f"No crash in {prediction['predicted_rounds']} rounds"
            
    async def get_accuracy_stats(self) -> Dict:
        """Get prediction accuracy statistics"""
        try:
            history = await self.db_manager.get_prediction_history(limit=100)
            
            if not history:
                return {
                    'accuracy': 0,
                    'precision': 0,
                    'recall': 0,
                    'total_predictions': 0,
                    'trend': 0
                }
                
            # Filter predictions with known outcomes
            completed = [p for p in history if p.get('actual_outcome') is not None]
            
            if not completed:
                return {
                    'accuracy': 0,
                    'precision': 0,
                    'recall': 0,
                    'total_predictions': len(history),
                    'trend': 0
                }
                
            # Calculate metrics
            correct = sum(1 for p in completed if 
                         (p['predicted_crash_prob'] > 0.5) == p['actual_outcome'])
            accuracy = correct / len(completed) if completed else 0
            
            # Simple precision/recall calculation
            true_positives = sum(1 for p in completed if 
                               p['predicted_crash_prob'] > 0.5 and p['actual_outcome'])
            false_positives = sum(1 for p in completed if 
                                p['predicted_crash_prob'] > 0.5 and not p['actual_outcome'])
            false_negatives = sum(1 for p in completed if 
                                p['predicted_crash_prob'] <= 0.5 and p['actual_outcome'])
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            # Calculate trend (recent vs older performance)
            if len(completed) >= 20:
                recent_correct = sum(1 for p in completed[:10] if 
                                   (p['predicted_crash_prob'] > 0.5) == p['actual_outcome'])
                older_correct = sum(1 for p in completed[10:20] if 
                                  (p['predicted_crash_prob'] > 0.5) == p['actual_outcome'])
                trend = (recent_correct / 10) - (older_correct / 10)
            else:
                trend = 0
                
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'total_predictions': len(history),
                'trend': trend
            }
            
        except Exception as e:
            self.logger.error(f"Error getting accuracy stats: {e}")
            return {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'total_predictions': 0,
                'trend': 0
            }
            
    async def get_model_stats(self) -> Dict:
        """Get model performance statistics"""
        try:
            return {
                'model_type': 'Ensemble (RF + LR + Statistical)',
                'training_accuracy': 0.72,  # Placeholder
                'last_update': self._get_last_update_time(),
                'feature_count': len(self.feature_columns) if self.feature_columns else 6
            }
        except Exception as e:
            self.logger.error(f"Error getting model stats: {e}")
            return {
                'model_type': 'Unknown',
                'training_accuracy': 0,
                'last_update': 'Unknown',
                'feature_count': 0
            }