"""
Discord bot commands for crash prediction
"""

import discord
from discord.ext import commands
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64

class PredictionCommands(commands.Cog):
    """Commands for crash prediction functionality"""
    
    def __init__(self, bot):
        self.bot = bot
        self.logger = logging.getLogger(__name__)
        
    @commands.command(name='predict')
    async def predict_crash(self, ctx):
        """Generate crash prediction"""
        try:
            # Get prediction from the predictor
            prediction = await self.bot.predictor.get_crash_prediction()
            
            if not prediction:
                await ctx.send("❌ Unable to generate prediction. Insufficient data.")
                return
                
            # Create embed for prediction
            embed = discord.Embed(
                title="🎯 Crash Prediction Analysis",
                color=0x00ff00 if prediction['confidence'] > 0.7 else 0xffff00,
                timestamp=datetime.utcnow()
            )
            
            # Add prediction fields
            embed.add_field(
                name="📊 Crash Probability",
                value=f"{prediction['crash_probability']:.1%} within next {prediction['rounds']} rounds",
                inline=False
            )
            
            embed.add_field(
                name="🎯 Confidence Level",
                value=f"{'High' if prediction['confidence'] > 0.7 else 'Medium' if prediction['confidence'] > 0.5 else 'Low'} ({prediction['confidence']:.1%})",
                inline=True
            )
            
            embed.add_field(
                name="📈 Trend Analysis",
                value=prediction['trend_description'],
                inline=True
            )
            
            embed.add_field(
                name="💡 Recommendation",
                value=prediction['recommendation'],
                inline=False
            )
            
            # Add data freshness
            embed.add_field(
                name="🔄 Last Updated",
                value=f"{prediction['last_update']} ago",
                inline=True
            )
            
            # Add footer with disclaimer
            embed.set_footer(
                text="⚠️ For analysis purposes only. Gamble responsibly!",
                icon_url="https://cdn.discordapp.com/emojis/⚠️.png"
            )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            self.logger.error(f"Prediction command error: {e}")
            await ctx.send("❌ Error generating prediction. Please try again later.")
            
    @commands.command(name='history')
    async def prediction_history(self, ctx):
        """Show prediction accuracy history"""
        try:
            # Get historical data
            history = await self.bot.predictor.get_prediction_history(limit=10)
            accuracy_stats = await self.bot.predictor.get_accuracy_stats()
            
            if not history:
                await ctx.send("❌ No prediction history available yet.")
                return
                
            # Create embed for history
            embed = discord.Embed(
                title="📈 Prediction Accuracy Report",
                color=0x0099ff,
                timestamp=datetime.utcnow()
            )
            
            # Add recent predictions
            recent_predictions = ""
            for i, pred in enumerate(history[:5], 1):
                outcome_icon = "✅" if pred['correct'] else "❌"
                recent_predictions += f"{i}. {pred['probability']:.0%} crash chance → {outcome_icon} {pred['outcome_description']}\n"
                
            embed.add_field(
                name="🔍 Recent Predictions",
                value=recent_predictions or "No recent predictions",
                inline=False
            )
            
            # Add accuracy statistics
            embed.add_field(
                name="📊 Overall Performance",
                value=f"**Accuracy**: {accuracy_stats['accuracy']:.1%} (last {accuracy_stats['total_predictions']} predictions)\n"
                      f"**Precision**: {accuracy_stats['precision']:.1%}\n"
                      f"**Recall**: {accuracy_stats['recall']:.1%}",
                inline=True
            )
            
            # Add trend information
            embed.add_field(
                name="📈 Performance Trend",
                value=f"{'📈 Improving' if accuracy_stats['trend'] > 0 else '📉 Declining' if accuracy_stats['trend'] < 0 else '➡️ Stable'}",
                inline=True
            )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            self.logger.error(f"History command error: {e}")
            await ctx.send("❌ Error retrieving history. Please try again later.")
            
    @commands.command(name='stats')
    async def data_statistics(self, ctx):
        """Show data collection and model statistics"""
        try:
            # Get statistics
            data_stats = await self.bot.data_collector.get_collection_stats()
            model_stats = await self.bot.predictor.get_model_stats()
            
            # Create embed for statistics
            embed = discord.Embed(
                title="📊 Data & Model Statistics",
                color=0x9932cc,
                timestamp=datetime.utcnow()
            )
            
            # Data collection stats
            embed.add_field(
                name="📦 Data Collection",
                value=f"**Total Games**: {data_stats['total_games']:,}\n"
                      f"**Crashes Recorded**: {data_stats['total_crashes']:,}\n"
                      f"**Crash Rate**: {data_stats['crash_rate']:.1%}\n"
                      f"**Data Range**: {data_stats['date_range']}",
                inline=True
            )
            
            # Model performance stats
            embed.add_field(
                name="🤖 Model Performance",
                value=f"**Model Type**: {model_stats['model_type']}\n"
                      f"**Training Accuracy**: {model_stats['training_accuracy']:.1%}\n"
                      f"**Last Updated**: {model_stats['last_update']}\n"
                      f"**Features Used**: {model_stats['feature_count']}",
                inline=True
            )
            
            # System health
            embed.add_field(
                name="⚡ System Health",
                value=f"**Uptime**: {data_stats['uptime']}\n"
                      f"**Last Data Sync**: {data_stats['last_sync']}\n"
                      f"**Status**: {'🟢 Healthy' if data_stats['healthy'] else '🔴 Issues'}",
                inline=False
            )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            self.logger.error(f"Stats command error: {e}")
            await ctx.send("❌ Error retrieving statistics. Please try again later.")
            
    @commands.command(name='help')
    async def help_command(self, ctx):
        """Show available commands and usage"""
        embed = discord.Embed(
            title="🤖 Growdice Crash Predictor Bot",
            description="Advanced crash prediction using machine learning and statistical analysis",
            color=0x00ff88
        )
        
        # Add commands
        embed.add_field(
            name="📋 Available Commands",
            value="**!predict** - Get crash prediction with confidence levels\n"
                  "**!history** - View prediction accuracy and past results\n"
                  "**!stats** - Display data collection and model statistics\n"
                  "**!help** - Show this help message",
            inline=False
        )
        
        # Add usage tips
        embed.add_field(
            name="💡 Usage Tips",
            value="• Predictions are updated every 5 minutes\n"
                  "• Higher confidence means more reliable predictions\n"
                  "• Check history to see model performance\n"
                  "• Use stats to monitor data quality",
            inline=False
        )
        
        # Add disclaimer
        embed.add_field(
            name="⚠️ Important Disclaimer",
            value="This bot is for educational and analytical purposes only. "
                  "Predictions are not guaranteed and should not be used as financial advice. "
                  "Always gamble responsibly and within your means.",
            inline=False
        )
        
        embed.set_footer(text="Made with ❤️ for the gaming community")
        
        await ctx.send(embed=embed)

async def setup(bot):
    await bot.add_cog(PredictionCommands(bot))