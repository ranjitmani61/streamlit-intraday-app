"""
Utility functions for the Professional Intraday Trading Assistant
"""

import pandas as pd
import numpy as np
import json
import csv
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

def load_configuration(config_file: str = 'config.py') -> Dict[str, Any]:
    """Load configuration from config file"""
    try:
        from config import (
            NSE_ALL_STOCKS, BSE_STOCKS, NIFTY_50_STOCKS, SECTORS,
            DEFAULT_WATCHLIST, TRADING_CONFIG, CHART_CONFIG,
            AI_CONFIG, TELEGRAM_CONFIG, ALERT_CONFIG,
            FIBONACCI_LEVELS, SUPPORT_RESISTANCE_CONFIG,
            MARKET_SESSIONS, DEFAULT_FEATURES, RISK_CONFIG,
            BACKTEST_CONFIG, PERFORMANCE_METRICS
        )
        
        return {
            'nse_stocks': NSE_ALL_STOCKS,
            'bse_stocks': BSE_STOCKS,
            'nifty_50': NIFTY_50_STOCKS,
            'sectors': SECTORS,
            'default_watchlist': DEFAULT_WATCHLIST,
            'trading_config': TRADING_CONFIG,
            'chart_config': CHART_CONFIG,
            'ai_config': AI_CONFIG,
            'telegram_config': TELEGRAM_CONFIG,
            'alert_config': ALERT_CONFIG,
            'fibonacci_levels': FIBONACCI_LEVELS,
            'support_resistance_config': SUPPORT_RESISTANCE_CONFIG,
            'market_sessions': MARKET_SESSIONS,
            'default_features': DEFAULT_FEATURES,
            'risk_config': RISK_CONFIG,
            'backtest_config': BACKTEST_CONFIG,
            'performance_metrics': PERFORMANCE_METRICS
        }
        
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        return get_default_configuration()

def get_default_configuration() -> Dict[str, Any]:
    """Get default configuration when config file is not available"""
    return {
        'nse_stocks': ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'SBIN'],
        'bse_stocks': ['RELIANCE', 'TCS', 'HDFCBANK'],
        'nifty_50': ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'SBIN'],
        'sectors': ['Banking', 'IT', 'Pharma', 'Auto', 'FMCG'],
        'default_watchlist': ['RELIANCE', 'TCS', 'HDFCBANK'],
        'trading_config': {
            'timeframes': ['1m', '5m', '15m', '30m', '1h'],
            'default_timeframe': '5m',
            'default_period': '1d'
        },
        'chart_config': {'height': 600, 'colors': {}},
        'ai_config': {'models': {}, 'features': []},
        'telegram_config': {'bot_token': '', 'chat_id': ''},
        'alert_config': {},
        'fibonacci_levels': [0.236, 0.382, 0.5, 0.618, 0.786],
        'support_resistance_config': {},
        'market_sessions': {},
        'default_features': {},
        'risk_config': {},
        'backtest_config': {},
        'performance_metrics': []
    }

def validate_stock_symbol(symbol: str) -> bool:
    """Validate if stock symbol is valid"""
    try:
        if not symbol or not isinstance(symbol, str):
            return False
        
        # Basic validation
        symbol = symbol.upper().strip()
        
        # Check length (typically 1-10 characters for Indian stocks)
        if len(symbol) < 1 or len(symbol) > 10:
            return False
        
        # Check for valid characters (alphanumeric only)
        if not symbol.replace('-', '').replace('&', '').isalnum():
            return False
        
        return True
        
    except Exception as e:
        print(f"Error validating symbol {symbol}: {str(e)}")
        return False

def format_currency(amount: float, currency: str = '₹') -> str:
    """Format currency amount for display"""
    try:
        if amount >= 10000000:  # 1 crore
            return f"{currency}{amount/10000000:.2f}Cr"
        elif amount >= 100000:  # 1 lakh
            return f"{currency}{amount/100000:.2f}L"
        elif amount >= 1000:  # 1 thousand
            return f"{currency}{amount/1000:.2f}K"
        else:
            return f"{currency}{amount:.2f}"
            
    except Exception as e:
        print(f"Error formatting currency: {str(e)}")
        return f"{currency}{amount}"

def format_percentage(value: float, decimals: int = 2) -> str:
    """Format percentage value for display"""
    try:
        if value >= 0:
            return f"+{value:.{decimals}f}%"
        else:
            return f"{value:.{decimals}f}%"
            
    except Exception as e:
        print(f"Error formatting percentage: {str(e)}")
        return f"{value}%"

def format_number(number: Union[int, float], decimals: int = 2) -> str:
    """Format large numbers with appropriate suffixes"""
    try:
        if abs(number) >= 10000000:  # 1 crore
            return f"{number/10000000:.{decimals}f}Cr"
        elif abs(number) >= 100000:  # 1 lakh
            return f"{number/100000:.{decimals}f}L"
        elif abs(number) >= 1000:  # 1 thousand
            return f"{number/1000:.{decimals}f}K"
        else:
            return f"{number:.{decimals}f}"
            
    except Exception as e:
        print(f"Error formatting number: {str(e)}")
        return str(number)

def calculate_position_size(capital: float, risk_percentage: float, 
                          entry_price: float, stop_loss: float) -> Dict[str, Any]:
    """Calculate position size based on risk management"""
    try:
        risk_amount = capital * (risk_percentage / 100)
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share <= 0:
            return {'error': 'Invalid stop loss level'}
        
        position_size = int(risk_amount / risk_per_share)
        position_value = position_size * entry_price
        actual_risk = position_size * risk_per_share
        
        return {
            'position_size': position_size,
            'position_value': position_value,
            'actual_risk': actual_risk,
            'risk_percentage': (actual_risk / capital) * 100,
            'risk_per_share': risk_per_share
        }
        
    except Exception as e:
        print(f"Error calculating position size: {str(e)}")
        return {'error': str(e)}

def calculate_risk_reward_ratio(entry_price: float, stop_loss: float, 
                               target_price: float) -> Dict[str, Any]:
    """Calculate risk-reward ratio for a trade"""
    try:
        risk = abs(entry_price - stop_loss)
        reward = abs(target_price - entry_price)
        
        if risk <= 0:
            return {'error': 'Invalid risk calculation'}
        
        ratio = reward / risk
        
        return {
            'risk': risk,
            'reward': reward,
            'ratio': ratio,
            'quality': get_risk_reward_quality(ratio)
        }
        
    except Exception as e:
        print(f"Error calculating risk-reward ratio: {str(e)}")
        return {'error': str(e)}

def get_risk_reward_quality(ratio: float) -> str:
    """Assess quality of risk-reward ratio"""
    if ratio >= 3:
        return "Excellent"
    elif ratio >= 2:
        return "Good"
    elif ratio >= 1.5:
        return "Fair"
    elif ratio >= 1:
        return "Acceptable"
    else:
        return "Poor"

def export_data_to_csv(data: Union[pd.DataFrame, List[Dict]], filename: str) -> str:
    """Export data to CSV format"""
    try:
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data
        
        if df.empty:
            return "No data to export"
        
        # Generate CSV string
        csv_string = df.to_csv(index=False)
        
        return csv_string
        
    except Exception as e:
        print(f"Error exporting to CSV: {str(e)}")
        return f"Error: {str(e)}"

def export_data_to_json(data: Union[pd.DataFrame, List[Dict], Dict]) -> str:
    """Export data to JSON format"""
    try:
        if isinstance(data, pd.DataFrame):
            json_data = data.to_dict('records')
        else:
            json_data = data
        
        return json.dumps(json_data, indent=2, default=str)
        
    except Exception as e:
        print(f"Error exporting to JSON: {str(e)}")
        return f'{{"error": "{str(e)}"}}'

def filter_stocks_by_criteria(stocks: List[str], criteria: Dict[str, Any]) -> List[str]:
    """Filter stocks based on various criteria"""
    try:
        filtered_stocks = stocks.copy()
        
        # Filter by sector if specified
        if 'sector' in criteria and criteria['sector'] != 'All':
            # This would require a sector mapping - simplified for now
            pass
        
        # Filter by market cap if specified
        if 'market_cap' in criteria:
            # This would require market cap data - simplified for now
            pass
        
        # Filter by price range if specified
        if 'price_range' in criteria:
            # This would require current price data - simplified for now
            pass
        
        return filtered_stocks
        
    except Exception as e:
        print(f"Error filtering stocks: {str(e)}")
        return stocks

def get_market_status() -> Dict[str, Any]:
    """Get current market status"""
    try:
        current_time = datetime.now()
        
        # NSE trading hours: 9:15 AM to 3:30 PM IST, Monday to Friday
        is_weekday = current_time.weekday() < 5
        
        market_open_time = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close_time = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
        
        is_market_hours = market_open_time <= current_time <= market_close_time
        is_open = is_weekday and is_market_hours
        
        # Determine session
        if not is_weekday:
            session = 'weekend'
        elif current_time < market_open_time:
            session = 'pre_market'
        elif current_time > market_close_time:
            session = 'after_hours'
        else:
            session = 'market_hours'
        
        # Calculate time to next session
        if session == 'pre_market':
            next_session_time = market_open_time
            next_session = 'market_open'
        elif session == 'market_hours':
            next_session_time = market_close_time
            next_session = 'market_close'
        elif session == 'after_hours':
            next_day = current_time + timedelta(days=1)
            next_session_time = next_day.replace(hour=9, minute=15, second=0, microsecond=0)
            next_session = 'market_open'
        else:  # weekend
            days_until_monday = (7 - current_time.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 1
            next_monday = current_time + timedelta(days=days_until_monday)
            next_session_time = next_monday.replace(hour=9, minute=15, second=0, microsecond=0)
            next_session = 'market_open'
        
        time_to_next = next_session_time - current_time
        
        return {
            'is_open': is_open,
            'session': session,
            'current_time': current_time,
            'next_session': next_session,
            'next_session_time': next_session_time,
            'time_to_next': time_to_next,
            'status_text': '🟢 Market Open' if is_open else '🔴 Market Closed'
        }
        
    except Exception as e:
        print(f"Error getting market status: {str(e)}")
        return {
            'is_open': False,
            'session': 'unknown',
            'status_text': '⚠️ Status Unknown'
        }

def calculate_trading_metrics(trades: List[Dict]) -> Dict[str, Any]:
    """Calculate comprehensive trading metrics"""
    try:
        if not trades:
            return get_empty_metrics()
        
        # Filter completed trades
        completed_trades = [t for t in trades if t.get('status') == 'closed' and 'pnl' in t]
        
        if not completed_trades:
            return get_empty_metrics()
        
        # Basic metrics
        total_trades = len(completed_trades)
        winning_trades = [t for t in completed_trades if t['pnl'] > 0]
        losing_trades = [t for t in completed_trades if t['pnl'] < 0]
        
        # P&L calculations
        total_pnl = sum(t['pnl'] for t in completed_trades)
        gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
        gross_loss = sum(abs(t['pnl']) for t in losing_trades) if losing_trades else 0
        
        # Ratios
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average metrics
        avg_win = gross_profit / len(winning_trades) if winning_trades else 0
        avg_loss = gross_loss / len(losing_trades) if losing_trades else 0
        avg_trade = total_pnl / total_trades if total_trades > 0 else 0
        
        # Risk metrics
        pnl_series = [t['pnl'] for t in completed_trades]
        max_drawdown = calculate_max_drawdown_from_trades(pnl_series)
        
        # Consecutive metrics
        max_consecutive_wins = calculate_max_consecutive(completed_trades, True)
        max_consecutive_losses = calculate_max_consecutive(completed_trades, False)
        
        # Additional metrics
        best_trade = max(pnl_series) if pnl_series else 0
        worst_trade = min(pnl_series) if pnl_series else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': avg_trade,
            'max_drawdown': max_drawdown,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'expectancy': avg_trade,
            'recovery_factor': total_pnl / max_drawdown if max_drawdown > 0 else 0
        }
        
    except Exception as e:
        print(f"Error calculating trading metrics: {str(e)}")
        return get_empty_metrics()

def get_empty_metrics() -> Dict[str, Any]:
    """Return empty metrics structure"""
    return {
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'win_rate': 0,
        'total_pnl': 0,
        'gross_profit': 0,
        'gross_loss': 0,
        'profit_factor': 0,
        'avg_win': 0,
        'avg_loss': 0,
        'avg_trade': 0,
        'max_drawdown': 0,
        'max_consecutive_wins': 0,
        'max_consecutive_losses': 0,
        'best_trade': 0,
        'worst_trade': 0,
        'expectancy': 0,
        'recovery_factor': 0
    }

def calculate_max_drawdown_from_trades(pnl_series: List[float]) -> float:
    """Calculate maximum drawdown from P&L series"""
    try:
        if not pnl_series:
            return 0
        
        cumulative_pnl = np.cumsum(pnl_series)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - running_max
        
        return abs(min(drawdown)) if len(drawdown) > 0 else 0
        
    except Exception as e:
        print(f"Error calculating max drawdown: {str(e)}")
        return 0

def calculate_max_consecutive(trades: List[Dict], wins: bool) -> int:
    """Calculate maximum consecutive wins or losses"""
    try:
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trades:
            is_win = trade.get('pnl', 0) > 0
            
            if (wins and is_win) or (not wins and not is_win):
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
        
    except Exception as e:
        print(f"Error calculating consecutive trades: {str(e)}")
        return 0

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare DataFrame for analysis"""
    try:
        if df.empty:
            return df
        
        # Remove NaN values
        df = df.dropna()
        
        # Reset index
        df = df.reset_index(drop=True)
        
        # Ensure datetime column if present
        if 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'])
        
        # Ensure numeric columns are proper type
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except Exception as e:
        print(f"Error cleaning DataFrame: {str(e)}")
        return df

def validate_data_quality(data: pd.DataFrame) -> Dict[str, Any]:
    """Validate data quality for analysis"""
    try:
        if data.empty:
            return {'valid': False, 'error': 'Empty dataset'}
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            return {
                'valid': False,
                'error': f'Missing columns: {", ".join(missing_columns)}'
            }
        
        # Check for sufficient data
        if len(data) < 10:
            return {'valid': False, 'error': 'Insufficient data points'}
        
        # Check for data consistency
        inconsistent_rows = 0
        for i in range(len(data)):
            if data['High'].iloc[i] < data['Low'].iloc[i]:
                inconsistent_rows += 1
            elif data['Close'].iloc[i] > data['High'].iloc[i] or data['Close'].iloc[i] < data['Low'].iloc[i]:
                inconsistent_rows += 1
        
        quality_score = max(0, 100 - (inconsistent_rows / len(data) * 100))
        
        return {
            'valid': True,
            'length': len(data),
            'inconsistent_rows': inconsistent_rows,
            'quality_score': quality_score,
            'quality_rating': get_quality_rating(quality_score)
        }
        
    except Exception as e:
        print(f"Error validating data quality: {str(e)}")
        return {'valid': False, 'error': str(e)}

def get_quality_rating(score: float) -> str:
    """Get quality rating based on score"""
    if score >= 95:
        return "Excellent"
    elif score >= 85:
        return "Good"
    elif score >= 70:
        return "Fair"
    elif score >= 50:
        return "Poor"
    else:
        return "Very Poor"

def safe_divide(numerator: float, denominator: float, default: float = 0) -> float:
    """Safely divide two numbers"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except Exception:
        return default

def normalize_symbol(symbol: str) -> str:
    """Normalize stock symbol for consistency"""
    try:
        if not symbol:
            return ""
        
        # Convert to uppercase and strip whitespace
        normalized = symbol.upper().strip()
        
        # Remove common suffixes for consistency
        normalized = normalized.replace('.NS', '').replace('.BO', '')
        
        return normalized
        
    except Exception as e:
        print(f"Error normalizing symbol: {str(e)}")
        return symbol

def get_symbol_with_exchange(symbol: str, exchange: str = 'NSE') -> str:
    """Get symbol with appropriate exchange suffix"""
    try:
        normalized = normalize_symbol(symbol)
        
        if exchange.upper() == 'BSE':
            return f"{normalized}.BO"
        else:  # Default to NSE
            return f"{normalized}.NS"
            
    except Exception as e:
        print(f"Error adding exchange suffix: {str(e)}")
        return symbol

def calculate_volatility(prices: pd.Series, window: int = 20) -> float:
    """Calculate price volatility"""
    try:
        if len(prices) < window:
            return 0
        
        returns = prices.pct_change().dropna()
        volatility = returns.rolling(window=window).std().iloc[-1]
        
        # Annualize volatility (assuming daily data)
        annualized_volatility = volatility * np.sqrt(252) * 100
        
        return annualized_volatility
        
    except Exception as e:
        print(f"Error calculating volatility: {str(e)}")
        return 0

def calculate_correlation(series1: pd.Series, series2: pd.Series) -> float:
    """Calculate correlation between two price series"""
    try:
        if len(series1) != len(series2) or len(series1) < 10:
            return 0
        
        correlation = series1.corr(series2)
        return correlation if not np.isnan(correlation) else 0
        
    except Exception as e:
        print(f"Error calculating correlation: {str(e)}")
        return 0

def get_session_state_safely(key: str, default: Any = None) -> Any:
    """Safely get value from Streamlit session state"""
    try:
        return st.session_state.get(key, default)
    except Exception as e:
        print(f"Error accessing session state key {key}: {str(e)}")
        return default

def set_session_state_safely(key: str, value: Any) -> bool:
    """Safely set value in Streamlit session state"""
    try:
        st.session_state[key] = value
        return True
    except Exception as e:
        print(f"Error setting session state key {key}: {str(e)}")
        return False

def format_datetime(dt: datetime, format_string: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format datetime for display"""
    try:
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)
        return dt.strftime(format_string)
    except Exception as e:
        print(f"Error formatting datetime: {str(e)}")
        return str(dt)

def get_color_for_value(value: float, positive_color: str = "#00ff00", 
                       negative_color: str = "#ff0000", neutral_color: str = "#888888") -> str:
    """Get color based on value (positive/negative/neutral)"""
    try:
        if value > 0:
            return positive_color
        elif value < 0:
            return negative_color
        else:
            return neutral_color
    except Exception:
        return neutral_color

def truncate_text(text: str, max_length: int = 50) -> str:
    """Truncate text to specified length"""
    try:
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."
    except Exception:
        return str(text)

def validate_timeframe(timeframe: str) -> bool:
    """Validate if timeframe is supported"""
    try:
        valid_timeframes = ['1m', '2m', '3m', '5m', '10m', '15m', '30m', '1h', '2h', '4h', '1d']
        return timeframe.lower() in valid_timeframes
    except Exception:
        return False

def validate_period(period: str) -> bool:
    """Validate if period is supported"""
    try:
        valid_periods = ['1d', '2d', '5d', '1mo', '2mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
        return period.lower() in valid_periods
    except Exception:
        return False

def create_notification_message(symbol: str, signal: str, price: float, 
                               confidence: int, details: Dict[str, Any] = None) -> str:
    """Create formatted notification message"""
    try:
        emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "🔵"
        
        message = f"{emoji} {signal} Signal for {symbol}\n"
        message += f"Price: ₹{price:.2f}\n"
        message += f"Confidence: {confidence}%\n"
        
        if details:
            if 'rsi' in details:
                message += f"RSI: {details['rsi']:.1f}\n"
            if 'macd' in details:
                message += f"MACD: {details['macd']:.3f}\n"
            if 'volume_ratio' in details:
                message += f"Volume: {details['volume_ratio']:.1f}x\n"
        
        message += f"Time: {datetime.now().strftime('%H:%M:%S')}"
        
        return message
        
    except Exception as e:
        print(f"Error creating notification message: {str(e)}")
        return f"{signal} signal for {symbol} at ₹{price:.2f}"

def log_error(error_message: str, context: str = ""):
    """Log error message with context"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"[{timestamp}] {context}: {error_message}" if context else f"[{timestamp}] {error_message}"
        print(full_message)
        
        # Could be extended to write to file or external logging service
        
    except Exception as e:
        print(f"Error logging message: {str(e)}")

def retry_operation(func, max_retries: int = 3, delay: float = 1.0, *args, **kwargs):
    """Retry an operation with exponential backoff"""
    import time
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            
            wait_time = delay * (2 ** attempt)
            print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time}s...")
            time.sleep(wait_time)

def get_performance_summary(metrics: Dict[str, Any]) -> str:
    """Get formatted performance summary"""
    try:
        summary = f"Performance Summary:\n"
        summary += f"Total Trades: {metrics.get('total_trades', 0)}\n"
        summary += f"Win Rate: {metrics.get('win_rate', 0):.1f}%\n"
        summary += f"Total P&L: {format_currency(metrics.get('total_pnl', 0))}\n"
        summary += f"Profit Factor: {metrics.get('profit_factor', 0):.2f}\n"
        summary += f"Max Drawdown: {format_currency(metrics.get('max_drawdown', 0))}\n"
        
        return summary
        
    except Exception as e:
        print(f"Error creating performance summary: {str(e)}")
        return "Performance data unavailable"
