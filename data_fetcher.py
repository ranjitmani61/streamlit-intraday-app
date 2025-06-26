"""
Enhanced data fetching module for comprehensive NSE/BSE market data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Optional, Dict, List, Tuple, Union
import warnings
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import time
warnings.filterwarnings('ignore')

class DataFetcher:
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 60  # 1 minute for intraday
        self.session = requests.Session()
        self.executor = ThreadPoolExecutor(max_workers=10)
    
    def get_realtime_data(self, symbol: str, period: str = '1d', interval: str = '5m') -> Optional[pd.DataFrame]:
        """Fetch real-time market data for NSE/BSE symbol"""
        try:
            # Add appropriate suffix
            if not symbol.endswith(('.NS', '.BO')):
                symbol = f"{symbol}.NS"  # Default to NSE
            
            # Check cache first
            cache_key = f"{symbol}_{period}_{interval}"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if (datetime.now() - timestamp).seconds < self.cache_timeout:
                    return cached_data
            
            # Fetch from yfinance
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                # Try BSE if NSE fails
                if symbol.endswith('.NS'):
                    symbol = symbol.replace('.NS', '.BO')
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                return None
            
            # Clean and prepare data
            data = self._clean_data(data)
            
            # Add technical columns
            data = self._add_technical_columns(data)
            
            # Cache the data
            self.cache[cache_key] = (data, datetime.now())
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def get_batch_data(self, symbols: List[str], period: str = '1d', interval: str = '5m') -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols concurrently"""
        try:
            # Use ThreadPoolExecutor for concurrent fetching
            futures = {}
            with ThreadPoolExecutor(max_workers=10) as executor:
                for symbol in symbols:
                    future = executor.submit(self.get_realtime_data, symbol, period, interval)
                    futures[symbol] = future
                
                # Collect results
                results = {}
                for symbol, future in futures.items():
                    try:
                        data = future.result(timeout=10)
                        if data is not None:
                            results[symbol] = data
                    except Exception as e:
                        print(f"Error fetching {symbol}: {str(e)}")
                        continue
                
                return results
                
        except Exception as e:
            print(f"Error in batch data fetching: {str(e)}")
            return {}
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price"""
        try:
            if not symbol.endswith(('.NS', '.BO')):
                symbol = f"{symbol}.NS"
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval='1m')
            
            if not data.empty:
                return float(data['Close'].iloc[-1])
            return None
            
        except Exception as e:
            print(f"Error getting current price for {symbol}: {str(e)}")
            return None
    
    def get_price_change(self, symbol: str) -> Dict[str, float]:
        """Get price change compared to previous close"""
        try:
            if not symbol.endswith(('.NS', '.BO')):
                symbol = f"{symbol}.NS"
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='2d', interval='1d')
            
            if len(data) >= 2:
                current_price = float(data['Close'].iloc[-1])
                previous_price = float(data['Close'].iloc[-2])
                change = current_price - previous_price
                change_pct = (change / previous_price) * 100
                
                return {
                    'current_price': current_price,
                    'previous_price': previous_price,
                    'change': change,
                    'change_pct': change_pct
                }
            
            return {'current_price': 0, 'previous_price': 0, 'change': 0, 'change_pct': 0}
            
        except Exception as e:
            print(f"Error getting price change for {symbol}: {str(e)}")
            return {'current_price': 0, 'previous_price': 0, 'change': 0, 'change_pct': 0}
    
    def detect_gap(self, symbol: str, threshold: float = 2.0) -> Dict[str, Union[bool, float]]:
        """Detect gap up/down from previous close"""
        try:
            if not symbol.endswith(('.NS', '.BO')):
                symbol = f"{symbol}.NS"
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='2d', interval='1d')
            
            if len(data) >= 2:
                yesterday_close = float(data['Close'].iloc[-2])
                today_open = float(data['Open'].iloc[-1])
                gap_pct = ((today_open - yesterday_close) / yesterday_close) * 100
                
                is_gap_up = gap_pct > threshold
                is_gap_down = gap_pct < -threshold
                
                return {
                    'has_gap': is_gap_up or is_gap_down,
                    'gap_type': 'up' if is_gap_up else 'down' if is_gap_down else 'none',
                    'gap_percentage': gap_pct,
                    'yesterday_close': yesterday_close,
                    'today_open': today_open
                }
            
            return {'has_gap': False, 'gap_type': 'none', 'gap_percentage': 0}
            
        except Exception as e:
            print(f"Error detecting gap for {symbol}: {str(e)}")
            return {'has_gap': False, 'gap_type': 'none', 'gap_percentage': 0}
    
    def detect_volume_surge(self, symbol: str, threshold: float = 2.0) -> Dict[str, Union[bool, float]]:
        """Detect volume surge compared to average"""
        try:
            data = self.get_realtime_data(symbol, period='5d', interval='5m')
            if data is None or len(data) < 20:
                return {'has_surge': False, 'volume_ratio': 0}
            
            # Calculate average volume (excluding current bar)
            avg_volume = data['Volume'][:-1].rolling(window=20).mean().iloc[-1]
            current_volume = data['Volume'].iloc[-1]
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            has_surge = volume_ratio > threshold
            
            return {
                'has_surge': has_surge,
                'volume_ratio': volume_ratio,
                'current_volume': current_volume,
                'average_volume': avg_volume
            }
            
        except Exception as e:
            print(f"Error detecting volume surge for {symbol}: {str(e)}")
            return {'has_surge': False, 'volume_ratio': 0}
    
    def get_market_breadth(self) -> Dict[str, any]:
        """Get market breadth data"""
        try:
            from config import NSE_ALL_STOCKS
            
            # Sample a subset for performance
            sample_stocks = NSE_ALL_STOCKS[:50]
            batch_data = self.get_batch_data(sample_stocks, period='1d', interval='5m')
            
            advances = 0
            declines = 0
            unchanged = 0
            
            for symbol, data in batch_data.items():
                if len(data) >= 2:
                    change_pct = ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
                    if change_pct > 0.1:
                        advances += 1
                    elif change_pct < -0.1:
                        declines += 1
                    else:
                        unchanged += 1
            
            total = advances + declines + unchanged
            
            return {
                'advances': advances,
                'declines': declines,
                'unchanged': unchanged,
                'advance_decline_ratio': advances / declines if declines > 0 else 0,
                'market_sentiment': 'Bullish' if advances > declines else 'Bearish'
            }
            
        except Exception as e:
            print(f"Error getting market breadth: {str(e)}")
            return {'advances': 0, 'declines': 0, 'unchanged': 0}
    
    def get_top_gainers_losers(self, market: str = 'NSE', count: int = 10) -> Tuple[List[Dict], List[Dict]]:
        """Get top gainers and losers from NSE/BSE"""
        try:
            from config import NSE_ALL_STOCKS, BSE_STOCKS
            
            stocks = NSE_ALL_STOCKS[:50] if market == 'NSE' else BSE_STOCKS[:30]
            batch_data = self.get_batch_data(stocks, period='1d', interval='5m')
            
            gainers = []
            losers = []
            
            for symbol, data in batch_data.items():
                if data is None or len(data) < 2:
                    continue
                
                try:
                    current_price = float(data['Close'].iloc[-1])
                    previous_price = float(data['Close'].iloc[-2])
                    change_pct = ((current_price - previous_price) / previous_price) * 100
                    volume = int(data['Volume'].iloc[-1])
                    
                    stock_data = {
                        'stock': symbol,
                        'price': current_price,
                        'change_pct': change_pct,
                        'volume': volume,
                        'sector': self._get_sector(symbol)
                    }
                    
                    if change_pct > 0:
                        gainers.append(stock_data)
                    else:
                        losers.append(stock_data)
                        
                except Exception:
                    continue
            
            # Sort and limit results
            gainers = sorted(gainers, key=lambda x: x['change_pct'], reverse=True)[:count]
            losers = sorted(losers, key=lambda x: x['change_pct'])[:count]
            
            return gainers, losers
            
        except Exception as e:
            print(f"Error fetching top gainers/losers: {str(e)}")
            return [], []
    
    def get_sector_performance(self) -> Dict[str, Dict]:
        """Get sector-wise performance"""
        try:
            from config import SECTORS
            
            sector_performance = {}
            
            # This would be implemented with sector-wise stock analysis
            # For now, return sample data structure
            for sector in SECTORS[:10]:  # Sample sectors
                sector_performance[sector] = {
                    'change_pct': np.random.uniform(-3, 3),
                    'volume_ratio': np.random.uniform(0.5, 2.5),
                    'stocks_up': np.random.randint(0, 10),
                    'stocks_down': np.random.randint(0, 10)
                }
            
            return sector_performance
            
        except Exception as e:
            print(f"Error getting sector performance: {str(e)}")
            return {}
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare market data"""
        try:
            # Remove any NaN values
            data = data.dropna()
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in data.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # Reset index to make datetime a column
            data = data.reset_index()
            
            # Ensure proper data types
            for col in ['Open', 'High', 'Low', 'Close']:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')
            
            return data
            
        except Exception as e:
            print(f"Error cleaning data: {str(e)}")
            return data
    
    def _add_technical_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical analysis columns"""
        try:
            # Typical Price
            data['Typical_Price'] = (data['High'] + data['Low'] + data['Close']) / 3
            
            # Price Range
            data['Price_Range'] = data['High'] - data['Low']
            
            # Volume-Price Trend
            data['VPT'] = (data['Volume'] * ((data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1))).cumsum()
            
            # True Range
            data['TR'] = np.maximum(
                data['High'] - data['Low'],
                np.maximum(
                    abs(data['High'] - data['Close'].shift(1)),
                    abs(data['Low'] - data['Close'].shift(1))
                )
            )
            
            return data
            
        except Exception as e:
            print(f"Error adding technical columns: {str(e)}")
            return data
    
    def _get_sector(self, symbol: str) -> str:
        """Get sector for a stock symbol"""
        # This would be implemented with a comprehensive sector mapping
        # For now, return a simple mapping
        sector_mapping = {
            'RELIANCE': 'Energy', 'TCS': 'IT', 'HDFCBANK': 'Banking',
            'INFY': 'IT', 'SBIN': 'Banking', 'ICICIBANK': 'Banking',
            'BHARTIARTL': 'Telecom', 'ITC': 'FMCG', 'KOTAKBANK': 'Banking',
            'LT': 'Infrastructure', 'HCLTECH': 'IT', 'ASIANPAINT': 'Paints',
            'MARUTI': 'Auto', 'BAJFINANCE': 'Finance', 'TITAN': 'Jewelry'
        }
        
        symbol_clean = symbol.replace('.NS', '').replace('.BO', '')
        return sector_mapping.get(symbol_clean, 'Others')
    
    def is_market_open(self) -> bool:
        """Check if NSE market is currently open"""
        try:
            current_time = datetime.now()
            
            # NSE trading hours: 9:15 AM to 3:30 PM IST, Monday to Friday
            if current_time.weekday() >= 5:  # Weekend
                return False
            
            market_open = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
            
            return market_open <= current_time <= market_close
            
        except Exception:
            return False
    
    def get_market_session(self) -> str:
        """Get current market session"""
        try:
            current_time = datetime.now().time()
            
            if current_time < datetime.strptime('09:15', '%H:%M').time():
                return 'pre_market'
            elif current_time <= datetime.strptime('15:30', '%H:%M').time():
                return 'market_hours'
            else:
                return 'after_hours'
                
        except Exception:
            return 'unknown'
    
    def clear_cache(self):
        """Clear the data cache"""
        self.cache.clear()
    
    def get_historical_data(self, symbol: str, period: str = '1y', interval: str = '1d') -> Optional[pd.DataFrame]:
        """Get historical data for backtesting"""
        try:
            if not symbol.endswith(('.NS', '.BO')):
                symbol = f"{symbol}.NS"
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if not data.empty:
                data = self._clean_data(data)
                data = self._add_technical_columns(data)
                return data
            
            return None
            
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {str(e)}")
            return None
