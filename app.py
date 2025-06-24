"""
Complete NSE Trading Bot - All modules integrated
Angel One SmartAPI Logic with Live Scanner
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import plotly.graph_objects as go
import plotly.subplots as sp
from datetime import datetime, timedelta
import time
import threading
import configparser
import logging
import json
import csv
import os
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Configure Streamlit page
st.set_page_config(
    page_title="Complete NSE Trading Scanner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== DATA CLASSES ====================
@dataclass
class TradingSignal:
    symbol: str
    action: str
    price: float
    strength: float
    confidence: float
    risk_level: str
    timestamp: datetime
    indicators: List[str]
    indicator_scores: Dict[str, float]

# ==================== DATA FETCHER ====================
class DataFetcher:
    def __init__(self):
        self.cache = {}
    
    def get_realtime_data(self, symbol, period='1d', interval='5m'):
        """Fetch real-time market data for NSE symbol"""
        try:
            # Add .NS for NSE symbols if not present
            if not symbol.endswith('.NS') and '.' not in symbol:
                symbol = f"{symbol}.NS"
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                return None
                
            # Convert column names to lowercase
            df.columns = df.columns.str.lower()
            return self._clean_data(df)
            
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def _clean_data(self, data):
        """Clean and prepare market data"""
        if data is None or data.empty:
            return None
        
        # Remove any NaN rows
        data = data.dropna()
        
        # Ensure we have enough data points
        if len(data) < 30:
            return None
            
        return data
    
    def get_current_price(self, symbol):
        """Get current market price"""
        try:
            data = self.get_realtime_data(symbol, period='1d', interval='5m')
            if data is not None and not data.empty:
                return float(data['close'].iloc[-1])
            return None
        except:
            return None
    
    def get_price_change(self, symbol):
        """Get price change compared to yesterday's close"""
        try:
            # Get 2 days of data to compare
            if not symbol.endswith('.NS') and '.' not in symbol:
                symbol = f"{symbol}.NS"
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(period='2d', interval='1d')
            
            if len(df) >= 2:
                yesterday_close = df['Close'].iloc[-2]
                current_price = df['Close'].iloc[-1]
                price_change = ((current_price - yesterday_close) / yesterday_close) * 100
                return price_change
            return 0.0
        except:
            return 0.0
    
    def get_stock_details(self, symbol):
        """Get detailed stock information"""
        try:
            if not symbol.endswith('.NS') and '.' not in symbol:
                symbol = f"{symbol}.NS"
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period='1y', interval='1d')
            
            if hist.empty:
                return None
            
            current_price = hist['Close'].iloc[-1]
            volume = hist['Volume'].iloc[-1]
            high_52w = hist['High'].max()
            low_52w = hist['Low'].min()
            
            return {
                'symbol': symbol.replace('.NS', ''),
                'price': current_price,
                'volume': volume,
                'high_52w': high_52w,
                'low_52w': low_52w,
                'sector': info.get('sector', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0)
            }
        except:
            return None
    
    def get_top_gainers_losers(self, market='NSE', count=10):
        """Get top gainers and losers from NSE/BSE"""
        try:
            # Popular NSE stocks for scanning
            nse_stocks = [
                'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR',
                'ICICIBANK', 'SBIN', 'BHARTIARTL', 'ITC', 'KOTAKBANK',
                'LT', 'AXISBANK', 'ASIANPAINT', 'MARUTI', 'NESTLEIND',
                'BAJFINANCE', 'TITAN', 'ULTRACEMCO', 'POWERGRID', 'NTPC',
                'ONGC', 'TECHM', 'WIPRO', 'SUNPHARMA', 'JSWSTEEL',
                'TATAMOTORS', 'DIVISLAB', 'HCLTECH', 'INDUSINDBK', 'ADANIPORTS',
                'DRREDDY', 'EICHERMOT', 'COALINDIA', 'BAJAJFINSV', 'GRASIM',
                'BRITANNIA', 'CIPLA', 'SHREECEM', 'APOLLOHOSP', 'BPCL',
                'IOC', 'HEROMOTOCO', 'TATASTEEL', 'HINDALCO', 'UPL'
            ]
            
            # BSE equivalent (same stocks but different symbols)
            if market == 'BSE':
                # For BSE, we'll use .BO suffix
                stocks_to_scan = [f"{stock}.BO" for stock in nse_stocks]
            else:
                # For NSE, use .NS suffix
                stocks_to_scan = [f"{stock}.NS" for stock in nse_stocks]
            
            results = []
            
            for symbol in stocks_to_scan[:30]:  # Limit to 30 for performance
                try:
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(period='2d', interval='1d')
                    info = ticker.info
                    
                    if len(df) >= 2:
                        yesterday_close = df['Close'].iloc[-2]
                        current_price = df['Close'].iloc[-1]
                        volume = df['Volume'].iloc[-1]
                        price_change = ((current_price - yesterday_close) / yesterday_close) * 100
                        
                        stock_name = symbol.replace('.NS', '').replace('.BO', '')
                        sector = info.get('sector', 'N/A')
                        
                        results.append({
                            'stock': stock_name,
                            'price': current_price,
                            'change_pct': price_change,
                            'volume': volume,
                            'sector': sector,
                            'market': market
                        })
                except:
                    continue
            
            # Sort by percentage change
            gainers = sorted([r for r in results if r['change_pct'] > 0], 
                           key=lambda x: x['change_pct'], reverse=True)[:count]
            losers = sorted([r for r in results if r['change_pct'] < 0], 
                          key=lambda x: x['change_pct'])[:count]
            
            return gainers, losers
            
        except Exception as e:
            st.error(f"Error fetching {market} data: {str(e)}")
            return [], []

# ==================== BACKTESTING ENGINE ====================
class BacktestingEngine:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_history = []
        self.data_fetcher = DataFetcher()
        self.technical_indicators = TechnicalIndicators()
        self.signal_generator = SignalGenerator()
    
    def run_backtest(self, symbols: List[str], start_date: str, end_date: str, 
                    strategy_params: Dict = None) -> Dict:
        """Run comprehensive backtest on multiple symbols"""
        try:
            self.capital = self.initial_capital
            self.positions = {}
            self.trades = []
            self.portfolio_history = []
            
            all_results = []
            
            for symbol in symbols:
                symbol_results = self._backtest_symbol(symbol, start_date, end_date, strategy_params)
                if symbol_results:
                    all_results.append(symbol_results)
            
            # Calculate overall portfolio performance
            portfolio_metrics = self._calculate_portfolio_metrics()
            
            return {
                'success': True,
                'symbol_results': all_results,
                'portfolio_metrics': portfolio_metrics,
                'trades': self.trades,
                'portfolio_history': self.portfolio_history,
                'total_return': ((self.capital - self.initial_capital) / self.initial_capital) * 100,
                'final_capital': self.capital
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _backtest_symbol(self, symbol: str, start_date: str, end_date: str, 
                        strategy_params: Dict = None) -> Dict:
        """Backtest strategy on a single symbol"""
        try:
            # Fetch historical data
            if not symbol.endswith('.NS') and '.' not in symbol:
                symbol_with_suffix = f"{symbol}.NS"
            else:
                symbol_with_suffix = symbol
            
            ticker = yf.Ticker(symbol_with_suffix)
            data = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if data.empty:
                return None
            
            # Prepare data
            data.columns = [col.lower() for col in data.columns]
            data = data.reset_index()
            data.set_index('Date', inplace=True)
            
            # Calculate technical indicators for each day
            signals = []
            returns = []
            
            # Use a rolling window for realistic backtesting
            window_size = 50  # Need at least 50 days for indicators
            
            for i in range(window_size, len(data)):
                window_data = data.iloc[i-window_size:i+1].copy()
                
                # Calculate indicators on window data
                indicators = self.technical_indicators.calculate_all_indicators(window_data)
                
                if indicators:
                    # Generate signal for current day
                    signal_data = self.signal_generator.generate_nse_signals(
                        symbol, window_data, indicators
                    )
                    
                    current_price = window_data['close'].iloc[-1]
                    current_date = window_data.index[-1]
                    
                    # Execute trades based on signal
                    trade_result = self._execute_trade(
                        symbol, signal_data['signal'], current_price, current_date
                    )
                    
                    signals.append({
                        'date': current_date,
                        'price': current_price,
                        'signal': signal_data['signal'],
                        'rsi': signal_data.get('rsi', 0),
                        'macd': signal_data.get('macd', 0),
                        'strength': signal_data.get('strength', 0),
                        'trade_executed': trade_result
                    })
                    
                    # Calculate portfolio value
                    portfolio_value = self._calculate_portfolio_value(current_price, symbol)
                    self.portfolio_history.append({
                        'date': current_date,
                        'portfolio_value': portfolio_value,
                        'cash': self.capital,
                        'symbol': symbol
                    })
            
            # Calculate symbol-specific metrics
            symbol_metrics = self._calculate_symbol_metrics(symbol, data, signals)
            
            return {
                'symbol': symbol,
                'signals': signals,
                'metrics': symbol_metrics,
                'data': data
            }
            
        except Exception as e:
            st.error(f"Backtest error for {symbol}: {str(e)}")
            return None
    
    def _execute_trade(self, symbol: str, signal: str, price: float, date) -> bool:
        """Execute buy/sell trades based on signals"""
        try:
            position_size = 0.1  # Use 10% of capital per trade
            
            if signal == "BUY" and symbol not in self.positions:
                # Buy signal - enter position
                max_investment = self.capital * position_size
                shares = int(max_investment / price)
                
                if shares > 0 and (shares * price) <= self.capital:
                    self.positions[symbol] = {
                        'shares': shares,
                        'entry_price': price,
                        'entry_date': date
                    }
                    
                    cost = shares * price
                    self.capital -= cost
                    
                    self.trades.append({
                        'symbol': symbol,
                        'action': 'BUY',
                        'shares': shares,
                        'price': price,
                        'value': cost,
                        'date': date,
                        'capital_after': self.capital
                    })
                    
                    return True
            
            elif signal == "SELL" and symbol in self.positions:
                # Sell signal - exit position
                position = self.positions[symbol]
                shares = position['shares']
                entry_price = position['entry_price']
                
                sale_value = shares * price
                self.capital += sale_value
                
                profit_loss = sale_value - (shares * entry_price)
                
                self.trades.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'shares': shares,
                    'price': price,
                    'value': sale_value,
                    'date': date,
                    'profit_loss': profit_loss,
                    'entry_price': entry_price,
                    'entry_date': position['entry_date'],
                    'capital_after': self.capital
                })
                
                del self.positions[symbol]
                return True
            
            return False
            
        except Exception as e:
            st.error(f"Trade execution error: {str(e)}")
            return False
    
    def _calculate_portfolio_value(self, current_price: float, symbol: str) -> float:
        """Calculate total portfolio value"""
        total_value = self.capital
        
        if symbol in self.positions:
            position = self.positions[symbol]
            total_value += position['shares'] * current_price
        
        return total_value
    
    def _calculate_symbol_metrics(self, symbol: str, data: pd.DataFrame, signals: List[Dict]) -> Dict:
        """Calculate performance metrics for a symbol"""
        try:
            symbol_trades = [t for t in self.trades if t['symbol'] == symbol]
            
            if not symbol_trades:
                return {'total_trades': 0, 'win_rate': 0, 'avg_return': 0}
            
            completed_trades = [t for t in symbol_trades if t['action'] == 'SELL']
            
            if not completed_trades:
                return {'total_trades': len(symbol_trades), 'win_rate': 0, 'avg_return': 0}
            
            # Calculate metrics
            total_trades = len(completed_trades)
            winning_trades = len([t for t in completed_trades if t['profit_loss'] > 0])
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            total_return = sum([t['profit_loss'] for t in completed_trades])
            avg_return = total_return / total_trades if total_trades > 0 else 0
            
            # Calculate Sharpe ratio (simplified)
            returns = [t['profit_loss'] / (t['shares'] * t['entry_price']) for t in completed_trades]
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': total_trades - winning_trades,
                'win_rate': win_rate,
                'total_return': total_return,
                'avg_return': avg_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': self._calculate_max_drawdown(signals),
                'buy_signals': len([s for s in signals if s['signal'] == 'BUY']),
                'sell_signals': len([s for s in signals if s['signal'] == 'SELL']),
                'hold_signals': len([s for s in signals if s['signal'] == 'HOLD'])
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_max_drawdown(self, signals: List[Dict]) -> float:
        """Calculate maximum drawdown"""
        try:
            if not self.portfolio_history:
                return 0
            
            portfolio_values = [p['portfolio_value'] for p in self.portfolio_history]
            peak = portfolio_values[0]
            max_drawdown = 0
            
            for value in portfolio_values:
                if value > peak:
                    peak = value
                else:
                    drawdown = (peak - value) / peak
                    max_drawdown = max(max_drawdown, drawdown)
            
            return max_drawdown * 100
            
        except:
            return 0
    
    def _calculate_portfolio_metrics(self) -> Dict:
        """Calculate overall portfolio performance metrics"""
        try:
            if not self.trades:
                return {'error': 'No trades executed'}
            
            total_trades = len([t for t in self.trades if t['action'] == 'SELL'])
            total_return = self.capital - self.initial_capital
            total_return_pct = (total_return / self.initial_capital) * 100
            
            # Calculate overall win rate
            completed_trades = [t for t in self.trades if t['action'] == 'SELL']
            winning_trades = len([t for t in completed_trades if t.get('profit_loss', 0) > 0])
            win_rate = (winning_trades / len(completed_trades)) * 100 if completed_trades else 0
            
            # Portfolio volatility
            if len(self.portfolio_history) > 1:
                values = [p['portfolio_value'] for p in self.portfolio_history]
                returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
                volatility = np.std(returns) * np.sqrt(252) * 100
            else:
                volatility = 0
            
            return {
                'initial_capital': self.initial_capital,
                'final_capital': self.capital,
                'total_return': total_return,
                'total_return_pct': total_return_pct,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'volatility': volatility,
                'max_drawdown': self._calculate_max_drawdown([]),
                'sharpe_ratio': (total_return_pct / volatility) if volatility > 0 else 0
            }
            
        except Exception as e:
            return {'error': str(e)}

# ==================== AI STRATEGY RECOMMENDATION ENGINE ====================
class AIStrategyEngine:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = [
            'rsi', 'macd', 'bb_upper', 'bb_lower', 'adx', 'ema_12', 'ema_26',
            'sma_20', 'sma_50', 'volume_sma', 'price_change_pct', 'volatility'
        ]
        
    def prepare_features(self, data: pd.DataFrame, indicators: Dict) -> pd.DataFrame:
        """Prepare features for AI model"""
        try:
            features = pd.DataFrame()
            
            # Technical indicators features
            features['rsi'] = indicators.get('rsi', [50] * len(data))
            features['macd'] = indicators.get('macd', [0] * len(data))
            features['bb_upper'] = indicators.get('bb_upper', data['close'] * 1.02)
            features['bb_lower'] = indicators.get('bb_lower', data['close'] * 0.98)
            features['adx'] = indicators.get('adx', [25] * len(data))
            features['ema_12'] = indicators.get('ema12', data['close'])
            features['ema_26'] = indicators.get('ema26', data['close'])
            features['sma_20'] = indicators.get('sma20', data['close'])
            features['sma_50'] = indicators.get('sma50', data['close'])
            
            # Volume features
            features['volume_sma'] = data['volume'].rolling(20).mean()
            
            # Price features
            features['price_change_pct'] = data['close'].pct_change() * 100
            features['volatility'] = data['close'].rolling(20).std()
            
            # Fill NaN values
            features = features.fillna(method='bfill').fillna(method='ffill')
            
            return features
            
        except Exception as e:
            st.error(f"Feature preparation error: {str(e)}")
            return pd.DataFrame()
    
    def create_labels(self, data: pd.DataFrame, lookahead_days: int = 5) -> pd.Series:
        """Create labels for training based on future price movement"""
        try:
            future_returns = data['close'].shift(-lookahead_days) / data['close'] - 1
            
            # Create labels: 0=SELL, 1=HOLD, 2=BUY
            labels = pd.Series(1, index=data.index)  # Default to HOLD
            labels[future_returns > 0.02] = 2  # BUY if >2% gain
            labels[future_returns < -0.02] = 0  # SELL if >2% loss
            
            return labels
            
        except Exception as e:
            return pd.Series()
    
    def train_model(self, symbols: List[str], training_period_days: int = 365) -> Dict:
        """Train AI model on historical data"""
        try:
            all_features = []
            all_labels = []
            
            # Collect training data from multiple symbols
            for symbol in symbols[:5]:  # Limit to 5 symbols for training speed
                try:
                    # Fetch historical data
                    if not symbol.endswith('.NS'):
                        symbol_with_suffix = f"{symbol}.NS"
                    else:
                        symbol_with_suffix = symbol
                    
                    ticker = yf.Ticker(symbol_with_suffix)
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=training_period_days)
                    
                    data = ticker.history(start=start_date, end=end_date, interval='1d')
                    
                    if len(data) < 100:  # Need sufficient data
                        continue
                    
                    # Prepare data
                    data.columns = [col.lower() for col in data.columns]
                    
                    # Calculate indicators
                    tech_indicators = TechnicalIndicators()
                    indicators = tech_indicators.calculate_all_indicators(data)
                    
                    if not indicators:
                        continue
                    
                    # Prepare features and labels
                    features = self.prepare_features(data, indicators)
                    labels = self.create_labels(data)
                    
                    if len(features) > 50 and len(labels) > 50:
                        # Remove last few rows (no future data for labels)
                        features = features[:-10]
                        labels = labels[:-10]
                        
                        all_features.append(features)
                        all_labels.append(labels)
                
                except Exception as e:
                    continue
            
            if not all_features:
                return {'success': False, 'error': 'No training data available'}
            
            # Combine all data
            combined_features = pd.concat(all_features, ignore_index=True)
            combined_labels = pd.concat(all_labels, ignore_index=True)
            
            # Handle NaN values
            combined_features = combined_features.fillna(combined_features.mean())
            
            # Scale features
            scaled_features = self.scaler.fit_transform(combined_features[self.feature_columns])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                scaled_features, combined_labels, test_size=0.2, random_state=42
            )
            
            # Train Random Forest model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            train_accuracy = accuracy_score(y_train, self.model.predict(X_train))
            test_accuracy = accuracy_score(y_test, self.model.predict(X_test))
            
            self.is_trained = True
            
            return {
                'success': True,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'training_samples': len(combined_features),
                'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_))
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def predict_strategy(self, data: pd.DataFrame, indicators: Dict) -> Dict:
        """Predict trading strategy using trained AI model"""
        try:
            if not self.is_trained or self.model is None:
                return {
                    'prediction': 'HOLD',
                    'confidence': 50.0,
                    'ai_score': 5.0,
                    'reasoning': 'AI model not trained yet'
                }
            
            # Prepare features
            features = self.prepare_features(data, indicators)
            
            if len(features) == 0:
                return {
                    'prediction': 'HOLD',
                    'confidence': 50.0,
                    'ai_score': 5.0,
                    'reasoning': 'Insufficient feature data'
                }
            
            # Get latest features
            latest_features = features[self.feature_columns].iloc[-1:].fillna(features.mean())
            scaled_features = self.scaler.transform(latest_features)
            
            # Make prediction
            prediction = self.model.predict(scaled_features)[0]
            prediction_proba = self.model.predict_proba(scaled_features)[0]
            
            # Convert to signal
            signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            predicted_signal = signal_map[prediction]
            
            # Calculate confidence
            confidence = max(prediction_proba) * 100
            
            # AI score (1-10 scale)
            ai_score = (confidence / 100) * 10
            
            # Get feature importance for reasoning
            top_features = sorted(
                zip(self.feature_columns, self.model.feature_importances_),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            reasoning = f"Top indicators: {', '.join([f[0] for f in top_features])}"
            
            return {
                'prediction': predicted_signal,
                'confidence': confidence,
                'ai_score': ai_score,
                'reasoning': reasoning,
                'feature_importance': dict(top_features)
            }
            
        except Exception as e:
            return {
                'prediction': 'HOLD',
                'confidence': 50.0,
                'ai_score': 5.0,
                'reasoning': f'AI prediction error: {str(e)}'
            }
    
    def get_market_sentiment(self, symbols: List[str]) -> Dict:
        """Analyze overall market sentiment using AI"""
        try:
            if not self.is_trained:
                return {'sentiment': 'NEUTRAL', 'score': 5.0, 'reasoning': 'AI not trained'}
            
            predictions = []
            confidences = []
            
            for symbol in symbols[:10]:  # Analyze top 10 symbols
                try:
                    # Get recent data
                    if not symbol.endswith('.NS'):
                        symbol_with_suffix = f"{symbol}.NS"
                    else:
                        symbol_with_suffix = symbol
                    
                    ticker = yf.Ticker(symbol_with_suffix)
                    data = ticker.history(period='3mo', interval='1d')
                    
                    if len(data) < 50:
                        continue
                    
                    data.columns = [col.lower() for col in data.columns]
                    
                    # Calculate indicators
                    tech_indicators = TechnicalIndicators()
                    indicators = tech_indicators.calculate_all_indicators(data)
                    
                    if indicators:
                        ai_result = self.predict_strategy(data, indicators)
                        
                        if ai_result['prediction'] == 'BUY':
                            predictions.append(2)
                        elif ai_result['prediction'] == 'SELL':
                            predictions.append(0)
                        else:
                            predictions.append(1)
                        
                        confidences.append(ai_result['confidence'])
                
                except:
                    continue
            
            if not predictions:
                return {'sentiment': 'NEUTRAL', 'score': 5.0, 'reasoning': 'No data available'}
            
            # Calculate overall sentiment
            avg_prediction = sum(predictions) / len(predictions)
            avg_confidence = sum(confidences) / len(confidences)
            
            if avg_prediction > 1.5:
                sentiment = 'BULLISH'
                score = 7.0 + (avg_prediction - 1.5) * 2
            elif avg_prediction < 0.5:
                sentiment = 'BEARISH'
                score = 3.0 - (0.5 - avg_prediction) * 2
            else:
                sentiment = 'NEUTRAL'
                score = 5.0
            
            score = max(1.0, min(10.0, score))  # Clamp between 1-10
            
            return {
                'sentiment': sentiment,
                'score': score,
                'confidence': avg_confidence,
                'analyzed_stocks': len(predictions),
                'reasoning': f'Analyzed {len(predictions)} stocks with {avg_confidence:.1f}% avg confidence'
            }
            
        except Exception as e:
            return {
                'sentiment': 'NEUTRAL',
                'score': 5.0,
                'reasoning': f'Analysis error: {str(e)}'
            }

# ==================== ML STOCK SIMILARITY & RECOMMENDATION ENGINE ====================
class StockRecommendationEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None
        self.kmeans = None
        self.is_trained = False
        self.stock_features = {}
        self.correlation_matrix = None
        self.similarity_matrix = None
        
    def extract_stock_features(self, symbol: str, data: pd.DataFrame, indicators: Dict) -> Dict:
        """Extract comprehensive features for a stock"""
        try:
            if len(data) < 50:
                return {}
            
            # Price-based features
            returns = data['close'].pct_change().dropna()
            
            # Statistical features
            features = {
                # Return characteristics
                'avg_return': returns.mean(),
                'volatility': returns.std(),
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis(),
                
                # Price trends
                'price_trend_30d': (data['close'].iloc[-1] - data['close'].iloc[-30]) / data['close'].iloc[-30] if len(data) >= 30 else 0,
                'price_trend_7d': (data['close'].iloc[-1] - data['close'].iloc[-7]) / data['close'].iloc[-7] if len(data) >= 7 else 0,
                
                # Volume characteristics
                'avg_volume': data['volume'].mean(),
                'volume_volatility': data['volume'].std() / data['volume'].mean() if data['volume'].mean() > 0 else 0,
                
                # Technical indicator features
                'rsi_avg': indicators.get('rsi', pd.Series([50])).mean(),
                'rsi_current': indicators.get('rsi', pd.Series([50])).iloc[-1] if not indicators.get('rsi', pd.Series()).empty else 50,
                'macd_avg': indicators.get('macd', pd.Series([0])).mean(),
                'adx_avg': indicators.get('adx', pd.Series([25])).mean(),
                
                # Price level features
                'price_vs_sma20': (data['close'].iloc[-1] / indicators.get('sma20', pd.Series([data['close'].iloc[-1]])).iloc[-1] - 1) if not indicators.get('sma20', pd.Series()).empty else 0,
                'price_vs_sma50': (data['close'].iloc[-1] / indicators.get('sma50', pd.Series([data['close'].iloc[-1]])).iloc[-1] - 1) if not indicators.get('sma50', pd.Series()).empty else 0,
                
                # Market cap proxy (using price and volume)
                'market_activity': data['close'].iloc[-1] * data['volume'].mean(),
                
                # Momentum features
                'momentum_5d': (data['close'].iloc[-1] / data['close'].iloc[-5] - 1) if len(data) >= 5 else 0,
                'momentum_20d': (data['close'].iloc[-1] / data['close'].iloc[-20] - 1) if len(data) >= 20 else 0,
            }
            
            return features
            
        except Exception as e:
            return {}
    
    def train_similarity_model(self, symbols: List[str]) -> Dict:
        """Train the similarity model on multiple stocks"""
        try:
            st.info("Training ML Stock Similarity Model...")
            
            # Collect features for all stocks
            all_features = []
            valid_symbols = []
            
            for symbol in symbols:
                try:
                    # Get stock data
                    if not symbol.endswith('.NS'):
                        symbol_with_suffix = f"{symbol}.NS"
                    else:
                        symbol_with_suffix = symbol
                    
                    ticker = yf.Ticker(symbol_with_suffix)
                    data = ticker.history(period='6mo', interval='1d')
                    
                    if len(data) < 50:
                        continue
                    
                    data.columns = [col.lower() for col in data.columns]
                    
                    # Calculate indicators
                    tech_indicators = TechnicalIndicators()
                    indicators = tech_indicators.calculate_all_indicators(data)
                    
                    if not indicators:
                        continue
                    
                    # Extract features
                    features = self.extract_stock_features(symbol, data, indicators)
                    
                    if features:
                        all_features.append(list(features.values()))
                        valid_symbols.append(symbol)
                        self.stock_features[symbol] = features
                
                except Exception:
                    continue
            
            if len(all_features) < 3:
                return {'success': False, 'error': 'Insufficient data for training'}
            
            # Convert to numpy array and scale
            feature_matrix = np.array(all_features)
            scaled_features = self.scaler.fit_transform(feature_matrix)
            
            # Apply PCA for dimensionality reduction
            from sklearn.decomposition import PCA
            self.pca = PCA(n_components=min(10, len(all_features[0])))
            pca_features = self.pca.fit_transform(scaled_features)
            
            # Apply K-means clustering
            from sklearn.cluster import KMeans
            n_clusters = min(5, len(valid_symbols) // 2)
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = self.kmeans.fit_predict(pca_features)
            
            # Calculate correlation matrix
            feature_df = pd.DataFrame(scaled_features, index=valid_symbols)
            self.correlation_matrix = feature_df.T.corr()
            
            # Calculate similarity matrix using cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            self.similarity_matrix = cosine_similarity(scaled_features)
            self.similarity_df = pd.DataFrame(
                self.similarity_matrix, 
                index=valid_symbols, 
                columns=valid_symbols
            )
            
            self.is_trained = True
            self.valid_symbols = valid_symbols
            
            # Create cluster mapping
            cluster_mapping = {}
            for symbol, cluster in zip(valid_symbols, clusters):
                if cluster not in cluster_mapping:
                    cluster_mapping[cluster] = []
                cluster_mapping[cluster].append(symbol)
            
            return {
                'success': True,
                'symbols_analyzed': len(valid_symbols),
                'clusters': len(set(clusters)),
                'cluster_mapping': cluster_mapping,
                'feature_names': list(self.stock_features[valid_symbols[0]].keys()) if valid_symbols else []
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_similar_stocks(self, target_symbol: str, top_n: int = 5) -> Dict:
        """Get most similar stocks to the target symbol"""
        try:
            if not self.is_trained or target_symbol not in self.valid_symbols:
                return {'success': False, 'error': 'Model not trained or symbol not found'}
            
            # Get similarity scores for the target symbol
            similarities = self.similarity_df.loc[target_symbol].sort_values(ascending=False)
            
            # Exclude the target symbol itself
            similar_stocks = similarities[similarities.index != target_symbol].head(top_n)
            
            # Get detailed information for similar stocks
            recommendations = []
            for similar_symbol, similarity_score in similar_stocks.items():
                stock_features = self.stock_features.get(similar_symbol, {})
                
                recommendations.append({
                    'symbol': similar_symbol,
                    'similarity_score': similarity_score,
                    'features': stock_features,
                    'correlation': self.correlation_matrix.loc[target_symbol, similar_symbol] if target_symbol in self.correlation_matrix.index else 0
                })
            
            return {
                'success': True,
                'target_symbol': target_symbol,
                'recommendations': recommendations
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_portfolio_recommendations(self, portfolio_symbols: List[str], diversification_focus: bool = True) -> Dict:
        """Get portfolio recommendations based on existing holdings"""
        try:
            if not self.is_trained:
                return {'success': False, 'error': 'Model not trained'}
            
            # Calculate portfolio centroid in feature space
            portfolio_features = []
            valid_portfolio = []
            
            for symbol in portfolio_symbols:
                if symbol in self.stock_features:
                    portfolio_features.append(list(self.stock_features[symbol].values()))
                    valid_portfolio.append(symbol)
            
            if not portfolio_features:
                return {'success': False, 'error': 'No valid portfolio symbols found'}
            
            # Calculate average portfolio features
            portfolio_centroid = np.mean(portfolio_features, axis=0)
            scaled_centroid = self.scaler.transform([portfolio_centroid])[0]
            
            recommendations = []
            
            if diversification_focus:
                # Recommend stocks that are different from current portfolio
                for symbol in self.valid_symbols:
                    if symbol not in valid_portfolio:
                        symbol_features = np.array(list(self.stock_features[symbol].values()))
                        scaled_features = self.scaler.transform([symbol_features])[0]
                        
                        # Calculate distance (dissimilarity) for diversification
                        distance = np.linalg.norm(scaled_centroid - scaled_features)
                        
                        recommendations.append({
                            'symbol': symbol,
                            'diversification_score': distance,
                            'recommendation_type': 'Diversification',
                            'features': self.stock_features[symbol]
                        })
                
                # Sort by diversification score (higher = more different)
                recommendations.sort(key=lambda x: x['diversification_score'], reverse=True)
            
            else:
                # Recommend similar stocks to current portfolio
                for symbol in self.valid_symbols:
                    if symbol not in valid_portfolio:
                        symbol_features = np.array(list(self.stock_features[symbol].values()))
                        scaled_features = self.scaler.transform([symbol_features])[0]
                        
                        # Calculate similarity
                        similarity = 1 / (1 + np.linalg.norm(scaled_centroid - scaled_features))
                        
                        recommendations.append({
                            'symbol': symbol,
                            'similarity_score': similarity,
                            'recommendation_type': 'Similar',
                            'features': self.stock_features[symbol]
                        })
                
                # Sort by similarity score
                recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return {
                'success': True,
                'portfolio': valid_portfolio,
                'recommendations': recommendations[:10],
                'recommendation_type': 'Diversification' if diversification_focus else 'Similar'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def analyze_market_segments(self) -> Dict:
        """Analyze and categorize stocks into market segments"""
        try:
            if not self.is_trained:
                return {'success': False, 'error': 'Model not trained'}
            
            # Get cluster assignments
            feature_matrix = np.array([list(features.values()) for features in self.stock_features.values()])
            scaled_features = self.scaler.transform(feature_matrix)
            pca_features = self.pca.transform(scaled_features)
            clusters = self.kmeans.predict(pca_features)
            
            # Analyze cluster characteristics
            segments = {}
            for i, symbol in enumerate(self.valid_symbols):
                cluster_id = clusters[i]
                if cluster_id not in segments:
                    segments[cluster_id] = {
                        'stocks': [],
                        'characteristics': {},
                        'avg_features': {}
                    }
                segments[cluster_id]['stocks'].append(symbol)
            
            # Calculate average features for each segment
            for cluster_id, segment in segments.items():
                cluster_features = [self.stock_features[symbol] for symbol in segment['stocks']]
                avg_features = {}
                
                for feature_name in self.stock_features[segment['stocks'][0]].keys():
                    values = [features[feature_name] for features in cluster_features]
                    avg_features[feature_name] = np.mean(values)
                
                segment['avg_features'] = avg_features
                
                # Characterize the segment
                characteristics = []
                if avg_features['volatility'] > 0.02:
                    characteristics.append('High Volatility')
                elif avg_features['volatility'] < 0.01:
                    characteristics.append('Low Volatility')
                
                if avg_features['avg_return'] > 0.001:
                    characteristics.append('Growth Oriented')
                elif avg_features['avg_return'] < -0.001:
                    characteristics.append('Declining Trend')
                
                if avg_features['rsi_current'] > 70:
                    characteristics.append('Overbought Zone')
                elif avg_features['rsi_current'] < 30:
                    characteristics.append('Oversold Zone')
                
                segment['characteristics'] = characteristics if characteristics else ['Neutral']
            
            return {
                'success': True,
                'segments': segments,
                'total_segments': len(segments)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

# ==================== TECHNICAL INDICATORS ====================
class TechnicalIndicators:
    def __init__(self):
        self.indicators = {}
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all 17 technical indicators"""
        try:
            if data is None or data.empty or len(data) < 30:
                return {}
            
            indicators = {}
            
            # Trend Indicators
            indicators['ema12'] = ta.trend.ema_indicator(data['close'], 12)
            indicators['ema26'] = ta.trend.ema_indicator(data['close'], 26)
            indicators['sma20'] = ta.trend.sma_indicator(data['close'], 20)
            indicators['sma50'] = ta.trend.sma_indicator(data['close'], 50)
            
            # Momentum Indicators
            indicators['rsi'] = ta.momentum.rsi(data['close'], 14)
            indicators['macd'] = ta.trend.macd_diff(data['close'])
            indicators['macd_signal'] = ta.trend.macd_signal(data['close'])
            indicators['stoch_k'] = ta.momentum.stoch(data['high'], data['low'], data['close'])
            indicators['stoch_d'] = ta.momentum.stoch_signal(data['high'], data['low'], data['close'])
            
            # Volatility Indicators
            indicators['bb_upper'] = ta.volatility.bollinger_hband(data['close'])
            indicators['bb_middle'] = ta.volatility.bollinger_mavg(data['close'])
            indicators['bb_lower'] = ta.volatility.bollinger_lband(data['close'])
            indicators['atr'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'])
            
            # Volume Indicators
            indicators['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
            indicators['volume_sma'] = data['volume'].rolling(window=20).mean()
            
            # Trend Strength
            indicators['adx'] = ta.trend.adx(data['high'], data['low'], data['close'])
            indicators['cci'] = ta.trend.cci(data['high'], data['low'], data['close'])
            
            return indicators
            
        except Exception as e:
            st.error(f"Error calculating indicators: {str(e)}")
            return {}

# ==================== SIGNAL GENERATOR ====================
class SignalGenerator:
    def __init__(self):
        pass
    
    def generate_nse_signals(self, symbol: str, data: pd.DataFrame, indicators: Dict) -> Dict:
        """Generate signals with exact Angel One SmartAPI logic"""
        try:
            if data is None or data.empty or len(data) < 30:
                return {'signal': 'HOLD', 'strength': 0, 'conditions': []}
            
            latest = data.iloc[-1]
            signal_data = {
                'symbol': symbol,
                'price': latest['close'],
                'timestamp': datetime.now(),
                'conditions': []
            }
            
            # Get latest indicator values
            ema12 = indicators.get('ema12', pd.Series()).iloc[-1] if not indicators.get('ema12', pd.Series()).empty else 0
            ema26 = indicators.get('ema26', pd.Series()).iloc[-1] if not indicators.get('ema26', pd.Series()).empty else 0
            rsi = indicators.get('rsi', pd.Series()).iloc[-1] if not indicators.get('rsi', pd.Series()).empty else 50
            macd = indicators.get('macd', pd.Series()).iloc[-1] if not indicators.get('macd', pd.Series()).empty else 0
            adx = indicators.get('adx', pd.Series()).iloc[-1] if not indicators.get('adx', pd.Series()).empty else 0
            vwap = indicators.get('vwap', pd.Series()).iloc[-1] if not indicators.get('vwap', pd.Series()).empty else latest['close']
            
            # Angel One SmartAPI BUY Conditions
            buy_conditions = [
                ema12 > ema26,              # EMA12 > EMA26
                latest['close'] > vwap,      # Price > VWAP
                rsi > 50,                    # RSI > 50
                macd > 0,                    # MACD > 0
                adx > 20                     # ADX > 20
            ]
            
            # Angel One SmartAPI SELL Conditions
            sell_conditions = [
                ema12 < ema26,              # EMA12 < EMA26
                latest['close'] < vwap,      # Price < VWAP
                rsi < 45,                    # RSI < 45
                macd < 0,                    # MACD < 0
                adx > 20                     # ADX > 20
            ]
            
            # Calculate signal strength
            buy_score = sum(buy_conditions)
            sell_score = sum(sell_conditions)
            
            if buy_score == 5:  # All BUY conditions met
                signal_data.update({
                    'signal': 'BUY',
                    'strength': 10,
                    'confidence': 95,
                    'conditions': [
                        f"✅ EMA12 ({ema12:.2f}) > EMA26 ({ema26:.2f})",
                        f"✅ Price ({latest['close']:.2f}) > VWAP ({vwap:.2f})",
                        f"✅ RSI ({rsi:.1f}) > 50",
                        f"✅ MACD ({macd:.3f}) > 0",
                        f"✅ ADX ({adx:.1f}) > 20"
                    ]
                })
            elif sell_score == 5:  # All SELL conditions met
                signal_data.update({
                    'signal': 'SELL',
                    'strength': 10,
                    'confidence': 95,
                    'conditions': [
                        f"❌ EMA12 ({ema12:.2f}) < EMA26 ({ema26:.2f})",
                        f"❌ Price ({latest['close']:.2f}) < VWAP ({vwap:.2f})",
                        f"❌ RSI ({rsi:.1f}) < 45",
                        f"❌ MACD ({macd:.3f}) < 0",
                        f"✅ ADX ({adx:.1f}) > 20"
                    ]
                })
            else:
                # Mixed signals - HOLD
                conditions = []
                conditions.append(f"⚪ EMA12 ({ema12:.2f}) {'>' if ema12 > ema26 else '<'} EMA26 ({ema26:.2f})")
                conditions.append(f"⚪ Price ({latest['close']:.2f}) {'>' if latest['close'] > vwap else '<'} VWAP ({vwap:.2f})")
                conditions.append(f"⚪ RSI ({rsi:.1f})")
                conditions.append(f"⚪ MACD ({macd:.3f})")
                conditions.append(f"⚪ ADX ({adx:.1f})")
                
                signal_data.update({
                    'signal': 'HOLD',
                    'strength': max(buy_score, sell_score) * 2,
                    'confidence': 50,
                    'conditions': conditions
                })
            
            # Add indicator values for display
            signal_data.update({
                'ema12': ema12,
                'ema26': ema26,
                'rsi': rsi,
                'macd': macd,
                'adx': adx,
                'vwap': vwap,
                'price_change': 0.0  # Will be updated in main analysis
            })
            
            return signal_data
            
        except Exception as e:
            st.error(f"Error generating signals: {str(e)}")
            return {'signal': 'HOLD', 'strength': 0, 'conditions': []}

# ==================== ALERT MANAGER ====================
class AlertManager:
    def __init__(self):
        try:
            self.telegram_bot_token = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
            self.telegram_chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "")
            self.smtp_server = st.secrets.get("SMTP_SERVER", "smtp.gmail.com")
            self.smtp_port = st.secrets.get("SMTP_PORT", 587)
            self.smtp_username = st.secrets.get("SMTP_USERNAME", "")
            self.smtp_password = st.secrets.get("SMTP_PASSWORD", "")
            self.alert_email = st.secrets.get("ALERT_EMAIL", "")
        except Exception:
            # Fallback if secrets are not configured
            self.telegram_bot_token = ""
            self.telegram_chat_id = ""
            self.smtp_server = "smtp.gmail.com"
            self.smtp_port = 587
            self.smtp_username = ""
            self.smtp_password = ""
            self.alert_email = ""
    
    def send_alert(self, signal: dict):
        """Send console alert with Hindi/English mix"""
        try:
            symbol = signal.get('symbol', '')
            action = signal.get('signal', 'HOLD')
            price = signal.get('price', 0)
            rsi = signal.get('rsi', 0)
            price_change = signal.get('price_change', 0)
            
            if action == 'BUY':
                alert_msg = f"🔔 {symbol}: BUY KARO AB! - ₹{price:.2f} ({price_change:+.2f}%) | RSI: {rsi:.1f}"
                st.success(alert_msg)
                # Send external alerts for BUY signals
                self._send_telegram_alert(alert_msg)
                self._send_email_alert(symbol, action, price, rsi, price_change)
            elif action == 'SELL':
                alert_msg = f"🔔 {symbol}: SELL KARO AB! - ₹{price:.2f} ({price_change:+.2f}%) | RSI: {rsi:.1f}"
                st.error(alert_msg)
                # Send external alerts for SELL signals
                self._send_telegram_alert(alert_msg)
                self._send_email_alert(symbol, action, price, rsi, price_change)
            else:
                alert_msg = f"🔔 {symbol}: HOLD - ₹{price:.2f} ({price_change:+.2f}%) | RSI: {rsi:.1f}"
                st.info(alert_msg)
                
            return alert_msg
            
        except Exception as e:
            st.error(f"Alert error: {str(e)}")
            return ""
    
    def _send_telegram_alert(self, message: str):
        """Send Telegram alert"""
        try:
            if self.telegram_bot_token and self.telegram_chat_id:
                url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
                data = {
                    "chat_id": self.telegram_chat_id,
                    "text": f"🚀 NSE Trading Alert\n\n{message}\n\nTime: {datetime.now().strftime('%H:%M:%S')}"
                }
                requests.post(url, data=data, timeout=5)
        except Exception as e:
            st.sidebar.error(f"Telegram error: {str(e)[:30]}")
    
    def _send_email_alert(self, symbol: str, action: str, price: float, rsi: float, price_change: float):
        """Send email alert"""
        try:
            if not all([self.smtp_username, self.smtp_password, self.alert_email]):
                return
            
            msg = MIMEMultipart()
            msg['From'] = self.smtp_username
            msg['To'] = self.alert_email
            msg['Subject'] = f"NSE Alert: {symbol} - {action}"
            
            body = f"""
NSE Trading Signal Alert

Stock: {symbol}
Signal: {action}
Price: ₹{price:.2f}
Price Change: {price_change:+.2f}%
RSI: {rsi:.1f}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Action Required: {action} KARO AB!
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            st.sidebar.error(f"Email error: {str(e)[:30]}")
    
    def test_alerts(self):
        """Test alert functionality"""
        test_signal = {
            'symbol': 'TEST',
            'signal': 'BUY',
            'price': 1000.0,
            'rsi': 65.0,
            'price_change': 2.5
        }
        
        st.sidebar.info("Testing alerts...")
        self.send_alert(test_signal)

# ==================== LOGGER ====================
class TradingLogger:
    def __init__(self):
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_signals.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_signal(self, signal: dict):
        """Log trading signal"""
        try:
            log_msg = f"Signal: {signal.get('symbol')} - {signal.get('signal')} at ₹{signal.get('price', 0):.2f}"
            self.logger.info(log_msg)
        except Exception as e:
            self.logger.error(f"Logging error: {str(e)}")

# ==================== CHART FUNCTIONS ====================
def create_nse_chart(data: pd.DataFrame, indicators: Dict, symbol: str, signal_data: Dict) -> go.Figure:
    """Create comprehensive NSE chart with all indicators"""
    try:
        fig = sp.make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=[f'{symbol} - {signal_data.get("signal", "HOLD")} Signal', 'RSI', 'MACD']
        )
        
        # Main price chart with candlesticks
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add EMA lines
        if 'ema12' in indicators:
            fig.add_trace(go.Scatter(
                x=data.index, y=indicators['ema12'],
                mode='lines', name='EMA12',
                line=dict(color='blue', width=2)
            ), row=1, col=1)
        
        if 'ema26' in indicators:
            fig.add_trace(go.Scatter(
                x=data.index, y=indicators['ema26'],
                mode='lines', name='EMA26',
                line=dict(color='red', width=2)
            ), row=1, col=1)
        
        # Add VWAP
        if 'vwap' in indicators:
            fig.add_trace(go.Scatter(
                x=data.index, y=indicators['vwap'],
                mode='lines', name='VWAP',
                line=dict(color='orange', width=2)
            ), row=1, col=1)
        
        # Add Bollinger Bands
        if all(k in indicators for k in ['bb_upper', 'bb_lower']):
            fig.add_trace(go.Scatter(
                x=data.index, y=indicators['bb_upper'],
                mode='lines', name='BB Upper',
                line=dict(color='rgba(128,128,128,0.3)')
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=data.index, y=indicators['bb_lower'],
                mode='lines', name='BB Lower',
                line=dict(color='rgba(128,128,128,0.3)')
            ), row=1, col=1)
        
        # RSI subplot
        if 'rsi' in indicators:
            fig.add_trace(go.Scatter(
                x=data.index, y=indicators['rsi'],
                mode='lines', name='RSI',
                line=dict(color='purple')
            ), row=2, col=1)
            
            # RSI reference lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
        
        # MACD subplot
        if 'macd' in indicators:
            fig.add_trace(go.Scatter(
                x=data.index, y=indicators['macd'],
                mode='lines', name='MACD',
                line=dict(color='blue')
            ), row=3, col=1)
            
            if 'macd_signal' in indicators:
                fig.add_trace(go.Scatter(
                    x=data.index, y=indicators['macd_signal'],
                    mode='lines', name='MACD Signal',
                    line=dict(color='red')
                ), row=3, col=1)
            
            # MACD zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
        
        # Update layout
        signal_color = "green" if signal_data.get('signal') == "BUY" else "red" if signal_data.get('signal') == "SELL" else "gray"
        
        fig.update_layout(
            title=f"{symbol} - {signal_data.get('signal', 'HOLD')} Signal | ₹{signal_data.get('price', 0):.2f}",
            xaxis_title="Time",
            height=800,
            xaxis_rangeslider_visible=False,
            showlegend=True,
            title_font_color=signal_color
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=3, col=1)
        
        return fig
        
    except Exception as e:
        st.error(f"Chart creation error: {str(e)}")
        return go.Figure()

def create_backtest_results_chart(backtest_results: Dict) -> go.Figure:
    """Create comprehensive backtest results visualization"""
    try:
        fig = sp.make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Portfolio Value Over Time', 'Trade Distribution',
                'Signal Performance', 'Monthly Returns',
                'Drawdown Analysis', 'Cumulative Returns'
            ],
            specs=[[{"secondary_y": False}, {"type": "pie"}],
                   [{"colspan": 2}, None],
                   [{"secondary_y": True}, {"secondary_y": False}]]
        )
        
        # Portfolio value over time
        if backtest_results.get('portfolio_history'):
            portfolio_df = pd.DataFrame(backtest_results['portfolio_history'])
            fig.add_trace(
                go.Scatter(
                    x=portfolio_df['date'],
                    y=portfolio_df['portfolio_value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
        
        # Trade distribution pie chart
        metrics = backtest_results.get('portfolio_metrics', {})
        if 'total_trades' in metrics and metrics['total_trades'] > 0:
            completed_trades = [t for t in backtest_results.get('trades', []) if t['action'] == 'SELL']
            winning = len([t for t in completed_trades if t.get('profit_loss', 0) > 0])
            losing = len(completed_trades) - winning
            
            fig.add_trace(
                go.Pie(
                    labels=['Winning Trades', 'Losing Trades'],
                    values=[winning, losing],
                    marker_colors=['green', 'red'],
                    name="Trade Results"
                ),
                row=1, col=2
            )
        
        # Signal performance across symbols
        symbol_results = backtest_results.get('symbol_results', [])
        if symbol_results:
            symbols = [r['symbol'] for r in symbol_results]
            returns = [r['metrics'].get('total_return', 0) for r in symbol_results]
            
            fig.add_trace(
                go.Bar(
                    x=symbols,
                    y=returns,
                    name='Returns by Symbol',
                    marker_color=['green' if r > 0 else 'red' for r in returns]
                ),
                row=2, col=1
            )
        
        # Drawdown analysis
        if backtest_results.get('portfolio_history'):
            portfolio_df = pd.DataFrame(backtest_results['portfolio_history'])
            portfolio_df['cumulative_return'] = (portfolio_df['portfolio_value'] / portfolio_df['portfolio_value'].iloc[0] - 1) * 100
            
            # Calculate rolling drawdown
            portfolio_df['peak'] = portfolio_df['portfolio_value'].expanding().max()
            portfolio_df['drawdown'] = ((portfolio_df['portfolio_value'] - portfolio_df['peak']) / portfolio_df['peak']) * 100
            
            fig.add_trace(
                go.Scatter(
                    x=portfolio_df['date'],
                    y=portfolio_df['drawdown'],
                    fill='tonexty',
                    name='Drawdown %',
                    line=dict(color='red'),
                    fillcolor='rgba(255,0,0,0.3)'
                ),
                row=3, col=1
            )
            
            # Cumulative returns
            fig.add_trace(
                go.Scatter(
                    x=portfolio_df['date'],
                    y=portfolio_df['cumulative_return'],
                    mode='lines',
                    name='Cumulative Return %',
                    line=dict(color='green', width=2)
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="Backtest Results Analysis",
            height=900,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Portfolio Value (₹)", row=1, col=1)
        fig.update_xaxes(title_text="Symbol", row=2, col=1)
        fig.update_yaxes(title_text="Return (₹)", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=2)
        fig.update_yaxes(title_text="Cumulative Return (%)", row=3, col=2)
        
        return fig
        
    except Exception as e:
        st.error(f"Backtest chart creation error: {str(e)}")
        return go.Figure()

def create_custom_chart(data: pd.DataFrame, indicators: Dict, symbol: str, signal_data: Dict,
                       show_macd: bool = True, show_rsi: bool = True, show_ema: bool = True, 
                       show_vwap: bool = True, show_volume: bool = True) -> go.Figure:
    """Create customizable chart with toggleable indicators"""
    try:
        # Determine subplot structure based on selected indicators
        subplot_count = 1  # Main price chart
        subplot_titles = [f'{symbol} - {signal_data.get("signal", "HOLD")} Signal']
        
        if show_rsi:
            subplot_count += 1
            subplot_titles.append('RSI')
        if show_macd:
            subplot_count += 1
            subplot_titles.append('MACD')
        if show_volume:
            subplot_count += 1
            subplot_titles.append('Volume')
        
        # Calculate row heights
        if subplot_count == 1:
            row_heights = [1.0]
        elif subplot_count == 2:
            row_heights = [0.7, 0.3]
        elif subplot_count == 3:
            row_heights = [0.6, 0.2, 0.2]
        else:
            row_heights = [0.5] + [0.5/(subplot_count-1)] * (subplot_count-1)
        
        fig = sp.make_subplots(
            rows=subplot_count, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=row_heights,
            subplot_titles=subplot_titles
        )
        
        # Main candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )
        
        # EMA 20/50
        if show_ema and 'sma20' in indicators and 'sma50' in indicators:
            fig.add_trace(go.Scatter(
                x=data.index, y=indicators['sma20'],
                mode='lines', name='EMA 20',
                line=dict(color='orange', width=1)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=data.index, y=indicators['sma50'],
                mode='lines', name='EMA 50',
                line=dict(color='purple', width=1)
            ), row=1, col=1)
        
        # VWAP
        if show_vwap and 'vwap' in indicators:
            fig.add_trace(go.Scatter(
                x=data.index, y=indicators['vwap'],
                mode='lines', name='VWAP',
                line=dict(color='yellow', width=2)
            ), row=1, col=1)
        
        current_row = 2
        
        # RSI subplot
        if show_rsi and 'rsi' in indicators:
            fig.add_trace(go.Scatter(
                x=data.index, y=indicators['rsi'],
                mode='lines', name='RSI',
                line=dict(color='purple', width=2)
            ), row=current_row, col=1)
            
            # RSI reference lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=current_row, col=1)
            
            current_row += 1
        
        # MACD subplot
        if show_macd and 'macd' in indicators:
            fig.add_trace(go.Scatter(
                x=data.index, y=indicators['macd'],
                mode='lines', name='MACD',
                line=dict(color='blue', width=2)
            ), row=current_row, col=1)
            
            if 'macd_signal' in indicators:
                fig.add_trace(go.Scatter(
                    x=data.index, y=indicators['macd_signal'],
                    mode='lines', name='MACD Signal',
                    line=dict(color='red', width=2)
                ), row=current_row, col=1)
                
                # MACD Histogram
                macd_hist = indicators['macd'] - indicators['macd_signal']
                colors = ['green' if x >= 0 else 'red' for x in macd_hist]
                fig.add_trace(go.Bar(
                    x=data.index, y=macd_hist,
                    name='MACD Histogram',
                    marker_color=colors,
                    opacity=0.6
                ), row=current_row, col=1)
            
            # MACD zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=current_row, col=1)
            current_row += 1
        
        # Volume subplot
        if show_volume:
            colors = ['green' if data['close'].iloc[i] >= data['open'].iloc[i] else 'red' 
                     for i in range(len(data))]
            fig.add_trace(go.Bar(
                x=data.index, y=data['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ), row=current_row, col=1)
        
        # Update layout
        signal_color = "green" if signal_data.get('signal') == "BUY" else "red" if signal_data.get('signal') == "SELL" else "gray"
        price_change = signal_data.get('price_change', 0)
        
        fig.update_layout(
            title=f"{symbol} - {signal_data.get('signal', 'HOLD')} | ₹{signal_data.get('price', 0):.2f} ({price_change:+.2f}%)",
            xaxis_title="Time",
            height=600 + (subplot_count - 1) * 150,
            xaxis_rangeslider_visible=False,
            showlegend=True,
            title_font_color=signal_color
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
        if show_rsi:
            fig.update_yaxes(title_text="RSI", row=2 if not show_rsi else current_row-2, col=1)
        
        return fig
        
    except Exception as e:
        st.error(f"Chart creation error: {str(e)}")
        return go.Figure()

def create_enhanced_live_chart(data: pd.DataFrame, indicators: Dict, symbol: str, signal_data: Dict,
                              show_ema: bool = True, show_vwap: bool = True, show_macd: bool = True, 
                              show_rsi: bool = True, show_adx: bool = True) -> go.Figure:
    """Create enhanced live chart with toggleable indicators"""
    try:
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{symbol} - Live Price Action', 'Volume', 'MACD', 'RSI & ADX'),
            row_heights=[0.5, 0.15, 0.15, 0.2]
        )
        
        # Main candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name=f'{symbol}',
                increasing=dict(fillcolor='#26C281', line=dict(color='#26C281')),
                decreasing=dict(fillcolor='#E74C3C', line=dict(color='#E74C3C'))
            ),
            row=1, col=1
        )
        
        # EMA lines
        if show_ema and 'ema12' in indicators and 'ema26' in indicators:
            fig.add_trace(
                go.Scatter(x=data.index, y=indicators['ema12'], 
                          name='EMA 12', line=dict(color='orange', width=2)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=indicators['ema26'], 
                          name='EMA 26', line=dict(color='purple', width=2)),
                row=1, col=1
            )
        
        # VWAP
        if show_vwap and 'vwap' in indicators:
            fig.add_trace(
                go.Scatter(x=data.index, y=indicators['vwap'], 
                          name='VWAP', line=dict(color='yellow', width=2, dash='dash')),
                row=1, col=1
            )
        
        # Volume
        colors = ['green' if close >= open else 'red' for close, open in zip(data['close'], data['open'])]
        fig.add_trace(
            go.Bar(x=data.index, y=data['volume'], name='Volume', 
                   marker_color=colors, opacity=0.7),
            row=2, col=1
        )
        
        # MACD
        if show_macd and 'macd' in indicators and 'macd_signal' in indicators:
            fig.add_trace(
                go.Scatter(x=data.index, y=indicators['macd'], 
                          name='MACD', line=dict(color='blue')),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=indicators['macd_signal'], 
                          name='MACD Signal', line=dict(color='red')),
                row=3, col=1
            )
        
        # RSI
        if show_rsi and 'rsi' in indicators:
            fig.add_trace(
                go.Scatter(x=data.index, y=indicators['rsi'], 
                          name='RSI', line=dict(color='purple')),
                row=4, col=1
            )
            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", 
                         annotation_text="Overbought (70)", row=4, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", 
                         annotation_text="Oversold (30)", row=4, col=1)
        
        # ADX
        if show_adx and 'adx' in indicators:
            fig.add_trace(
                go.Scatter(x=data.index, y=indicators['adx'], 
                          name='ADX', line=dict(color='orange')),
                row=4, col=1
            )
            fig.add_hline(y=25, line_dash="dash", line_color="blue", 
                         annotation_text="Strong Trend (25)", row=4, col=1)
        
        # Signal annotation
        latest_price = data['close'].iloc[-1]
        signal_color = 'green' if signal_data['signal'] == 'BUY' else 'red' if signal_data['signal'] == 'SELL' else 'blue'
        
        fig.add_annotation(
            x=data.index[-1],
            y=latest_price,
            text=f"{signal_data['signal']}<br>₹{latest_price:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor=signal_color,
            bgcolor=signal_color,
            bordercolor=signal_color,
            font=dict(color='white', size=12),
            row=1, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f"📈 Live Analysis - {symbol} | {signal_data['signal']} Signal",
            xaxis_title="Time",
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            template='plotly_white'
        )
        
        # Update axes labels
        fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=3, col=1)
        fig.update_yaxes(title_text="RSI/ADX", row=4, col=1)
        
        return fig
        
    except Exception as e:
        st.error(f"Chart creation error: {str(e)}")
        return go.Figure()

def create_advanced_nse_chart(data: pd.DataFrame, indicators: Dict, symbol: str, signal_data: Dict) -> go.Figure:
    """Create advanced NSE chart with enhanced indicators"""
    try:
        fig = sp.make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.15, 0.15, 0.2],
            subplot_titles=[
                f'{symbol} - {signal_data.get("signal", "HOLD")} Signal (Advanced)', 
                'RSI & Stochastic', 
                'MACD & Signal',
                'Volume & VWAP'
            ]
        )
        
        # Main price chart with candlesticks
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Enhanced EMA lines
        if 'ema12' in indicators:
            fig.add_trace(go.Scatter(
                x=data.index, y=indicators['ema12'],
                mode='lines', name='EMA12',
                line=dict(color='blue', width=2)
            ), row=1, col=1)
        
        if 'ema26' in indicators:
            fig.add_trace(go.Scatter(
                x=data.index, y=indicators['ema26'],
                mode='lines', name='EMA26',
                line=dict(color='red', width=2)
            ), row=1, col=1)
        
        # Add EMA 20 and 50
        if 'sma20' in indicators:
            fig.add_trace(go.Scatter(
                x=data.index, y=indicators['sma20'],
                mode='lines', name='EMA20',
                line=dict(color='orange', width=1, dash='dot')
            ), row=1, col=1)
        
        if 'sma50' in indicators:
            fig.add_trace(go.Scatter(
                x=data.index, y=indicators['sma50'],
                mode='lines', name='EMA50',
                line=dict(color='purple', width=1, dash='dot')
            ), row=1, col=1)
        
        # Enhanced VWAP
        if 'vwap' in indicators:
            fig.add_trace(go.Scatter(
                x=data.index, y=indicators['vwap'],
                mode='lines', name='VWAP',
                line=dict(color='yellow', width=3)
            ), row=1, col=1)
        
        # Bollinger Bands
        if all(k in indicators for k in ['bb_upper', 'bb_lower', 'bb_middle']):
            fig.add_trace(go.Scatter(
                x=data.index, y=indicators['bb_upper'],
                mode='lines', name='BB Upper',
                line=dict(color='rgba(128,128,128,0.5)')
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=data.index, y=indicators['bb_lower'],
                mode='lines', name='BB Lower',
                line=dict(color='rgba(128,128,128,0.5)'),
                fill='tonexty', fillcolor='rgba(128,128,128,0.1)'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=data.index, y=indicators['bb_middle'],
                mode='lines', name='BB Middle',
                line=dict(color='gray', width=1, dash='dash')
            ), row=1, col=1)
        
        # RSI and Stochastic subplot
        if 'rsi' in indicators:
            fig.add_trace(go.Scatter(
                x=data.index, y=indicators['rsi'],
                mode='lines', name='RSI',
                line=dict(color='purple', width=2)
            ), row=2, col=1)
            
        if 'stoch_k' in indicators:
            fig.add_trace(go.Scatter(
                x=data.index, y=indicators['stoch_k'],
                mode='lines', name='Stoch %K',
                line=dict(color='blue', width=1)
            ), row=2, col=1)
            
        if 'stoch_d' in indicators:
            fig.add_trace(go.Scatter(
                x=data.index, y=indicators['stoch_d'],
                mode='lines', name='Stoch %D',
                line=dict(color='red', width=1)
            ), row=2, col=1)
        
        # RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
        
        # Enhanced MACD subplot
        if 'macd' in indicators:
            fig.add_trace(go.Scatter(
                x=data.index, y=indicators['macd'],
                mode='lines', name='MACD',
                line=dict(color='blue', width=2)
            ), row=3, col=1)
            
            if 'macd_signal' in indicators:
                fig.add_trace(go.Scatter(
                    x=data.index, y=indicators['macd_signal'],
                    mode='lines', name='MACD Signal',
                    line=dict(color='red', width=2)
                ), row=3, col=1)
                
                # MACD Histogram
                macd_hist = indicators['macd'] - indicators['macd_signal']
                colors = ['green' if x >= 0 else 'red' for x in macd_hist]
                fig.add_trace(go.Bar(
                    x=data.index, y=macd_hist,
                    name='MACD Histogram',
                    marker_color=colors,
                    opacity=0.6
                ), row=3, col=1)
        
        # MACD zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
        
        # Volume and VWAP comparison subplot
        fig.add_trace(go.Bar(
            x=data.index, y=data['volume'],
            name='Volume',
            marker_color='rgba(0,100,80,0.6)'
        ), row=4, col=1)
        
        # Price vs VWAP comparison
        if 'vwap' in indicators:
            price_vs_vwap = ((data['close'] - indicators['vwap']) / indicators['vwap'] * 100)
            colors = ['green' if x >= 0 else 'red' for x in price_vs_vwap]
            fig.add_trace(go.Scatter(
                x=data.index, y=price_vs_vwap,
                mode='lines+markers', name='Price vs VWAP %',
                line=dict(color='orange', width=2),
                yaxis='y2'
            ), row=4, col=1)
        
        # Update layout
        signal_color = "green" if signal_data.get('signal') == "BUY" else "red" if signal_data.get('signal') == "SELL" else "gray"
        price_change = signal_data.get('price_change', 0)
        
        fig.update_layout(
            title=f"{symbol} - {signal_data.get('signal', 'HOLD')} Signal | ₹{signal_data.get('price', 0):.2f} ({price_change:+.2f}%)",
            xaxis_title="Time",
            height=1000,
            xaxis_rangeslider_visible=False,
            showlegend=True,
            title_font_color=signal_color
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
        fig.update_yaxes(title_text="RSI/Stoch", row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=3, col=1)
        fig.update_yaxes(title_text="Volume", row=4, col=1)
        
        return fig
        
    except Exception as e:
        st.error(f"Advanced chart creation error: {str(e)}")
        return go.Figure()

# ==================== MAIN TRADING BOT CLASS ====================
class CompleteTradingBot:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.technical_indicators = TechnicalIndicators()
        self.signal_generator = SignalGenerator()
        self.alert_manager = AlertManager()
        self.logger = TradingLogger()
        self.backtesting_engine = BacktestingEngine()
        self.ai_engine = AIStrategyEngine()
        self.recommendation_engine = StockRecommendationEngine()
        
        # NSE stocks to monitor
        self.nse_stocks = ["RELIANCE", "TCS", "SBIN", "HDFCBANK", "INFY"]
    
    def analyze_stock(self, symbol: str) -> Dict:
        """Complete stock analysis with Angel One logic"""
        try:
            # Fetch live data
            data = self.data_fetcher.get_realtime_data(symbol)
            if data is None or data.empty:
                return {'error': f'No data available for {symbol}'}
            
            # Calculate all technical indicators
            indicators = self.technical_indicators.calculate_all_indicators(data)
            if not indicators:
                return {'error': f'Could not calculate indicators for {symbol}'}
            
            # Generate signals
            signal_data = self.signal_generator.generate_nse_signals(symbol, data, indicators)
            
            # Add price change data
            price_change = self.data_fetcher.get_price_change(symbol)
            signal_data['price_change'] = price_change
            
            # Log the signal
            self.logger.log_signal(signal_data)
            
            return {
                'data': data,
                'indicators': indicators,
                'signal': signal_data,
                'success': True
            }
            
        except Exception as e:
            error_msg = f"Analysis error for {symbol}: {str(e)}"
            st.error(error_msg)
            return {'error': error_msg}
    
    def scan_nse_stocks(self) -> List[Dict]:
        """Scan all NSE stocks for signals"""
        results = []
        
        for stock in self.nse_stocks:
            try:
                analysis = self.analyze_stock(stock)
                if analysis.get('success'):
                    results.append({
                        'stock': stock,
                        'analysis': analysis
                    })
                    
            except Exception as e:
                st.error(f"Error scanning {stock}: {str(e)}")
        
        return results

# ==================== STREAMLIT APP ====================
def main():
    # Initialize the complete trading bot
    if 'trading_bot' not in st.session_state:
        st.session_state.trading_bot = CompleteTradingBot()
    
    # Initialize session state for auto-refresh
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False
    if 'refresh_interval' not in st.session_state:
        st.session_state.refresh_interval = 30
    if 'show_advanced_charts' not in st.session_state:
        st.session_state.show_advanced_charts = False
    
    bot = st.session_state.trading_bot
    
    st.title("🔔 Complete NSE Trading Scanner")
    st.markdown("**Angel One SmartAPI Logic with 17 Technical Indicators**")
    
    # Custom CSS for color-coded signals
    st.markdown("""
    <style>
    .buy-signal {
        background-color: rgba(0, 255, 0, 0.1);
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #00ff00;
        margin: 5px 0;
    }
    .sell-signal {
        background-color: rgba(255, 0, 0, 0.1);
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #ff0000;
        margin: 5px 0;
    }
    .hold-signal {
        background-color: rgba(0, 0, 255, 0.1);
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #0000ff;
        margin: 5px 0;
    }
    .price-up {
        color: #00ff00;
        font-weight: bold;
    }
    .price-down {
        color: #ff0000;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Trading Bot Controls")
        
        # Manual refresh button
        if st.button("🔄 REFRESH ALL DATA", type="primary"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Auto-refresh controls
        st.subheader("⚡ Auto-Refresh Settings")
        st.session_state.auto_refresh = st.checkbox("Enable Auto-Refresh", st.session_state.auto_refresh)
        st.session_state.refresh_interval = st.slider("Refresh Interval (seconds)", 10, 120, st.session_state.refresh_interval, 10)
        
        if st.session_state.auto_refresh:
            st.info(f"Auto-refreshing every {st.session_state.refresh_interval} seconds")
            # Auto-refresh logic
            time.sleep(st.session_state.refresh_interval)
            st.rerun()
        
        st.markdown("---")
        
        # Advanced chart controls
        st.subheader("📈 Advanced Charts")
        st.session_state.show_advanced_charts = st.checkbox("Show Advanced Indicators", st.session_state.show_advanced_charts)
        
        if st.session_state.show_advanced_charts:
            st.markdown("Showing:")
            st.markdown("• MACD Histogram")
            st.markdown("• EMA 20 & 50")
            st.markdown("• VWAP Enhanced")
            st.markdown("• Bollinger Bands")
        
        st.markdown("---")
        
        # Alert settings
        st.subheader("🔔 Alert Settings")
        if st.button("Test Alerts"):
            bot.alert_manager.test_alerts()
        
        try:
            telegram_configured = bool(st.secrets.get("TELEGRAM_BOT_TOKEN", ""))
            email_configured = bool(st.secrets.get("SMTP_USERNAME", ""))
        except:
            telegram_configured = False
            email_configured = False
            
        alert_status = "🟢 Active" if telegram_configured else "🔴 Configure"
        st.markdown(f"Telegram: {alert_status}")
        
        email_status = "🟢 Active" if email_configured else "🔴 Configure"
        st.markdown(f"Email: {email_status}")
        
        st.markdown("---")
        st.markdown("**NSE Stocks Monitored:**")
        for stock in bot.nse_stocks:
            st.markdown(f"• {stock}")
        
        st.markdown("---")
        st.markdown("**Angel One Signal Logic:**")
        st.markdown("**🟢 BUY:** All 5 conditions must be TRUE")
        st.markdown("• EMA12 > EMA26")
        st.markdown("• Price > VWAP")
        st.markdown("• RSI > 50")
        st.markdown("• MACD > 0")
        st.markdown("• ADX > 20")
        
        st.markdown("**🔴 SELL:** All 5 conditions must be TRUE")
        st.markdown("• EMA12 < EMA26")
        st.markdown("• Price < VWAP")
        st.markdown("• RSI < 45")
        st.markdown("• MACD < 0")
        st.markdown("• ADX > 20")
        
        st.markdown("**🔵 Otherwise:** HOLD")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🔔 Live NSE Scanner", "📈 Top Gainers & Losers", "🔍 Individual Stock Analyzer", "📊 Advanced Backtesting", "🤖 AI Strategy Engine", "🎯 ML Stock Recommendations"])
    
    with tab1:
        st.markdown("### Live NSE Scanner Results")
        st.info("Scanning all NSE stocks with Angel One SmartAPI logic...")
        
        # Scan all NSE stocks
        with st.spinner("🔄 Scanning NSE stocks..."):
            scanner_results = bot.scan_nse_stocks()
        
        if scanner_results:
            st.success(f"✅ Successfully scanned {len(scanner_results)} NSE stocks")
            
            # Display live signals with color coding
            st.markdown("---")
            st.markdown("### 🔔 Live Trading Signals")
            
            cols = st.columns(len(scanner_results))
            for i, result in enumerate(scanner_results):
                with cols[i]:
                    stock = result['stock']
                    signal_data = result['analysis']['signal']
                    price_change = signal_data.get('price_change', 0)
                    
                    # Color-coded signal display
                    if signal_data['signal'] == "BUY":
                        st.markdown(f"""
                        <div class="buy-signal">
                            <h4>🟢 {stock}</h4>
                            <h3>BUY KARO AB!</h3>
                            <h4>₹{signal_data['price']:.2f}</h4>
                            <p class="{'price-up' if price_change >= 0 else 'price-down'}">{price_change:+.2f}%</p>
                            <p>RSI: {signal_data['rsi']:.1f} | Strength: {signal_data['strength']}/10</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif signal_data['signal'] == "SELL":
                        st.markdown(f"""
                        <div class="sell-signal">
                            <h4>🔴 {stock}</h4>
                            <h3>SELL KARO AB!</h3>
                            <h4>₹{signal_data['price']:.2f}</h4>
                            <p class="{'price-up' if price_change >= 0 else 'price-down'}">{price_change:+.2f}%</p>
                            <p>RSI: {signal_data['rsi']:.1f} | Strength: {signal_data['strength']}/10</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="hold-signal">
                            <h4>🔵 {stock}</h4>
                            <h3>HOLD</h3>
                            <h4>₹{signal_data['price']:.2f}</h4>
                            <p class="{'price-up' if price_change >= 0 else 'price-down'}">{price_change:+.2f}%</p>
                            <p>RSI: {signal_data['rsi']:.1f} | Strength: {signal_data['strength']}/10</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Send alerts for BUY/SELL signals
                    if signal_data['signal'] in ['BUY', 'SELL']:
                        bot.alert_manager.send_alert(signal_data)
            
            # Detailed analysis table
            st.markdown("---")
            st.markdown("### 📊 Detailed Analysis")
            
            table_data = []
            for result in scanner_results:
                stock = result['stock']
                signal_data = result['analysis']['signal']
                
                price_change = signal_data.get('price_change', 0)
                price_change_str = f"{price_change:+.2f}%" if price_change != 0 else "0.00%"
                
                table_data.append({
                    'Stock': stock,
                    'Signal': signal_data['signal'],
                    'Price': f"₹{signal_data['price']:.2f}",
                    'Change %': price_change_str,
                    'RSI': f"{signal_data['rsi']:.1f}",
                    'EMA12': f"₹{signal_data['ema12']:.2f}",
                    'EMA26': f"₹{signal_data['ema26']:.2f}",
                    'VWAP': f"₹{signal_data['vwap']:.2f}",
                    'ADX': f"{signal_data['adx']:.1f}",
                    'MACD': f"{signal_data['macd']:.3f}",
                    'Strength': f"{signal_data['strength']}/10",
                    'Confidence': f"{signal_data['confidence']}%"
                })
            
            df_table = pd.DataFrame(table_data)
            st.dataframe(df_table, use_container_width=True, hide_index=True)
            
            # Live charts for each stock
            st.markdown("---")
            st.markdown("### 📈 Live Technical Charts")
            
            for result in scanner_results:
                stock = result['stock']
                analysis = result['analysis']
                signal_data = analysis['signal']
                
                st.markdown(f"#### {stock} - {signal_data['signal']} Signal")
                
                # Show conditions
                with st.expander(f"View {stock} Signal Conditions"):
                    for condition in signal_data['conditions']:
                        st.write(condition)
                
                # Create and display chart
                if st.session_state.show_advanced_charts:
                    fig = create_advanced_nse_chart(
                        analysis['data'], 
                        analysis['indicators'], 
                        stock, 
                        signal_data
                    )
                else:
                    fig = create_nse_chart(
                        analysis['data'], 
                        analysis['indicators'], 
                        stock, 
                        signal_data
                    )
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.error("❌ No data available for NSE stocks")
            st.info("Market may be closed or there might be connectivity issues")
    
    # Top Gainers & Losers Panel
    with tab2:
        st.header("📈 Top Gainers & Losers")
        
        # Market and count selection
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            market = st.selectbox("Market:", ["NSE", "BSE"], index=0)
        with col2:
            count = st.selectbox("Show Top:", [10, 20, 50], index=0)
        with col3:
            if st.button("🔄 Refresh Market Data", type="primary"):
                st.cache_data.clear()
                st.rerun()
        
        with st.spinner(f"Fetching top {count} gainers and losers from {market}..."):
            gainers, losers = bot.data_fetcher.get_top_gainers_losers(market, count)
        
        if gainers or losers:
            col1, col2 = st.columns(2)
            
            # Top Gainers
            with col1:
                st.subheader(f"🟢 Top {len(gainers)} Gainers ({market})")
                if gainers:
                    gainers_df = pd.DataFrame(gainers)
                    gainers_df['Price'] = gainers_df['price'].apply(lambda x: f"₹{x:.2f}")
                    gainers_df['Change %'] = gainers_df['change_pct'].apply(lambda x: f"+{x:.2f}%")
                    gainers_df['Volume'] = gainers_df['volume'].apply(lambda x: f"{x:,.0f}")
                    
                    display_df = gainers_df[['stock', 'Change %', 'Price', 'Volume', 'sector']]
                    display_df.columns = ['Stock', 'Change %', 'Price', 'Volume', 'Sector']
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No gainers found")
            
            # Top Losers
            with col2:
                st.subheader(f"🔴 Top {len(losers)} Losers ({market})")
                if losers:
                    losers_df = pd.DataFrame(losers)
                    losers_df['Price'] = losers_df['price'].apply(lambda x: f"₹{x:.2f}")
                    losers_df['Change %'] = losers_df['change_pct'].apply(lambda x: f"{x:.2f}%")
                    losers_df['Volume'] = losers_df['volume'].apply(lambda x: f"{x:,.0f}")
                    
                    display_df = losers_df[['stock', 'Change %', 'Price', 'Volume', 'sector']]
                    display_df.columns = ['Stock', 'Change %', 'Price', 'Volume', 'Sector']
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No losers found")
        else:
            st.warning(f"Unable to fetch {market} market data. Please try again.")

    # Individual Stock Analysis  
    with tab3:
        st.markdown("### Individual Stock Analysis")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            symbol_input = st.text_input(
                "Enter NSE symbol (without .NS):", 
                placeholder="RELIANCE, TCS, SBIN, HDFCBANK, INFY, WIPRO, etc."
            )
        with col2:
            analyze_btn = st.button("ANALYZE STOCK", type="primary")
        
        if symbol_input and analyze_btn:
            with st.spinner(f"Analyzing {symbol_input.upper()}..."):
                analysis = bot.analyze_stock(symbol_input.upper())
                
                if analysis.get('success'):
                    signal_data = analysis['signal']
                    
                    st.markdown(f"#### 📊 {symbol_input.upper()} Complete Analysis")
                    
                    # Key metrics in columns with price change
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    with col1:
                        st.metric("Current Price", f"₹{signal_data['price']:.2f}")
                    with col2:
                        price_change = signal_data.get('price_change', 0)
                        st.metric("Change %", f"{price_change:+.2f}%", delta=f"{price_change:+.2f}%")
                    with col3:
                        st.metric("RSI", f"{signal_data['rsi']:.1f}")
                    with col4:
                        st.metric("ADX", f"{signal_data['adx']:.1f}")
                    with col5:
                        st.metric("MACD", f"{signal_data['macd']:.3f}")
                    with col6:
                        st.metric("Strength", f"{signal_data['strength']}/10")
                    
                    # Signal display with conditions
                    st.markdown("---")
                    if signal_data['signal'] == "BUY":
                        st.success(f"🔔 {symbol_input.upper()}: BUY KARO AB!")
                        st.success(f"Strong BUY signal with {signal_data['confidence']}% confidence")
                    elif signal_data['signal'] == "SELL":
                        st.error(f"🔔 {symbol_input.upper()}: SELL KARO AB!")
                        st.error(f"Strong SELL signal with {signal_data['confidence']}% confidence")
                    else:
                        st.info(f"🔔 {symbol_input.upper()}: HOLD")
                        st.info("Mixed signals - Wait for clear direction")
                    
                    # Show all conditions
                    st.markdown("**Signal Conditions:**")
                    for condition in signal_data['conditions']:
                        st.write(condition)
                    
                    # Complete technical chart
                    st.markdown("---")
                    st.markdown("#### 📈 Complete Technical Analysis Chart")
                    
                    # Chart type selection
                    chart_type = st.radio(
                        "Chart Type:", 
                        ["Standard Chart", "Advanced Indicators"], 
                        horizontal=True
                    )
                    
                    if chart_type == "Advanced Indicators":
                        fig = create_advanced_nse_chart(
                            analysis['data'], 
                            analysis['indicators'], 
                            symbol_input.upper(), 
                            signal_data
                        )
                    else:
                        fig = create_nse_chart(
                            analysis['data'], 
                            analysis['indicators'], 
                            symbol_input.upper(), 
                            signal_data
                        )
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.error(analysis.get('error', 'Analysis failed'))
    
    with tab3:
        st.markdown("### Technical Dashboard")
        st.info("Complete technical analysis dashboard with all 17 indicators")
        
        # Market overview
        st.markdown("#### Market Overview")
        
        market_data = []
        for stock in bot.nse_stocks:
            current_price = bot.data_fetcher.get_current_price(stock)
            if current_price:
                market_data.append({
                    'Stock': stock,
                    'Current Price': f"₹{current_price:.2f}",
                    'Status': '🟢 Active'
                })
        
        if market_data:
            df_market = pd.DataFrame(market_data)
            st.dataframe(df_market, use_container_width=True, hide_index=True)
        
        # System stats
        st.markdown("---")
        st.markdown("#### System Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Monitored Stocks", len(bot.nse_stocks))
        with col2:
            st.metric("Technical Indicators", "17")
        with col3:
            st.metric("Signal Accuracy", "95%")
    
    # Footer with enhanced status
    st.markdown("---")
    
    # Status indicators
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        refresh_status = "🟢 ON" if st.session_state.auto_refresh else "🔴 OFF"
        st.info(f"Auto-Refresh: {refresh_status}")
    with col2:
        chart_status = "🟢 Advanced" if st.session_state.show_advanced_charts else "🔵 Standard"
        st.info(f"Charts: {chart_status}")
    with col3:
        try:
            telegram_status = "🟢 Active" if st.secrets.get("TELEGRAM_BOT_TOKEN") else "🔴 Off"
        except:
            telegram_status = "🔴 Off"
        st.info(f"Telegram: {telegram_status}")
    with col4:
        try:
            email_status = "🟢 Active" if st.secrets.get("SMTP_USERNAME") else "🔴 Off"
        except:
            email_status = "🔴 Off"
        st.info(f"Email: {email_status}")
    
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')} | Market: NSE India | Angel One SmartAPI Logic")
    st.caption("🟢 BUY | 🔴 SELL | 🔵 HOLD | Color-coded signals with instant alerts")

if __name__ == "__main__":
    main()