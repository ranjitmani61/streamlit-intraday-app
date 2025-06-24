import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.subplots as sp
from datetime import datetime, timedelta
import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class TradingSignal:
    symbol: str
    action: str  # BUY or SELL
    price: float
    strength: float
    confidence: float
    risk_level: str
    timestamp: datetime
    indicators: List[str]
    indicator_scores: Dict[str, float]

class TechnicalAnalyzer:
    """Simplified technical analysis without TA-Lib dependency"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.indicators = {}
    
    def calculate_sma(self, period: int = 20) -> pd.Series:
        """Simple Moving Average"""
        return self.data['Close'].rolling(window=period).mean()
    
    def calculate_ema(self, period: int = 12) -> pd.Series:
        """Exponential Moving Average"""
        return self.data['Close'].ewm(span=period).mean()
    
    def calculate_rsi(self, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """MACD Indicator"""
        ema_fast = self.data['Close'].ewm(span=fast).mean()
        ema_slow = self.data['Close'].ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def calculate_bollinger_bands(self, period: int = 20, std: int = 2) -> Dict:
        """Bollinger Bands"""
        sma = self.data['Close'].rolling(window=period).mean()
        std_dev = self.data['Close'].rolling(window=period).std()
        
        return {
            'upper': sma + (std_dev * std),
            'middle': sma,
            'lower': sma - (std_dev * std)
        }
    
    def calculate_stochastic(self, k_period: int = 14, d_period: int = 3) -> Dict:
        """Stochastic Oscillator"""
        low_min = self.data['Low'].rolling(window=k_period).min()
        high_max = self.data['High'].rolling(window=k_period).max()
        k_percent = 100 * ((self.data['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'k': k_percent,
            'd': d_percent
        }
    
    def calculate_atr(self, period: int = 14) -> pd.Series:
        """Average True Range"""
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data['Close'].shift())
        low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    def calculate_vwap(self) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        return (typical_price * self.data['Volume']).cumsum() / self.data['Volume'].cumsum()
    
    def calculate_all_indicators(self) -> Dict:
        """Calculate all available indicators"""
        try:
            # Trend Indicators
            self.indicators['SMA_20'] = self.calculate_sma(20)
            self.indicators['SMA_50'] = self.calculate_sma(50)
            self.indicators['EMA_12'] = self.calculate_ema(12)
            self.indicators['EMA_26'] = self.calculate_ema(26)
            
            # Momentum Indicators
            self.indicators['RSI'] = self.calculate_rsi(14)
            macd_data = self.calculate_macd()
            self.indicators['MACD'] = macd_data['macd']
            self.indicators['MACD_Signal'] = macd_data['signal']
            self.indicators['MACD_Histogram'] = macd_data['histogram']
            
            # Volatility Indicators
            bb_data = self.calculate_bollinger_bands()
            self.indicators['BB_Upper'] = bb_data['upper']
            self.indicators['BB_Middle'] = bb_data['middle']
            self.indicators['BB_Lower'] = bb_data['lower']
            self.indicators['ATR'] = self.calculate_atr()
            
            # Volume Indicators
            self.indicators['VWAP'] = self.calculate_vwap()
            
            # Momentum Oscillators
            stoch_data = self.calculate_stochastic()
            self.indicators['Stoch_K'] = stoch_data['k']
            self.indicators['Stoch_D'] = stoch_data['d']
            
            return self.indicators
            
        except Exception as e:
            st.error(f"Error calculating indicators: {str(e)}")
            return {}

class SignalGenerator:
    """Generate trading signals based on technical indicators"""
    
    def __init__(self, data: pd.DataFrame, indicators: Dict):
        self.data = data
        self.indicators = indicators
    
    def generate_signals(self, symbol: str) -> List[TradingSignal]:
        """Generate buy/sell signals"""
        signals = []
        
        if not self.indicators or len(self.data) < 50:
            return signals
        
        try:
            current_price = float(self.data['Close'].iloc[-1])
            
            # Calculate signal strength for BUY
            buy_score = self._calculate_buy_score()
            if buy_score >= 6.0:
                signals.append(TradingSignal(
                    symbol=symbol,
                    action="BUY",
                    price=current_price,
                    strength=buy_score,
                    confidence=min(buy_score * 1.2, 10),
                    risk_level=self._calculate_risk_level(),
                    timestamp=datetime.now(),
                    indicators=self._get_active_indicators('BUY'),
                    indicator_scores=self._get_indicator_scores('BUY')
                ))
            
            # Calculate signal strength for SELL
            sell_score = self._calculate_sell_score()
            if sell_score >= 6.0:
                signals.append(TradingSignal(
                    symbol=symbol,
                    action="SELL",
                    price=current_price,
                    strength=sell_score,
                    confidence=min(sell_score * 1.2, 10),
                    risk_level=self._calculate_risk_level(),
                    timestamp=datetime.now(),
                    indicators=self._get_active_indicators('SELL'),
                    indicator_scores=self._get_indicator_scores('SELL')
                ))
                
        except Exception as e:
            st.error(f"Error generating signals: {str(e)}")
        
        return signals
    
    def _calculate_buy_score(self) -> float:
        """Calculate buy signal score"""
        score = 0
        
        try:
            # RSI oversold
            if 'RSI' in self.indicators:
                rsi = self.indicators['RSI'].iloc[-1]
                if rsi < 30:
                    score += 2.0
                elif rsi < 50:
                    score += 1.0
            
            # MACD bullish crossover
            if 'MACD' in self.indicators and 'MACD_Signal' in self.indicators:
                macd = self.indicators['MACD'].iloc[-1]
                macd_signal = self.indicators['MACD_Signal'].iloc[-1]
                if macd > macd_signal:
                    score += 1.5
            
            # Price above EMA
            if 'EMA_12' in self.indicators:
                ema12 = self.indicators['EMA_12'].iloc[-1]
                current_price = self.data['Close'].iloc[-1]
                if current_price > ema12:
                    score += 1.0
            
            # Bollinger Bands oversold
            if 'BB_Lower' in self.indicators:
                bb_lower = self.indicators['BB_Lower'].iloc[-1]
                current_price = self.data['Close'].iloc[-1]
                if current_price <= bb_lower:
                    score += 1.5
            
            # Stochastic oversold
            if 'Stoch_K' in self.indicators:
                stoch_k = self.indicators['Stoch_K'].iloc[-1]
                if stoch_k < 20:
                    score += 1.0
            
            # Price above VWAP
            if 'VWAP' in self.indicators:
                vwap = self.indicators['VWAP'].iloc[-1]
                current_price = self.data['Close'].iloc[-1]
                if current_price > vwap:
                    score += 0.5
                    
        except Exception:
            pass
            
        return min(score, 10)
    
    def _calculate_sell_score(self) -> float:
        """Calculate sell signal score"""
        score = 0
        
        try:
            # RSI overbought
            if 'RSI' in self.indicators:
                rsi = self.indicators['RSI'].iloc[-1]
                if rsi > 70:
                    score += 2.0
                elif rsi > 50:
                    score += 1.0
            
            # MACD bearish crossover
            if 'MACD' in self.indicators and 'MACD_Signal' in self.indicators:
                macd = self.indicators['MACD'].iloc[-1]
                macd_signal = self.indicators['MACD_Signal'].iloc[-1]
                if macd < macd_signal:
                    score += 1.5
            
            # Price below EMA
            if 'EMA_12' in self.indicators:
                ema12 = self.indicators['EMA_12'].iloc[-1]
                current_price = self.data['Close'].iloc[-1]
                if current_price < ema12:
                    score += 1.0
            
            # Bollinger Bands overbought
            if 'BB_Upper' in self.indicators:
                bb_upper = self.indicators['BB_Upper'].iloc[-1]
                current_price = self.data['Close'].iloc[-1]
                if current_price >= bb_upper:
                    score += 1.5
            
            # Stochastic overbought
            if 'Stoch_K' in self.indicators:
                stoch_k = self.indicators['Stoch_K'].iloc[-1]
                if stoch_k > 80:
                    score += 1.0
            
            # Price below VWAP
            if 'VWAP' in self.indicators:
                vwap = self.indicators['VWAP'].iloc[-1]
                current_price = self.data['Close'].iloc[-1]
                if current_price < vwap:
                    score += 0.5
                    
        except Exception:
            pass
            
        return min(score, 10)
    
    def _calculate_risk_level(self) -> str:
        """Calculate risk level"""
        risk_factors = 0
        
        try:
            # High ATR indicates high volatility
            if 'ATR' in self.indicators:
                atr = self.indicators['ATR'].iloc[-1]
                atr_mean = self.indicators['ATR'].mean()
                if atr > atr_mean * 1.5:
                    risk_factors += 1
            
            # Extreme RSI values
            if 'RSI' in self.indicators:
                rsi = self.indicators['RSI'].iloc[-1]
                if rsi > 80 or rsi < 20:
                    risk_factors += 1
                    
        except Exception:
            pass
        
        if risk_factors >= 2:
            return 'HIGH'
        elif risk_factors == 1:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _get_active_indicators(self, action: str) -> List[str]:
        """Get list of active indicators for the signal"""
        active = []
        
        try:
            if action == 'BUY':
                if 'RSI' in self.indicators and self.indicators['RSI'].iloc[-1] < 50:
                    active.append('RSI')
                if 'MACD' in self.indicators and 'MACD_Signal' in self.indicators:
                    if self.indicators['MACD'].iloc[-1] > self.indicators['MACD_Signal'].iloc[-1]:
                        active.append('MACD')
                if 'EMA_12' in self.indicators and self.data['Close'].iloc[-1] > self.indicators['EMA_12'].iloc[-1]:
                    active.append('EMA')
            else:  # SELL
                if 'RSI' in self.indicators and self.indicators['RSI'].iloc[-1] > 50:
                    active.append('RSI')
                if 'MACD' in self.indicators and 'MACD_Signal' in self.indicators:
                    if self.indicators['MACD'].iloc[-1] < self.indicators['MACD_Signal'].iloc[-1]:
                        active.append('MACD')
                if 'EMA_12' in self.indicators and self.data['Close'].iloc[-1] < self.indicators['EMA_12'].iloc[-1]:
                    active.append('EMA')
                    
        except Exception:
            pass
            
        return active
    
    def _get_indicator_scores(self, action: str) -> Dict[str, float]:
        """Get individual indicator scores"""
        scores = {}
        
        try:
            if 'RSI' in self.indicators:
                rsi = self.indicators['RSI'].iloc[-1]
                if action == 'BUY' and rsi < 50:
                    scores['RSI'] = max(0, (50 - rsi) / 10)
                elif action == 'SELL' and rsi > 50:
                    scores['RSI'] = max(0, (rsi - 50) / 10)
            
            if 'MACD' in self.indicators:
                scores['MACD'] = 1.5
                
        except Exception:
            pass
            
        return scores

def fetch_stock_data(symbol: str, period: str = "3mo", interval: str = "1h") -> Optional[pd.DataFrame]:
    """Fetch stock data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            return None
            
        # Clean data
        data = data.dropna()
        return data
        
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def create_live_chart(data: pd.DataFrame, symbol: str) -> go.Figure:
    """Create interactive chart with indicators"""
    
    fig = sp.make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.2, 0.15, 0.15],
        subplot_titles=[f'{symbol} Price & Technical Indicators', 'MACD', 'RSI', 'Volume']
    )
    
    # Main price chart with Bollinger Bands
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add indicators to main chart
    if 'BB_Upper' in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=indicators['BB_Upper'], 
                                name='BB Upper', line=dict(color='rgba(255,0,0,0.3)')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=indicators['BB_Lower'], 
                                name='BB Lower', line=dict(color='rgba(255,0,0,0.3)')), row=1, col=1)
    
    if 'EMA_12' in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=indicators['EMA_12'], 
                                name='EMA 12', line=dict(color='blue')), row=1, col=1)
    
    if 'SMA_20' in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=indicators['SMA_20'], 
                                name='SMA 20', line=dict(color='orange')), row=1, col=1)
    
    if 'VWAP' in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=indicators['VWAP'], 
                                name='VWAP', line=dict(color='purple')), row=1, col=1)
    
    # MACD
    if 'MACD' in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=indicators['MACD'], 
                                name='MACD', line=dict(color='blue')), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=indicators['MACD_Signal'], 
                                name='Signal', line=dict(color='red')), row=2, col=1)
        fig.add_trace(go.Bar(x=data.index, y=indicators['MACD_Histogram'], 
                            name='Histogram'), row=2, col=1)
    
    # RSI
    if 'RSI' in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=indicators['RSI'], 
                                name='RSI', line=dict(color='purple')), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # Volume
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume'), row=4, col=1)
    
    fig.update_layout(
        title=f'{symbol} Technical Analysis Dashboard',
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True
    )
    
    return fig

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("📈 Live Trading Signals")
    st.markdown("Real-time market analysis and live buy/sell alerts")
    
    # Market status indicator
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.markdown("🔴 **LIVE**")
    with col2:
        current_time = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"⏰ {current_time}")
    with col3:
        if datetime.now().weekday() < 5 and 9 <= datetime.now().hour < 16:
            st.markdown("🟢 **Market Open**")
        else:
            st.markdown("🔴 **Market Closed**")
    
    # Sidebar configuration
    st.sidebar.header("📊 Live Trading Settings")
    
    # Market and symbol selection
    market_type = st.sidebar.selectbox(
        "Select Market",
        ["Indian Stocks (NSE)", "US Stocks"],
        index=0
    )
    
    if market_type == "Indian Stocks (NSE)":
        default_symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS', 'ITC.NS']
        symbol = st.sidebar.selectbox("Select Stock", default_symbols, help="Choose Indian stock")
    else:
        default_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META']
        symbol = st.sidebar.selectbox("Select Stock", default_symbols, help="Choose US stock")
    
    # Intraday settings
    period = st.sidebar.selectbox(
        "Intraday Period",
        ["1d"],
        index=0,
        help="Intraday analysis"
    )
    
    interval = st.sidebar.selectbox(
        "Time Frame",
        ["1m", "5m", "15m"],
        index=0,
        help="1m for quick signals, 5m for medium-term"
    )
    
    # Signal threshold
    signal_threshold = st.sidebar.slider(
        "Alert Sensitivity",
        min_value=1.0,
        max_value=10.0,
        value=5.0,
        step=0.5,
        help="Lower = more alerts, Higher = only strong signals"
    )
    
    # Live updates
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔴 Live Updates")
    
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    auto_refresh = st.sidebar.checkbox("Auto Refresh (15s)", value=True)
    
    if auto_refresh:
        st.sidebar.success("🔴 LIVE MODE")
        time.sleep(15)
        st.rerun()
    else:
        st.sidebar.info("Manual refresh mode")
    
    # Main content
    if symbol:
        # Create tabs
        tab1, tab2 = st.tabs(["🚨 Live Trading Alerts", "📊 Quick Analysis"])
        
        with tab1:
            st.subheader("🚨 Live Intraday Engine")
            
            # NSE stocks exactly like your original
            stocks = ["RELIANCE.NS", "SBIN.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]
            
            st.markdown("### 🔔 Live NSE Scanner")
            st.info("📊 Scanning: RELIANCE, SBIN, TCS, HDFCBANK, INFY")
            
            def fetch_live_data(symbol):
                """Fetch live 5-minute data"""
                try:
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(period="1d", interval="5m")
                    if df.empty:
                        return None
                    df.columns = df.columns.str.lower()
                    return df
                except:
                    return None
            
            def add_indicators(df):
                """Add technical indicators exactly like your original"""
                df['ema12'] = ta.trend.ema_indicator(df['close'], 12)
                df['ema26'] = ta.trend.ema_indicator(df['close'], 26)
                df['rsi'] = ta.momentum.rsi(df['close'])
                df['macd'] = ta.trend.macd_diff(df['close'])
                df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
                df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
                return df
            
            def generate_signal(df):
                """Generate signal exactly like your original logic"""
                if df.empty:
                    return "HOLD"
                    
                latest = df.iloc[-1]
                
                # BUY conditions
                if (
                    latest['ema12'] > latest['ema26']
                    and latest['close'] > latest['vwap']
                    and latest['rsi'] > 50
                    and latest['macd'] > 0
                    and latest['adx'] > 20
                ):
                    return "BUY"
                
                # SELL conditions
                elif (
                    latest['ema12'] < latest['ema26']
                    and latest['close'] < latest['vwap']
                    and latest['rsi'] < 45
                    and latest['macd'] < 0
                    and latest['adx'] > 20
                ):
                    return "SELL"
                
                else:
                    return "HOLD"
            
            # Live analysis
            results = []
            with st.spinner("🔄 Fetching live data..."):
                for stock in stocks:
                    try:
                        df = fetch_live_data(stock)
                        if df is not None and len(df) > 30:
                            df = add_indicators(df)
                            signal = generate_signal(df)
                            current_price = df['close'].iloc[-1]
                            latest = df.iloc[-1]
                            
                            results.append({
                                'stock': stock.replace('.NS', ''),
                                'signal': signal,
                                'price': current_price,
                                'rsi': latest['rsi'],
                                'ema12': latest['ema12'],
                                'ema26': latest['ema26'],
                                'vwap': latest['vwap'],
                                'adx': latest['adx'],
                                'macd': latest['macd']
                            })
                        else:
                            st.warning(f"❌ No data for {stock}")
                    except Exception as e:
                        st.error(f"❌ Error analyzing {stock}: {str(e)}")
            
            # Display results
            if results:
                st.success(f"✅ Analyzed {len(results)} stocks")
                
                # Display results exactly like your original format
                st.markdown("---")
                for result in results:
                    signal = result['signal']
                    stock_name = result['stock']
                    price = result['price']
                    rsi = result['rsi']
                    
                    if signal == "BUY":
                        st.success(f"🔔 {stock_name}: BUY - ₹{price:.2f} | RSI: {rsi:.1f}")
                    elif signal == "SELL":
                        st.error(f"🔔 {stock_name}: SELL - ₹{price:.2f} | RSI: {rsi:.1f}")
                    else:
                        st.info(f"🔔 {stock_name}: HOLD - ₹{price:.2f} | RSI: {rsi:.1f}")
                
                # Detailed table
                st.markdown("---")
                st.markdown("### 📊 Detailed Analysis")
                
                table_data = []
                for result in results:
                    table_data.append({
                        'Stock': result['stock'],
                        'Signal': result['signal'],
                        'Price': f"₹{result['price']:.2f}",
                        'RSI': f"{result['rsi']:.1f}",
                        'EMA12': f"₹{result['ema12']:.2f}",
                        'EMA26': f"₹{result['ema26']:.2f}",
                        'VWAP': f"₹{result['vwap']:.2f}",
                        'ADX': f"{result['adx']:.1f}",
                        'MACD': f"{result['macd']:.3f}"
                    })
                
                df_table = pd.DataFrame(table_data)
                st.dataframe(df_table, use_container_width=True, hide_index=True)
                
                # Show charts for all stocks
                st.markdown("---")
                st.markdown("### 📈 Live Charts")
                
                for i, result in enumerate(results):
                    stock_name = result['stock']
                    
                    # Fetch fresh data for chart
                    df_chart = fetch_live_data(f"{stock_name}.NS")
                    if df_chart is not None and len(df_chart) > 30:
                        df_chart = add_indicators(df_chart)
                        
                        # Create chart
                        fig = go.Figure()
                        
                        # Candlestick
                        fig.add_trace(go.Candlestick(
                            x=df_chart.index,
                            open=df_chart['open'],
                            high=df_chart['high'],
                            low=df_chart['low'],
                            close=df_chart['close'],
                            name='Price'
                        ))
                        
                        # Add indicators
                        fig.add_trace(go.Scatter(
                            x=df_chart.index,
                            y=df_chart['ema12'],
                            mode='lines',
                            name='EMA12',
                            line=dict(color='blue', width=1.5)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=df_chart.index,
                            y=df_chart['ema26'],
                            mode='lines',
                            name='EMA26',
                            line=dict(color='red', width=1.5)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=df_chart.index,
                            y=df_chart['vwap'],
                            mode='lines',
                            name='VWAP',
                            line=dict(color='orange', width=1.5)
                        ))
                        
                        # Update layout
                        signal_color = "green" if result['signal'] == "BUY" else "red" if result['signal'] == "SELL" else "gray"
                        fig.update_layout(
                            title=f"{stock_name} - {result['signal']} Signal | ₹{result['price']:.2f}",
                            xaxis_title="Time",
                            yaxis_title="Price (₹)",
                            height=400,
                            xaxis_rangeslider_visible=False,
                            showlegend=True,
                            title_font_color=signal_color
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("❌ No data available - Market may be closed")
                st.info("Try refreshing the page")
            
            # Individual stock analysis
            st.markdown("---")
            st.markdown("### 🎯 Individual Analysis")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                manual_symbol = st.text_input("Enter NSE symbol:", placeholder="RELIANCE.NS, TCS.NS, SBIN.NS")
            with col2:
                if st.button("ANALYZE"):
                    if manual_symbol:
                        with st.spinner(f"Analyzing {manual_symbol.upper()}..."):
                            df = fetch_live_data(manual_symbol.upper())
                            if df is not None and len(df) > 30:
                                df = add_indicators(df)
                                signal = generate_signal(df)
                                current_price = df['close'].iloc[-1]
                                latest = df.iloc[-1]
                                
                                stock_name = manual_symbol.upper().replace('.NS', '')
                                
                                st.markdown(f"#### 📊 {stock_name} Analysis")
                                
                                # Metrics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Price", f"₹{current_price:.2f}")
                                with col2:
                                    st.metric("RSI", f"{latest['rsi']:.1f}")
                                with col3:
                                    st.metric("ADX", f"{latest['adx']:.1f}")
                                with col4:
                                    st.metric("MACD", f"{latest['macd']:.3f}")
                                
                                # Signal
                                if signal == "BUY":
                                    st.success(f"🔔 {stock_name}: BUY")
                                    st.write("✅ EMA12 > EMA26")
                                    st.write("✅ Price > VWAP") 
                                    st.write("✅ RSI > 50")
                                    st.write("✅ MACD > 0")
                                    st.write("✅ ADX > 20")
                                elif signal == "SELL":
                                    st.error(f"🔔 {stock_name}: SELL")
                                    st.write("❌ EMA12 < EMA26")
                                    st.write("❌ Price < VWAP")
                                    st.write("❌ RSI < 45") 
                                    st.write("❌ MACD < 0")
                                    st.write("✅ ADX > 20")
                                else:
                                    st.info(f"🔔 {stock_name}: HOLD")
                                
                                # Display live chart
                                st.markdown(f"#### 📈 {stock_name} Live Chart")
                                
                                # Create candlestick chart
                                fig = go.Figure()
                                
                                # Candlestick
                                fig.add_trace(go.Candlestick(
                                    x=df.index,
                                    open=df['open'],
                                    high=df['high'],
                                    low=df['low'],
                                    close=df['close'],
                                    name='Price'
                                ))
                                
                                # Add EMA lines
                                fig.add_trace(go.Scatter(
                                    x=df.index,
                                    y=df['ema12'],
                                    mode='lines',
                                    name='EMA12',
                                    line=dict(color='blue', width=2)
                                ))
                                
                                fig.add_trace(go.Scatter(
                                    x=df.index,
                                    y=df['ema26'],
                                    mode='lines',
                                    name='EMA26',
                                    line=dict(color='red', width=2)
                                ))
                                
                                fig.add_trace(go.Scatter(
                                    x=df.index,
                                    y=df['vwap'],
                                    mode='lines',
                                    name='VWAP',
                                    line=dict(color='orange', width=2)
                                ))
                                
                                # Update layout
                                fig.update_layout(
                                    title=f"{stock_name} - Live Intraday Chart (5min)",
                                    xaxis_title="Time",
                                    yaxis_title="Price (₹)",
                                    height=500,
                                    xaxis_rangeslider_visible=False,
                                    showlegend=True
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error("Could not fetch data")


        with tab3:
            st.subheader("🚨 Trading Signals")
            
            if data is not None and len(data) > 0 and indicators:
                # Generate signals
                signal_gen = SignalGenerator(data, indicators)
                signals = signal_gen.generate_signals(symbol.upper())
                
                # Filter signals by threshold
                filtered_signals = [s for s in signals if s.strength >= signal_threshold]
                
                if filtered_signals:
                    st.success(f"Found {len(filtered_signals)} trading signal(s) for {symbol.upper()}")
                    
                    for signal in filtered_signals:
                        # Signal card
                        color = "green" if signal.action == "BUY" else "red"
                        emoji = "🟢" if signal.action == "BUY" else "🔴"
                        
                        # Create alert box
                        st.markdown(f"""
                        <div style="border: 2px solid {color}; border-radius: 10px; padding: 15px; margin: 15px 0; background-color: rgba({'0,255,0' if signal.action == 'BUY' else '255,0,0'},0.1);">
                            <h2 style="color: {color}; margin-top: 0;">{emoji} {signal.action} ALERT - {signal.symbol}</h2>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                                <div><strong>💰 Price:</strong> ${signal.price:.2f}</div>
                                <div><strong>⏰ Time:</strong> {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</div>
                                <div><strong>💪 Signal Strength:</strong> {signal.strength:.1f}/10</div>
                                <div><strong>🎯 Confidence:</strong> {signal.confidence:.1f}/10</div>
                                <div><strong>⚠️ Risk Level:</strong> {signal.risk_level}</div>
                                <div><strong>📊 Active Indicators:</strong> {len(signal.indicators)}</div>
                            </div>
                            <div style="margin-top: 10px;">
                                <strong>📈 Contributing Indicators:</strong> {', '.join(signal.indicators)}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add recommendation
                        if signal.action == "BUY":
                            if signal.strength >= 8:
                                st.success("🔥 STRONG BUY recommendation - Multiple indicators align!")
                            elif signal.strength >= 6:
                                st.info("📈 BUY signal detected - Consider entering position")
                        else:
                            if signal.strength >= 8:
                                st.error("🚨 STRONG SELL recommendation - Multiple indicators align!")
                            elif signal.strength >= 6:
                                st.warning("📉 SELL signal detected - Consider exiting position")
                
                # Always show current market analysis
                st.markdown("---")
                st.subheader(f"📊 Current Market Analysis for {symbol.upper()}")
                
                current_price = data['Close'].iloc[-1]
                
                # Market condition analysis
                analysis_text = []
                
                if 'RSI' in indicators:
                    rsi = indicators['RSI'].iloc[-1]
                    if rsi < 30:
                        analysis_text.append(f"🔥 RSI is oversold at {rsi:.1f} - Potential buying opportunity")
                    elif rsi > 70:
                        analysis_text.append(f"⚠️ RSI is overbought at {rsi:.1f} - Consider taking profits")
                    else:
                        analysis_text.append(f"RSI is neutral at {rsi:.1f}")
                
                if 'EMA_12' in indicators:
                    ema12 = indicators['EMA_12'].iloc[-1]
                    if current_price > ema12:
                        analysis_text.append(f"📈 Price ${current_price:.2f} is above EMA12 ${ema12:.2f} - Bullish trend")
                    else:
                        analysis_text.append(f"📉 Price ${current_price:.2f} is below EMA12 ${ema12:.2f} - Bearish trend")
                
                if 'BB_Upper' in indicators and 'BB_Lower' in indicators:
                    bb_upper = indicators['BB_Upper'].iloc[-1]
                    bb_lower = indicators['BB_Lower'].iloc[-1]
                    if current_price >= bb_upper:
                        analysis_text.append("⚠️ Price is at upper Bollinger Band - Potential resistance")
                    elif current_price <= bb_lower:
                        analysis_text.append("🔥 Price is at lower Bollinger Band - Potential support")
                
                for text in analysis_text:
                    st.write(text)
                
                if not filtered_signals:
                    st.info(f"No strong signals detected for {symbol.upper()} at current threshold ({signal_threshold}). Market analysis shows neutral conditions or signals below threshold.")
            else:
                st.warning("Load data in the Analysis tab first to generate signals.")
        
        with tab3:
            st.subheader("📈 Technical Indicators Summary")
            
            if data is not None and indicators:
                # Create indicator summary table
                indicator_data = []
                
                try:
                    if 'RSI' in indicators:
                        rsi_val = indicators['RSI'].iloc[-1]
                        rsi_signal = "Oversold" if rsi_val < 30 else "Overbought" if rsi_val > 70 else "Neutral"
                        indicator_data.append(["RSI", f"{rsi_val:.2f}", rsi_signal])
                    
                    if 'MACD' in indicators and 'MACD_Signal' in indicators:
                        macd_val = indicators['MACD'].iloc[-1]
                        macd_signal_val = indicators['MACD_Signal'].iloc[-1]
                        macd_signal = "Bullish" if macd_val > macd_signal_val else "Bearish"
                        indicator_data.append(["MACD", f"{macd_val:.4f}", macd_signal])
                    
                    if 'EMA_12' in indicators:
                        ema_val = indicators['EMA_12'].iloc[-1]
                        current_price = data['Close'].iloc[-1]
                        ema_signal = "Above EMA" if current_price > ema_val else "Below EMA"
                        indicator_data.append(["EMA 12", f"${ema_val:.2f}", ema_signal])
                    
                    if 'BB_Upper' in indicators and 'BB_Lower' in indicators:
                        bb_upper = indicators['BB_Upper'].iloc[-1]
                        bb_lower = indicators['BB_Lower'].iloc[-1]
                        current_price = data['Close'].iloc[-1]
                        if current_price >= bb_upper:
                            bb_signal = "Above Upper Band"
                        elif current_price <= bb_lower:
                            bb_signal = "Below Lower Band"
                        else:
                            bb_signal = "Within Bands"
                        indicator_data.append(["Bollinger Bands", f"${bb_upper:.2f} / ${bb_lower:.2f}", bb_signal])
                    
                    if 'Stoch_K' in indicators:
                        stoch_k = indicators['Stoch_K'].iloc[-1]
                        stoch_signal = "Oversold" if stoch_k < 20 else "Overbought" if stoch_k > 80 else "Neutral"
                        indicator_data.append(["Stochastic %K", f"{stoch_k:.2f}", stoch_signal])
                    
                    if 'ATR' in indicators:
                        atr_val = indicators['ATR'].iloc[-1]
                        indicator_data.append(["ATR", f"${atr_val:.2f}", "Volatility Measure"])
                    
                    if 'VWAP' in indicators:
                        vwap_val = indicators['VWAP'].iloc[-1]
                        current_price = data['Close'].iloc[-1]
                        vwap_signal = "Above VWAP" if current_price > vwap_val else "Below VWAP"
                        indicator_data.append(["VWAP", f"${vwap_val:.2f}", vwap_signal])
                    
                    if indicator_data:
                        df_indicators = pd.DataFrame(indicator_data, columns=["Indicator", "Value", "Signal"])
                        st.dataframe(df_indicators, use_container_width=True)
                    else:
                        st.warning("No indicator data available")
                        
                except Exception as e:
                    st.error(f"Error displaying indicators: {str(e)}")
            else:
                st.warning("Load data in the Analysis tab first to view indicators.")
        
        with tab5:
            st.subheader("ℹ️ Trading Bot Information")
            
            st.markdown("""
            ### Features
            - **Real-time Analysis**: Live market data from Yahoo Finance
            - **17 Technical Indicators**: Comprehensive technical analysis
            - **Automated Signals**: Buy/sell signals with confidence scoring
            - **Interactive Charts**: Professional trading charts with indicators
            - **Risk Assessment**: Automatic risk level calculation
            
            ### Available Indicators
            
            **Trend Indicators:**
            - Simple Moving Average (SMA)
            - Exponential Moving Average (EMA)
            - MACD (Moving Average Convergence Divergence)
            
            **Momentum Indicators:**
            - RSI (Relative Strength Index)
            - Stochastic Oscillator
            
            **Volatility Indicators:**
            - Bollinger Bands
            - Average True Range (ATR)
            
            **Volume Indicators:**
            - VWAP (Volume Weighted Average Price)
            
            ### Signal Interpretation
            - **Strength**: Signal power from 1-10 (higher is stronger)
            - **Confidence**: Algorithm confidence in the signal
            - **Risk Level**: LOW, MEDIUM, or HIGH based on market conditions
            
            ### Usage Tips
            1. Use multiple timeframes for better analysis
            2. Combine signals with fundamental analysis
            3. Always consider risk management
            4. This is for educational purposes only
            
            **Disclaimer**: This tool is for educational purposes only. Always do your own research and consider consulting with a financial advisor before making investment decisions.
            """)

if __name__ == "__main__":
    main()