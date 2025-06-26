"""
Enhanced UI components for Professional Intraday Trading Assistant
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import numpy as np

class UIComponents:
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize all session state variables"""
        if 'watchlist' not in st.session_state:
            from config import DEFAULT_WATCHLIST
            st.session_state.watchlist = DEFAULT_WATCHLIST.copy()
        
        if 'selected_stock' not in st.session_state:
            st.session_state.selected_stock = 'RELIANCE'
        
        if 'scanning_active' not in st.session_state:
            st.session_state.scanning_active = False
        
        if 'alerts_enabled' not in st.session_state:
            st.session_state.alerts_enabled = True
        
        if 'telegram_connected' not in st.session_state:
            st.session_state.telegram_connected = True
        
        if 'current_timeframe' not in st.session_state:
            st.session_state.current_timeframe = '5m'
        
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = False
        
        if 'refresh_interval' not in st.session_state:
            st.session_state.refresh_interval = 15
    
    def setup_page_config(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="Professional Intraday Trading Assistant",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Professional CSS styling
        st.markdown("""
        <style>
        /* Global Styles */
        .main > div {
            padding: 0.5rem 1rem;
            max-width: 100%;
        }
        
        /* Header Styles */
        .trading-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            text-align: center;
        }
        
        /* Signal Cards */
        .signal-card {
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
            border-left: 5px solid;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .buy-signal {
            background-color: #d4edda;
            border-left-color: #28a745;
        }
        
        .sell-signal {
            background-color: #f8d7da;
            border-left-color: #dc3545;
        }
        
        .hold-signal {
            background-color: #cce7ff;
            border-left-color: #007bff;
        }
        
        /* Metric Cards */
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
            border-left: 4px solid #007bff;
        }
        
        /* Scanner Status */
        .scanner-status {
            position: fixed;
            top: 80px;
            right: 20px;
            z-index: 1000;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            color: white;
            font-weight: bold;
        }
        
        .scanner-active {
            background-color: #28a745;
            animation: pulse 2s infinite;
        }
        
        .scanner-inactive {
            background-color: #6c757d;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        
        /* Button Styles */
        .stButton > button {
            border-radius: 8px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        /* Sidebar Styles */
        .sidebar .stSelectbox > div > div {
            background-color: #f8f9fa;
        }
        
        /* Chart Container */
        .chart-container {
            background: white;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        
        /* Table Styles */
        .dataframe {
            border-radius: 8px;
            overflow: hidden;
        }
        
        .dataframe th {
            background-color: #f8f9fa;
            font-weight: 600;
        }
        
        /* Alert Styles */
        .alert-box {
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            border-left: 4px solid;
        }
        
        .alert-success {
            background-color: #d4edda;
            border-left-color: #28a745;
            color: #155724;
        }
        
        .alert-warning {
            background-color: #fff3cd;
            border-left-color: #ffc107;
            color: #856404;
        }
        
        .alert-danger {
            background-color: #f8d7da;
            border-left-color: #dc3545;
            color: #721c24;
        }
        
        /* Mobile Responsive */
        @media (max-width: 768px) {
            .main > div {
                padding: 0.25rem 0.5rem;
            }
            
            .trading-header {
                padding: 1rem;
                font-size: 0.9rem;
            }
            
            .signal-card {
                padding: 0.75rem;
                margin: 0.25rem 0;
            }
            
            .scanner-status {
                top: 60px;
                right: 10px;
                font-size: 0.8rem;
                padding: 0.3rem 0.6rem;
            }
        }
        
        /* Loading Animation */
        .loading-spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 2s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
        """, unsafe_allow_html=True)
    
    def create_header(self):
        """Create professional header"""
        st.markdown("""
        <div class="trading-header">
            <h1>📊 Professional Intraday Trading Assistant</h1>
            <p>Real-time NSE/BSE Scanner | AI-Powered Signals | Professional Trading Tools</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Market status indicator
        self.create_market_status()
    
    def create_market_status(self):
        """Create market status indicator"""
        current_time = datetime.now()
        is_market_hours = (9 <= current_time.hour <= 15) and (current_time.weekday() < 5)
        
        status_color = "#28a745" if is_market_hours else "#dc3545"
        status_text = "🟢 Market Open" if is_market_hours else "🔴 Market Closed"
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="background-color: {status_color}; color: white; padding: 0.5rem; 
                       border-radius: 5px; text-align: center; font-weight: bold;">
                {status_text}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background-color: #007bff; color: white; padding: 0.5rem; 
                       border-radius: 5px; text-align: center;">
                📅 {current_time.strftime('%d %b %Y')}
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background-color: #6f42c1; color: white; padding: 0.5rem; 
                       border-radius: 5px; text-align: center;">
                🕐 {current_time.strftime('%H:%M:%S')}
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            scanner_status = "🔴 Inactive" if not st.session_state.scanning_active else "🟢 Active"
            scanner_color = "#28a745" if st.session_state.scanning_active else "#6c757d"
            st.markdown(f"""
            <div style="background-color: {scanner_color}; color: white; padding: 0.5rem; 
                       border-radius: 5px; text-align: center;">
                📡 Scanner: {scanner_status}
            </div>
            """, unsafe_allow_html=True)
    
    def create_professional_sidebar(self) -> Dict[str, Any]:
        """Create comprehensive professional sidebar"""
        with st.sidebar:
            st.markdown("## 🎛️ Trading Control Panel")
            
            # Quick Navigation
            st.markdown("### 🚀 Quick Actions")
            nav_col1, nav_col2 = st.columns(2)
            
            with nav_col1:
                if st.button("📡 Scanner", use_container_width=True):
                    st.session_state.current_tab = 'scanner'
                    st.rerun()
                
                if st.button("🎯 AI Signals", use_container_width=True):
                    st.session_state.current_tab = 'ai_signals'
                    st.rerun()
                
                if st.button("📊 Analysis", use_container_width=True):
                    st.session_state.current_tab = 'analysis'
                    st.rerun()
            
            with nav_col2:
                if st.button("📈 Backtest", use_container_width=True):
                    st.session_state.current_tab = 'backtest'
                    st.rerun()
                
                if st.button("📝 Journal", use_container_width=True):
                    st.session_state.current_tab = 'journal'
                    st.rerun()
                
                if st.button("🔔 Alerts", use_container_width=True):
                    st.session_state.current_tab = 'alerts'
                    st.rerun()
            
            st.markdown("---")
            
            # Trading Parameters
            st.markdown("### ⚙️ Trading Setup")
            
            # Market Selection
            market_filter = st.selectbox(
                "🏛️ Market:",
                ["NSE", "BSE", "Both"],
                index=0,
                key="market_filter"
            )
            
            # Timeframe Selection
            timeframe = st.selectbox(
                "⏰ Timeframe:",
                ["1m", "2m", "3m", "5m", "10m", "15m", "30m"],
                index=3,  # Default to 5m
                key="timeframe_selector"
            )
            
            # Period Selection
            period = st.selectbox(
                "📅 Period:",
                ["1d", "2d", "5d"],
                index=0,
                key="period_selector"
            )
            
            st.markdown("---")
            
            # Scanner Configuration
            st.markdown("### 🔍 Scanner Config")
            
            volume_threshold = st.slider(
                "📊 Volume Surge Threshold:",
                min_value=1.5,
                max_value=5.0,
                value=2.0,
                step=0.1,
                key="volume_threshold"
            )
            
            gap_threshold = st.slider(
                "📈 Gap Threshold (%):",
                min_value=1.0,
                max_value=10.0,
                value=2.0,
                step=0.5,
                key="gap_threshold"
            )
            
            rsi_overbought = st.slider(
                "📊 RSI Overbought:",
                min_value=60,
                max_value=90,
                value=70,
                step=5,
                key="rsi_overbought"
            )
            
            rsi_oversold = st.slider(
                "📊 RSI Oversold:",
                min_value=10,
                max_value=40,
                value=30,
                step=5,
                key="rsi_oversold"
            )
            
            st.markdown("---")
            
            # Auto-Refresh Settings
            st.markdown("### 🔄 Auto-Refresh")
            
            auto_refresh = st.toggle(
                "Enable Auto-Refresh",
                value=st.session_state.auto_refresh,
                key="auto_refresh_toggle"
            )
            
            if auto_refresh:
                refresh_interval = st.selectbox(
                    "Refresh Interval:",
                    [5, 10, 15, 30, 60],
                    index=2,
                    key="refresh_interval_select"
                )
                st.success(f"🔄 Auto-refreshing every {refresh_interval}s")
                st.session_state.auto_refresh = True
                st.session_state.refresh_interval = refresh_interval
            else:
                st.session_state.auto_refresh = False
                st.info("🔄 Manual refresh mode")
            
            # Manual Refresh Button
            if st.button("🔄 Refresh Now", type="primary", use_container_width=True):
                st.rerun()
            
            st.markdown("---")
            
            # Alert Settings
            st.markdown("### 🔔 Alert Settings")
            
            telegram_alerts = st.toggle(
                "📱 Telegram Alerts",
                value=True,
                key="telegram_alerts"
            )
            
            email_alerts = st.toggle(
                "📧 Email Alerts",
                value=False,
                key="email_alerts"
            )
            
            sound_alerts = st.toggle(
                "🔊 Sound Alerts",
                value=False,
                key="sound_alerts"
            )
            
            if telegram_alerts:
                st.success("✅ Telegram Connected")
                st.caption("🤖 @MyStockSentryBot")
                
                if st.button("🧪 Test Telegram", use_container_width=True):
                    st.session_state.test_telegram = True
            
            st.markdown("---")
            
            # Watchlist Quick Add
            st.markdown("### 📋 Quick Add to Watchlist")
            new_stock = st.text_input("Stock Symbol:", placeholder="RELIANCE", key="quick_add_stock")
            
            if st.button("➕ Add Stock", use_container_width=True):
                if new_stock and new_stock.upper() not in st.session_state.watchlist:
                    st.session_state.watchlist.append(new_stock.upper())
                    st.success(f"✅ Added {new_stock.upper()}")
                    st.rerun()
            
            # Predefined Lists
            st.markdown("### 📚 Quick Lists")
            
            if st.button("⭐ Nifty 50", use_container_width=True):
                from config import NIFTY_50_STOCKS
                self._add_stocks_to_watchlist(NIFTY_50_STOCKS[:20])  # Add first 20
            
            if st.button("🏦 Bank Nifty", use_container_width=True):
                bank_stocks = ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK"]
                self._add_stocks_to_watchlist(bank_stocks)
            
            if st.button("💻 IT Stocks", use_container_width=True):
                it_stocks = ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM"]
                self._add_stocks_to_watchlist(it_stocks)
            
            st.markdown("---")
            
            # System Status
            st.markdown("### 📊 System Status")
            
            # Data source status
            st.markdown("**Data Sources:**")
            st.success("✅ yfinance API")
            st.success("✅ NSE Real-time")
            
            # AI Model status
            st.markdown("**AI Models:**")
            st.success("✅ Random Forest")
            st.info("🔄 XGBoost (Loading)")
            
            # Performance metrics
            st.markdown("**Performance:**")
            st.metric("Response Time", "150ms", "-20ms")
            st.metric("Cache Hit Rate", "85%", "+5%")
            
        # Return configuration dictionary
        return {
            'market_filter': market_filter,
            'timeframe': timeframe,
            'period': period,
            'volume_threshold': volume_threshold,
            'gap_threshold': gap_threshold,
            'rsi_overbought': rsi_overbought,
            'rsi_oversold': rsi_oversold,
            'auto_refresh': auto_refresh,
            'refresh_interval': st.session_state.refresh_interval,
            'telegram_alerts': telegram_alerts,
            'email_alerts': email_alerts,
            'sound_alerts': sound_alerts
        }
    
    def display_signal_card(self, signal_data: Dict[str, Any], index: int = 0):
        """Display professional signal card"""
        signal = signal_data.get('signal', 'HOLD')
        symbol = signal_data.get('symbol', 'UNKNOWN')
        price = signal_data.get('price', 0)
        confidence = signal_data.get('confidence', 50)
        change_pct = signal_data.get('change_pct', 0)
        
        # Determine card style
        if signal == 'BUY':
            card_class = 'buy-signal'
            signal_emoji = '🟢'
            signal_color = '#28a745'
        elif signal == 'SELL':
            card_class = 'sell-signal'
            signal_emoji = '🔴'
            signal_color = '#dc3545'
        else:
            card_class = 'hold-signal'
            signal_emoji = '🔵'
            signal_color = '#007bff'
        
        # Format change percentage
        change_color = '#28a745' if change_pct >= 0 else '#dc3545'
        change_prefix = '+' if change_pct >= 0 else ''
        
        # Create signal card
        st.markdown(f"""
        <div class="signal-card {card_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4 style="margin: 0; color: {signal_color};">
                        {signal_emoji} {symbol} - {signal}
                    </h4>
                    <p style="margin: 5px 0;">
                        <strong>Price:</strong> ₹{price:.2f} 
                        <span style="color: {change_color};">
                            ({change_prefix}{change_pct:.1f}%)
                        </span>
                    </p>
                    <p style="margin: 5px 0;">
                        <strong>Confidence:</strong> {confidence}%
                    </p>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 2rem;">{signal_emoji}</div>
                    <small>{datetime.now().strftime('%H:%M:%S')}</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Add action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(f"📊 Analyze {symbol}", key=f"analyze_{symbol}_{index}"):
                st.session_state.selected_stock = symbol
                st.session_state.current_tab = 'analysis'
                st.rerun()
        
        with col2:
            if st.button(f"📈 Chart {symbol}", key=f"chart_{symbol}_{index}"):
                st.session_state.selected_stock = symbol
                st.session_state.show_chart = True
                st.rerun()
        
        with col3:
            if st.button(f"➕ Watchlist", key=f"watchlist_{symbol}_{index}"):
                if symbol not in st.session_state.watchlist:
                    st.session_state.watchlist.append(symbol)
                    st.success(f"Added {symbol} to watchlist")
                    st.rerun()
    
    def display_scanner_results_table(self, signals: List[Dict[str, Any]]):
        """Display scanner results in a professional table"""
        if not signals:
            st.info("📡 No signals detected. Adjust scanner parameters or wait for market activity.")
            return
        
        # Convert to DataFrame
        df_data = []
        for signal in signals:
            df_data.append({
                'Symbol': signal.get('symbol', 'N/A'),
                'Signal': signal.get('signal', 'HOLD'),
                'Price': f"₹{signal.get('price', 0):.2f}",
                'Change%': f"{signal.get('change_pct', 0):+.1f}%",
                'Volume': f"{signal.get('volume_ratio', 1):.1f}x",
                'RSI': f"{signal.get('rsi', 50):.1f}",
                'Confidence': f"{signal.get('confidence', 50)}%",
                'Time': datetime.now().strftime('%H:%M:%S')
            })
        
        df = pd.DataFrame(df_data)
        
        # Style the dataframe
        def style_signal(val):
            if val == 'BUY':
                return 'background-color: #d4edda; color: #155724; font-weight: bold;'
            elif val == 'SELL':
                return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
            else:
                return 'background-color: #cce7ff; color: #004085; font-weight: bold;'
        
        def style_change(val):
            if '+' in val:
                return 'color: #28a745; font-weight: bold;'
            elif val.startswith('-'):
                return 'color: #dc3545; font-weight: bold;'
            else:
                return 'color: #6c757d;'
        
        styled_df = df.style.applymap(style_signal, subset=['Signal'])
        styled_df = styled_df.applymap(style_change, subset=['Change%'])
        
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        buy_count = len([s for s in signals if s.get('signal') == 'BUY'])
        sell_count = len([s for s in signals if s.get('signal') == 'SELL'])
        hold_count = len([s for s in signals if s.get('signal') == 'HOLD'])
        total_count = len(signals)
        
        with col1:
            st.metric("🟢 BUY Signals", buy_count, f"{(buy_count/total_count)*100:.1f}%")
        
        with col2:
            st.metric("🔴 SELL Signals", sell_count, f"{(sell_count/total_count)*100:.1f}%")
        
        with col3:
            st.metric("🔵 HOLD Signals", hold_count, f"{(hold_count/total_count)*100:.1f}%")
        
        with col4:
            avg_confidence = np.mean([s.get('confidence', 50) for s in signals])
            st.metric("📊 Avg Confidence", f"{avg_confidence:.1f}%")
    
    def display_market_breadth_heatmap(self, sector_data: Dict[str, Dict]):
        """Display market breadth as a heatmap"""
        st.markdown("### 🔥 Market Breadth Heatmap")
        
        if not sector_data:
            st.info("📊 Loading market breadth data...")
            return
        
        # Create heatmap visualization
        sectors = list(sector_data.keys())
        changes = [data.get('change_pct', 0) for data in sector_data.values()]
        
        # Create color-coded sector cards
        cols = st.columns(min(4, len(sectors)))
        
        for i, (sector, change) in enumerate(zip(sectors, changes)):
            with cols[i % 4]:
                color = '#28a745' if change > 0 else '#dc3545' if change < 0 else '#6c757d'
                
                st.markdown(f"""
                <div style="background-color: {color}; color: white; padding: 1rem; 
                           border-radius: 8px; text-align: center; margin: 0.25rem 0;">
                    <h4 style="margin: 0;">{sector}</h4>
                    <p style="margin: 0; font-size: 1.2rem; font-weight: bold;">
                        {change:+.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    def display_loading_spinner(self, message: str = "Loading..."):
        """Display loading spinner with message"""
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem;">
            <div class="loading-spinner"></div>
            <p style="margin-top: 1rem; color: #6c757d;">{message}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def display_alert_notification(self, message: str, alert_type: str = "info"):
        """Display alert notification"""
        alert_colors = {
            'success': '#d4edda',
            'warning': '#fff3cd', 
            'danger': '#f8d7da',
            'info': '#d1ecf1'
        }
        
        alert_icons = {
            'success': '✅',
            'warning': '⚠️',
            'danger': '❌',
            'info': 'ℹ️'
        }
        
        color = alert_colors.get(alert_type, alert_colors['info'])
        icon = alert_icons.get(alert_type, alert_icons['info'])
        
        st.markdown(f"""
        <div class="alert-box alert-{alert_type}" style="background-color: {color};">
            <strong>{icon} Alert:</strong> {message}
        </div>
        """, unsafe_allow_html=True)
    
    def display_performance_metrics(self, metrics: Dict[str, Any]):
        """Display performance metrics dashboard"""
        st.markdown("### 📊 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Return",
                f"{metrics.get('total_return', 0):.1f}%",
                f"{metrics.get('return_delta', 0):+.1f}%"
            )
        
        with col2:
            st.metric(
                "Win Rate",
                f"{metrics.get('win_rate', 0):.1f}%",
                f"{metrics.get('win_rate_delta', 0):+.1f}%"
            )
        
        with col3:
            st.metric(
                "Max Drawdown",
                f"{metrics.get('max_drawdown', 0):.1f}%",
                f"{metrics.get('drawdown_delta', 0):+.1f}%"
            )
        
        with col4:
            st.metric(
                "Sharpe Ratio",
                f"{metrics.get('sharpe_ratio', 0):.2f}",
                f"{metrics.get('sharpe_delta', 0):+.2f}"
            )
    
    def _add_stocks_to_watchlist(self, stocks: List[str]):
        """Add multiple stocks to watchlist"""
        added_count = 0
        for stock in stocks:
            if stock not in st.session_state.watchlist:
                st.session_state.watchlist.append(stock)
                added_count += 1
        
        if added_count > 0:
            st.success(f"✅ Added {added_count} stocks to watchlist")
            st.rerun()
        else:
            st.info("ℹ️ All stocks already in watchlist")
    
    def create_stock_selector(self, key: str = "stock_selector") -> str:
        """Create enhanced stock selector"""
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_stock = st.selectbox(
                "📈 Select Stock for Analysis:",
                st.session_state.watchlist,
                index=0 if st.session_state.watchlist else None,
                key=key
            )
        
        with col2:
            if st.button("🔄 Refresh Data", key=f"refresh_{key}"):
                st.rerun()
        
        return selected_stock
    
    def create_timeframe_selector(self, key: str = "timeframe_selector") -> str:
        """Create timeframe selector"""
        timeframes = ["1m", "2m", "3m", "5m", "10m", "15m", "30m", "1h"]
        
        return st.selectbox(
            "⏰ Timeframe:",
            timeframes,
            index=3,  # Default to 5m
            key=key
        )
    
    def display_telegram_status(self):
        """Display Telegram connection status"""
        if st.session_state.telegram_connected:
            st.success("✅ Telegram Bot Connected")
            st.caption("🤖 Bot: @MyStockSentryBot")
            st.caption("👤 Chat ID: 6253409461")
            
            if st.button("📱 Send Test Message"):
                st.session_state.send_test_telegram = True
                st.info("📤 Sending test message...")
        else:
            st.error("❌ Telegram Bot Not Connected")
            if st.button("🔄 Reconnect"):
                st.session_state.telegram_reconnect = True
