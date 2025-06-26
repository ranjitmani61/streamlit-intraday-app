"""
Professional Intraday Trading Assistant
Real-time NSE/BSE Scanner with AI-powered signals
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import asyncio
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from config import *
from data_fetcher import DataFetcher
from indicators import TechnicalIndicators
from ui_components import UIComponents
from chart_utils import ChartUtils
from alert_manager import AlertManager
from telegram_bot import TelegramBot
from ai_signals import AISignalGenerator
from market_scanner import MarketScanner
from trading_journal import TradingJournal
from backtesting import BacktestEngine
from fibonacci import FibonacciCalculator
from support_resistance import SupportResistanceCalculator
from utils import *

class TradingApp:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.indicators = TechnicalIndicators()
        self.ui = UIComponents()
        self.chart_utils = ChartUtils()
        self.alert_manager = AlertManager()
        self.telegram_bot = TelegramBot()
        self.ai_signals = AISignalGenerator()
        self.market_scanner = MarketScanner()
        self.journal = TradingJournal()
        self.backtest_engine = BacktestEngine()
        self.fibonacci = FibonacciCalculator()
        self.support_resistance = SupportResistanceCalculator()
        
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'watchlist' not in st.session_state:
            st.session_state.watchlist = DEFAULT_WATCHLIST.copy()
        
        if 'scanning_active' not in st.session_state:
            st.session_state.scanning_active = False
        
        if 'alerts_history' not in st.session_state:
            st.session_state.alerts_history = []
        
        if 'trade_journal' not in st.session_state:
            st.session_state.trade_journal = []
        
        if 'current_timeframe' not in st.session_state:
            st.session_state.current_timeframe = '5m'
        
        if 'market_session' not in st.session_state:
            st.session_state.market_session = 'market_hours'
        
        if 'volume_surge_threshold' not in st.session_state:
            st.session_state.volume_surge_threshold = 2.0
        
        if 'gap_threshold' not in st.session_state:
            st.session_state.gap_threshold = 2.0
    
    def run(self):
        """Main application runner"""
        # Setup page config
        self.ui.setup_page_config()
        
        # Create header
        self.ui.create_header()
        
        # Create sidebar controls
        sidebar_config = self.ui.create_professional_sidebar()
        
        # Main content area
        self.render_main_content(sidebar_config)
        
        # Handle auto-refresh
        if sidebar_config.get('auto_refresh', False):
            time.sleep(sidebar_config.get('refresh_interval', 30))
            st.rerun()
    
    def render_main_content(self, config):
        """Render main trading dashboard"""
        # Create tabs for different features
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📡 Live Scanner", "📊 Market Analysis", "🎯 AI Signals", 
            "📈 Backtesting", "📝 Journal", "🔔 Alerts", "📋 Watchlist"
        ])
        
        with tab1:
            self.render_live_scanner(config)
        
        with tab2:
            self.render_market_analysis(config)
        
        with tab3:
            self.render_ai_signals(config)
        
        with tab4:
            self.render_backtesting(config)
        
        with tab5:
            self.render_trading_journal(config)
        
        with tab6:
            self.render_alerts_panel(config)
        
        with tab7:
            self.render_watchlist_manager(config)
    
    def render_live_scanner(self, config):
        """Render live market scanner"""
        st.markdown("## 📡 Real-time Market Scanner")
        
        # Scanner controls
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔍 Start Scanning", type="primary"):
                st.session_state.scanning_active = True
                self.start_live_scanning(config)
        
        with col2:
            if st.button("⏹️ Stop Scanning"):
                st.session_state.scanning_active = False
        
        with col3:
            market_filter = st.selectbox("Market:", ["NSE", "BSE", "Both"], index=0)
        
        with col4:
            sector_filter = st.selectbox("Sector:", ["All"] + SECTORS, index=0)
        
        # Scanner results
        if st.session_state.scanning_active:
            self.display_scanner_results(config, market_filter, sector_filter)
        else:
            st.info("📡 Scanner is stopped. Click 'Start Scanning' to begin real-time analysis.")
    
    def start_live_scanning(self, config):
        """Start live market scanning"""
        with st.spinner("🔍 Scanning markets..."):
            # Get all stocks based on filter
            stocks_to_scan = self.get_stocks_by_filter(config.get('market_filter', 'NSE'))
            
            # Perform scanning
            scanner_results = self.market_scanner.scan_markets(
                stocks_to_scan,
                timeframe=config.get('timeframe', '5m'),
                volume_threshold=config.get('volume_surge_threshold', 2.0),
                gap_threshold=config.get('gap_threshold', 2.0)
            )
            
            # Display results
            self.display_live_results(scanner_results)
    
    def display_scanner_results(self, config, market_filter, sector_filter):
        """Display real-time scanner results"""
        # Create columns for different signal types
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🟢 BUY Signals")
            buy_signals = self.get_buy_signals(market_filter, sector_filter)
            for signal in buy_signals[:10]:  # Show top 10
                self.display_signal_card(signal, "buy")
        
        with col2:
            st.markdown("### 🔴 SELL Signals")
            sell_signals = self.get_sell_signals(market_filter, sector_filter)
            for signal in sell_signals[:10]:  # Show top 10
                self.display_signal_card(signal, "sell")
        
        with col3:
            st.markdown("### ⚡ Volume Surges")
            volume_surges = self.get_volume_surges(market_filter, sector_filter)
            for surge in volume_surges[:10]:  # Show top 10
                self.display_surge_card(surge)
    
    def render_market_analysis(self, config):
        """Render comprehensive market analysis"""
        st.markdown("## 📊 Market Analysis Dashboard")
        
        # Stock selector
        selected_stock = st.selectbox(
            "Select Stock for Analysis:",
            st.session_state.watchlist,
            key="analysis_stock_selector"
        )
        
        if selected_stock:
            # Get market data
            data = self.data_fetcher.get_realtime_data(
                selected_stock,
                period=config.get('period', '1d'),
                interval=config.get('timeframe', '5m')
            )
            
            if data is not None and not data.empty:
                # Calculate indicators
                indicators = self.indicators.calculate_all_indicators(data)
                
                # Create comprehensive chart
                chart = self.create_analysis_chart(data, indicators, selected_stock)
                st.plotly_chart(chart, use_container_width=True)
                
                # Display key metrics
                self.display_key_metrics(data, indicators)
                
                # Support/Resistance levels
                self.display_support_resistance(data, selected_stock)
                
                # Fibonacci levels
                self.display_fibonacci_levels(data, selected_stock)
            else:
                st.error(f"❌ Could not fetch data for {selected_stock}")
    
    def create_analysis_chart(self, data, indicators, symbol):
        """Create comprehensive analysis chart"""
        return self.chart_utils.create_comprehensive_chart(
            data, indicators, symbol,
            show_volume=True,
            show_indicators=True,
            show_support_resistance=True,
            show_fibonacci=True
        )
    
    def render_ai_signals(self, config):
        """Render AI-powered signal analysis"""
        st.markdown("## 🎯 AI-Powered Trading Signals")
        
        # AI model selection
        col1, col2 = st.columns(2)
        with col1:
            model_type = st.selectbox("AI Model:", ["Random Forest", "XGBoost", "Neural Network"], index=0)
        with col2:
            confidence_threshold = st.slider("Confidence Threshold:", 0.5, 0.95, 0.75, 0.05)
        
        # Generate AI signals for watchlist
        if st.button("🤖 Generate AI Signals", type="primary"):
            with st.spinner("🧠 AI analyzing markets..."):
                ai_signals = self.ai_signals.generate_signals_batch(
                    st.session_state.watchlist,
                    model_type=model_type.lower().replace(" ", "_"),
                    confidence_threshold=confidence_threshold
                )
                
                self.display_ai_signals(ai_signals)
    
    def display_ai_signals(self, signals):
        """Display AI-generated signals"""
        if not signals:
            st.warning("⚠️ No signals generated. Try adjusting the confidence threshold.")
            return
        
        # Sort by confidence
        signals = sorted(signals, key=lambda x: x['confidence'], reverse=True)
        
        # Display top signals
        for i, signal in enumerate(signals[:20]):  # Show top 20
            self.display_ai_signal_card(signal, i)
    
    def render_backtesting(self, config):
        """Render backtesting interface"""
        st.markdown("## 📈 Strategy Backtesting")
        
        # Backtesting parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            backtest_symbol = st.selectbox("Stock:", st.session_state.watchlist, key="backtest_stock")
        with col2:
            backtest_period = st.selectbox("Period:", ["1mo", "3mo", "6mo", "1y"], index=1)
        with col3:
            strategy_type = st.selectbox("Strategy:", ["ADX+RSI", "MACD+EMA", "Custom"], index=0)
        
        # Strategy parameters
        if strategy_type == "Custom":
            self.render_custom_strategy_params()
        
        # Run backtest
        if st.button("🚀 Run Backtest", type="primary"):
            with st.spinner("📊 Running backtest..."):
                results = self.backtest_engine.run_backtest(
                    symbol=backtest_symbol,
                    period=backtest_period,
                    strategy=strategy_type,
                    parameters=self.get_strategy_parameters()
                )
                
                self.display_backtest_results(results)
    
    def render_trading_journal(self, config):
        """Render trading journal interface"""
        st.markdown("## 📝 Trading Journal")
        
        # Add new trade
        with st.expander("➕ Add New Trade", expanded=False):
            self.render_add_trade_form()
        
        # Display existing trades
        if st.session_state.trade_journal:
            self.display_trade_history()
        else:
            st.info("📝 No trades recorded yet. Add your first trade above!")
    
    def render_alerts_panel(self, config):
        """Render alerts and notifications panel"""
        st.markdown("## 🔔 Alerts & Notifications")
        
        # Alert settings
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### ⚙️ Alert Settings")
            price_alerts = st.toggle("Price Alerts", value=True)
            volume_alerts = st.toggle("Volume Alerts", value=True)
            signal_alerts = st.toggle("Signal Alerts", value=True)
            
        with col2:
            st.markdown("### 📱 Notification Channels")
            telegram_alerts = st.toggle("Telegram", value=True)
            email_alerts = st.toggle("Email", value=False)
            browser_alerts = st.toggle("Browser", value=True)
        
        # Recent alerts
        st.markdown("### 📋 Recent Alerts")
        if st.session_state.alerts_history:
            self.display_alerts_history()
        else:
            st.info("🔔 No alerts yet. Enable scanning to receive real-time alerts.")
    
    def render_watchlist_manager(self, config):
        """Render watchlist management interface"""
        st.markdown("## 📋 Watchlist Manager")
        
        # Current watchlist
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### 📈 Current Watchlist")
            for i, stock in enumerate(st.session_state.watchlist):
                col_stock, col_remove = st.columns([4, 1])
                with col_stock:
                    st.write(f"{i+1}. {stock}")
                with col_remove:
                    if st.button("❌", key=f"remove_{stock}"):
                        st.session_state.watchlist.remove(stock)
                        st.rerun()
        
        with col2:
            st.markdown("### ➕ Add Stock")
            new_stock = st.text_input("Stock Symbol:", placeholder="RELIANCE")
            if st.button("Add to Watchlist"):
                if new_stock and new_stock not in st.session_state.watchlist:
                    st.session_state.watchlist.append(new_stock.upper())
                    st.success(f"✅ Added {new_stock.upper()} to watchlist")
                    st.rerun()
        
        # Predefined watchlists
        st.markdown("### 📚 Predefined Lists")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📈 Top Gainers"):
                gainers, _ = self.data_fetcher.get_top_gainers_losers()
                self.add_stocks_to_watchlist([g['stock'].replace('.NS', '') for g in gainers[:10]])
        
        with col2:
            if st.button("📉 Top Losers"):
                _, losers = self.data_fetcher.get_top_gainers_losers()
                self.add_stocks_to_watchlist([l['stock'].replace('.NS', '') for l in losers[:10]])
        
        with col3:
            if st.button("⭐ Nifty 50"):
                self.add_stocks_to_watchlist(NIFTY_50_STOCKS)
    
    def get_stocks_by_filter(self, market_filter):
        """Get stocks based on market filter"""
        if market_filter == "NSE":
            return NSE_ALL_STOCKS
        elif market_filter == "BSE":
            return BSE_STOCKS
        else:
            return NSE_ALL_STOCKS + BSE_STOCKS
    
    def get_buy_signals(self, market_filter, sector_filter):
        """Get current buy signals"""
        # This would be implemented with real-time scanning
        return []
    
    def get_sell_signals(self, market_filter, sector_filter):
        """Get current sell signals"""
        # This would be implemented with real-time scanning
        return []
    
    def get_volume_surges(self, market_filter, sector_filter):
        """Get current volume surges"""
        # This would be implemented with real-time scanning
        return []
    
    def display_signal_card(self, signal, signal_type):
        """Display individual signal card"""
        color = "#d4edda" if signal_type == "buy" else "#f8d7da"
        st.markdown(f"""
        <div style="background-color: {color}; padding: 10px; border-radius: 5px; margin: 5px 0;">
            <strong>{signal.get('symbol', 'N/A')}</strong><br>
            Price: ₹{signal.get('price', 0):.2f}<br>
            Signal: {signal_type.upper()}<br>
            Confidence: {signal.get('confidence', 0):.1%}
        </div>
        """, unsafe_allow_html=True)
    
    def display_surge_card(self, surge):
        """Display volume surge card"""
        st.markdown(f"""
        <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 5px 0;">
            <strong>{surge.get('symbol', 'N/A')}</strong><br>
            Volume: {surge.get('volume_ratio', 0):.1f}x<br>
            Price: ₹{surge.get('price', 0):.2f}<br>
            Change: {surge.get('change_pct', 0):.1f}%
        </div>
        """, unsafe_allow_html=True)
    
    def add_stocks_to_watchlist(self, stocks):
        """Add multiple stocks to watchlist"""
        added_count = 0
        for stock in stocks:
            if stock not in st.session_state.watchlist:
                st.session_state.watchlist.append(stock)
                added_count += 1
        
        if added_count > 0:
            st.success(f"✅ Added {added_count} stocks to watchlist")
            st.rerun()

# Main application entry point
def main():
    """Main application entry point"""
    app = TradingApp()
    app.run()

if __name__ == "__main__":
    main()
