"""
Enhanced configuration settings for the Professional Intraday Trading Bot
"""

# NSE Complete Stock List (Top 200+ stocks for better coverage)
NSE_ALL_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN", "ICICIBANK", "BHARTIARTL", "ITC",
    "KOTAKBANK", "LT", "HCLTECH", "ASIANPAINT", "MARUTI", "BAJFINANCE", "TITAN",
    "NESTLEIND", "ULTRACEMCO", "WIPRO", "ADANIGREEN", "TATAMOTORS", "ONGC", "NTPC",
    "POWERGRID", "COALINDIA", "HINDALCO", "TECHM", "SUNPHARMA", "JSWSTEEL",
    "INDUSINDBK", "DRREDDY", "CIPLA", "BPCL", "GRASIM", "EICHERMOT", "TATACONSUM",
    "DIVISLAB", "BRITANNIA", "APOLLOHOSP", "BAJAJFINSV", "BAJAJ-AUTO", "AXISBANK",
    "HEROMOTOCO", "HDFCLIFE", "SBILIFE", "SHREECEM", "PIDILITIND", "DABUR",
    "GODREJCP", "HINDUNILVR", "MARICO", "COLPAL", "BERGEPAINT", "KANSAINER",
    "ASTRAL", "RELAXO", "BATAINDIA", "VIPIND", "ABCAPITAL", "ABFRL", "ACC",
    "ADANIENT", "ADANIPORTS", "ADANIPOWER", "ADANITRANS", "AFFLE", "AIAENG",
    "AJANTPHARM", "ALKEM", "AMBUJACEM", "APOLLOTYRE", "AUROPHARMA", "AVANTI",
    "BAJAJHLDNG", "BALKRISIND", "BANDHANBNK", "BANKBARODA", "BANKINDIA", "BEML",
    "BEL", "BHARATFORG", "BHARTIARTL", "BHEL", "BIOCON", "BOSCHLTD", "BSOFT",
    "CADILAHC", "CANBK", "CANFINHOME", "CENTURYTEX", "CESC", "CHAMBLFERT",
    "CHOLAFIN", "CIPLA", "CONCOR", "CROMPTON", "CUB", "CUMMINSIND", "DEEPAKNTR",
    "DELTACORP", "DHANI", "DISHTV", "DLF", "DMART", "DRREDDY", "EDELWEISS",
    "FEDERALBNK", "GAIL", "GMRINFRA", "GNFC", "GRASIM", "HAVELLS", "HCLTECH",
    "HDFC", "HDFCAMC", "HDFCBANK", "HDFCLIFE", "HINDALCO", "HINDCOPPER",
    "HINDPETRO", "HINDUNILVR", "IBULHSGFIN", "ICICIBANK", "ICICIGI", "ICICIPRULI",
    "IDEA", "IDFCFIRSTB", "IEX", "IFCI", "IGL", "INDHOTEL", "INDIACEM",
    "INDIAMART", "INDIANB", "INDIGO", "INDUSINDBK", "INDUSTOWER", "INFIBEAM",
    "INFY", "IOC", "IRCTC", "ITC", "JINDALSTEL", "JSWENERGY", "JSWSTEEL",
    "JUBLFOOD", "JUSTDIAL", "KADILA", "KAJARIACER", "KOTAKBANK", "KPITTECH",
    "L&TFH", "LALPATHLAB", "LICHSGFIN", "LT", "LTTS", "LUPIN", "M&M",
    "M&MFIN", "MANAPPURAM", "MARICO", "MARUTI", "MCDOWELL-N", "MCX", "MINDTREE",
    "MOTHERSUMI", "MPHASIS", "MRF", "MUTHOOTFIN", "NATIONALUM", "NAUKRI",
    "NAVINFLUOR", "NESTLEIND", "NMDC", "NTPC", "OBEROIRLTY", "OFSS", "ONGC",
    "PAGEIND", "PEL", "PERSISTENT", "PETRONET", "PFC", "PIDILITIND", "PIIND",
    "PNB", "POLYCAB", "POWERGRID", "PVR", "QUESS", "RAMCOCEM", "RBLBANK",
    "RECLTD", "RELIANCE", "SAIL", "SBICARD", "SBILIFE", "SBIN", "SHREECEM",
    "SIEMENS", "SRF", "STAR", "SUNPHARMA", "SUNTV", "TATACOMM", "TATACONSUM",
    "TATAMOTORS", "TATAPOWER", "TATASTEEL", "TCS", "TECHM", "TITAN", "TORNTPHARM",
    "TORNTPOWER", "TRENT", "TRIDENT", "TVSMOTOR", "UBL", "ULTRACEMCO", "UPL",
    "VEDL", "VOLTAS", "WIPRO", "YESBANK", "ZEEL", "ZYDUSLIFE"
]

# BSE Stocks List
BSE_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN", "ICICIBANK", "BHARTIARTL",
    "ITC", "KOTAKBANK", "LT", "HCLTECH", "ASIANPAINT", "MARUTI", "BAJFINANCE",
    "TITAN", "NESTLEIND", "ULTRACEMCO", "WIPRO", "ADANIGREEN", "TATAMOTORS"
]

# Nifty 50 Stocks
NIFTY_50_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN", "ICICIBANK", "BHARTIARTL",
    "ITC", "KOTAKBANK", "LT", "HCLTECH", "ASIANPAINT", "MARUTI", "BAJFINANCE",
    "TITAN", "NESTLEIND", "ULTRACEMCO", "WIPRO", "ADANIGREEN", "TATAMOTORS",
    "ONGC", "NTPC", "POWERGRID", "COALINDIA", "HINDALCO", "TECHM", "SUNPHARMA",
    "JSWSTEEL", "INDUSINDBK", "DRREDDY", "CIPLA", "BPCL", "GRASIM", "EICHERMOT",
    "TATACONSUM", "DIVISLAB", "BRITANNIA", "APOLLOHOSP", "BAJAJFINSV", "BAJAJ-AUTO",
    "AXISBANK", "HEROMOTOCO", "HDFCLIFE", "SBILIFE", "SHREECEM", "PIDILITIND",
    "DABUR", "GODREJCP", "HINDUNILVR", "MARICO"
]

# Market Sectors
SECTORS = [
    "Banking", "IT", "Pharma", "Auto", "FMCG", "Energy", "Metals", "Real Estate",
    "Telecom", "Infrastructure", "Finance", "Media", "Textiles", "Chemicals",
    "Cement", "Airlines", "Oil & Gas", "Power", "Capital Goods", "Consumer Durables"
]

# Default Watchlist
DEFAULT_WATCHLIST = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN", "ICICIBANK", "BHARTIARTL",
    "ITC", "KOTAKBANK", "LT", "HCLTECH", "ASIANPAINT", "MARUTI", "BAJFINANCE",
    "TITAN", "NESTLEIND", "ULTRACEMCO", "WIPRO", "ADANIGREEN", "TATAMOTORS"
]

# Trading Configuration
TRADING_CONFIG = {
    'timeframes': ['1m', '2m', '3m', '5m', '10m', '15m', '30m', '1h'],
    'intraday_timeframes': ['1m', '2m', '3m', '5m', '10m'],
    'periods': ['1d', '2d', '5d', '1mo'],
    'intraday_periods': ['1d', '2d'],
    'auto_refresh_intervals': [5, 10, 15, 30, 60],
    'default_timeframe': '5m',
    'default_period': '1d',
    'default_auto_refresh': 15,
    'volume_surge_threshold': 2.0,
    'gap_threshold': 2.0,
    'adx_threshold': 25,
    'rsi_overbought': 70,
    'rsi_oversold': 30
}

# Chart Configuration
CHART_CONFIG = {
    'height': 600,
    'mobile_height': 400,
    'colors': {
        'buy': '#00ff00',
        'sell': '#ff0000',
        'hold': '#0066ff',
        'candlestick_up': '#00ff88',
        'candlestick_down': '#ff4444',
        'volume_up': '#26a69a',
        'volume_down': '#ef5350',
        'support': '#2196f3',
        'resistance': '#f44336',
        'fibonacci': '#9c27b0'
    },
    'indicators': {
        'ema_20': {'color': '#ff9800', 'width': 2},
        'ema_50': {'color': '#e91e63', 'width': 2},
        'vwap': {'color': '#00bcd4', 'width': 2},
        'bollinger_upper': {'color': '#9e9e9e', 'width': 1},
        'bollinger_lower': {'color': '#9e9e9e', 'width': 1}
    }
}

# AI Model Configuration
AI_CONFIG = {
    'models': {
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5
        },
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1
        },
        'neural_network': {
            'hidden_layers': [64, 32, 16],
            'epochs': 100,
            'batch_size': 32
        }
    },
    'features': [
        'rsi', 'macd', 'adx', 'ema_12', 'ema_26', 'vwap', 'bb_width',
        'volume_ratio', 'price_change', 'volatility', 'momentum'
    ],
    'lookback_period': 50,
    'prediction_horizon': 5
}

# Telegram Bot Configuration
TELEGRAM_CONFIG = {
    'bot_token': '7248457164:AAF-IAycn_9fGcJtm4IifjA68QaDPnvwivg',
    'chat_id': '6253409461',
    'bot_username': '@MyStockSentryBot',
    'commands': {
        'buylist': 'Get current BUY signals',
        'selllist': 'Get current SELL signals',
        'status': 'Get bot status',
        'watchlist': 'Get current watchlist',
        'alerts': 'Toggle alerts on/off'
    }
}

# Alert Configuration
ALERT_CONFIG = {
    'price_change_threshold': 2.0,  # Percentage
    'volume_surge_threshold': 2.0,  # Multiple of average
    'rsi_extreme_threshold': {'oversold': 30, 'overbought': 70},
    'gap_threshold': 2.0,  # Percentage
    'breakout_threshold': 1.5,  # Percentage above resistance
    'breakdown_threshold': 1.5,  # Percentage below support
    'alert_cooldown': 300,  # Seconds between similar alerts
    'max_alerts_per_hour': 20
}

# Fibonacci Levels
FIBONACCI_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

# Support/Resistance Configuration
SUPPORT_RESISTANCE_CONFIG = {
    'lookback_period': 50,
    'min_touches': 2,
    'tolerance': 0.5,  # Percentage tolerance for level identification
    'strength_threshold': 3  # Minimum strength for significant levels
}

# Session Timings (IST)
MARKET_SESSIONS = {
    'pre_market': {'start': '09:00', 'end': '09:15'},
    'market_hours': {'start': '09:15', 'end': '15:30'},
    'after_hours': {'start': '15:30', 'end': '16:00'}
}

# Feature Toggles
DEFAULT_FEATURES = {
    'live_scanner': True,
    'ai_signals': True,
    'volume_analysis': True,
    'gap_analysis': True,
    'support_resistance': True,
    'fibonacci_levels': True,
    'market_breadth': True,
    'sentiment_analysis': False,  # Requires external API
    'fii_dii_data': False,  # Requires external API
    'news_integration': False,  # Requires external API
    'advanced_backtesting': True,
    'trade_journaling': True,
    'telegram_alerts': True,
    'email_alerts': False,
    'sound_alerts': False,
    'auto_refresh': True
}

# Risk Management
RISK_CONFIG = {
    'max_position_size': 0.05,  # 5% of portfolio
    'stop_loss': 0.02,  # 2% stop loss
    'take_profit': 0.04,  # 4% take profit
    'max_daily_trades': 10,
    'max_daily_loss': 0.03,  # 3% of portfolio
    'position_sizing': 'fixed'  # 'fixed', 'volatility_based', 'kelly'
}

# Backtesting Configuration
BACKTEST_CONFIG = {
    'initial_capital': 100000,
    'commission': 0.001,  # 0.1% per trade
    'slippage': 0.0005,  # 0.05% slippage
    'lookback_periods': [30, 60, 90, 180, 365],
    'benchmark': 'NIFTY50',
    'risk_free_rate': 0.06  # 6% annual risk-free rate
}

# Performance Metrics
PERFORMANCE_METRICS = [
    'total_return', 'annualized_return', 'volatility', 'sharpe_ratio',
    'max_drawdown', 'win_rate', 'profit_factor', 'calmar_ratio',
    'sortino_ratio', 'information_ratio', 'beta', 'alpha'
]
