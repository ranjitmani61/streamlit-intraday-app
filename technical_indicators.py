"""
Technical Indicators Module
Implementation of 17 technical indicators for trading analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import talib

class TechnicalIndicators:
    def __init__(self, config):
        """Initialize technical indicators with configuration"""
        self.config = config
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate all 17 technical indicators
        
        Args:
            data (pd.DataFrame): OHLCV data
            
        Returns:
            dict: Dictionary containing all indicator values
        """
        indicators = {}
        
        try:
            # Trend Indicators
            indicators.update(self.calculate_trend_indicators(data))
            
            # Momentum Indicators
            indicators.update(self.calculate_momentum_indicators(data))
            
            # Volatility Indicators
            indicators.update(self.calculate_volatility_indicators(data))
            
            # Volume Indicators
            indicators.update(self.calculate_volume_indicators(data))
            
            # Support/Resistance Indicators
            indicators.update(self.calculate_support_resistance_indicators(data))
            
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
        
        return indicators
    
    def calculate_trend_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trend-based indicators"""
        indicators = {}
        
        try:
            # 1. Exponential Moving Average (EMA)
            ema_short = self.config.getint('INDICATORS', 'ema_short')
            ema_long = self.config.getint('INDICATORS', 'ema_long')
            indicators['EMA_short'] = talib.EMA(data['Close'], timeperiod=ema_short)
            indicators['EMA_long'] = talib.EMA(data['Close'], timeperiod=ema_long)
            indicators['EMA_signal'] = np.where(
                indicators['EMA_short'] > indicators['EMA_long'], 1, -1
            )
            
            # 2. Simple Moving Average (SMA)
            sma_period = self.config.getint('INDICATORS', 'sma_period')
            indicators['SMA'] = talib.SMA(data['Close'], timeperiod=sma_period)
            indicators['SMA_signal'] = np.where(data['Close'] > indicators['SMA'], 1, -1)
            
            # 3. MACD (Moving Average Convergence Divergence)
            macd_fast = self.config.getint('INDICATORS', 'macd_fast')
            macd_slow = self.config.getint('INDICATORS', 'macd_slow')
            macd_signal = self.config.getint('INDICATORS', 'macd_signal')
            
            macd_line, macd_signal_line, macd_histogram = talib.MACD(
                data['Close'], fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal
            )
            indicators['MACD'] = macd_line
            indicators['MACD_signal_line'] = macd_signal_line
            indicators['MACD_histogram'] = macd_histogram
            indicators['MACD_signal'] = np.where(macd_line > macd_signal_line, 1, -1)
            
            # 4. ADX (Average Directional Index)
            adx_period = self.config.getint('INDICATORS', 'adx_period')
            indicators['ADX'] = talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=adx_period)
            indicators['ADX_signal'] = np.where(indicators['ADX'] > 25, 1, 0)  # Strong trend if ADX > 25
            
            # 5. SuperTrend
            supertrend_period = self.config.getint('INDICATORS', 'supertrend_period')
            supertrend_multiplier = self.config.getfloat('INDICATORS', 'supertrend_multiplier')
            supertrend, supertrend_signal = self.calculate_supertrend(
                data, supertrend_period, supertrend_multiplier
            )
            indicators['SuperTrend'] = supertrend
            indicators['SuperTrend_signal'] = supertrend_signal
            
            # 6. Parabolic SAR
            sar_acceleration = self.config.getfloat('INDICATORS', 'sar_acceleration')
            sar_maximum = self.config.getfloat('INDICATORS', 'sar_maximum')
            indicators['SAR'] = talib.SAR(
                data['High'], data['Low'], acceleration=sar_acceleration, maximum=sar_maximum
            )
            indicators['SAR_signal'] = np.where(data['Close'] > indicators['SAR'], 1, -1)
            
            # 7. Ichimoku Cloud
            ichimoku_conversion = self.config.getint('INDICATORS', 'ichimoku_conversion')
            ichimoku_base = self.config.getint('INDICATORS', 'ichimoku_base')
            ichimoku_leading_span_b = self.config.getint('INDICATORS', 'ichimoku_leading_span_b')
            
            ichimoku = self.calculate_ichimoku(data, ichimoku_conversion, ichimoku_base, ichimoku_leading_span_b)
            indicators.update(ichimoku)
            
        except Exception as e:
            print(f"Error calculating trend indicators: {str(e)}")
        
        return indicators
    
    def calculate_momentum_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate momentum-based indicators"""
        indicators = {}
        
        try:
            # 8. RSI (Relative Strength Index)
            rsi_period = self.config.getint('INDICATORS', 'rsi_period')
            indicators['RSI'] = talib.RSI(data['Close'], timeperiod=rsi_period)
            indicators['RSI_signal'] = np.where(
                (indicators['RSI'] < 30), 1,  # Oversold - buy signal
                np.where((indicators['RSI'] > 70), -1, 0)  # Overbought - sell signal
            )
            
            # 9. Stochastic RSI
            stoch_k = self.config.getint('INDICATORS', 'stoch_k')
            stoch_d = self.config.getint('INDICATORS', 'stoch_d')
            stoch_rsi = self.calculate_stochastic_rsi(data['Close'], rsi_period, stoch_k, stoch_d)
            indicators.update(stoch_rsi)
            
            # 10. CCI (Commodity Channel Index)
            cci_period = self.config.getint('INDICATORS', 'cci_period')
            indicators['CCI'] = talib.CCI(data['High'], data['Low'], data['Close'], timeperiod=cci_period)
            indicators['CCI_signal'] = np.where(
                (indicators['CCI'] < -100), 1,  # Oversold
                np.where((indicators['CCI'] > 100), -1, 0)  # Overbought
            )
            
            # 11. MFI (Money Flow Index)
            mfi_period = self.config.getint('INDICATORS', 'mfi_period')
            indicators['MFI'] = talib.MFI(
                data['High'], data['Low'], data['Close'], data['Volume'], timeperiod=mfi_period
            )
            indicators['MFI_signal'] = np.where(
                (indicators['MFI'] < 20), 1,  # Oversold
                np.where((indicators['MFI'] > 80), -1, 0)  # Overbought
            )
            
        except Exception as e:
            print(f"Error calculating momentum indicators: {str(e)}")
        
        return indicators
    
    def calculate_volatility_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volatility-based indicators"""
        indicators = {}
        
        try:
            # 12. Bollinger Bands
            bb_period = self.config.getint('INDICATORS', 'bb_period')
            bb_std = self.config.getint('INDICATORS', 'bb_std')
            
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                data['Close'], timeperiod=bb_period, nbdevup=bb_std, nbdevdn=bb_std
            )
            indicators['BB_upper'] = bb_upper
            indicators['BB_middle'] = bb_middle
            indicators['BB_lower'] = bb_lower
            indicators['BB_signal'] = np.where(
                data['Close'] <= bb_lower, 1,  # Price at lower band - buy
                np.where(data['Close'] >= bb_upper, -1, 0)  # Price at upper band - sell
            )
            
            # 13. ATR (Average True Range)
            atr_period = self.config.getint('INDICATORS', 'atr_period')
            indicators['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=atr_period)
            
            # 14. Donchian Channel
            donchian_period = self.config.getint('INDICATORS', 'donchian_period')
            donchian = self.calculate_donchian_channel(data, donchian_period)
            indicators.update(donchian)
            
        except Exception as e:
            print(f"Error calculating volatility indicators: {str(e)}")
        
        return indicators
    
    def calculate_volume_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume-based indicators"""
        indicators = {}
        
        try:
            # 15. VWAP (Volume Weighted Average Price)
            vwap_period = self.config.getint('INDICATORS', 'vwap_period')
            indicators['VWAP'] = self.calculate_vwap(data, vwap_period)
            indicators['VWAP_signal'] = np.where(data['Close'] > indicators['VWAP'], 1, -1)
            
            # 16. OBV (On-Balance Volume)
            indicators['OBV'] = talib.OBV(data['Close'], data['Volume'])
            # Calculate OBV signal based on trend
            obv_sma = talib.SMA(indicators['OBV'], timeperiod=10)
            indicators['OBV_signal'] = np.where(indicators['OBV'] > obv_sma, 1, -1)
            
        except Exception as e:
            print(f"Error calculating volume indicators: {str(e)}")
        
        return indicators
    
    def calculate_support_resistance_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate support/resistance indicators"""
        indicators = {}
        
        try:
            # 17. Pivot Points
            pivot_points = self.calculate_pivot_points(data)
            indicators.update(pivot_points)
            
        except Exception as e:
            print(f"Error calculating support/resistance indicators: {str(e)}")
        
        return indicators
    
    def calculate_supertrend(self, data: pd.DataFrame, period: int, multiplier: float):
        """Calculate SuperTrend indicator"""
        try:
            # Calculate ATR
            atr = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=period)
            
            # Calculate basic bands
            high_low_avg = (data['High'] + data['Low']) / 2
            basic_upper_band = high_low_avg + (multiplier * atr)
            basic_lower_band = high_low_avg - (multiplier * atr)
            
            # Initialize arrays
            final_upper_band = np.zeros(len(data))
            final_lower_band = np.zeros(len(data))
            supertrend = np.zeros(len(data))
            signal = np.zeros(len(data))
            
            for i in range(1, len(data)):
                # Final Upper Band
                if basic_upper_band[i] < final_upper_band[i-1] or data['Close'].iloc[i-1] > final_upper_band[i-1]:
                    final_upper_band[i] = basic_upper_band[i]
                else:
                    final_upper_band[i] = final_upper_band[i-1]
                
                # Final Lower Band
                if basic_lower_band[i] > final_lower_band[i-1] or data['Close'].iloc[i-1] < final_lower_band[i-1]:
                    final_lower_band[i] = basic_lower_band[i]
                else:
                    final_lower_band[i] = final_lower_band[i-1]
                
                # SuperTrend
                if supertrend[i-1] == final_upper_band[i-1] and data['Close'].iloc[i] <= final_upper_band[i]:
                    supertrend[i] = final_upper_band[i]
                elif supertrend[i-1] == final_upper_band[i-1] and data['Close'].iloc[i] > final_upper_band[i]:
                    supertrend[i] = final_lower_band[i]
                elif supertrend[i-1] == final_lower_band[i-1] and data['Close'].iloc[i] >= final_lower_band[i]:
                    supertrend[i] = final_lower_band[i]
                elif supertrend[i-1] == final_lower_band[i-1] and data['Close'].iloc[i] < final_lower_band[i]:
                    supertrend[i] = final_upper_band[i]
                else:
                    supertrend[i] = supertrend[i-1]
                
                # Signal
                signal[i] = 1 if data['Close'].iloc[i] > supertrend[i] else -1
            
            return pd.Series(supertrend, index=data.index), pd.Series(signal, index=data.index)
            
        except Exception as e:
            print(f"Error calculating SuperTrend: {str(e)}")
            return pd.Series(np.zeros(len(data)), index=data.index), pd.Series(np.zeros(len(data)), index=data.index)
    
    def calculate_ichimoku(self, data: pd.DataFrame, conversion_period: int, base_period: int, leading_span_b_period: int):
        """Calculate Ichimoku Cloud indicators"""
        try:
            # Conversion Line (Tenkan-sen)
            conversion_line = (data['High'].rolling(window=conversion_period).max() + 
                             data['Low'].rolling(window=conversion_period).min()) / 2
            
            # Base Line (Kijun-sen)
            base_line = (data['High'].rolling(window=base_period).max() + 
                        data['Low'].rolling(window=base_period).min()) / 2
            
            # Leading Span A (Senkou Span A)
            leading_span_a = ((conversion_line + base_line) / 2).shift(base_period)
            
            # Leading Span B (Senkou Span B)
            leading_span_b = ((data['High'].rolling(window=leading_span_b_period).max() + 
                              data['Low'].rolling(window=leading_span_b_period).min()) / 2).shift(base_period)
            
            # Lagging Span (Chikou Span)
            lagging_span = data['Close'].shift(-base_period)
            
            # Ichimoku Signal
            ichimoku_signal = np.where(
                (data['Close'] > leading_span_a) & (data['Close'] > leading_span_b) & 
                (conversion_line > base_line), 1,
                np.where(
                    (data['Close'] < leading_span_a) & (data['Close'] < leading_span_b) & 
                    (conversion_line < base_line), -1, 0
                )
            )
            
            return {
                'Ichimoku_conversion': conversion_line,
                'Ichimoku_base': base_line,
                'Ichimoku_leading_span_a': leading_span_a,
                'Ichimoku_leading_span_b': leading_span_b,
                'Ichimoku_lagging_span': lagging_span,
                'Ichimoku_signal': ichimoku_signal
            }
            
        except Exception as e:
            print(f"Error calculating Ichimoku: {str(e)}")
            return {}
    
    def calculate_stochastic_rsi(self, close_prices: pd.Series, rsi_period: int, stoch_k: int, stoch_d: int):
        """Calculate Stochastic RSI"""
        try:
            rsi = talib.RSI(close_prices, timeperiod=rsi_period)
            
            # Calculate Stochastic of RSI
            stoch_rsi_k = ((rsi - rsi.rolling(window=stoch_k).min()) / 
                          (rsi.rolling(window=stoch_k).max() - rsi.rolling(window=stoch_k).min())) * 100
            
            stoch_rsi_d = stoch_rsi_k.rolling(window=stoch_d).mean()
            
            stoch_rsi_signal = np.where(
                (stoch_rsi_k < 20) & (stoch_rsi_d < 20), 1,  # Oversold
                np.where((stoch_rsi_k > 80) & (stoch_rsi_d > 80), -1, 0)  # Overbought
            )
            
            return {
                'Stoch_RSI_K': stoch_rsi_k,
                'Stoch_RSI_D': stoch_rsi_d,
                'Stoch_RSI_signal': stoch_rsi_signal
            }
            
        except Exception as e:
            print(f"Error calculating Stochastic RSI: {str(e)}")
            return {}
    
    def calculate_donchian_channel(self, data: pd.DataFrame, period: int):
        """Calculate Donchian Channel"""
        try:
            donchian_upper = data['High'].rolling(window=period).max()
            donchian_lower = data['Low'].rolling(window=period).min()
            donchian_middle = (donchian_upper + donchian_lower) / 2
            
            donchian_signal = np.where(
                data['Close'] >= donchian_upper, 1,  # Breakout above upper channel
                np.where(data['Close'] <= donchian_lower, -1, 0)  # Breakdown below lower channel
            )
            
            return {
                'Donchian_upper': donchian_upper,
                'Donchian_middle': donchian_middle,
                'Donchian_lower': donchian_lower,
                'Donchian_signal': donchian_signal
            }
            
        except Exception as e:
            print(f"Error calculating Donchian Channel: {str(e)}")
            return {}
    
    def calculate_vwap(self, data: pd.DataFrame, period: int):
        """Calculate Volume Weighted Average Price"""
        try:
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            vwap = (typical_price * data['Volume']).rolling(window=period).sum() / data['Volume'].rolling(window=period).sum()
            return vwap
            
        except Exception as e:
            print(f"Error calculating VWAP: {str(e)}")
            return pd.Series(np.zeros(len(data)), index=data.index)
    
    def calculate_pivot_points(self, data: pd.DataFrame):
        """Calculate Pivot Points and Support/Resistance levels"""
        try:
            # Use previous day's high, low, close for pivot calculation
            high_prev = data['High'].shift(1)
            low_prev = data['Low'].shift(1)
            close_prev = data['Close'].shift(1)
            
            # Pivot Point
            pivot = (high_prev + low_prev + close_prev) / 3
            
            # Support and Resistance levels
            r1 = 2 * pivot - low_prev
            s1 = 2 * pivot - high_prev
            r2 = pivot + (high_prev - low_prev)
            s2 = pivot - (high_prev - low_prev)
            r3 = high_prev + 2 * (pivot - low_prev)
            s3 = low_prev - 2 * (high_prev - pivot)
            
            # Generate signals based on price relative to pivot levels
            pivot_signal = np.where(
                data['Close'] > r1, 1,  # Above resistance - bullish
                np.where(data['Close'] < s1, -1, 0)  # Below support - bearish
            )
            
            return {
                'Pivot': pivot,
                'R1': r1, 'R2': r2, 'R3': r3,
                'S1': s1, 'S2': s2, 'S3': s3,
                'Pivot_signal': pivot_signal
            }
            
        except Exception as e:
            print(f"Error calculating Pivot Points: {str(e)}")
            return {}
