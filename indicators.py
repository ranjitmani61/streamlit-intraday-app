"""
Comprehensive technical indicators module for advanced stock analysis
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class TechnicalIndicators:
    def __init__(self):
        pass
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive technical indicators"""
        try:
            if data.empty or len(data) < 50:
                return self._get_default_indicators()
            
            indicators = {}
            
            # Trend Indicators
            indicators.update(self._calculate_trend_indicators(data))
            
            # Momentum Indicators
            indicators.update(self._calculate_momentum_indicators(data))
            
            # Volatility Indicators
            indicators.update(self._calculate_volatility_indicators(data))
            
            # Volume Indicators
            indicators.update(self._calculate_volume_indicators(data))
            
            # Custom Indicators
            indicators.update(self._calculate_custom_indicators(data))
            
            return indicators
            
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            return self._get_default_indicators()
    
    def _calculate_trend_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trend-based indicators"""
        indicators = {}
        close = data['Close']
        high = data['High']
        low = data['Low']
        
        try:
            # Moving Averages
            indicators['sma_5'] = ta.trend.sma_indicator(close, window=5).iloc[-1]
            indicators['sma_10'] = ta.trend.sma_indicator(close, window=10).iloc[-1]
            indicators['sma_20'] = ta.trend.sma_indicator(close, window=20).iloc[-1]
            indicators['sma_50'] = ta.trend.sma_indicator(close, window=50).iloc[-1]
            
            indicators['ema_5'] = ta.trend.ema_indicator(close, window=5).iloc[-1]
            indicators['ema_10'] = ta.trend.ema_indicator(close, window=10).iloc[-1]
            indicators['ema_12'] = ta.trend.ema_indicator(close, window=12).iloc[-1]
            indicators['ema_20'] = ta.trend.ema_indicator(close, window=20).iloc[-1]
            indicators['ema_26'] = ta.trend.ema_indicator(close, window=26).iloc[-1]
            indicators['ema_50'] = ta.trend.ema_indicator(close, window=50).iloc[-1]
            
            # MACD
            macd_line = ta.trend.macd_diff(close)
            indicators['macd'] = macd_line.iloc[-1] if not macd_line.empty else 0
            indicators['macd_signal'] = ta.trend.macd_signal(close).iloc[-1]
            indicators['macd_histogram'] = ta.trend.macd(close).iloc[-1]
            
            # ADX (Average Directional Index)
            indicators['adx'] = ta.trend.adx(high, low, close, window=14).iloc[-1]
            indicators['adx_pos'] = ta.trend.adx_pos(high, low, close, window=14).iloc[-1]
            indicators['adx_neg'] = ta.trend.adx_neg(high, low, close, window=14).iloc[-1]
            
            # Parabolic SAR
            psar_up = ta.trend.psar_up(high, low, close)
            psar_down = ta.trend.psar_down(high, low, close)
            indicators['psar'] = psar_up.iloc[-1] if pd.notna(psar_up.iloc[-1]) else psar_down.iloc[-1]
            
            # Ichimoku
            indicators['ichimoku_conversion'] = ta.trend.ichimoku_conversion_line(high, low).iloc[-1]
            indicators['ichimoku_base'] = ta.trend.ichimoku_base_line(high, low).iloc[-1]
            indicators['ichimoku_a'] = ta.trend.ichimoku_a(high, low).iloc[-1]
            indicators['ichimoku_b'] = ta.trend.ichimoku_b(high, low).iloc[-1]
            
        except Exception as e:
            print(f"Error in trend indicators: {str(e)}")
            current_price = close.iloc[-1] if not close.empty else 100
            indicators.update({
                'sma_5': current_price, 'sma_10': current_price, 'sma_20': current_price, 'sma_50': current_price,
                'ema_5': current_price, 'ema_10': current_price, 'ema_12': current_price, 'ema_20': current_price,
                'ema_26': current_price, 'ema_50': current_price, 'macd': 0, 'macd_signal': 0,
                'macd_histogram': 0, 'adx': 25, 'adx_pos': 25, 'adx_neg': 25, 'psar': current_price,
                'ichimoku_conversion': current_price, 'ichimoku_base': current_price,
                'ichimoku_a': current_price, 'ichimoku_b': current_price
            })
        
        return indicators
    
    def _calculate_momentum_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate momentum-based indicators"""
        indicators = {}
        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']
        
        try:
            # RSI (Multiple timeframes)
            indicators['rsi_14'] = ta.momentum.rsi(close, window=14).iloc[-1]
            indicators['rsi_21'] = ta.momentum.rsi(close, window=21).iloc[-1]
            indicators['rsi_50'] = ta.momentum.rsi(close, window=50).iloc[-1]
            
            # Stochastic Oscillator
            indicators['stoch_k'] = ta.momentum.stoch(high, low, close).iloc[-1]
            indicators['stoch_d'] = ta.momentum.stoch_signal(high, low, close).iloc[-1]
            
            # Williams %R
            indicators['williams_r'] = ta.momentum.williams_r(high, low, close).iloc[-1]
            
            # ROC (Rate of Change)
            indicators['roc_12'] = ta.momentum.roc(close, window=12).iloc[-1]
            indicators['roc_25'] = ta.momentum.roc(close, window=25).iloc[-1]
            
            # Money Flow Index
            indicators['mfi'] = ta.volume.money_flow_index(high, low, close, volume).iloc[-1]
            
            # Commodity Channel Index
            indicators['cci'] = ta.trend.cci(high, low, close).iloc[-1]
            
            # Awesome Oscillator
            indicators['ao'] = ta.momentum.awesome_oscillator(high, low).iloc[-1]
            
            # Ultimate Oscillator
            indicators['uo'] = ta.momentum.ultimate_oscillator(high, low, close).iloc[-1]
            
        except Exception as e:
            print(f"Error in momentum indicators: {str(e)}")
            indicators.update({
                'rsi_14': 50, 'rsi_21': 50, 'rsi_50': 50, 'stoch_k': 50, 'stoch_d': 50,
                'williams_r': -50, 'roc_12': 0, 'roc_25': 0, 'mfi': 50, 'cci': 0,
                'ao': 0, 'uo': 50
            })
        
        return indicators
    
    def _calculate_volatility_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volatility-based indicators"""
        indicators = {}
        close = data['Close']
        high = data['High']
        low = data['Low']
        
        try:
            # Bollinger Bands
            bb_high = ta.volatility.bollinger_hband(close, window=20, window_dev=2)
            bb_low = ta.volatility.bollinger_lband(close, window=20, window_dev=2)
            bb_mid = ta.volatility.bollinger_mavg(close, window=20)
            
            indicators['bb_upper'] = bb_high.iloc[-1]
            indicators['bb_lower'] = bb_low.iloc[-1]
            indicators['bb_middle'] = bb_mid.iloc[-1]
            indicators['bb_width'] = ((bb_high.iloc[-1] - bb_low.iloc[-1]) / bb_mid.iloc[-1]) * 100
            indicators['bb_percent'] = (close.iloc[-1] - bb_low.iloc[-1]) / (bb_high.iloc[-1] - bb_low.iloc[-1])
            
            # Keltner Channels
            indicators['kc_upper'] = ta.volatility.keltner_channel_hband(high, low, close).iloc[-1]
            indicators['kc_lower'] = ta.volatility.keltner_channel_lband(high, low, close).iloc[-1]
            indicators['kc_middle'] = ta.volatility.keltner_channel_mband(high, low, close).iloc[-1]
            
            # Average True Range
            indicators['atr_14'] = ta.volatility.average_true_range(high, low, close, window=14).iloc[-1]
            indicators['atr_21'] = ta.volatility.average_true_range(high, low, close, window=21).iloc[-1]
            
            # Donchian Channels
            indicators['dc_upper'] = ta.volatility.donchian_channel_hband(high, low, close).iloc[-1]
            indicators['dc_lower'] = ta.volatility.donchian_channel_lband(high, low, close).iloc[-1]
            indicators['dc_middle'] = ta.volatility.donchian_channel_mband(high, low, close).iloc[-1]
            
        except Exception as e:
            print(f"Error in volatility indicators: {str(e)}")
            current_price = close.iloc[-1] if not close.empty else 100
            indicators.update({
                'bb_upper': current_price * 1.02, 'bb_lower': current_price * 0.98,
                'bb_middle': current_price, 'bb_width': 4.0, 'bb_percent': 0.5,
                'kc_upper': current_price * 1.02, 'kc_lower': current_price * 0.98,
                'kc_middle': current_price, 'atr_14': current_price * 0.02,
                'atr_21': current_price * 0.02, 'dc_upper': current_price * 1.05,
                'dc_lower': current_price * 0.95, 'dc_middle': current_price
            })
        
        return indicators
    
    def _calculate_volume_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume-based indicators"""
        indicators = {}
        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']
        
        try:
            # VWAP (Volume Weighted Average Price)
            indicators['vwap'] = ta.volume.volume_weighted_average_price(high, low, close, volume).iloc[-1]
            
            # On Balance Volume
            indicators['obv'] = ta.volume.on_balance_volume(close, volume).iloc[-1]
            
            # Accumulation/Distribution Line
            indicators['ad'] = ta.volume.acc_dist_index(high, low, close, volume).iloc[-1]
            
            # Chaikin Money Flow
            indicators['cmf'] = ta.volume.chaikin_money_flow(high, low, close, volume).iloc[-1]
            
            # Force Index
            indicators['fi'] = ta.volume.force_index(close, volume).iloc[-1]
            
            # Ease of Movement
            indicators['eom'] = ta.volume.ease_of_movement(high, low, volume).iloc[-1]
            
            # Volume SMA
            indicators['volume_sma_10'] = ta.trend.sma_indicator(volume, window=10).iloc[-1]
            indicators['volume_sma_20'] = ta.trend.sma_indicator(volume, window=20).iloc[-1]
            
            # Volume Ratio
            current_volume = volume.iloc[-1]
            avg_volume = indicators['volume_sma_20']
            indicators['volume_ratio'] = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Positive/Negative Volume Index
            indicators['pvi'] = ta.volume.positive_volume_index(close, volume).iloc[-1]
            indicators['nvi'] = ta.volume.negative_volume_index(close, volume).iloc[-1]
            
        except Exception as e:
            print(f"Error in volume indicators: {str(e)}")
            current_price = close.iloc[-1] if not close.empty else 100
            current_volume = volume.iloc[-1] if not volume.empty else 1000000
            indicators.update({
                'vwap': current_price, 'obv': current_volume, 'ad': 0, 'cmf': 0,
                'fi': 0, 'eom': 0, 'volume_sma_10': current_volume,
                'volume_sma_20': current_volume, 'volume_ratio': 1.0,
                'pvi': 1000, 'nvi': 1000
            })
        
        return indicators
    
    def _calculate_custom_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate custom indicators for intraday trading"""
        indicators = {}
        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']
        
        try:
            # Intraday Momentum
            indicators['intraday_momentum'] = self._calculate_intraday_momentum(data)
            
            # Price Position
            indicators['price_position'] = self._calculate_price_position(data)
            
            # Volatility Ratio
            indicators['volatility_ratio'] = self._calculate_volatility_ratio(data)
            
            # Trend Strength
            indicators['trend_strength'] = self._calculate_trend_strength(data)
            
            # Volume Momentum
            indicators['volume_momentum'] = self._calculate_volume_momentum(data)
            
            # Support/Resistance Proximity
            indicators['sr_proximity'] = self._calculate_sr_proximity(data)
            
        except Exception as e:
            print(f"Error in custom indicators: {str(e)}")
            indicators.update({
                'intraday_momentum': 0, 'price_position': 0.5, 'volatility_ratio': 1.0,
                'trend_strength': 0, 'volume_momentum': 0, 'sr_proximity': 0.5
            })
        
        return indicators
    
    def _calculate_intraday_momentum(self, data: pd.DataFrame) -> float:
        """Calculate intraday momentum"""
        try:
            if len(data) < 10:
                return 0
            
            recent_prices = data['Close'].tail(10)
            momentum = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0] * 100
            return momentum
        except:
            return 0
    
    def _calculate_price_position(self, data: pd.DataFrame) -> float:
        """Calculate price position within recent range"""
        try:
            if len(data) < 20:
                return 0.5
            
            recent_data = data.tail(20)
            high_20 = recent_data['High'].max()
            low_20 = recent_data['Low'].min()
            current_price = data['Close'].iloc[-1]
            
            if high_20 == low_20:
                return 0.5
            
            position = (current_price - low_20) / (high_20 - low_20)
            return position
        except:
            return 0.5
    
    def _calculate_volatility_ratio(self, data: pd.DataFrame) -> float:
        """Calculate current volatility vs average"""
        try:
            if len(data) < 20:
                return 1.0
            
            # Current volatility (last 5 periods)
            recent_data = data.tail(5)
            current_vol = recent_data['Close'].std()
            
            # Average volatility (last 20 periods)
            avg_vol = data['Close'].tail(20).rolling(window=5).std().mean()
            
            if avg_vol == 0:
                return 1.0
            
            return current_vol / avg_vol
        except:
            return 1.0
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength using EMA convergence"""
        try:
            if len(data) < 50:
                return 0
            
            ema_12 = ta.trend.ema_indicator(data['Close'], window=12).iloc[-1]
            ema_26 = ta.trend.ema_indicator(data['Close'], window=26).iloc[-1]
            ema_50 = ta.trend.ema_indicator(data['Close'], window=50).iloc[-1]
            
            # Calculate trend strength based on EMA alignment
            if ema_12 > ema_26 > ema_50:
                strength = ((ema_12 - ema_50) / ema_50) * 100
            elif ema_12 < ema_26 < ema_50:
                strength = -((ema_50 - ema_12) / ema_50) * 100
            else:
                strength = 0
            
            return max(-10, min(10, strength))  # Normalize to -10 to 10
        except:
            return 0
    
    def _calculate_volume_momentum(self, data: pd.DataFrame) -> float:
        """Calculate volume momentum"""
        try:
            if len(data) < 10:
                return 0
            
            recent_volume = data['Volume'].tail(5).mean()
            avg_volume = data['Volume'].tail(20).mean()
            
            if avg_volume == 0:
                return 0
            
            volume_momentum = (recent_volume - avg_volume) / avg_volume * 100
            return max(-50, min(50, volume_momentum))  # Normalize to -50 to 50
        except:
            return 0
    
    def _calculate_sr_proximity(self, data: pd.DataFrame) -> float:
        """Calculate proximity to support/resistance levels"""
        try:
            if len(data) < 20:
                return 0.5
            
            current_price = data['Close'].iloc[-1]
            recent_highs = data['High'].tail(20).rolling(window=5).max()
            recent_lows = data['Low'].tail(20).rolling(window=5).min()
            
            resistance = recent_highs.max()
            support = recent_lows.min()
            
            if resistance == support:
                return 0.5
            
            proximity = (current_price - support) / (resistance - support)
            return proximity
        except:
            return 0.5
    
    def generate_signal(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive trading signal"""
        try:
            current_price = data['Close'].iloc[-1]
            
            # Multi-factor signal generation
            signal_factors = []
            
            # Trend Factor (30% weight)
            trend_score = self._calculate_trend_score(indicators)
            signal_factors.append(('trend', trend_score, 0.30))
            
            # Momentum Factor (25% weight)
            momentum_score = self._calculate_momentum_score(indicators)
            signal_factors.append(('momentum', momentum_score, 0.25))
            
            # Volume Factor (20% weight)
            volume_score = self._calculate_volume_score(indicators)
            signal_factors.append(('volume', volume_score, 0.20))
            
            # Volatility Factor (15% weight)
            volatility_score = self._calculate_volatility_score(indicators)
            signal_factors.append(('volatility', volatility_score, 0.15))
            
            # Custom Factor (10% weight)
            custom_score = self._calculate_custom_score(indicators)
            signal_factors.append(('custom', custom_score, 0.10))
            
            # Calculate weighted signal
            total_score = sum(score * weight for _, score, weight in signal_factors)
            
            # Determine signal
            if total_score > 0.6:
                signal = "BUY"
                strength = min(10, int(total_score * 10))
                confidence = min(95, int(total_score * 100))
            elif total_score < -0.6:
                signal = "SELL"
                strength = min(10, int(abs(total_score) * 10))
                confidence = min(95, int(abs(total_score) * 100))
            else:
                signal = "HOLD"
                strength = 5
                confidence = 50
            
            # Generate detailed analysis
            analysis = self._generate_signal_analysis(indicators, signal_factors)
            
            return {
                'signal': signal,
                'strength': strength,
                'confidence': confidence,
                'price': current_price,
                'total_score': total_score,
                'factors': {name: score for name, score, _ in signal_factors},
                'analysis': analysis,
                'rsi': indicators.get('rsi_14', 50),
                'macd': indicators.get('macd', 0),
                'adx': indicators.get('adx', 25),
                'volume_ratio': indicators.get('volume_ratio', 1.0)
            }
            
        except Exception as e:
            print(f"Error generating signal: {str(e)}")
            return self._get_default_signal(data)
    
    def _calculate_trend_score(self, indicators: Dict[str, Any]) -> float:
        """Calculate trend score (-1 to 1)"""
        try:
            score = 0
            
            # EMA alignment
            if indicators['ema_12'] > indicators['ema_26']:
                score += 0.3
            else:
                score -= 0.3
            
            # MACD
            if indicators['macd'] > 0:
                score += 0.2
            else:
                score -= 0.2
            
            # ADX trend strength
            if indicators['adx'] > 25:
                if indicators['adx_pos'] > indicators['adx_neg']:
                    score += 0.3
                else:
                    score -= 0.3
            
            # Price vs Moving Averages
            current_price = indicators.get('current_price', 100)
            if current_price > indicators['ema_20']:
                score += 0.2
            else:
                score -= 0.2
            
            return max(-1, min(1, score))
        except:
            return 0
    
    def _calculate_momentum_score(self, indicators: Dict[str, Any]) -> float:
        """Calculate momentum score (-1 to 1)"""
        try:
            score = 0
            
            # RSI
            rsi = indicators['rsi_14']
            if rsi > 70:
                score -= 0.3  # Overbought
            elif rsi > 50:
                score += 0.3  # Bullish momentum
            elif rsi < 30:
                score += 0.3  # Oversold (potential reversal)
            else:
                score -= 0.3  # Bearish momentum
            
            # Stochastic
            if indicators['stoch_k'] > 80:
                score -= 0.2
            elif indicators['stoch_k'] > 50:
                score += 0.2
            elif indicators['stoch_k'] < 20:
                score += 0.2
            else:
                score -= 0.2
            
            # ROC
            if indicators['roc_12'] > 0:
                score += 0.3
            else:
                score -= 0.3
            
            # Intraday momentum
            intraday_mom = indicators.get('intraday_momentum', 0)
            if intraday_mom > 1:
                score += 0.2
            elif intraday_mom < -1:
                score -= 0.2
            
            return max(-1, min(1, score))
        except:
            return 0
    
    def _calculate_volume_score(self, indicators: Dict[str, Any]) -> float:
        """Calculate volume score (-1 to 1)"""
        try:
            score = 0
            
            # Volume ratio
            vol_ratio = indicators.get('volume_ratio', 1.0)
            if vol_ratio > 2:
                score += 0.4  # High volume
            elif vol_ratio > 1.5:
                score += 0.2
            elif vol_ratio < 0.5:
                score -= 0.3  # Low volume
            
            # OBV trend
            obv = indicators.get('obv', 0)
            if obv > 0:
                score += 0.3
            else:
                score -= 0.3
            
            # CMF
            cmf = indicators.get('cmf', 0)
            if cmf > 0.1:
                score += 0.3
            elif cmf < -0.1:
                score -= 0.3
            
            return max(-1, min(1, score))
        except:
            return 0
    
    def _calculate_volatility_score(self, indicators: Dict[str, Any]) -> float:
        """Calculate volatility score (-1 to 1)"""
        try:
            score = 0
            
            # Bollinger Bands position
            bb_percent = indicators.get('bb_percent', 0.5)
            if bb_percent > 0.8:
                score -= 0.3  # Near upper band
            elif bb_percent < 0.2:
                score += 0.3  # Near lower band
            
            # ATR relative to price
            atr = indicators.get('atr_14', 0)
            current_price = indicators.get('current_price', 100)
            atr_percent = (atr / current_price) * 100 if current_price > 0 else 0
            
            if atr_percent > 3:
                score -= 0.2  # High volatility
            elif atr_percent < 1:
                score += 0.2  # Low volatility
            
            # Volatility ratio
            vol_ratio = indicators.get('volatility_ratio', 1.0)
            if vol_ratio > 1.5:
                score -= 0.3
            elif vol_ratio < 0.7:
                score += 0.3
            
            return max(-1, min(1, score))
        except:
            return 0
    
    def _calculate_custom_score(self, indicators: Dict[str, Any]) -> float:
        """Calculate custom score (-1 to 1)"""
        try:
            score = 0
            
            # Price position
            price_pos = indicators.get('price_position', 0.5)
            if price_pos > 0.8:
                score += 0.3
            elif price_pos < 0.2:
                score -= 0.3
            
            # Trend strength
            trend_strength = indicators.get('trend_strength', 0)
            score += trend_strength / 10  # Normalize
            
            # Volume momentum
            vol_momentum = indicators.get('volume_momentum', 0)
            score += vol_momentum / 50  # Normalize
            
            return max(-1, min(1, score))
        except:
            return 0
    
    def _generate_signal_analysis(self, indicators: Dict[str, Any], factors: List) -> List[str]:
        """Generate human-readable signal analysis"""
        analysis = []
        
        try:
            # Trend analysis
            if indicators['ema_12'] > indicators['ema_26']:
                analysis.append("✅ Bullish trend: EMA12 > EMA26")
            else:
                analysis.append("❌ Bearish trend: EMA12 < EMA26")
            
            # RSI analysis
            rsi = indicators['rsi_14']
            if rsi > 70:
                analysis.append(f"⚠️ Overbought: RSI {rsi:.1f}")
            elif rsi < 30:
                analysis.append(f"💡 Oversold: RSI {rsi:.1f}")
            else:
                analysis.append(f"📊 RSI neutral: {rsi:.1f}")
            
            # Volume analysis
            vol_ratio = indicators.get('volume_ratio', 1.0)
            if vol_ratio > 2:
                analysis.append(f"🔥 High volume: {vol_ratio:.1f}x avg")
            elif vol_ratio < 0.5:
                analysis.append(f"📉 Low volume: {vol_ratio:.1f}x avg")
            
            # MACD analysis
            macd = indicators['macd']
            if macd > 0:
                analysis.append("✅ MACD bullish")
            else:
                analysis.append("❌ MACD bearish")
            
            # ADX analysis
            adx = indicators['adx']
            if adx > 25:
                analysis.append(f"💪 Strong trend: ADX {adx:.1f}")
            else:
                analysis.append(f"😐 Weak trend: ADX {adx:.1f}")
            
        except Exception as e:
            analysis.append("⚠️ Error in analysis generation")
        
        return analysis
    
    def _get_default_indicators(self) -> Dict[str, Any]:
        """Return default indicator values when calculation fails"""
        return {
            # Trend indicators
            'sma_5': 100, 'sma_10': 100, 'sma_20': 100, 'sma_50': 100,
            'ema_5': 100, 'ema_10': 100, 'ema_12': 100, 'ema_20': 100,
            'ema_26': 100, 'ema_50': 100, 'macd': 0, 'macd_signal': 0,
            'macd_histogram': 0, 'adx': 25, 'adx_pos': 25, 'adx_neg': 25,
            'psar': 100, 'ichimoku_conversion': 100, 'ichimoku_base': 100,
            'ichimoku_a': 100, 'ichimoku_b': 100,
            
            # Momentum indicators
            'rsi_14': 50, 'rsi_21': 50, 'rsi_50': 50, 'stoch_k': 50, 'stoch_d': 50,
            'williams_r': -50, 'roc_12': 0, 'roc_25': 0, 'mfi': 50, 'cci': 0,
            'ao': 0, 'uo': 50,
            
            # Volatility indicators
            'bb_upper': 102, 'bb_lower': 98, 'bb_middle': 100, 'bb_width': 4.0,
            'bb_percent': 0.5, 'kc_upper': 102, 'kc_lower': 98, 'kc_middle': 100,
            'atr_14': 2.0, 'atr_21': 2.0, 'dc_upper': 105, 'dc_lower': 95,
            'dc_middle': 100,
            
            # Volume indicators
            'vwap': 100, 'obv': 1000000, 'ad': 0, 'cmf': 0, 'fi': 0, 'eom': 0,
            'volume_sma_10': 1000000, 'volume_sma_20': 1000000, 'volume_ratio': 1.0,
            'pvi': 1000, 'nvi': 1000,
            
            # Custom indicators
            'intraday_momentum': 0, 'price_position': 0.5, 'volatility_ratio': 1.0,
            'trend_strength': 0, 'volume_momentum': 0, 'sr_proximity': 0.5
        }
    
    def _get_default_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Return default signal when generation fails"""
        current_price = data['Close'].iloc[-1] if not data.empty else 100
        
        return {
            'signal': 'HOLD',
            'strength': 5,
            'confidence': 50,
            'price': current_price,
            'total_score': 0,
            'factors': {'trend': 0, 'momentum': 0, 'volume': 0, 'volatility': 0, 'custom': 0},
            'analysis': ['⚠️ Error generating analysis'],
            'rsi': 50,
            'macd': 0,
            'adx': 25,
            'volume_ratio': 1.0
        }
