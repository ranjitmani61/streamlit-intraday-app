"""
Real-time market scanner for detecting trading opportunities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class MarketScanner:
    def __init__(self):
        self.scanning_active = False
        self.scan_results = []
        self.scan_history = []
        self.last_scan_time = None
        self.scan_count = 0
        
        # Scanner configuration
        self.config = {
            'volume_surge_threshold': 2.0,
            'gap_threshold': 2.0,
            'price_change_threshold': 1.5,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'adx_threshold': 25,
            'min_confidence': 60
        }
    
    def scan_markets(self, symbols: List[str], timeframe: str = '5m', 
                    volume_threshold: float = 2.0, gap_threshold: float = 2.0) -> List[Dict[str, Any]]:
        """Scan multiple stocks for trading opportunities"""
        try:
            self.scanning_active = True
            self.scan_count += 1
            scan_start_time = datetime.now()
            
            print(f"Starting market scan #{self.scan_count} for {len(symbols)} symbols...")
            
            # Update configuration
            self.config.update({
                'volume_surge_threshold': volume_threshold,
                'gap_threshold': gap_threshold
            })
            
            # Perform concurrent scanning
            scan_results = self._scan_symbols_concurrent(symbols, timeframe)
            
            # Filter and sort results
            filtered_results = self._filter_scan_results(scan_results)
            
            # Update scan history
            scan_summary = {
                'scan_id': self.scan_count,
                'timestamp': scan_start_time,
                'symbols_scanned': len(symbols),
                'opportunities_found': len(filtered_results),
                'scan_duration': (datetime.now() - scan_start_time).total_seconds(),
                'timeframe': timeframe
            }
            
            self.scan_history.append(scan_summary)
            self.last_scan_time = scan_start_time
            self.scan_results = filtered_results
            
            print(f"Scan completed: {len(filtered_results)} opportunities found in {scan_summary['scan_duration']:.2f}s")
            
            return filtered_results
            
        except Exception as e:
            print(f"Error in market scan: {str(e)}")
            return []
        finally:
            self.scanning_active = False
    
    def _scan_symbols_concurrent(self, symbols: List[str], timeframe: str) -> List[Dict[str, Any]]:
        """Scan symbols concurrently for better performance"""
        try:
            results = []
            
            # Use ThreadPoolExecutor for concurrent processing
            with ThreadPoolExecutor(max_workers=10) as executor:
                # Submit scanning tasks
                future_to_symbol = {
                    executor.submit(self._scan_single_symbol, symbol, timeframe): symbol 
                    for symbol in symbols
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_symbol, timeout=60):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        print(f"Error scanning {symbol}: {str(e)}")
                        continue
            
            return results
            
        except Exception as e:
            print(f"Error in concurrent scanning: {str(e)}")
            return []
    
    def _scan_single_symbol(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Scan a single symbol for opportunities"""
        try:
            from data_fetcher import DataFetcher
            from indicators import TechnicalIndicators
            
            # Get market data
            data_fetcher = DataFetcher()
            indicators = TechnicalIndicators()
            
            data = data_fetcher.get_realtime_data(symbol, period='1d', interval=timeframe)
            if data is None or len(data) < 20:
                return None
            
            # Calculate technical indicators
            tech_indicators = indicators.calculate_all_indicators(data)
            
            # Detect various opportunities
            opportunities = []
            
            # Volume surge detection
            volume_surge = self._detect_volume_surge(data, symbol)
            if volume_surge:
                opportunities.append(volume_surge)
            
            # Gap detection
            gap_opportunity = self._detect_gap_opportunity(data, symbol)
            if gap_opportunity:
                opportunities.append(gap_opportunity)
            
            # Breakout detection
            breakout = self._detect_breakout(data, tech_indicators, symbol)
            if breakout:
                opportunities.append(breakout)
            
            # Trend reversal detection
            reversal = self._detect_trend_reversal(data, tech_indicators, symbol)
            if reversal:
                opportunities.append(reversal)
            
            # Momentum signals
            momentum = self._detect_momentum_signals(data, tech_indicators, symbol)
            if momentum:
                opportunities.append(momentum)
            
            # Return best opportunity
            if opportunities:
                # Sort by confidence and return the best one
                best_opportunity = max(opportunities, key=lambda x: x.get('confidence', 0))
                return best_opportunity
            
            return None
            
        except Exception as e:
            print(f"Error scanning {symbol}: {str(e)}")
            return None
    
    def _detect_volume_surge(self, data: pd.DataFrame, symbol: str) -> Optional[Dict[str, Any]]:
        """Detect volume surge opportunities"""
        try:
            if len(data) < 20:
                return None
            
            # Calculate volume metrics
            current_volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            if volume_ratio < self.config['volume_surge_threshold']:
                return None
            
            # Get price information
            current_price = data['Close'].iloc[-1]
            previous_close = data['Close'].iloc[-2]
            price_change = ((current_price - previous_close) / previous_close) * 100
            
            # Calculate confidence based on volume surge strength
            confidence = min(95, 50 + (volume_ratio - 1) * 20)
            
            return {
                'symbol': symbol,
                'opportunity_type': 'volume_surge',
                'signal': 'BUY' if price_change > 0 else 'WATCH',
                'price': current_price,
                'change_pct': price_change,
                'volume_ratio': volume_ratio,
                'current_volume': current_volume,
                'average_volume': avg_volume,
                'confidence': confidence,
                'timestamp': datetime.now(),
                'description': f"Volume surge {volume_ratio:.1f}x average",
                'action_required': 'Monitor for breakout' if price_change > 0 else 'Watch for direction'
            }
            
        except Exception as e:
            print(f"Error detecting volume surge for {symbol}: {str(e)}")
            return None
    
    def _detect_gap_opportunity(self, data: pd.DataFrame, symbol: str) -> Optional[Dict[str, Any]]:
        """Detect gap up/down opportunities"""
        try:
            if len(data) < 2:
                return None
            
            # Get gap information
            yesterday_close = data['Close'].iloc[-2]
            today_open = data['Open'].iloc[-1]
            current_price = data['Close'].iloc[-1]
            
            gap_pct = ((today_open - yesterday_close) / yesterday_close) * 100
            
            if abs(gap_pct) < self.config['gap_threshold']:
                return None
            
            gap_type = 'gap_up' if gap_pct > 0 else 'gap_down'
            signal = 'BUY' if gap_pct > 0 else 'SELL'
            
            # Check if gap is being filled
            gap_fill = False
            if gap_pct > 0 and current_price < today_open:
                gap_fill = True
            elif gap_pct < 0 and current_price > today_open:
                gap_fill = True
            
            confidence = min(90, 60 + abs(gap_pct) * 5)
            
            return {
                'symbol': symbol,
                'opportunity_type': gap_type,
                'signal': signal,
                'price': current_price,
                'gap_percentage': gap_pct,
                'yesterday_close': yesterday_close,
                'today_open': today_open,
                'gap_filling': gap_fill,
                'confidence': confidence,
                'timestamp': datetime.now(),
                'description': f"Gap {gap_type.replace('_', ' ')} {abs(gap_pct):.1f}%",
                'action_required': 'Monitor gap fill' if gap_fill else 'Ride the momentum'
            }
            
        except Exception as e:
            print(f"Error detecting gap for {symbol}: {str(e)}")
            return None
    
    def _detect_breakout(self, data: pd.DataFrame, indicators: Dict[str, Any], symbol: str) -> Optional[Dict[str, Any]]:
        """Detect breakout opportunities"""
        try:
            if len(data) < 20:
                return None
            
            current_price = data['Close'].iloc[-1]
            
            # Calculate support and resistance levels
            recent_data = data.tail(20)
            resistance = recent_data['High'].max()
            support = recent_data['Low'].min()
            
            # Check for breakout
            breakout_type = None
            if current_price > resistance * 1.01:  # 1% above resistance
                breakout_type = 'resistance_breakout'
                signal = 'BUY'
            elif current_price < support * 0.99:  # 1% below support
                breakout_type = 'support_breakdown'
                signal = 'SELL'
            else:
                return None
            
            # Confirm with volume
            volume_ratio = indicators.get('volume_ratio', 1.0)
            if volume_ratio < 1.5:  # Need decent volume for breakout
                return None
            
            # Calculate confidence
            price_distance = abs(current_price - (resistance if breakout_type == 'resistance_breakout' else support))
            confidence = min(85, 70 + volume_ratio * 10)
            
            return {
                'symbol': symbol,
                'opportunity_type': breakout_type,
                'signal': signal,
                'price': current_price,
                'resistance_level': resistance,
                'support_level': support,
                'volume_ratio': volume_ratio,
                'confidence': confidence,
                'timestamp': datetime.now(),
                'description': f"{breakout_type.replace('_', ' ').title()}",
                'action_required': 'Enter position' if signal == 'BUY' else 'Exit/Short position'
            }
            
        except Exception as e:
            print(f"Error detecting breakout for {symbol}: {str(e)}")
            return None
    
    def _detect_trend_reversal(self, data: pd.DataFrame, indicators: Dict[str, Any], symbol: str) -> Optional[Dict[str, Any]]:
        """Detect trend reversal opportunities"""
        try:
            current_price = data['Close'].iloc[-1]
            rsi = indicators.get('rsi_14', 50)
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            stoch_k = indicators.get('stoch_k', 50)
            
            # Bullish reversal conditions
            bullish_reversal = (
                rsi < 35 and  # Oversold
                macd > macd_signal and  # MACD crossover
                stoch_k < 30  # Stochastic oversold
            )
            
            # Bearish reversal conditions
            bearish_reversal = (
                rsi > 65 and  # Overbought
                macd < macd_signal and  # MACD crossover down
                stoch_k > 70  # Stochastic overbought
            )
            
            if not (bullish_reversal or bearish_reversal):
                return None
            
            reversal_type = 'bullish_reversal' if bullish_reversal else 'bearish_reversal'
            signal = 'BUY' if bullish_reversal else 'SELL'
            
            # Calculate confidence based on multiple confirmations
            confirmations = 0
            if bullish_reversal:
                if rsi < 30:
                    confirmations += 1
                if macd > macd_signal:
                    confirmations += 1
                if stoch_k < 25:
                    confirmations += 1
            else:
                if rsi > 70:
                    confirmations += 1
                if macd < macd_signal:
                    confirmations += 1
                if stoch_k > 75:
                    confirmations += 1
            
            confidence = 50 + confirmations * 15
            
            return {
                'symbol': symbol,
                'opportunity_type': reversal_type,
                'signal': signal,
                'price': current_price,
                'rsi': rsi,
                'macd': macd,
                'stoch_k': stoch_k,
                'confirmations': confirmations,
                'confidence': confidence,
                'timestamp': datetime.now(),
                'description': f"{reversal_type.replace('_', ' ').title()}",
                'action_required': 'Prepare for reversal trade'
            }
            
        except Exception as e:
            print(f"Error detecting trend reversal for {symbol}: {str(e)}")
            return None
    
    def _detect_momentum_signals(self, data: pd.DataFrame, indicators: Dict[str, Any], symbol: str) -> Optional[Dict[str, Any]]:
        """Detect momentum-based signals"""
        try:
            current_price = data['Close'].iloc[-1]
            rsi = indicators.get('rsi_14', 50)
            adx = indicators.get('adx', 20)
            ema_12 = indicators.get('ema_12', current_price)
            ema_26 = indicators.get('ema_26', current_price)
            volume_ratio = indicators.get('volume_ratio', 1.0)
            
            # Strong momentum conditions
            strong_bullish = (
                rsi > 55 and rsi < 75 and  # Good RSI range
                adx > self.config['adx_threshold'] and  # Strong trend
                ema_12 > ema_26 and  # Uptrend
                current_price > ema_12 and  # Price above EMA
                volume_ratio > 1.2  # Decent volume
            )
            
            strong_bearish = (
                rsi < 45 and rsi > 25 and  # Good RSI range
                adx > self.config['adx_threshold'] and  # Strong trend
                ema_12 < ema_26 and  # Downtrend
                current_price < ema_12 and  # Price below EMA
                volume_ratio > 1.2  # Decent volume
            )
            
            if not (strong_bullish or strong_bearish):
                return None
            
            momentum_type = 'bullish_momentum' if strong_bullish else 'bearish_momentum'
            signal = 'BUY' if strong_bullish else 'SELL'
            
            # Calculate confidence
            strength_score = 0
            if strong_bullish:
                strength_score += min(20, (rsi - 50) * 2)
                strength_score += min(15, adx - 25)
                strength_score += min(10, (volume_ratio - 1) * 10)
            else:
                strength_score += min(20, (50 - rsi) * 2)
                strength_score += min(15, adx - 25)
                strength_score += min(10, (volume_ratio - 1) * 10)
            
            confidence = 50 + strength_score
            
            return {
                'symbol': symbol,
                'opportunity_type': momentum_type,
                'signal': signal,
                'price': current_price,
                'rsi': rsi,
                'adx': adx,
                'ema_12': ema_12,
                'ema_26': ema_26,
                'volume_ratio': volume_ratio,
                'confidence': confidence,
                'timestamp': datetime.now(),
                'description': f"{momentum_type.replace('_', ' ').title()}",
                'action_required': 'Follow the momentum'
            }
            
        except Exception as e:
            print(f"Error detecting momentum for {symbol}: {str(e)}")
            return None
    
    def _filter_scan_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter and sort scan results"""
        try:
            if not results:
                return []
            
            # Filter by minimum confidence
            filtered = [r for r in results if r.get('confidence', 0) >= self.config['min_confidence']]
            
            # Remove duplicates (same symbol)
            seen_symbols = set()
            unique_results = []
            for result in filtered:
                symbol = result.get('symbol')
                if symbol not in seen_symbols:
                    seen_symbols.add(symbol)
                    unique_results.append(result)
            
            # Sort by confidence
            sorted_results = sorted(unique_results, key=lambda x: x.get('confidence', 0), reverse=True)
            
            # Limit results to top 50
            return sorted_results[:50]
            
        except Exception as e:
            print(f"Error filtering scan results: {str(e)}")
            return results
    
    def get_scan_statistics(self) -> Dict[str, Any]:
        """Get scanning statistics"""
        try:
            if not self.scan_history:
                return {'total_scans': 0, 'avg_opportunities': 0, 'avg_duration': 0}
            
            total_scans = len(self.scan_history)
            total_opportunities = sum(scan['opportunities_found'] for scan in self.scan_history)
            total_duration = sum(scan['scan_duration'] for scan in self.scan_history)
            
            return {
                'total_scans': total_scans,
                'avg_opportunities': total_opportunities / total_scans,
                'avg_duration': total_duration / total_scans,
                'last_scan': self.last_scan_time,
                'current_opportunities': len(self.scan_results)
            }
            
        except Exception as e:
            print(f"Error getting scan statistics: {str(e)}")
            return {'total_scans': 0, 'avg_opportunities': 0, 'avg_duration': 0}
    
    def get_opportunity_breakdown(self) -> Dict[str, int]:
        """Get breakdown of opportunity types"""
        try:
            breakdown = {}
            
            for result in self.scan_results:
                opp_type = result.get('opportunity_type', 'unknown')
                breakdown[opp_type] = breakdown.get(opp_type, 0) + 1
            
            return breakdown
            
        except Exception as e:
            print(f"Error getting opportunity breakdown: {str(e)}")
            return {}
    
    def get_top_opportunities(self, limit: int = 10, opportunity_type: str = None) -> List[Dict[str, Any]]:
        """Get top opportunities from last scan"""
        try:
            results = self.scan_results
            
            # Filter by opportunity type if specified
            if opportunity_type:
                results = [r for r in results if r.get('opportunity_type') == opportunity_type]
            
            # Sort by confidence and return top results
            sorted_results = sorted(results, key=lambda x: x.get('confidence', 0), reverse=True)
            return sorted_results[:limit]
            
        except Exception as e:
            print(f"Error getting top opportunities: {str(e)}")
            return []
    
    def start_continuous_scanning(self, symbols: List[str], interval: int = 60):
        """Start continuous scanning in background"""
        try:
            def scan_loop():
                while self.scanning_active:
                    self.scan_markets(symbols)
                    time.sleep(interval)
            
            self.scanning_active = True
            scan_thread = threading.Thread(target=scan_loop, daemon=True)
            scan_thread.start()
            
            print(f"Started continuous scanning for {len(symbols)} symbols every {interval}s")
            
        except Exception as e:
            print(f"Error starting continuous scanning: {str(e)}")
    
    def stop_continuous_scanning(self):
        """Stop continuous scanning"""
        self.scanning_active = False
        print("Continuous scanning stopped")
    
    def clear_scan_history(self):
        """Clear scan history"""
        self.scan_history.clear()
        self.scan_results.clear()
        self.scan_count = 0
        print("Scan history cleared")
