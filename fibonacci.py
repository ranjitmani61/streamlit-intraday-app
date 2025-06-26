"""
Fibonacci retracement and extension calculations for technical analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta

class FibonacciCalculator:
    def __init__(self):
        # Standard Fibonacci levels
        self.retracement_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        self.extension_levels = [1.272, 1.414, 1.618, 2.0, 2.618, 3.14, 4.236]
        
        # Color mapping for different levels
        self.level_colors = {
            0.0: '#000000',      # Black
            0.236: '#FF6B6B',    # Red
            0.382: '#4ECDC4',    # Teal
            0.5: '#45B7D1',      # Blue
            0.618: '#96CEB4',    # Green
            0.786: '#FFEAA7',    # Yellow
            1.0: '#000000',      # Black
            1.272: '#DDA0DD',    # Plum
            1.414: '#98D8C8',    # Mint
            1.618: '#F7DC6F',    # Gold
            2.0: '#BB8FCE',      # Purple
            2.618: '#85C1E9',    # Light Blue
            3.14: '#F8C471',     # Orange
            4.236: '#EC7063'     # Coral
        }
    
    def calculate_fibonacci_retracements(self, data: pd.DataFrame, 
                                       lookback_period: int = 50) -> Dict[str, Any]:
        """Calculate Fibonacci retracement levels"""
        try:
            if len(data) < lookback_period:
                return {'error': 'Insufficient data for Fibonacci calculation'}
            
            # Get recent data
            recent_data = data.tail(lookback_period)
            
            # Find swing high and low
            swing_high = recent_data['High'].max()
            swing_low = recent_data['Low'].min()
            
            # Find the dates of swing points
            high_date = recent_data[recent_data['High'] == swing_high]['Datetime'].iloc[0]
            low_date = recent_data[recent_data['Low'] == swing_low]['Datetime'].iloc[0]
            
            # Determine trend direction
            trend_direction = 'uptrend' if high_date > low_date else 'downtrend'
            
            # Calculate retracement levels
            price_range = swing_high - swing_low
            
            retracement_prices = {}
            for level in self.retracement_levels:
                if trend_direction == 'uptrend':
                    # For uptrend, retracement from high
                    price = swing_high - (price_range * level)
                else:
                    # For downtrend, retracement from low
                    price = swing_low + (price_range * level)
                
                retracement_prices[level] = {
                    'price': price,
                    'percentage': level * 100,
                    'color': self.level_colors.get(level, '#888888')
                }
            
            # Calculate current price position
            current_price = data['Close'].iloc[-1]
            current_retracement = self._calculate_current_retracement(
                current_price, swing_high, swing_low, trend_direction
            )
            
            # Find nearest support and resistance levels
            support_resistance = self._find_fibonacci_support_resistance(
                current_price, retracement_prices, trend_direction
            )
            
            return {
                'swing_high': swing_high,
                'swing_low': swing_low,
                'high_date': high_date,
                'low_date': low_date,
                'trend_direction': trend_direction,
                'price_range': price_range,
                'retracement_levels': retracement_prices,
                'current_price': current_price,
                'current_retracement': current_retracement,
                'nearest_support': support_resistance['support'],
                'nearest_resistance': support_resistance['resistance'],
                'key_levels': self._identify_key_levels(retracement_prices, current_price)
            }
            
        except Exception as e:
            print(f"Error calculating Fibonacci retracements: {str(e)}")
            return {'error': str(e)}
    
    def calculate_fibonacci_extensions(self, data: pd.DataFrame, 
                                     swing_points: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Calculate Fibonacci extension levels"""
        try:
            if swing_points:
                # Use provided swing points
                swing_high = swing_points['high']
                swing_low = swing_points['low']
                retracement_point = swing_points.get('retracement', data['Close'].iloc[-1])
            else:
                # Auto-detect swing points
                fib_data = self.calculate_fibonacci_retracements(data)
                if 'error' in fib_data:
                    return fib_data
                
                swing_high = fib_data['swing_high']
                swing_low = fib_data['swing_low']
                retracement_point = data['Close'].iloc[-1]
            
            # Calculate base range (A to B)
            base_range = abs(swing_high - swing_low)
            
            # Determine extension direction
            if retracement_point > swing_high:
                # Upward extension from high
                extension_direction = 'up'
                base_point = swing_high
            else:
                # Downward extension from low
                extension_direction = 'down'
                base_point = swing_low
            
            # Calculate extension levels
            extension_prices = {}
            for level in self.extension_levels:
                if extension_direction == 'up':
                    price = base_point + (base_range * level)
                else:
                    price = base_point - (base_range * level)
                
                extension_prices[level] = {
                    'price': price,
                    'percentage': level * 100,
                    'color': self.level_colors.get(level, '#888888')
                }
            
            # Calculate potential profit targets
            current_price = data['Close'].iloc[-1]
            profit_targets = self._calculate_profit_targets(
                current_price, extension_prices, extension_direction
            )
            
            return {
                'swing_high': swing_high,
                'swing_low': swing_low,
                'retracement_point': retracement_point,
                'base_range': base_range,
                'extension_direction': extension_direction,
                'extension_levels': extension_prices,
                'current_price': current_price,
                'profit_targets': profit_targets,
                'risk_reward_ratios': self._calculate_risk_reward_ratios(
                    current_price, extension_prices, base_point
                )
            }
            
        except Exception as e:
            print(f"Error calculating Fibonacci extensions: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_current_retracement(self, current_price: float, swing_high: float, 
                                     swing_low: float, trend_direction: str) -> Dict[str, Any]:
        """Calculate current price retracement level"""
        try:
            price_range = swing_high - swing_low
            
            if trend_direction == 'uptrend':
                retracement_amount = swing_high - current_price
                retracement_percentage = (retracement_amount / price_range) * 100
            else:
                retracement_amount = current_price - swing_low
                retracement_percentage = (retracement_amount / price_range) * 100
            
            # Find closest Fibonacci level
            closest_level = min(self.retracement_levels, 
                              key=lambda x: abs(x * 100 - retracement_percentage))
            
            return {
                'percentage': retracement_percentage,
                'amount': retracement_amount,
                'closest_fib_level': closest_level,
                'distance_to_fib': abs(closest_level * 100 - retracement_percentage)
            }
            
        except Exception as e:
            print(f"Error calculating current retracement: {str(e)}")
            return {'percentage': 0, 'amount': 0, 'closest_fib_level': 0.5}
    
    def _find_fibonacci_support_resistance(self, current_price: float, 
                                         retracement_prices: Dict[float, Dict], 
                                         trend_direction: str) -> Dict[str, Optional[Dict]]:
        """Find nearest Fibonacci support and resistance levels"""
        try:
            support = None
            resistance = None
            
            # Sort levels by price
            sorted_levels = sorted(retracement_prices.items(), key=lambda x: x[1]['price'])
            
            for level, data in sorted_levels:
                price = data['price']
                
                if price < current_price:
                    # Potential support
                    support = {'level': level, 'price': price, 'distance': current_price - price}
                elif price > current_price and resistance is None:
                    # First resistance above current price
                    resistance = {'level': level, 'price': price, 'distance': price - current_price}
                    break
            
            return {'support': support, 'resistance': resistance}
            
        except Exception as e:
            print(f"Error finding support/resistance: {str(e)}")
            return {'support': None, 'resistance': None}
    
    def _identify_key_levels(self, retracement_prices: Dict[float, Dict], 
                           current_price: float) -> List[Dict[str, Any]]:
        """Identify key Fibonacci levels for trading"""
        try:
            key_levels = []
            
            # Important Fibonacci levels for trading
            important_levels = [0.382, 0.5, 0.618]
            
            for level in important_levels:
                if level in retracement_prices:
                    level_data = retracement_prices[level]
                    distance = abs(level_data['price'] - current_price)
                    distance_percentage = (distance / current_price) * 100
                    
                    # Determine significance
                    if distance_percentage < 1:
                        significance = 'Very Close'
                    elif distance_percentage < 3:
                        significance = 'Close'
                    elif distance_percentage < 5:
                        significance = 'Moderate'
                    else:
                        significance = 'Distant'
                    
                    key_levels.append({
                        'level': level,
                        'price': level_data['price'],
                        'distance': distance,
                        'distance_percentage': distance_percentage,
                        'significance': significance,
                        'trading_action': self._suggest_trading_action(level, distance_percentage)
                    })
            
            # Sort by distance from current price
            key_levels.sort(key=lambda x: x['distance'])
            
            return key_levels
            
        except Exception as e:
            print(f"Error identifying key levels: {str(e)}")
            return []
    
    def _suggest_trading_action(self, level: float, distance_percentage: float) -> str:
        """Suggest trading action based on Fibonacci level"""
        try:
            if distance_percentage < 1:
                if level == 0.382:
                    return "Watch for bounce (weak support)"
                elif level == 0.5:
                    return "Key level - watch for reaction"
                elif level == 0.618:
                    return "Strong support/resistance level"
            elif distance_percentage < 3:
                return "Approaching key level - prepare for action"
            else:
                return "Monitor for future reference"
            
        except Exception:
            return "Monitor"
    
    def _calculate_profit_targets(self, current_price: float, extension_prices: Dict[float, Dict], 
                                direction: str) -> List[Dict[str, Any]]:
        """Calculate profit targets based on Fibonacci extensions"""
        try:
            profit_targets = []
            
            for level, data in extension_prices.items():
                price = data['price']
                
                # Only consider levels that make sense for profit taking
                if ((direction == 'up' and price > current_price) or 
                    (direction == 'down' and price < current_price)):
                    
                    profit_percentage = abs((price - current_price) / current_price) * 100
                    
                    profit_targets.append({
                        'level': level,
                        'price': price,
                        'profit_percentage': profit_percentage,
                        'target_priority': self._get_target_priority(level)
                    })
            
            # Sort by level (closest first)
            profit_targets.sort(key=lambda x: x['level'])
            
            return profit_targets
            
        except Exception as e:
            print(f"Error calculating profit targets: {str(e)}")
            return []
    
    def _get_target_priority(self, level: float) -> str:
        """Get priority level for profit targets"""
        if level <= 1.272:
            return "Primary"
        elif level <= 1.618:
            return "Secondary"
        elif level <= 2.618:
            return "Extended"
        else:
            return "Extreme"
    
    def _calculate_risk_reward_ratios(self, current_price: float, extension_prices: Dict[float, Dict], 
                                    base_point: float) -> List[Dict[str, Any]]:
        """Calculate risk-reward ratios for different targets"""
        try:
            risk_reward_ratios = []
            
            # Assume risk is distance to base point (simplified)
            risk = abs(current_price - base_point)
            
            for level, data in extension_prices.items():
                price = data['price']
                reward = abs(price - current_price)
                
                if risk > 0:
                    ratio = reward / risk
                    
                    risk_reward_ratios.append({
                        'level': level,
                        'price': price,
                        'risk': risk,
                        'reward': reward,
                        'ratio': ratio,
                        'quality': self._assess_risk_reward_quality(ratio)
                    })
            
            # Sort by ratio (best first)
            risk_reward_ratios.sort(key=lambda x: x['ratio'], reverse=True)
            
            return risk_reward_ratios
            
        except Exception as e:
            print(f"Error calculating risk-reward ratios: {str(e)}")
            return []
    
    def _assess_risk_reward_quality(self, ratio: float) -> str:
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
    
    def find_fibonacci_clusters(self, data: pd.DataFrame, timeframes: List[int] = None) -> Dict[str, Any]:
        """Find Fibonacci level clusters across multiple timeframes"""
        try:
            if timeframes is None:
                timeframes = [20, 50, 100]
            
            all_levels = []
            
            # Calculate Fibonacci levels for different timeframes
            for timeframe in timeframes:
                if len(data) >= timeframe:
                    fib_data = self.calculate_fibonacci_retracements(data, timeframe)
                    
                    if 'error' not in fib_data:
                        for level, level_data in fib_data['retracement_levels'].items():
                            all_levels.append({
                                'timeframe': timeframe,
                                'level': level,
                                'price': level_data['price']
                            })
            
            # Find clusters (levels close to each other)
            clusters = self._find_price_clusters(all_levels)
            
            # Get current price for context
            current_price = data['Close'].iloc[-1]
            
            # Assess cluster significance
            significant_clusters = self._assess_cluster_significance(clusters, current_price)
            
            return {
                'timeframes_analyzed': timeframes,
                'total_levels': len(all_levels),
                'clusters_found': len(clusters),
                'significant_clusters': significant_clusters,
                'current_price': current_price,
                'nearest_cluster': self._find_nearest_cluster(significant_clusters, current_price)
            }
            
        except Exception as e:
            print(f"Error finding Fibonacci clusters: {str(e)}")
            return {'error': str(e)}
    
    def _find_price_clusters(self, levels: List[Dict], tolerance: float = 0.02) -> List[Dict[str, Any]]:
        """Find clusters of Fibonacci levels"""
        try:
            if not levels:
                return []
            
            # Sort levels by price
            sorted_levels = sorted(levels, key=lambda x: x['price'])
            
            clusters = []
            current_cluster = [sorted_levels[0]]
            
            for i in range(1, len(sorted_levels)):
                current_level = sorted_levels[i]
                last_in_cluster = current_cluster[-1]
                
                # Check if current level is close to the cluster
                price_diff = abs(current_level['price'] - last_in_cluster['price'])
                relative_diff = price_diff / last_in_cluster['price']
                
                if relative_diff <= tolerance:
                    current_cluster.append(current_level)
                else:
                    # Start new cluster
                    if len(current_cluster) >= 2:  # Only keep clusters with 2+ levels
                        clusters.append({
                            'levels': current_cluster,
                            'count': len(current_cluster),
                            'average_price': np.mean([l['price'] for l in current_cluster]),
                            'price_range': max(l['price'] for l in current_cluster) - min(l['price'] for l in current_cluster)
                        })
                    
                    current_cluster = [current_level]
            
            # Don't forget the last cluster
            if len(current_cluster) >= 2:
                clusters.append({
                    'levels': current_cluster,
                    'count': len(current_cluster),
                    'average_price': np.mean([l['price'] for l in current_cluster]),
                    'price_range': max(l['price'] for l in current_cluster) - min(l['price'] for l in current_cluster)
                })
            
            return clusters
            
        except Exception as e:
            print(f"Error finding price clusters: {str(e)}")
            return []
    
    def _assess_cluster_significance(self, clusters: List[Dict], current_price: float) -> List[Dict[str, Any]]:
        """Assess significance of Fibonacci clusters"""
        try:
            significant_clusters = []
            
            for cluster in clusters:
                # Calculate significance score
                significance_score = 0
                
                # More levels in cluster = higher significance
                significance_score += cluster['count'] * 2
                
                # Levels from different timeframes = higher significance
                timeframes = set(level['timeframe'] for level in cluster['levels'])
                significance_score += len(timeframes) * 3
                
                # Important Fibonacci levels = higher significance
                important_levels = [0.382, 0.5, 0.618]
                for level_data in cluster['levels']:
                    if level_data['level'] in important_levels:
                        significance_score += 5
                
                # Distance from current price
                distance = abs(cluster['average_price'] - current_price)
                distance_percentage = (distance / current_price) * 100
                
                # Closer to current price = more relevant
                if distance_percentage < 2:
                    significance_score += 10
                elif distance_percentage < 5:
                    significance_score += 5
                
                # Determine significance level
                if significance_score >= 20:
                    significance = "Very High"
                elif significance_score >= 15:
                    significance = "High"
                elif significance_score >= 10:
                    significance = "Moderate"
                else:
                    significance = "Low"
                
                significant_clusters.append({
                    'cluster': cluster,
                    'significance_score': significance_score,
                    'significance': significance,
                    'distance_from_current': distance,
                    'distance_percentage': distance_percentage,
                    'trading_relevance': self._assess_trading_relevance(distance_percentage, significance_score)
                })
            
            # Sort by significance score
            significant_clusters.sort(key=lambda x: x['significance_score'], reverse=True)
            
            return significant_clusters
            
        except Exception as e:
            print(f"Error assessing cluster significance: {str(e)}")
            return []
    
    def _assess_trading_relevance(self, distance_percentage: float, significance_score: int) -> str:
        """Assess trading relevance of a cluster"""
        if distance_percentage < 1 and significance_score >= 15:
            return "Immediate attention - strong support/resistance"
        elif distance_percentage < 2 and significance_score >= 10:
            return "High priority - key level approaching"
        elif distance_percentage < 5:
            return "Monitor closely - potential target/support"
        else:
            return "Long-term reference level"
    
    def _find_nearest_cluster(self, clusters: List[Dict], current_price: float) -> Optional[Dict[str, Any]]:
        """Find the nearest significant cluster to current price"""
        try:
            if not clusters:
                return None
            
            nearest = min(clusters, key=lambda x: x['distance_from_current'])
            
            return {
                'cluster_info': nearest,
                'action_required': self._suggest_cluster_action(nearest, current_price)
            }
            
        except Exception as e:
            print(f"Error finding nearest cluster: {str(e)}")
            return None
    
    def _suggest_cluster_action(self, cluster_info: Dict, current_price: float) -> str:
        """Suggest action based on nearest cluster"""
        try:
            distance_pct = cluster_info['distance_percentage']
            significance = cluster_info['significance']
            cluster_price = cluster_info['cluster']['average_price']
            
            if distance_pct < 0.5:
                return f"Price at strong {significance.lower()} cluster - expect reaction"
            elif distance_pct < 2:
                direction = "approaching resistance" if cluster_price > current_price else "approaching support"
                return f"Price {direction} - prepare for potential reversal"
            else:
                return "Monitor for future reference"
                
        except Exception:
            return "Monitor cluster"
