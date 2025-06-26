"""
Support and resistance level calculation and analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SupportResistanceCalculator:
    def __init__(self):
        self.config = {
            'lookback_period': 50,
            'min_touches': 2,
            'tolerance_percentage': 0.5,  # 0.5% tolerance for level identification
            'strength_threshold': 3,
            'volume_weight': 0.3,
            'time_decay': 0.1
        }
        
        # Level strength mapping
        self.strength_levels = {
            1: 'Weak',
            2: 'Moderate', 
            3: 'Strong',
            4: 'Very Strong',
            5: 'Critical'
        }
    
    def calculate_support_resistance(self, data: pd.DataFrame, 
                                   method: str = 'pivot_points') -> Dict[str, Any]:
        """Calculate support and resistance levels using specified method"""
        try:
            if len(data) < self.config['lookback_period']:
                return {'error': 'Insufficient data for support/resistance calculation'}
            
            if method == 'pivot_points':
                return self._calculate_pivot_point_levels(data)
            elif method == 'fractals':
                return self._calculate_fractal_levels(data)
            elif method == 'moving_averages':
                return self._calculate_ma_levels(data)
            elif method == 'volume_profile':
                return self._calculate_volume_profile_levels(data)
            elif method == 'fibonacci':
                return self._calculate_fibonacci_levels(data)
            else:
                # Default to comprehensive analysis
                return self._calculate_comprehensive_levels(data)
                
        except Exception as e:
            print(f"Error calculating support/resistance: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_pivot_point_levels(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate support and resistance using pivot points"""
        try:
            # Get recent data for calculation
            recent_data = data.tail(self.config['lookback_period'])
            
            # Find pivot highs and lows
            pivot_highs = self._find_pivot_highs(recent_data)
            pivot_lows = self._find_pivot_lows(recent_data)
            
            # Calculate support levels from pivot lows
            support_levels = self._calculate_level_clusters(pivot_lows, 'support')
            
            # Calculate resistance levels from pivot highs
            resistance_levels = self._calculate_level_clusters(pivot_highs, 'resistance')
            
            # Get current price for context
            current_price = data['Close'].iloc[-1]
            
            # Find nearest levels
            nearest_support = self._find_nearest_level(support_levels, current_price, 'below')
            nearest_resistance = self._find_nearest_level(resistance_levels, current_price, 'above')
            
            # Calculate level strength
            for level in support_levels + resistance_levels:
                level['strength'] = self._calculate_level_strength(level, recent_data)
                level['strength_text'] = self.strength_levels.get(level['strength'], 'Unknown')
            
            return {
                'method': 'pivot_points',
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance,
                'current_price': current_price,
                'total_levels': len(support_levels) + len(resistance_levels),
                'analysis_period': f"Last {self.config['lookback_period']} periods",
                'level_quality': self._assess_level_quality(support_levels + resistance_levels)
            }
            
        except Exception as e:
            print(f"Error in pivot point calculation: {str(e)}")
            return {'error': str(e)}
    
    def _find_pivot_highs(self, data: pd.DataFrame, window: int = 5) -> List[Dict[str, Any]]:
        """Find pivot high points"""
        try:
            pivot_highs = []
            
            for i in range(window, len(data) - window):
                current_high = data['High'].iloc[i]
                is_pivot = True
                
                # Check if current high is higher than surrounding highs
                for j in range(i - window, i + window + 1):
                    if j != i and data['High'].iloc[j] >= current_high:
                        is_pivot = False
                        break
                
                if is_pivot:
                    pivot_highs.append({
                        'price': current_high,
                        'date': data['Datetime'].iloc[i] if 'Datetime' in data.columns else i,
                        'index': i,
                        'volume': data['Volume'].iloc[i],
                        'type': 'resistance'
                    })
            
            return pivot_highs
            
        except Exception as e:
            print(f"Error finding pivot highs: {str(e)}")
            return []
    
    def _find_pivot_lows(self, data: pd.DataFrame, window: int = 5) -> List[Dict[str, Any]]:
        """Find pivot low points"""
        try:
            pivot_lows = []
            
            for i in range(window, len(data) - window):
                current_low = data['Low'].iloc[i]
                is_pivot = True
                
                # Check if current low is lower than surrounding lows
                for j in range(i - window, i + window + 1):
                    if j != i and data['Low'].iloc[j] <= current_low:
                        is_pivot = False
                        break
                
                if is_pivot:
                    pivot_lows.append({
                        'price': current_low,
                        'date': data['Datetime'].iloc[i] if 'Datetime' in data.columns else i,
                        'index': i,
                        'volume': data['Volume'].iloc[i],
                        'type': 'support'
                    })
            
            return pivot_lows
            
        except Exception as e:
            print(f"Error finding pivot lows: {str(e)}")
            return []
    
    def _calculate_level_clusters(self, pivot_points: List[Dict], level_type: str) -> List[Dict[str, Any]]:
        """Calculate support/resistance level clusters"""
        try:
            if not pivot_points:
                return []
            
            # Group pivot points into clusters
            clusters = []
            tolerance = self.config['tolerance_percentage'] / 100
            
            # Sort pivot points by price
            sorted_points = sorted(pivot_points, key=lambda x: x['price'])
            
            current_cluster = [sorted_points[0]]
            
            for point in sorted_points[1:]:
                # Check if point belongs to current cluster
                cluster_center = np.mean([p['price'] for p in current_cluster])
                price_diff = abs(point['price'] - cluster_center) / cluster_center
                
                if price_diff <= tolerance:
                    current_cluster.append(point)
                else:
                    # Process current cluster and start new one
                    if len(current_cluster) >= self.config['min_touches']:
                        cluster_level = self._create_level_from_cluster(current_cluster, level_type)
                        if cluster_level:
                            clusters.append(cluster_level)
                    
                    current_cluster = [point]
            
            # Don't forget the last cluster
            if len(current_cluster) >= self.config['min_touches']:
                cluster_level = self._create_level_from_cluster(current_cluster, level_type)
                if cluster_level:
                    clusters.append(cluster_level)
            
            return clusters
            
        except Exception as e:
            print(f"Error calculating level clusters: {str(e)}")
            return []
    
    def _create_level_from_cluster(self, cluster: List[Dict], level_type: str) -> Optional[Dict[str, Any]]:
        """Create a support/resistance level from a cluster of pivot points"""
        try:
            if not cluster:
                return None
            
            # Calculate level price (weighted average)
            total_volume = sum(point['volume'] for point in cluster)
            if total_volume > 0:
                weighted_price = sum(point['price'] * point['volume'] for point in cluster) / total_volume
            else:
                weighted_price = np.mean([point['price'] for point in cluster])
            
            # Calculate level metrics
            touch_count = len(cluster)
            price_range = max(point['price'] for point in cluster) - min(point['price'] for point in cluster)
            avg_volume = np.mean([point['volume'] for point in cluster])
            
            # Calculate recency (more recent = higher weight)
            latest_touch = max(cluster, key=lambda x: x['index'])
            recency_score = 1.0 - (latest_touch['index'] / 100)  # Normalized recency
            
            return {
                'price': weighted_price,
                'type': level_type,
                'touch_count': touch_count,
                'price_range': price_range,
                'average_volume': avg_volume,
                'latest_touch_index': latest_touch['index'],
                'latest_touch_date': latest_touch['date'],
                'recency_score': recency_score,
                'cluster_points': cluster,
                'confidence': self._calculate_level_confidence(touch_count, price_range, avg_volume)
            }
            
        except Exception as e:
            print(f"Error creating level from cluster: {str(e)}")
            return None
    
    def _calculate_level_confidence(self, touch_count: int, price_range: float, avg_volume: float) -> float:
        """Calculate confidence score for a support/resistance level"""
        try:
            confidence = 0.0
            
            # Touch count contribution (more touches = higher confidence)
            touch_score = min(touch_count / 5, 1.0) * 40  # Max 40 points
            confidence += touch_score
            
            # Price tightness contribution (tighter range = higher confidence)
            tightness_score = max(0, (1 - price_range / 100)) * 30  # Max 30 points
            confidence += tightness_score
            
            # Volume contribution (higher volume = higher confidence)
            volume_score = min(avg_volume / 1000000, 1.0) * 30  # Max 30 points (normalized)
            confidence += volume_score
            
            return min(confidence, 100.0)  # Cap at 100%
            
        except Exception as e:
            print(f"Error calculating level confidence: {str(e)}")
            return 50.0
    
    def _calculate_level_strength(self, level: Dict[str, Any], data: pd.DataFrame) -> int:
        """Calculate strength rating for a support/resistance level"""
        try:
            strength = 1
            
            # Touch count factor
            if level['touch_count'] >= 5:
                strength += 2
            elif level['touch_count'] >= 3:
                strength += 1
            
            # Volume factor
            avg_volume = data['Volume'].mean()
            if level['average_volume'] > avg_volume * 1.5:
                strength += 1
            
            # Recency factor
            if level['recency_score'] > 0.7:
                strength += 1
            
            # Confidence factor
            if level['confidence'] > 80:
                strength += 1
            
            return min(strength, 5)  # Cap at 5
            
        except Exception as e:
            print(f"Error calculating level strength: {str(e)}")
            return 1
    
    def _find_nearest_level(self, levels: List[Dict], current_price: float, direction: str) -> Optional[Dict[str, Any]]:
        """Find nearest support or resistance level"""
        try:
            if not levels:
                return None
            
            filtered_levels = []
            
            for level in levels:
                if direction == 'above' and level['price'] > current_price:
                    filtered_levels.append(level)
                elif direction == 'below' and level['price'] < current_price:
                    filtered_levels.append(level)
            
            if not filtered_levels:
                return None
            
            # Find closest level
            nearest = min(filtered_levels, key=lambda x: abs(x['price'] - current_price))
            
            # Add distance information
            nearest['distance'] = abs(nearest['price'] - current_price)
            nearest['distance_percentage'] = (nearest['distance'] / current_price) * 100
            
            return nearest
            
        except Exception as e:
            print(f"Error finding nearest level: {str(e)}")
            return None
    
    def _calculate_fractal_levels(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate support/resistance using fractal analysis"""
        try:
            # Williams Fractal implementation
            fractal_highs = []
            fractal_lows = []
            
            for i in range(2, len(data) - 2):
                # Fractal high: current high is higher than 2 highs before and after
                if (data['High'].iloc[i] > data['High'].iloc[i-1] and 
                    data['High'].iloc[i] > data['High'].iloc[i-2] and
                    data['High'].iloc[i] > data['High'].iloc[i+1] and 
                    data['High'].iloc[i] > data['High'].iloc[i+2]):
                    
                    fractal_highs.append({
                        'price': data['High'].iloc[i],
                        'date': data['Datetime'].iloc[i] if 'Datetime' in data.columns else i,
                        'index': i,
                        'volume': data['Volume'].iloc[i],
                        'type': 'resistance'
                    })
                
                # Fractal low: current low is lower than 2 lows before and after
                if (data['Low'].iloc[i] < data['Low'].iloc[i-1] and 
                    data['Low'].iloc[i] < data['Low'].iloc[i-2] and
                    data['Low'].iloc[i] < data['Low'].iloc[i+1] and 
                    data['Low'].iloc[i] < data['Low'].iloc[i+2]):
                    
                    fractal_lows.append({
                        'price': data['Low'].iloc[i],
                        'date': data['Datetime'].iloc[i] if 'Datetime' in data.columns else i,
                        'index': i,
                        'volume': data['Volume'].iloc[i],
                        'type': 'support'
                    })
            
            # Calculate level clusters
            support_levels = self._calculate_level_clusters(fractal_lows, 'support')
            resistance_levels = self._calculate_level_clusters(fractal_highs, 'resistance')
            
            current_price = data['Close'].iloc[-1]
            
            return {
                'method': 'fractals',
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'nearest_support': self._find_nearest_level(support_levels, current_price, 'below'),
                'nearest_resistance': self._find_nearest_level(resistance_levels, current_price, 'above'),
                'current_price': current_price,
                'fractal_highs_count': len(fractal_highs),
                'fractal_lows_count': len(fractal_lows)
            }
            
        except Exception as e:
            print(f"Error in fractal calculation: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_ma_levels(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate support/resistance using moving averages"""
        try:
            import ta
            
            # Calculate different moving averages
            ma_periods = [20, 50, 100, 200]
            current_price = data['Close'].iloc[-1]
            
            support_levels = []
            resistance_levels = []
            
            for period in ma_periods:
                if len(data) >= period:
                    ma = ta.trend.sma_indicator(data['Close'], window=period).iloc[-1]
                    
                    level_data = {
                        'price': ma,
                        'type': 'support' if ma < current_price else 'resistance',
                        'ma_period': period,
                        'touch_count': self._count_ma_touches(data, ma, period),
                        'confidence': 70 + (period / 10),  # Longer MA = higher confidence
                        'strength': 3 if period >= 50 else 2
                    }
                    
                    if ma < current_price:
                        support_levels.append(level_data)
                    else:
                        resistance_levels.append(level_data)
            
            return {
                'method': 'moving_averages',
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'nearest_support': self._find_nearest_level(support_levels, current_price, 'below'),
                'nearest_resistance': self._find_nearest_level(resistance_levels, current_price, 'above'),
                'current_price': current_price,
                'ma_periods_used': ma_periods
            }
            
        except Exception as e:
            print(f"Error in MA calculation: {str(e)}")
            return {'error': str(e)}
    
    def _count_ma_touches(self, data: pd.DataFrame, ma_level: float, period: int) -> int:
        """Count how many times price touched a moving average"""
        try:
            import ta
            
            ma_series = ta.trend.sma_indicator(data['Close'], window=period)
            touches = 0
            tolerance = ma_level * 0.01  # 1% tolerance
            
            for i in range(len(data)):
                if abs(data['Low'].iloc[i] - ma_series.iloc[i]) <= tolerance:
                    touches += 1
                elif abs(data['High'].iloc[i] - ma_series.iloc[i]) <= tolerance:
                    touches += 1
            
            return touches
            
        except Exception as e:
            print(f"Error counting MA touches: {str(e)}")
            return 0
    
    def _calculate_volume_profile_levels(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate support/resistance using volume profile"""
        try:
            # Simplified volume profile implementation
            price_bins = 50
            price_min = data['Low'].min()
            price_max = data['High'].max()
            
            bin_size = (price_max - price_min) / price_bins
            volume_at_price = {}
            
            for i in range(len(data)):
                # Calculate volume for each price level
                price_range = data['High'].iloc[i] - data['Low'].iloc[i]
                volume_per_point = data['Volume'].iloc[i] / max(price_range, 0.01)
                
                # Distribute volume across the price range
                low_bin = int((data['Low'].iloc[i] - price_min) / bin_size)
                high_bin = int((data['High'].iloc[i] - price_min) / bin_size)
                
                for bin_idx in range(low_bin, high_bin + 1):
                    price_level = price_min + (bin_idx * bin_size)
                    volume_at_price[price_level] = volume_at_price.get(price_level, 0) + volume_per_point
            
            # Find high volume nodes (potential support/resistance)
            sorted_levels = sorted(volume_at_price.items(), key=lambda x: x[1], reverse=True)
            top_volume_levels = sorted_levels[:10]  # Top 10 volume levels
            
            current_price = data['Close'].iloc[-1]
            support_levels = []
            resistance_levels = []
            
            for price, volume in top_volume_levels:
                level_data = {
                    'price': price,
                    'volume': volume,
                    'type': 'support' if price < current_price else 'resistance',
                    'confidence': min(90, volume / max(volume_at_price.values()) * 100),
                    'strength': 4 if volume > np.mean(list(volume_at_price.values())) * 2 else 3
                }
                
                if price < current_price:
                    support_levels.append(level_data)
                else:
                    resistance_levels.append(level_data)
            
            return {
                'method': 'volume_profile',
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'nearest_support': self._find_nearest_level(support_levels, current_price, 'below'),
                'nearest_resistance': self._find_nearest_level(resistance_levels, current_price, 'above'),
                'current_price': current_price,
                'total_volume_levels': len(volume_at_price),
                'point_of_control': max(volume_at_price.items(), key=lambda x: x[1])[0]
            }
            
        except Exception as e:
            print(f"Error in volume profile calculation: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_fibonacci_levels(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate support/resistance using Fibonacci retracements"""
        try:
            from fibonacci import FibonacciCalculator
            
            fib_calc = FibonacciCalculator()
            fib_data = fib_calc.calculate_fibonacci_retracements(data)
            
            if 'error' in fib_data:
                return fib_data
            
            current_price = data['Close'].iloc[-1]
            support_levels = []
            resistance_levels = []
            
            for level, level_data in fib_data['retracement_levels'].items():
                price = level_data['price']
                
                sr_level = {
                    'price': price,
                    'fibonacci_level': level,
                    'type': 'support' if price < current_price else 'resistance',
                    'confidence': 80 if level in [0.382, 0.5, 0.618] else 60,
                    'strength': 4 if level in [0.382, 0.618] else 3
                }
                
                if price < current_price:
                    support_levels.append(sr_level)
                else:
                    resistance_levels.append(sr_level)
            
            return {
                'method': 'fibonacci',
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'nearest_support': self._find_nearest_level(support_levels, current_price, 'below'),
                'nearest_resistance': self._find_nearest_level(resistance_levels, current_price, 'above'),
                'current_price': current_price,
                'swing_high': fib_data['swing_high'],
                'swing_low': fib_data['swing_low'],
                'trend_direction': fib_data['trend_direction']
            }
            
        except Exception as e:
            print(f"Error in Fibonacci calculation: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_comprehensive_levels(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive support/resistance using multiple methods"""
        try:
            # Calculate using different methods
            pivot_results = self._calculate_pivot_point_levels(data)
            fractal_results = self._calculate_fractal_levels(data)
            ma_results = self._calculate_ma_levels(data)
            
            # Combine all levels
            all_support = []
            all_resistance = []
            
            for result in [pivot_results, fractal_results, ma_results]:
                if 'error' not in result:
                    all_support.extend(result.get('support_levels', []))
                    all_resistance.extend(result.get('resistance_levels', []))
            
            # Find consensus levels (levels confirmed by multiple methods)
            consensus_support = self._find_consensus_levels(all_support)
            consensus_resistance = self._find_consensus_levels(all_resistance)
            
            current_price = data['Close'].iloc[-1]
            
            return {
                'method': 'comprehensive',
                'support_levels': consensus_support,
                'resistance_levels': consensus_resistance,
                'nearest_support': self._find_nearest_level(consensus_support, current_price, 'below'),
                'nearest_resistance': self._find_nearest_level(consensus_resistance, current_price, 'above'),
                'current_price': current_price,
                'total_methods_used': 3,
                'level_consensus': self._calculate_level_consensus(consensus_support + consensus_resistance),
                'trading_range': self._identify_trading_range(consensus_support, consensus_resistance, current_price)
            }
            
        except Exception as e:
            print(f"Error in comprehensive calculation: {str(e)}")
            return {'error': str(e)}
    
    def _find_consensus_levels(self, levels: List[Dict]) -> List[Dict[str, Any]]:
        """Find levels that are confirmed by multiple methods"""
        try:
            if not levels:
                return []
            
            consensus_levels = []
            tolerance = self.config['tolerance_percentage'] / 100
            
            # Group levels by price proximity
            for level in levels:
                # Check if this level is close to any existing consensus level
                found_consensus = False
                
                for consensus in consensus_levels:
                    price_diff = abs(level['price'] - consensus['price']) / consensus['price']
                    
                    if price_diff <= tolerance:
                        # Add to existing consensus
                        consensus['methods'] += 1
                        consensus['combined_confidence'] = (consensus['combined_confidence'] + level.get('confidence', 70)) / 2
                        consensus['combined_strength'] = max(consensus['combined_strength'], level.get('strength', 2))
                        found_consensus = True
                        break
                
                if not found_consensus:
                    # Create new consensus level
                    consensus_levels.append({
                        'price': level['price'],
                        'type': level['type'],
                        'methods': 1,
                        'combined_confidence': level.get('confidence', 70),
                        'combined_strength': level.get('strength', 2),
                        'touch_count': level.get('touch_count', 1)
                    })
            
            # Filter levels with multiple method confirmation
            confirmed_levels = [level for level in consensus_levels if level['methods'] >= 2]
            
            # Sort by combined confidence
            confirmed_levels.sort(key=lambda x: x['combined_confidence'], reverse=True)
            
            return confirmed_levels
            
        except Exception as e:
            print(f"Error finding consensus levels: {str(e)}")
            return []
    
    def _calculate_level_consensus(self, levels: List[Dict]) -> Dict[str, Any]:
        """Calculate consensus metrics for levels"""
        try:
            if not levels:
                return {'quality': 'No levels', 'score': 0}
            
            total_confidence = sum(level.get('combined_confidence', 70) for level in levels)
            avg_confidence = total_confidence / len(levels)
            
            multi_method_levels = len([level for level in levels if level.get('methods', 1) >= 2])
            consensus_percentage = (multi_method_levels / len(levels)) * 100
            
            # Calculate overall quality
            if avg_confidence > 80 and consensus_percentage > 70:
                quality = 'Excellent'
                score = 90
            elif avg_confidence > 70 and consensus_percentage > 50:
                quality = 'Good'
                score = 75
            elif avg_confidence > 60:
                quality = 'Fair'
                score = 60
            else:
                quality = 'Poor'
                score = 40
            
            return {
                'quality': quality,
                'score': score,
                'average_confidence': avg_confidence,
                'consensus_percentage': consensus_percentage,
                'total_levels': len(levels),
                'multi_method_levels': multi_method_levels
            }
            
        except Exception as e:
            print(f"Error calculating level consensus: {str(e)}")
            return {'quality': 'Unknown', 'score': 50}
    
    def _identify_trading_range(self, support_levels: List[Dict], resistance_levels: List[Dict], 
                              current_price: float) -> Optional[Dict[str, Any]]:
        """Identify if stock is in a trading range"""
        try:
            if not support_levels or not resistance_levels:
                return None
            
            # Find the range around current price
            nearby_support = [s for s in support_levels if current_price - s['price'] < current_price * 0.1]
            nearby_resistance = [r for r in resistance_levels if r['price'] - current_price < current_price * 0.1]
            
            if not nearby_support or not nearby_resistance:
                return None
            
            # Get strongest nearby levels
            strongest_support = max(nearby_support, key=lambda x: x.get('combined_strength', 0))
            strongest_resistance = max(nearby_resistance, key=lambda x: x.get('combined_strength', 0))
            
            range_size = strongest_resistance['price'] - strongest_support['price']
            range_percentage = (range_size / current_price) * 100
            
            # Price position within range
            position_in_range = (current_price - strongest_support['price']) / range_size
            
            return {
                'in_trading_range': True,
                'support_price': strongest_support['price'],
                'resistance_price': strongest_resistance['price'],
                'range_size': range_size,
                'range_percentage': range_percentage,
                'position_in_range': position_in_range,
                'range_quality': 'Strong' if range_percentage > 5 else 'Narrow',
                'trading_strategy': self._suggest_range_strategy(position_in_range, range_percentage)
            }
            
        except Exception as e:
            print(f"Error identifying trading range: {str(e)}")
            return None
    
    def _suggest_range_strategy(self, position_in_range: float, range_percentage: float) -> str:
        """Suggest trading strategy based on range analysis"""
        try:
            if range_percentage < 3:
                return "Range too narrow for range trading"
            
            if position_in_range < 0.3:
                return "Near support - consider buying on bounce"
            elif position_in_range > 0.7:
                return "Near resistance - consider selling on rejection"
            else:
                return "Mid-range - wait for move to extremes"
                
        except Exception:
            return "Monitor range boundaries"
    
    def _assess_level_quality(self, levels: List[Dict]) -> Dict[str, Any]:
        """Assess overall quality of support/resistance levels"""
        try:
            if not levels:
                return {'quality': 'No levels', 'recommendations': []}
            
            high_confidence_levels = [l for l in levels if l.get('confidence', 0) > 80]
            strong_levels = [l for l in levels if l.get('strength', 0) >= 4]
            
            quality_score = 0
            recommendations = []
            
            # Quality factors
            if len(high_confidence_levels) >= 3:
                quality_score += 30
                recommendations.append("Multiple high-confidence levels identified")
            
            if len(strong_levels) >= 2:
                quality_score += 25
                recommendations.append("Strong levels present for reliable trading")
            
            if len(levels) >= 5:
                quality_score += 20
                recommendations.append("Good number of levels for analysis")
            
            recent_levels = [l for l in levels if l.get('recency_score', 0) > 0.7]
            if recent_levels:
                quality_score += 25
                recommendations.append("Recent price action confirms levels")
            
            # Determine overall quality
            if quality_score >= 80:
                quality = 'Excellent'
            elif quality_score >= 60:
                quality = 'Good'
            elif quality_score >= 40:
                quality = 'Fair'
            else:
                quality = 'Poor'
                recommendations.append("Consider using additional analysis methods")
            
            return {
                'quality': quality,
                'score': quality_score,
                'high_confidence_count': len(high_confidence_levels),
                'strong_levels_count': len(strong_levels),
                'total_levels': len(levels),
                'recommendations': recommendations
            }
            
        except Exception as e:
            print(f"Error assessing level quality: {str(e)}")
            return {'quality': 'Unknown', 'recommendations': []}
