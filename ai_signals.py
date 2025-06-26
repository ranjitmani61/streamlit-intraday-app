"""
AI-powered signal generation using machine learning models
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import ta
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class AISignalGenerator:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = [
            'rsi_14', 'rsi_21', 'macd', 'macd_signal', 'adx', 'ema_12', 'ema_26',
            'ema_20', 'ema_50', 'bb_percent', 'volume_ratio', 'price_position',
            'volatility_ratio', 'trend_strength', 'volume_momentum', 'sr_proximity',
            'stoch_k', 'williams_r', 'roc_12', 'mfi', 'atr_14', 'obv_trend'
        ]
        self.target_mapping = {'SELL': 0, 'HOLD': 1, 'BUY': 2}
        self.reverse_mapping = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models"""
        try:
            # Random Forest Model
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
            
            # Additional models can be added here
            self.models['random_forest_aggressive'] = RandomForestClassifier(
                n_estimators=150,
                max_depth=15,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42,
                class_weight='balanced'
            )
            
            # Initialize scalers
            for model_name in self.models.keys():
                self.scalers[model_name] = StandardScaler()
                
        except Exception as e:
            print(f"Error initializing models: {str(e)}")
    
    def generate_signals_batch(self, symbols: List[str], model_type: str = 'random_forest',
                             confidence_threshold: float = 0.75) -> List[Dict[str, Any]]:
        """Generate AI signals for multiple symbols"""
        try:
            signals = []
            
            for symbol in symbols:
                signal = self.generate_single_signal(symbol, model_type, confidence_threshold)
                if signal:
                    signals.append(signal)
            
            # Sort by confidence
            signals = sorted(signals, key=lambda x: x['confidence'], reverse=True)
            return signals
            
        except Exception as e:
            print(f"Error generating batch signals: {str(e)}")
            return []
    
    def generate_single_signal(self, symbol: str, model_type: str = 'random_forest',
                             confidence_threshold: float = 0.75) -> Optional[Dict[str, Any]]:
        """Generate AI signal for a single symbol"""
        try:
            from data_fetcher import DataFetcher
            from indicators import TechnicalIndicators
            
            # Fetch data
            data_fetcher = DataFetcher()
            indicators = TechnicalIndicators()
            
            data = data_fetcher.get_realtime_data(symbol, period='3mo', interval='1h')
            if data is None or len(data) < 100:
                return None
            
            # Calculate indicators
            tech_indicators = indicators.calculate_all_indicators(data)
            
            # Prepare features
            features = self._prepare_features(data, tech_indicators)
            if features is None:
                return None
            
            # Train model if not already trained
            if model_type not in self.models or not hasattr(self.models[model_type], 'feature_importances_'):
                self._train_model(data, tech_indicators, model_type)
            
            # Generate prediction
            prediction = self._predict_signal(features, model_type)
            if prediction is None:
                return None
            
            signal, confidence = prediction
            
            # Apply confidence threshold
            if confidence < confidence_threshold:
                signal = 'HOLD'
            
            # Get current price
            current_price = data['Close'].iloc[-1]
            change_pct = ((current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
            
            return {
                'symbol': symbol,
                'signal': signal,
                'confidence': confidence * 100,
                'price': current_price,
                'change_pct': change_pct,
                'model_type': model_type,
                'features_used': len(self.feature_columns),
                'timestamp': datetime.now(),
                'ai_score': self._calculate_ai_score(features, tech_indicators),
                'risk_level': self._calculate_risk_level(features, tech_indicators),
                'recommendation': self._generate_recommendation(signal, confidence, tech_indicators)
            }
            
        except Exception as e:
            print(f"Error generating signal for {symbol}: {str(e)}")
            return None
    
    def _prepare_features(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Optional[np.ndarray]:
        """Prepare features for ML model"""
        try:
            features = []
            
            # Technical indicators
            features.extend([
                indicators.get('rsi_14', 50),
                indicators.get('rsi_21', 50),
                indicators.get('macd', 0),
                indicators.get('macd_signal', 0),
                indicators.get('adx', 25),
                indicators.get('ema_12', 0),
                indicators.get('ema_26', 0),
                indicators.get('ema_20', 0),
                indicators.get('ema_50', 0),
                indicators.get('bb_percent', 0.5),
                indicators.get('volume_ratio', 1.0),
                indicators.get('price_position', 0.5),
                indicators.get('volatility_ratio', 1.0),
                indicators.get('trend_strength', 0),
                indicators.get('volume_momentum', 0),
                indicators.get('sr_proximity', 0.5),
                indicators.get('stoch_k', 50),
                indicators.get('williams_r', -50),
                indicators.get('roc_12', 0),
                indicators.get('mfi', 50),
                indicators.get('atr_14', 0)
            ])
            
            # OBV trend (simplified)
            obv_trend = 1 if indicators.get('obv', 0) > 0 else -1
            features.append(obv_trend)
            
            # Additional engineered features
            features.extend(self._engineer_additional_features(data, indicators))
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            print(f"Error preparing features: {str(e)}")
            return None
    
    def _engineer_additional_features(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> List[float]:
        """Engineer additional features"""
        try:
            additional_features = []
            
            # Price momentum
            if len(data) >= 5:
                price_momentum = (data['Close'].iloc[-1] - data['Close'].iloc[-5]) / data['Close'].iloc[-5]
                additional_features.append(price_momentum)
            else:
                additional_features.append(0)
            
            # Volume trend
            if len(data) >= 10:
                recent_volume = data['Volume'].tail(5).mean()
                older_volume = data['Volume'].tail(10).head(5).mean()
                volume_trend = (recent_volume - older_volume) / older_volume if older_volume > 0 else 0
                additional_features.append(volume_trend)
            else:
                additional_features.append(0)
            
            # Volatility trend
            if len(data) >= 20:
                recent_volatility = data['Close'].tail(10).std()
                older_volatility = data['Close'].tail(20).head(10).std()
                volatility_trend = (recent_volatility - older_volatility) / older_volatility if older_volatility > 0 else 0
                additional_features.append(volatility_trend)
            else:
                additional_features.append(0)
            
            # Support/resistance strength
            current_price = data['Close'].iloc[-1]
            recent_high = data['High'].tail(20).max()
            recent_low = data['Low'].tail(20).min()
            
            resistance_distance = (recent_high - current_price) / current_price
            support_distance = (current_price - recent_low) / current_price
            
            additional_features.extend([resistance_distance, support_distance])
            
            return additional_features
            
        except Exception as e:
            print(f"Error engineering features: {str(e)}")
            return [0] * 5  # Return default values
    
    def _train_model(self, data: pd.DataFrame, indicators: Dict[str, Any], model_type: str = 'random_forest'):
        """Train ML model with historical data"""
        try:
            # Generate training data
            training_data = self._generate_training_data(data, indicators)
            if training_data is None or len(training_data) < 50:
                print("Insufficient training data")
                return
            
            X = training_data.drop(['target'], axis=1)
            y = training_data['target']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = self.scalers[model_type]
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = self.models[model_type]
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Model {model_type} accuracy: {accuracy:.3f}")
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
    
    def _generate_training_data(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Generate training data with labels"""
        try:
            if len(data) < 100:
                return None
            
            training_rows = []
            
            # Use sliding window to create training samples
            window_size = 50
            prediction_horizon = 5
            
            for i in range(window_size, len(data) - prediction_horizon):
                # Get features for current window
                window_data = data.iloc[i-window_size:i]
                window_indicators = self._calculate_window_indicators(window_data)
                
                if window_indicators is None:
                    continue
                
                features = self._prepare_window_features(window_data, window_indicators)
                if features is None:
                    continue
                
                # Generate label based on future price movement
                current_price = data['Close'].iloc[i]
                future_price = data['Close'].iloc[i + prediction_horizon]
                
                price_change = (future_price - current_price) / current_price
                
                # Define thresholds for classification
                buy_threshold = 0.02  # 2% gain
                sell_threshold = -0.02  # 2% loss
                
                if price_change > buy_threshold:
                    label = 2  # BUY
                elif price_change < sell_threshold:
                    label = 0  # SELL
                else:
                    label = 1  # HOLD
                
                # Create feature row
                feature_dict = {}
                for j, col in enumerate(self.feature_columns[:len(features[0])]):
                    feature_dict[col] = features[0][j]
                
                feature_dict['target'] = label
                training_rows.append(feature_dict)
            
            if not training_rows:
                return None
            
            return pd.DataFrame(training_rows)
            
        except Exception as e:
            print(f"Error generating training data: {str(e)}")
            return None
    
    def _calculate_window_indicators(self, window_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Calculate indicators for a data window"""
        try:
            from indicators import TechnicalIndicators
            
            indicators = TechnicalIndicators()
            return indicators.calculate_all_indicators(window_data)
            
        except Exception as e:
            print(f"Error calculating window indicators: {str(e)}")
            return None
    
    def _prepare_window_features(self, window_data: pd.DataFrame, indicators: Dict[str, Any]) -> Optional[np.ndarray]:
        """Prepare features for a data window"""
        try:
            features = []
            
            # Use same feature preparation as main function
            features.extend([
                indicators.get('rsi_14', 50),
                indicators.get('rsi_21', 50),
                indicators.get('macd', 0),
                indicators.get('macd_signal', 0),
                indicators.get('adx', 25),
                indicators.get('ema_12', 0),
                indicators.get('ema_26', 0),
                indicators.get('ema_20', 0),
                indicators.get('ema_50', 0),
                indicators.get('bb_percent', 0.5),
                indicators.get('volume_ratio', 1.0),
                indicators.get('price_position', 0.5),
                indicators.get('volatility_ratio', 1.0),
                indicators.get('trend_strength', 0),
                indicators.get('volume_momentum', 0),
                indicators.get('sr_proximity', 0.5),
                indicators.get('stoch_k', 50),
                indicators.get('williams_r', -50),
                indicators.get('roc_12', 0),
                indicators.get('mfi', 50),
                indicators.get('atr_14', 0)
            ])
            
            # OBV trend
            obv_trend = 1 if indicators.get('obv', 0) > 0 else -1
            features.append(obv_trend)
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            print(f"Error preparing window features: {str(e)}")
            return None
    
    def _predict_signal(self, features: np.ndarray, model_type: str) -> Optional[Tuple[str, float]]:
        """Generate prediction using trained model"""
        try:
            model = self.models[model_type]
            scaler = self.scalers[model_type]
            
            # Check if model is trained
            if not hasattr(model, 'feature_importances_'):
                # Use rule-based prediction if model not trained
                return self._rule_based_prediction(features)
            
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Get prediction and probabilities
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
            
            signal = self.reverse_mapping[prediction]
            confidence = probabilities[prediction]
            
            return signal, confidence
            
        except Exception as e:
            print(f"Error predicting signal: {str(e)}")
            return self._rule_based_prediction(features)
    
    def _rule_based_prediction(self, features: np.ndarray) -> Tuple[str, float]:
        """Fallback rule-based prediction"""
        try:
            # Extract key features (assuming standard order)
            rsi = features[0][0] if len(features[0]) > 0 else 50
            macd = features[0][2] if len(features[0]) > 2 else 0
            adx = features[0][4] if len(features[0]) > 4 else 25
            ema_12 = features[0][5] if len(features[0]) > 5 else 0
            ema_26 = features[0][6] if len(features[0]) > 6 else 0
            
            score = 0
            
            # RSI signals
            if rsi > 70:
                score -= 0.3  # Overbought
            elif rsi > 50:
                score += 0.2  # Bullish
            elif rsi < 30:
                score += 0.3  # Oversold
            else:
                score -= 0.2  # Bearish
            
            # MACD signals
            if macd > 0:
                score += 0.3
            else:
                score -= 0.3
            
            # EMA trend
            if ema_12 > ema_26:
                score += 0.3
            else:
                score -= 0.3
            
            # ADX strength
            if adx > 25:
                score *= 1.2  # Strong trend
            
            # Determine signal
            if score > 0.3:
                return 'BUY', min(0.8, abs(score))
            elif score < -0.3:
                return 'SELL', min(0.8, abs(score))
            else:
                return 'HOLD', 0.6
                
        except Exception as e:
            print(f"Error in rule-based prediction: {str(e)}")
            return 'HOLD', 0.5
    
    def _calculate_ai_score(self, features: np.ndarray, indicators: Dict[str, Any]) -> float:
        """Calculate AI confidence score"""
        try:
            # Combine multiple factors for AI score
            technical_score = self._calculate_technical_score(indicators)
            volume_score = self._calculate_volume_score(indicators)
            momentum_score = self._calculate_momentum_score(indicators)
            
            # Weighted average
            ai_score = (technical_score * 0.4 + volume_score * 0.3 + momentum_score * 0.3)
            return max(0, min(100, ai_score * 100))
            
        except Exception as e:
            print(f"Error calculating AI score: {str(e)}")
            return 50.0
    
    def _calculate_technical_score(self, indicators: Dict[str, Any]) -> float:
        """Calculate technical analysis score"""
        try:
            score = 0
            
            # EMA alignment
            if indicators.get('ema_12', 0) > indicators.get('ema_26', 0):
                score += 0.25
            
            # MACD
            if indicators.get('macd', 0) > 0:
                score += 0.25
            
            # ADX
            if indicators.get('adx', 0) > 25:
                score += 0.25
            
            # RSI
            rsi = indicators.get('rsi_14', 50)
            if 40 <= rsi <= 60:
                score += 0.25
            
            return score
            
        except Exception as e:
            return 0.5
    
    def _calculate_volume_score(self, indicators: Dict[str, Any]) -> float:
        """Calculate volume-based score"""
        try:
            score = 0
            
            # Volume ratio
            vol_ratio = indicators.get('volume_ratio', 1.0)
            if vol_ratio > 1.5:
                score += 0.5
            elif vol_ratio > 1.2:
                score += 0.3
            
            # Volume momentum
            vol_momentum = indicators.get('volume_momentum', 0)
            if vol_momentum > 0:
                score += 0.3
            
            # OBV
            if indicators.get('obv', 0) > 0:
                score += 0.2
            
            return min(1.0, score)
            
        except Exception as e:
            return 0.5
    
    def _calculate_momentum_score(self, indicators: Dict[str, Any]) -> float:
        """Calculate momentum score"""
        try:
            score = 0
            
            # RSI momentum
            rsi = indicators.get('rsi_14', 50)
            if 50 < rsi < 70:
                score += 0.3
            elif 30 < rsi < 50:
                score += 0.1
            
            # ROC
            roc = indicators.get('roc_12', 0)
            if roc > 0:
                score += 0.3
            
            # Stochastic
            stoch = indicators.get('stoch_k', 50)
            if 50 < stoch < 80:
                score += 0.2
            
            # Trend strength
            trend_strength = indicators.get('trend_strength', 0)
            if trend_strength > 0:
                score += 0.2
            
            return min(1.0, score)
            
        except Exception as e:
            return 0.5
    
    def _calculate_risk_level(self, features: np.ndarray, indicators: Dict[str, Any]) -> str:
        """Calculate risk level for the signal"""
        try:
            risk_score = 0
            
            # Volatility
            volatility = indicators.get('atr_14', 0)
            if volatility > 5:
                risk_score += 2
            elif volatility > 3:
                risk_score += 1
            
            # RSI extremes
            rsi = indicators.get('rsi_14', 50)
            if rsi > 80 or rsi < 20:
                risk_score += 2
            elif rsi > 70 or rsi < 30:
                risk_score += 1
            
            # Volume surge
            vol_ratio = indicators.get('volume_ratio', 1.0)
            if vol_ratio > 3:
                risk_score += 1
            
            # Determine risk level
            if risk_score >= 4:
                return 'HIGH'
            elif risk_score >= 2:
                return 'MEDIUM'
            else:
                return 'LOW'
                
        except Exception as e:
            return 'MEDIUM'
    
    def _generate_recommendation(self, signal: str, confidence: float, indicators: Dict[str, Any]) -> str:
        """Generate trading recommendation"""
        try:
            if signal == 'BUY' and confidence > 0.8:
                return "Strong BUY - High confidence AI signal"
            elif signal == 'BUY' and confidence > 0.6:
                return "BUY - Good AI signal with decent confidence"
            elif signal == 'SELL' and confidence > 0.8:
                return "Strong SELL - High confidence AI signal"
            elif signal == 'SELL' and confidence > 0.6:
                return "SELL - Good AI signal with decent confidence"
            else:
                return "HOLD - Wait for better setup"
                
        except Exception as e:
            return "HOLD - Analysis inconclusive"
    
    def get_model_performance(self, model_type: str = 'random_forest') -> Dict[str, Any]:
        """Get model performance metrics"""
        try:
            model = self.models.get(model_type)
            
            if not model or not hasattr(model, 'feature_importances_'):
                return {'status': 'not_trained', 'accuracy': 0, 'features': 0}
            
            # Get feature importance
            feature_importance = dict(zip(self.feature_columns, model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                'status': 'trained',
                'n_estimators': getattr(model, 'n_estimators', 0),
                'max_depth': getattr(model, 'max_depth', 0),
                'top_features': top_features,
                'feature_count': len(self.feature_columns)
            }
            
        except Exception as e:
            print(f"Error getting model performance: {str(e)}")
            return {'status': 'error', 'accuracy': 0, 'features': 0}
