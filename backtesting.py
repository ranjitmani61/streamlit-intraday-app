"""
Comprehensive backtesting engine for trading strategies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

class BacktestEngine:
    def __init__(self):
        self.results = {}
        self.trades = []
        self.portfolio_history = []
        
        # Default configuration
        self.config = {
            'initial_capital': 100000,
            'commission': 0.001,  # 0.1%
            'slippage': 0.0005,   # 0.05%
            'position_size': 0.05,  # 5% per position
            'stop_loss': 0.02,    # 2%
            'take_profit': 0.04,  # 4%
            'max_positions': 10
        }
    
    def run_backtest(self, symbol: str, period: str = '1y', strategy: str = 'ADX+RSI', 
                    parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run comprehensive backtest"""
        try:
            print(f"Running backtest for {symbol} using {strategy} strategy...")
            
            # Get historical data
            from data_fetcher import DataFetcher
            data_fetcher = DataFetcher()
            
            data = data_fetcher.get_historical_data(symbol, period=period, interval='1h')
            if data is None or len(data) < 100:
                return {'error': 'Insufficient data for backtesting'}
            
            # Initialize backtest
            self._initialize_backtest(parameters or {})
            
            # Run strategy
            if strategy == 'ADX+RSI':
                signals = self._adx_rsi_strategy(data, parameters or {})
            elif strategy == 'MACD+EMA':
                signals = self._macd_ema_strategy(data, parameters or {})
            elif strategy == 'Custom':
                signals = self._custom_strategy(data, parameters or {})
            else:
                signals = self._adx_rsi_strategy(data, parameters or {})
            
            # Execute trades
            portfolio_value = self._execute_backtest_trades(data, signals)
            
            # Calculate performance metrics
            performance = self._calculate_performance_metrics(portfolio_value, data)
            
            # Generate detailed results
            results = {
                'symbol': symbol,
                'strategy': strategy,
                'period': period,
                'start_date': data['Datetime'].iloc[0].strftime('%Y-%m-%d'),
                'end_date': data['Datetime'].iloc[-1].strftime('%Y-%m-%d'),
                'initial_capital': self.config['initial_capital'],
                'final_capital': portfolio_value[-1] if portfolio_value else self.config['initial_capital'],
                'total_return': performance['total_return'],
                'annualized_return': performance['annualized_return'],
                'max_drawdown': performance['max_drawdown'],
                'sharpe_ratio': performance['sharpe_ratio'],
                'win_rate': performance['win_rate'],
                'profit_factor': performance['profit_factor'],
                'total_trades': len(self.trades),
                'winning_trades': len([t for t in self.trades if t['pnl'] > 0]),
                'losing_trades': len([t for t in self.trades if t['pnl'] < 0]),
                'avg_trade_duration': performance['avg_trade_duration'],
                'portfolio_history': portfolio_value,
                'trades': self.trades,
                'equity_curve': self._create_equity_curve(portfolio_value, data),
                'monthly_returns': self._calculate_monthly_returns(portfolio_value, data),
                'drawdown_periods': self._calculate_drawdown_periods(portfolio_value),
                'trade_analysis': self._analyze_trades()
            }
            
            self.results = results
            return results
            
        except Exception as e:
            print(f"Error running backtest: {str(e)}")
            return {'error': str(e)}
    
    def _initialize_backtest(self, parameters: Dict[str, Any]):
        """Initialize backtest with parameters"""
        try:
            # Update configuration with parameters
            self.config.update(parameters)
            
            # Reset state
            self.trades = []
            self.portfolio_history = []
            
        except Exception as e:
            print(f"Error initializing backtest: {str(e)}")
    
    def _adx_rsi_strategy(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
        """ADX + RSI strategy implementation"""
        try:
            from indicators import TechnicalIndicators
            
            # Default parameters
            adx_threshold = parameters.get('adx_threshold', 25)
            rsi_oversold = parameters.get('rsi_oversold', 30)
            rsi_overbought = parameters.get('rsi_overbought', 70)
            ema_fast = parameters.get('ema_fast', 12)
            ema_slow = parameters.get('ema_slow', 26)
            
            signals = []
            indicators = TechnicalIndicators()
            
            # Calculate indicators for each row
            for i in range(50, len(data)):  # Start after enough data for indicators
                window_data = data.iloc[max(0, i-50):i+1]
                tech_indicators = indicators.calculate_all_indicators(window_data)
                
                # Get current values
                current_price = data['Close'].iloc[i]
                adx = tech_indicators.get('adx', 0)
                rsi = tech_indicators.get('rsi_14', 50)
                ema_12 = tech_indicators.get('ema_12', current_price)
                ema_26 = tech_indicators.get('ema_26', current_price)
                
                # Signal generation logic
                signal = 'HOLD'
                
                # BUY conditions
                if (adx > adx_threshold and 
                    rsi < rsi_oversold and 
                    ema_12 > ema_26 and 
                    current_price > ema_12):
                    signal = 'BUY'
                
                # SELL conditions
                elif (adx > adx_threshold and 
                      rsi > rsi_overbought and 
                      ema_12 < ema_26 and 
                      current_price < ema_12):
                    signal = 'SELL'
                
                signals.append({
                    'timestamp': data['Datetime'].iloc[i],
                    'price': current_price,
                    'signal': signal,
                    'adx': adx,
                    'rsi': rsi,
                    'ema_12': ema_12,
                    'ema_26': ema_26
                })
            
            return pd.DataFrame(signals)
            
        except Exception as e:
            print(f"Error in ADX+RSI strategy: {str(e)}")
            return pd.DataFrame()
    
    def _macd_ema_strategy(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
        """MACD + EMA strategy implementation"""
        try:
            from indicators import TechnicalIndicators
            
            # Default parameters
            ema_period = parameters.get('ema_period', 20)
            macd_fast = parameters.get('macd_fast', 12)
            macd_slow = parameters.get('macd_slow', 26)
            macd_signal = parameters.get('macd_signal', 9)
            
            signals = []
            indicators = TechnicalIndicators()
            
            for i in range(50, len(data)):
                window_data = data.iloc[max(0, i-50):i+1]
                tech_indicators = indicators.calculate_all_indicators(window_data)
                
                current_price = data['Close'].iloc[i]
                ema_20 = tech_indicators.get('ema_20', current_price)
                macd = tech_indicators.get('macd', 0)
                macd_sig = tech_indicators.get('macd_signal', 0)
                
                signal = 'HOLD'
                
                # BUY: MACD above signal and price above EMA
                if macd > macd_sig and current_price > ema_20:
                    signal = 'BUY'
                
                # SELL: MACD below signal and price below EMA
                elif macd < macd_sig and current_price < ema_20:
                    signal = 'SELL'
                
                signals.append({
                    'timestamp': data['Datetime'].iloc[i],
                    'price': current_price,
                    'signal': signal,
                    'macd': macd,
                    'macd_signal': macd_sig,
                    'ema_20': ema_20
                })
            
            return pd.DataFrame(signals)
            
        except Exception as e:
            print(f"Error in MACD+EMA strategy: {str(e)}")
            return pd.DataFrame()
    
    def _custom_strategy(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
        """Custom strategy implementation"""
        try:
            # Implement custom strategy logic here
            # For now, return simple moving average crossover
            
            fast_ma = parameters.get('fast_ma', 10)
            slow_ma = parameters.get('slow_ma', 20)
            
            signals = []
            
            for i in range(slow_ma, len(data)):
                current_price = data['Close'].iloc[i]
                
                fast_avg = data['Close'].iloc[i-fast_ma:i].mean()
                slow_avg = data['Close'].iloc[i-slow_ma:i].mean()
                
                signal = 'HOLD'
                
                if fast_avg > slow_avg:
                    signal = 'BUY'
                elif fast_avg < slow_avg:
                    signal = 'SELL'
                
                signals.append({
                    'timestamp': data['Datetime'].iloc[i],
                    'price': current_price,
                    'signal': signal,
                    'fast_ma': fast_avg,
                    'slow_ma': slow_avg
                })
            
            return pd.DataFrame(signals)
            
        except Exception as e:
            print(f"Error in custom strategy: {str(e)}")
            return pd.DataFrame()
    
    def _execute_backtest_trades(self, data: pd.DataFrame, signals: pd.DataFrame) -> List[float]:
        """Execute trades based on signals"""
        try:
            portfolio_value = [self.config['initial_capital']]
            current_capital = self.config['initial_capital']
            positions = {}  # symbol -> {'quantity': int, 'entry_price': float, 'entry_time': datetime}
            
            for _, signal_row in signals.iterrows():
                signal = signal_row['signal']
                price = signal_row['price']
                timestamp = signal_row['timestamp']
                
                if signal == 'BUY' and len(positions) < self.config['max_positions']:
                    # Enter long position
                    position_value = current_capital * self.config['position_size']
                    quantity = int(position_value / price)
                    
                    if quantity > 0:
                        # Apply slippage and commission
                        entry_price = price * (1 + self.config['slippage'])
                        commission = position_value * self.config['commission']
                        
                        positions[f"long_{timestamp}"] = {
                            'type': 'long',
                            'quantity': quantity,
                            'entry_price': entry_price,
                            'entry_time': timestamp,
                            'stop_loss': entry_price * (1 - self.config['stop_loss']),
                            'take_profit': entry_price * (1 + self.config['take_profit'])
                        }
                        
                        current_capital -= (quantity * entry_price + commission)
                
                elif signal == 'SELL':
                    # Close long positions or enter short
                    for pos_id, position in list(positions.items()):
                        if position['type'] == 'long':
                            # Close long position
                            exit_price = price * (1 - self.config['slippage'])
                            commission = position['quantity'] * exit_price * self.config['commission']
                            
                            pnl = (exit_price - position['entry_price']) * position['quantity'] - commission
                            current_capital += position['quantity'] * exit_price
                            
                            # Record trade
                            self.trades.append({
                                'entry_time': position['entry_time'],
                                'exit_time': timestamp,
                                'entry_price': position['entry_price'],
                                'exit_price': exit_price,
                                'quantity': position['quantity'],
                                'type': 'long',
                                'pnl': pnl,
                                'duration': (timestamp - position['entry_time']).total_seconds() / 3600  # hours
                            })
                            
                            del positions[pos_id]
                
                # Check stop loss and take profit
                for pos_id, position in list(positions.items()):
                    if position['type'] == 'long':
                        if price <= position['stop_loss'] or price >= position['take_profit']:
                            # Exit position
                            exit_price = price * (1 - self.config['slippage'])
                            commission = position['quantity'] * exit_price * self.config['commission']
                            
                            pnl = (exit_price - position['entry_price']) * position['quantity'] - commission
                            current_capital += position['quantity'] * exit_price
                            
                            # Record trade
                            exit_reason = 'stop_loss' if price <= position['stop_loss'] else 'take_profit'
                            self.trades.append({
                                'entry_time': position['entry_time'],
                                'exit_time': timestamp,
                                'entry_price': position['entry_price'],
                                'exit_price': exit_price,
                                'quantity': position['quantity'],
                                'type': 'long',
                                'pnl': pnl,
                                'duration': (timestamp - position['entry_time']).total_seconds() / 3600,
                                'exit_reason': exit_reason
                            })
                            
                            del positions[pos_id]
                
                # Calculate current portfolio value (cash + open positions)
                open_positions_value = sum(
                    pos['quantity'] * price for pos in positions.values()
                )
                total_value = current_capital + open_positions_value
                portfolio_value.append(total_value)
            
            return portfolio_value
            
        except Exception as e:
            print(f"Error executing backtest trades: {str(e)}")
            return [self.config['initial_capital']]
    
    def _calculate_performance_metrics(self, portfolio_values: List[float], data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        try:
            if len(portfolio_values) < 2:
                return self._get_empty_performance()
            
            initial_value = portfolio_values[0]
            final_value = portfolio_values[-1]
            
            # Basic returns
            total_return = ((final_value - initial_value) / initial_value) * 100
            
            # Annualized return
            days = len(portfolio_values)
            years = days / 365.25
            annualized_return = (((final_value / initial_value) ** (1/years)) - 1) * 100 if years > 0 else 0
            
            # Volatility and Sharpe ratio
            returns = pd.Series(portfolio_values).pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            risk_free_rate = 0.06  # 6% annual risk-free rate
            excess_returns = returns.mean() * 252 - risk_free_rate
            sharpe_ratio = excess_returns / volatility if volatility != 0 else 0
            
            # Maximum drawdown
            max_drawdown = self._calculate_max_drawdown(portfolio_values)
            
            # Trade-based metrics
            if self.trades:
                winning_trades = [t for t in self.trades if t['pnl'] > 0]
                losing_trades = [t for t in self.trades if t['pnl'] < 0]
                
                win_rate = (len(winning_trades) / len(self.trades)) * 100
                
                avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
                avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
                
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
                
                avg_trade_duration = np.mean([t['duration'] for t in self.trades])
            else:
                win_rate = 0
                profit_factor = 0
                avg_trade_duration = 0
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility * 100,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_trade_duration': avg_trade_duration
            }
            
        except Exception as e:
            print(f"Error calculating performance metrics: {str(e)}")
            return self._get_empty_performance()
    
    def _get_empty_performance(self) -> Dict[str, float]:
        """Return empty performance metrics"""
        return {
            'total_return': 0,
            'annualized_return': 0,
            'volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_trade_duration': 0
        }
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        try:
            if len(portfolio_values) < 2:
                return 0
            
            values = pd.Series(portfolio_values)
            rolling_max = values.expanding().max()
            drawdown = (values - rolling_max) / rolling_max * 100
            
            return abs(drawdown.min())
            
        except Exception as e:
            print(f"Error calculating max drawdown: {str(e)}")
            return 0
    
    def _create_equity_curve(self, portfolio_values: List[float], data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create equity curve data"""
        try:
            equity_curve = []
            
            for i, value in enumerate(portfolio_values):
                if i < len(data):
                    equity_curve.append({
                        'date': data['Datetime'].iloc[i].strftime('%Y-%m-%d'),
                        'portfolio_value': value,
                        'return': ((value - portfolio_values[0]) / portfolio_values[0]) * 100
                    })
            
            return equity_curve
            
        except Exception as e:
            print(f"Error creating equity curve: {str(e)}")
            return []
    
    def _calculate_monthly_returns(self, portfolio_values: List[float], data: pd.DataFrame) -> Dict[str, float]:
        """Calculate monthly returns"""
        try:
            if len(portfolio_values) != len(data):
                return {}
            
            df = pd.DataFrame({
                'date': [data['Datetime'].iloc[i] for i in range(len(portfolio_values))],
                'value': portfolio_values
            })
            
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Resample to monthly
            monthly_values = df.resample('M').last()
            monthly_returns = monthly_values.pct_change().dropna()
            
            return {
                index.strftime('%Y-%m'): value * 100 
                for index, value in monthly_returns['value'].items()
            }
            
        except Exception as e:
            print(f"Error calculating monthly returns: {str(e)}")
            return {}
    
    def _calculate_drawdown_periods(self, portfolio_values: List[float]) -> List[Dict[str, Any]]:
        """Calculate drawdown periods"""
        try:
            if len(portfolio_values) < 2:
                return []
            
            values = pd.Series(portfolio_values)
            rolling_max = values.expanding().max()
            drawdown = (values - rolling_max) / rolling_max * 100
            
            # Find drawdown periods
            drawdown_periods = []
            in_drawdown = False
            start_idx = 0
            
            for i, dd in enumerate(drawdown):
                if dd < -1 and not in_drawdown:  # Start of drawdown (>1%)
                    in_drawdown = True
                    start_idx = i
                elif dd >= -0.1 and in_drawdown:  # End of drawdown
                    in_drawdown = False
                    max_dd = drawdown[start_idx:i+1].min()
                    duration = i - start_idx
                    
                    drawdown_periods.append({
                        'start': start_idx,
                        'end': i,
                        'duration': duration,
                        'max_drawdown': abs(max_dd)
                    })
            
            return drawdown_periods
            
        except Exception as e:
            print(f"Error calculating drawdown periods: {str(e)}")
            return []
    
    def _analyze_trades(self) -> Dict[str, Any]:
        """Analyze trade patterns"""
        try:
            if not self.trades:
                return {}
            
            # Basic statistics
            total_trades = len(self.trades)
            profitable_trades = [t for t in self.trades if t['pnl'] > 0]
            losing_trades = [t for t in self.trades if t['pnl'] < 0]
            
            # Duration analysis
            durations = [t['duration'] for t in self.trades]
            avg_duration = np.mean(durations)
            median_duration = np.median(durations)
            
            # P&L analysis
            total_pnl = sum(t['pnl'] for t in self.trades)
            best_trade = max(self.trades, key=lambda t: t['pnl'])
            worst_trade = min(self.trades, key=lambda t: t['pnl'])
            
            # Consecutive wins/losses
            consecutive_wins = self._calculate_consecutive_results(True)
            consecutive_losses = self._calculate_consecutive_results(False)
            
            return {
                'total_trades': total_trades,
                'profitable_trades': len(profitable_trades),
                'losing_trades': len(losing_trades),
                'avg_duration_hours': avg_duration,
                'median_duration_hours': median_duration,
                'total_pnl': total_pnl,
                'best_trade_pnl': best_trade['pnl'],
                'worst_trade_pnl': worst_trade['pnl'],
                'max_consecutive_wins': consecutive_wins,
                'max_consecutive_losses': consecutive_losses,
                'avg_win': np.mean([t['pnl'] for t in profitable_trades]) if profitable_trades else 0,
                'avg_loss': np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            }
            
        except Exception as e:
            print(f"Error analyzing trades: {str(e)}")
            return {}
    
    def _calculate_consecutive_results(self, wins: bool) -> int:
        """Calculate maximum consecutive wins or losses"""
        try:
            if not self.trades:
                return 0
            
            max_consecutive = 0
            current_consecutive = 0
            
            for trade in self.trades:
                is_win = trade['pnl'] > 0
                
                if (wins and is_win) or (not wins and not is_win):
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0
            
            return max_consecutive
            
        except Exception as e:
            print(f"Error calculating consecutive results: {str(e)}")
            return 0
    
    def optimize_parameters(self, symbol: str, strategy: str, param_ranges: Dict[str, List]) -> Dict[str, Any]:
        """Optimize strategy parameters"""
        try:
            best_result = None
            best_params = None
            best_sharpe = -999
            
            # Generate parameter combinations (simplified grid search)
            param_combinations = self._generate_param_combinations(param_ranges)
            
            for params in param_combinations[:20]:  # Limit to 20 combinations for performance
                result = self.run_backtest(symbol, strategy=strategy, parameters=params)
                
                if 'error' not in result:
                    sharpe = result.get('sharpe_ratio', -999)
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_result = result
                        best_params = params
            
            return {
                'best_parameters': best_params,
                'best_result': best_result,
                'optimization_runs': len(param_combinations)
            }
            
        except Exception as e:
            print(f"Error optimizing parameters: {str(e)}")
            return {'error': str(e)}
    
    def _generate_param_combinations(self, param_ranges: Dict[str, List]) -> List[Dict[str, Any]]:
        """Generate parameter combinations for optimization"""
        try:
            import itertools
            
            if not param_ranges:
                return [{}]
            
            keys = list(param_ranges.keys())
            values = list(param_ranges.values())
            
            combinations = []
            for combination in itertools.product(*values):
                param_dict = dict(zip(keys, combination))
                combinations.append(param_dict)
            
            return combinations
            
        except Exception as e:
            print(f"Error generating parameter combinations: {str(e)}")
            return [{}]
    
    def compare_strategies(self, symbol: str, strategies: List[str], period: str = '1y') -> Dict[str, Any]:
        """Compare multiple strategies"""
        try:
            results = {}
            
            for strategy in strategies:
                result = self.run_backtest(symbol, period=period, strategy=strategy)
                if 'error' not in result:
                    results[strategy] = {
                        'total_return': result['total_return'],
                        'sharpe_ratio': result['sharpe_ratio'],
                        'max_drawdown': result['max_drawdown'],
                        'win_rate': result['win_rate'],
                        'total_trades': result['total_trades']
                    }
            
            # Find best strategy
            best_strategy = max(results.keys(), key=lambda s: results[s]['sharpe_ratio']) if results else None
            
            return {
                'results': results,
                'best_strategy': best_strategy,
                'comparison_metrics': self._create_comparison_table(results)
            }
            
        except Exception as e:
            print(f"Error comparing strategies: {str(e)}")
            return {'error': str(e)}
    
    def _create_comparison_table(self, results: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """Create comparison table for strategies"""
        try:
            comparison = []
            
            for strategy, metrics in results.items():
                comparison.append({
                    'Strategy': strategy,
                    'Total Return (%)': f"{metrics['total_return']:.2f}",
                    'Sharpe Ratio': f"{metrics['sharpe_ratio']:.2f}",
                    'Max Drawdown (%)': f"{metrics['max_drawdown']:.2f}",
                    'Win Rate (%)': f"{metrics['win_rate']:.2f}",
                    'Total Trades': metrics['total_trades']
                })
            
            return comparison
            
        except Exception as e:
            print(f"Error creating comparison table: {str(e)}")
            return []
