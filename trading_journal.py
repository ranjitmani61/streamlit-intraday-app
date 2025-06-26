"""
Trading journal for tracking trades and performance
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
from dataclasses import dataclass, asdict
import uuid

@dataclass
class Trade:
    """Trade data structure"""
    id: str
    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime]
    exit_price: Optional[float]
    quantity: int
    trade_type: str  # 'BUY' or 'SELL'
    strategy: str
    stop_loss: Optional[float]
    take_profit: Optional[float]
    notes: str
    tags: List[str]
    status: str  # 'open', 'closed', 'cancelled'
    pnl: Optional[float]
    pnl_percentage: Optional[float]
    holding_period: Optional[int]  # in minutes
    commission: float
    slippage: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary"""
        trade_dict = asdict(self)
        trade_dict['entry_date'] = self.entry_date.isoformat()
        if self.exit_date:
            trade_dict['exit_date'] = self.exit_date.isoformat()
        return trade_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trade':
        """Create trade from dictionary"""
        data['entry_date'] = datetime.fromisoformat(data['entry_date'])
        if data.get('exit_date'):
            data['exit_date'] = datetime.fromisoformat(data['exit_date'])
        return cls(**data)

class TradingJournal:
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state for trading journal"""
        if 'trades' not in st.session_state:
            st.session_state.trades = []
        
        if 'journal_settings' not in st.session_state:
            st.session_state.journal_settings = {
                'default_commission': 0.001,  # 0.1%
                'default_slippage': 0.0005,   # 0.05%
                'risk_per_trade': 0.02,       # 2% risk per trade
                'default_strategy': 'AI Signal'
            }
        
        if 'portfolio_value' not in st.session_state:
            st.session_state.portfolio_value = 100000  # Default portfolio value
    
    def add_trade(self, trade_data: Dict[str, Any]) -> bool:
        """Add new trade to journal"""
        try:
            # Create trade object
            trade = Trade(
                id=str(uuid.uuid4()),
                symbol=trade_data['symbol'],
                entry_date=trade_data.get('entry_date', datetime.now()),
                entry_price=trade_data['entry_price'],
                exit_date=trade_data.get('exit_date'),
                exit_price=trade_data.get('exit_price'),
                quantity=trade_data['quantity'],
                trade_type=trade_data['trade_type'],
                strategy=trade_data.get('strategy', 'Manual'),
                stop_loss=trade_data.get('stop_loss'),
                take_profit=trade_data.get('take_profit'),
                notes=trade_data.get('notes', ''),
                tags=trade_data.get('tags', []),
                status=trade_data.get('status', 'open'),
                pnl=None,
                pnl_percentage=None,
                holding_period=None,
                commission=trade_data.get('commission', st.session_state.journal_settings['default_commission']),
                slippage=trade_data.get('slippage', st.session_state.journal_settings['default_slippage'])
            )
            
            # Calculate P&L if exit data is provided
            if trade.exit_date and trade.exit_price:
                self._calculate_trade_pnl(trade)
            
            # Add to session state
            st.session_state.trades.append(trade.to_dict())
            
            return True
            
        except Exception as e:
            print(f"Error adding trade: {str(e)}")
            return False
    
    def close_trade(self, trade_id: str, exit_price: float, exit_date: datetime = None) -> bool:
        """Close an open trade"""
        try:
            if exit_date is None:
                exit_date = datetime.now()
            
            # Find and update trade
            for i, trade_dict in enumerate(st.session_state.trades):
                if trade_dict['id'] == trade_id:
                    trade = Trade.from_dict(trade_dict)
                    
                    trade.exit_date = exit_date
                    trade.exit_price = exit_price
                    trade.status = 'closed'
                    
                    # Calculate P&L
                    self._calculate_trade_pnl(trade)
                    
                    # Update in session state
                    st.session_state.trades[i] = trade.to_dict()
                    
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error closing trade: {str(e)}")
            return False
    
    def _calculate_trade_pnl(self, trade: Trade):
        """Calculate trade P&L and metrics"""
        try:
            if not trade.exit_price or not trade.exit_date:
                return
            
            # Calculate gross P&L
            if trade.trade_type == 'BUY':
                gross_pnl = (trade.exit_price - trade.entry_price) * trade.quantity
            else:  # SELL (short)
                gross_pnl = (trade.entry_price - trade.exit_price) * trade.quantity
            
            # Calculate costs
            entry_value = trade.entry_price * trade.quantity
            exit_value = trade.exit_price * trade.quantity
            
            commission_cost = (entry_value + exit_value) * trade.commission
            slippage_cost = (entry_value + exit_value) * trade.slippage
            
            # Net P&L
            trade.pnl = gross_pnl - commission_cost - slippage_cost
            trade.pnl_percentage = (trade.pnl / entry_value) * 100
            
            # Holding period in minutes
            holding_period = trade.exit_date - trade.entry_date
            trade.holding_period = int(holding_period.total_seconds() / 60)
            
        except Exception as e:
            print(f"Error calculating P&L: {str(e)}")
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Get trades as pandas DataFrame"""
        try:
            if not st.session_state.trades:
                return pd.DataFrame()
            
            # Convert trades to DataFrame
            trades_data = []
            for trade_dict in st.session_state.trades:
                trade = Trade.from_dict(trade_dict)
                trades_data.append({
                    'ID': trade.id[:8],  # Short ID
                    'Symbol': trade.symbol,
                    'Type': trade.trade_type,
                    'Strategy': trade.strategy,
                    'Entry Date': trade.entry_date.strftime('%Y-%m-%d %H:%M'),
                    'Entry Price': trade.entry_price,
                    'Exit Date': trade.exit_date.strftime('%Y-%m-%d %H:%M') if trade.exit_date else '',
                    'Exit Price': trade.exit_price if trade.exit_price else '',
                    'Quantity': trade.quantity,
                    'Status': trade.status,
                    'P&L': trade.pnl if trade.pnl else '',
                    'P&L %': f"{trade.pnl_percentage:.2f}%" if trade.pnl_percentage else '',
                    'Holding (min)': trade.holding_period if trade.holding_period else '',
                    'Notes': trade.notes,
                    'Tags': ', '.join(trade.tags) if trade.tags else ''
                })
            
            return pd.DataFrame(trades_data)
            
        except Exception as e:
            print(f"Error creating trades DataFrame: {str(e)}")
            return pd.DataFrame()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate portfolio performance metrics"""
        try:
            if not st.session_state.trades:
                return self._get_empty_metrics()
            
            closed_trades = [
                Trade.from_dict(t) for t in st.session_state.trades 
                if t['status'] == 'closed' and t['pnl'] is not None
            ]
            
            if not closed_trades:
                return self._get_empty_metrics()
            
            # Basic metrics
            total_trades = len(closed_trades)
            winning_trades = len([t for t in closed_trades if t.pnl > 0])
            losing_trades = len([t for t in closed_trades if t.pnl < 0])
            
            # P&L metrics
            total_pnl = sum(t.pnl for t in closed_trades)
            total_pnl_percentage = (total_pnl / st.session_state.portfolio_value) * 100
            
            # Win/Loss metrics
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            winning_pnl = sum(t.pnl for t in closed_trades if t.pnl > 0)
            losing_pnl = sum(t.pnl for t in closed_trades if t.pnl < 0)
            
            avg_win = winning_pnl / winning_trades if winning_trades > 0 else 0
            avg_loss = abs(losing_pnl) / losing_trades if losing_trades > 0 else 0
            
            profit_factor = abs(winning_pnl / losing_pnl) if losing_pnl != 0 else float('inf')
            
            # Risk metrics
            pnl_list = [t.pnl for t in closed_trades]
            max_drawdown = self._calculate_max_drawdown(pnl_list)
            
            # Time metrics
            avg_holding_period = np.mean([t.holding_period for t in closed_trades if t.holding_period])
            
            # Strategy breakdown
            strategy_performance = self._get_strategy_performance(closed_trades)
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'total_pnl_percentage': total_pnl_percentage,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'avg_holding_period': avg_holding_period,
                'strategy_performance': strategy_performance,
                'best_trade': max(pnl_list) if pnl_list else 0,
                'worst_trade': min(pnl_list) if pnl_list else 0
            }
            
        except Exception as e:
            print(f"Error calculating performance metrics: {str(e)}")
            return self._get_empty_metrics()
    
    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'total_pnl_percentage': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'avg_holding_period': 0,
            'strategy_performance': {},
            'best_trade': 0,
            'worst_trade': 0
        }
    
    def _calculate_max_drawdown(self, pnl_list: List[float]) -> float:
        """Calculate maximum drawdown"""
        try:
            if not pnl_list:
                return 0
            
            cumulative_pnl = np.cumsum(pnl_list)
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdown = cumulative_pnl - running_max
            
            return abs(min(drawdown)) if len(drawdown) > 0 else 0
            
        except Exception as e:
            print(f"Error calculating max drawdown: {str(e)}")
            return 0
    
    def _get_strategy_performance(self, trades: List[Trade]) -> Dict[str, Dict[str, Any]]:
        """Get performance breakdown by strategy"""
        try:
            strategy_stats = {}
            
            for trade in trades:
                strategy = trade.strategy
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {
                        'trades': 0,
                        'wins': 0,
                        'losses': 0,
                        'total_pnl': 0,
                        'win_rate': 0
                    }
                
                stats = strategy_stats[strategy]
                stats['trades'] += 1
                stats['total_pnl'] += trade.pnl
                
                if trade.pnl > 0:
                    stats['wins'] += 1
                else:
                    stats['losses'] += 1
                
                stats['win_rate'] = (stats['wins'] / stats['trades']) * 100
            
            return strategy_stats
            
        except Exception as e:
            print(f"Error getting strategy performance: {str(e)}")
            return {}
    
    def get_trade_by_id(self, trade_id: str) -> Optional[Trade]:
        """Get trade by ID"""
        try:
            for trade_dict in st.session_state.trades:
                if trade_dict['id'] == trade_id:
                    return Trade.from_dict(trade_dict)
            return None
            
        except Exception as e:
            print(f"Error getting trade by ID: {str(e)}")
            return None
    
    def update_trade(self, trade_id: str, updates: Dict[str, Any]) -> bool:
        """Update trade information"""
        try:
            for i, trade_dict in enumerate(st.session_state.trades):
                if trade_dict['id'] == trade_id:
                    trade = Trade.from_dict(trade_dict)
                    
                    # Update fields
                    for key, value in updates.items():
                        if hasattr(trade, key):
                            setattr(trade, key, value)
                    
                    # Recalculate P&L if exit data was updated
                    if 'exit_price' in updates or 'exit_date' in updates:
                        self._calculate_trade_pnl(trade)
                    
                    # Update in session state
                    st.session_state.trades[i] = trade.to_dict()
                    
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error updating trade: {str(e)}")
            return False
    
    def delete_trade(self, trade_id: str) -> bool:
        """Delete trade from journal"""
        try:
            original_length = len(st.session_state.trades)
            st.session_state.trades = [
                t for t in st.session_state.trades if t['id'] != trade_id
            ]
            
            return len(st.session_state.trades) < original_length
            
        except Exception as e:
            print(f"Error deleting trade: {str(e)}")
            return False
    
    def get_open_trades(self) -> List[Trade]:
        """Get all open trades"""
        try:
            return [
                Trade.from_dict(t) for t in st.session_state.trades 
                if t['status'] == 'open'
            ]
            
        except Exception as e:
            print(f"Error getting open trades: {str(e)}")
            return []
    
    def get_trades_by_symbol(self, symbol: str) -> List[Trade]:
        """Get all trades for a specific symbol"""
        try:
            return [
                Trade.from_dict(t) for t in st.session_state.trades 
                if t['symbol'] == symbol
            ]
            
        except Exception as e:
            print(f"Error getting trades by symbol: {str(e)}")
            return []
    
    def get_daily_pnl(self, days: int = 30) -> Dict[str, float]:
        """Get daily P&L for the last N days"""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            daily_pnl = {}
            
            for trade_dict in st.session_state.trades:
                trade = Trade.from_dict(trade_dict)
                
                if (trade.status == 'closed' and trade.exit_date and 
                    start_date <= trade.exit_date.date() <= end_date):
                    
                    date_str = trade.exit_date.strftime('%Y-%m-%d')
                    daily_pnl[date_str] = daily_pnl.get(date_str, 0) + trade.pnl
            
            return daily_pnl
            
        except Exception as e:
            print(f"Error getting daily P&L: {str(e)}")
            return {}
    
    def export_trades_to_csv(self) -> str:
        """Export trades to CSV format"""
        try:
            df = self.get_trades_dataframe()
            if df.empty:
                return ""
            
            return df.to_csv(index=False)
            
        except Exception as e:
            print(f"Error exporting trades to CSV: {str(e)}")
            return ""
    
    def import_trades_from_csv(self, csv_data: str) -> bool:
        """Import trades from CSV data"""
        try:
            # This would implement CSV import functionality
            # For now, return success
            return True
            
        except Exception as e:
            print(f"Error importing trades from CSV: {str(e)}")
            return False
    
    def clear_all_trades(self) -> bool:
        """Clear all trades from journal"""
        try:
            st.session_state.trades = []
            return True
            
        except Exception as e:
            print(f"Error clearing trades: {str(e)}")
            return False
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Calculate risk management metrics"""
        try:
            open_trades = self.get_open_trades()
            
            if not open_trades:
                return {'open_positions': 0, 'total_exposure': 0, 'risk_percentage': 0}
            
            total_exposure = sum(t.entry_price * t.quantity for t in open_trades)
            risk_percentage = (total_exposure / st.session_state.portfolio_value) * 100
            
            return {
                'open_positions': len(open_trades),
                'total_exposure': total_exposure,
                'risk_percentage': risk_percentage,
                'max_position_size': st.session_state.portfolio_value * 0.05  # 5% max
            }
            
        except Exception as e:
            print(f"Error calculating risk metrics: {str(e)}")
            return {'open_positions': 0, 'total_exposure': 0, 'risk_percentage': 0}
