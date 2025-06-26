"""
Professional chart utilities for trading dashboard
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import ta

class ChartUtils:
    def __init__(self):
        self.default_colors = {
            'candlestick_up': '#00ff88',
            'candlestick_down': '#ff4444',
            'volume_up': '#26a69a',
            'volume_down': '#ef5350',
            'ema_20': '#ff9800',
            'ema_50': '#e91e63',
            'vwap': '#00bcd4',
            'support': '#2196f3',
            'resistance': '#f44336',
            'fibonacci': '#9c27b0'
        }
    
    def create_comprehensive_chart(self, data: pd.DataFrame, indicators: Dict[str, Any], 
                                 symbol: str, **kwargs) -> go.Figure:
        """Create comprehensive trading chart with all features"""
        try:
            if data.empty:
                return self._create_empty_chart(symbol)
            
            # Create subplots
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=(f'{symbol} - Price Action', 'Volume', 'RSI', 'MACD'),
                row_heights=[0.5, 0.2, 0.15, 0.15]
            )
            
            # Add candlestick chart
            self._add_candlestick(fig, data, row=1)
            
            # Add technical indicators
            if kwargs.get('show_indicators', True):
                self._add_moving_averages(fig, data, indicators, row=1)
                self._add_vwap(fig, data, indicators, row=1)
                self._add_bollinger_bands(fig, data, indicators, row=1)
            
            # Add support/resistance levels
            if kwargs.get('show_support_resistance', True):
                self._add_support_resistance_levels(fig, data, row=1)
            
            # Add Fibonacci levels
            if kwargs.get('show_fibonacci', True):
                self._add_fibonacci_levels(fig, data, row=1)
            
            # Add volume
            if kwargs.get('show_volume', True):
                self._add_volume(fig, data, row=2)
            
            # Add RSI
            self._add_rsi(fig, data, indicators, row=3)
            
            # Add MACD
            self._add_macd(fig, data, indicators, row=4)
            
            # Update layout
            self._update_chart_layout(fig, symbol)
            
            return fig
            
        except Exception as e:
            print(f"Error creating comprehensive chart: {str(e)}")
            return self._create_error_chart(symbol, str(e))
    
    def create_simple_chart(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """Create simple candlestick chart"""
        try:
            if data.empty:
                return self._create_empty_chart(symbol)
            
            fig = go.Figure()
            
            # Add candlestick
            fig.add_trace(go.Candlestick(
                x=data['Datetime'] if 'Datetime' in data.columns else data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name=symbol,
                increasing_line_color=self.default_colors['candlestick_up'],
                decreasing_line_color=self.default_colors['candlestick_down']
            ))
            
            fig.update_layout(
                title=f'{symbol} - Price Chart',
                xaxis_title='Time',
                yaxis_title='Price (₹)',
                height=400
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating simple chart: {str(e)}")
            return self._create_error_chart(symbol, str(e))
    
    def create_scanner_chart(self, data: pd.DataFrame, symbol: str, 
                           signal_data: Dict[str, Any]) -> go.Figure:
        """Create chart for scanner results with signal indicators"""
        try:
            if data.empty:
                return self._create_empty_chart(symbol)
            
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=(f'{symbol} - {signal_data.get("signal", "HOLD")} Signal', 'Volume'),
                row_heights=[0.7, 0.3]
            )
            
            # Add candlestick
            self._add_candlestick(fig, data, row=1)
            
            # Add signal markers
            self._add_signal_markers(fig, data, signal_data, row=1)
            
            # Add volume
            self._add_volume(fig, data, row=2)
            
            # Update layout for scanner
            fig.update_layout(
                height=300,
                title_text=f'{symbol} - {signal_data.get("signal", "HOLD")} Signal',
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating scanner chart: {str(e)}")
            return self._create_error_chart(symbol, str(e))
    
    def _add_candlestick(self, fig: go.Figure, data: pd.DataFrame, row: int = 1):
        """Add candlestick trace to figure"""
        try:
            fig.add_trace(
                go.Candlestick(
                    x=data['Datetime'] if 'Datetime' in data.columns else data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price',
                    increasing_line_color=self.default_colors['candlestick_up'],
                    decreasing_line_color=self.default_colors['candlestick_down']
                ),
                row=row, col=1
            )
        except Exception as e:
            print(f"Error adding candlestick: {str(e)}")
    
    def _add_moving_averages(self, fig: go.Figure, data: pd.DataFrame, 
                           indicators: Dict[str, Any], row: int = 1):
        """Add moving averages to chart"""
        try:
            # EMA 20
            if len(data) >= 20:
                ema_20 = ta.trend.ema_indicator(data['Close'], window=20)
                fig.add_trace(
                    go.Scatter(
                        x=data['Datetime'] if 'Datetime' in data.columns else data.index,
                        y=ema_20,
                        mode='lines',
                        name='EMA 20',
                        line=dict(color=self.default_colors['ema_20'], width=2)
                    ),
                    row=row, col=1
                )
            
            # EMA 50
            if len(data) >= 50:
                ema_50 = ta.trend.ema_indicator(data['Close'], window=50)
                fig.add_trace(
                    go.Scatter(
                        x=data['Datetime'] if 'Datetime' in data.columns else data.index,
                        y=ema_50,
                        mode='lines',
                        name='EMA 50',
                        line=dict(color=self.default_colors['ema_50'], width=2)
                    ),
                    row=row, col=1
                )
        except Exception as e:
            print(f"Error adding moving averages: {str(e)}")
    
    def _add_vwap(self, fig: go.Figure, data: pd.DataFrame, 
                  indicators: Dict[str, Any], row: int = 1):
        """Add VWAP to chart"""
        try:
            if len(data) >= 10:
                vwap = ta.volume.volume_weighted_average_price(
                    data['High'], data['Low'], data['Close'], data['Volume']
                )
                fig.add_trace(
                    go.Scatter(
                        x=data['Datetime'] if 'Datetime' in data.columns else data.index,
                        y=vwap,
                        mode='lines',
                        name='VWAP',
                        line=dict(color=self.default_colors['vwap'], width=2, dash='dash')
                    ),
                    row=row, col=1
                )
        except Exception as e:
            print(f"Error adding VWAP: {str(e)}")
    
    def _add_bollinger_bands(self, fig: go.Figure, data: pd.DataFrame, 
                           indicators: Dict[str, Any], row: int = 1):
        """Add Bollinger Bands to chart"""
        try:
            if len(data) >= 20:
                bb_high = ta.volatility.bollinger_hband(data['Close'])
                bb_low = ta.volatility.bollinger_lband(data['Close'])
                
                # Upper band
                fig.add_trace(
                    go.Scatter(
                        x=data['Datetime'] if 'Datetime' in data.columns else data.index,
                        y=bb_high,
                        mode='lines',
                        name='BB Upper',
                        line=dict(color='gray', width=1),
                        opacity=0.7
                    ),
                    row=row, col=1
                )
                
                # Lower band
                fig.add_trace(
                    go.Scatter(
                        x=data['Datetime'] if 'Datetime' in data.columns else data.index,
                        y=bb_low,
                        mode='lines',
                        name='BB Lower',
                        line=dict(color='gray', width=1),
                        fill='tonexty',
                        fillcolor='rgba(128,128,128,0.1)',
                        opacity=0.7
                    ),
                    row=row, col=1
                )
        except Exception as e:
            print(f"Error adding Bollinger Bands: {str(e)}")
    
    def _add_volume(self, fig: go.Figure, data: pd.DataFrame, row: int = 2):
        """Add volume bars to chart"""
        try:
            colors = []
            for i in range(len(data)):
                if i == 0:
                    colors.append(self.default_colors['volume_up'])
                else:
                    if data['Close'].iloc[i] >= data['Close'].iloc[i-1]:
                        colors.append(self.default_colors['volume_up'])
                    else:
                        colors.append(self.default_colors['volume_down'])
            
            fig.add_trace(
                go.Bar(
                    x=data['Datetime'] if 'Datetime' in data.columns else data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=row, col=1
            )
        except Exception as e:
            print(f"Error adding volume: {str(e)}")
    
    def _add_rsi(self, fig: go.Figure, data: pd.DataFrame, 
                 indicators: Dict[str, Any], row: int = 3):
        """Add RSI indicator"""
        try:
            if len(data) >= 14:
                rsi = ta.momentum.rsi(data['Close'], window=14)
                
                fig.add_trace(
                    go.Scatter(
                        x=data['Datetime'] if 'Datetime' in data.columns else data.index,
                        y=rsi,
                        mode='lines',
                        name='RSI',
                        line=dict(color='purple', width=2)
                    ),
                    row=row, col=1
                )
                
                # Add overbought/oversold lines
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=row, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=row, col=1)
                fig.add_hline(y=50, line_dash="dot", line_color="gray", row=row, col=1)
        except Exception as e:
            print(f"Error adding RSI: {str(e)}")
    
    def _add_macd(self, fig: go.Figure, data: pd.DataFrame, 
                  indicators: Dict[str, Any], row: int = 4):
        """Add MACD indicator"""
        try:
            if len(data) >= 26:
                macd_line = ta.trend.macd_diff(data['Close'])
                macd_signal = ta.trend.macd_signal(data['Close'])
                macd_histogram = ta.trend.macd(data['Close'])
                
                # MACD line
                fig.add_trace(
                    go.Scatter(
                        x=data['Datetime'] if 'Datetime' in data.columns else data.index,
                        y=macd_line,
                        mode='lines',
                        name='MACD',
                        line=dict(color='blue', width=2)
                    ),
                    row=row, col=1
                )
                
                # Signal line
                fig.add_trace(
                    go.Scatter(
                        x=data['Datetime'] if 'Datetime' in data.columns else data.index,
                        y=macd_signal,
                        mode='lines',
                        name='Signal',
                        line=dict(color='red', width=2)
                    ),
                    row=row, col=1
                )
                
                # Histogram
                colors = ['green' if val >= 0 else 'red' for val in macd_histogram]
                fig.add_trace(
                    go.Bar(
                        x=data['Datetime'] if 'Datetime' in data.columns else data.index,
                        y=macd_histogram,
                        name='Histogram',
                        marker_color=colors,
                        opacity=0.7
                    ),
                    row=row, col=1
                )
        except Exception as e:
            print(f"Error adding MACD: {str(e)}")
    
    def _add_support_resistance_levels(self, fig: go.Figure, data: pd.DataFrame, row: int = 1):
        """Add support and resistance levels"""
        try:
            # Simple support/resistance calculation
            if len(data) >= 20:
                recent_data = data.tail(20)
                resistance = recent_data['High'].max()
                support = recent_data['Low'].min()
                
                # Add horizontal lines
                fig.add_hline(
                    y=resistance, 
                    line_dash="dash", 
                    line_color=self.default_colors['resistance'],
                    annotation_text="Resistance",
                    row=row, col=1
                )
                
                fig.add_hline(
                    y=support, 
                    line_dash="dash", 
                    line_color=self.default_colors['support'],
                    annotation_text="Support",
                    row=row, col=1
                )
        except Exception as e:
            print(f"Error adding support/resistance: {str(e)}")
    
    def _add_fibonacci_levels(self, fig: go.Figure, data: pd.DataFrame, row: int = 1):
        """Add Fibonacci retracement levels"""
        try:
            if len(data) >= 20:
                recent_data = data.tail(20)
                high = recent_data['High'].max()
                low = recent_data['Low'].min()
                
                fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
                
                for level in fib_levels:
                    price = high - (high - low) * level
                    fig.add_hline(
                        y=price,
                        line_dash="dot",
                        line_color=self.default_colors['fibonacci'],
                        annotation_text=f"Fib {level:.1%}",
                        opacity=0.6,
                        row=row, col=1
                    )
        except Exception as e:
            print(f"Error adding Fibonacci levels: {str(e)}")
    
    def _add_signal_markers(self, fig: go.Figure, data: pd.DataFrame, 
                          signal_data: Dict[str, Any], row: int = 1):
        """Add buy/sell signal markers"""
        try:
            signal = signal_data.get('signal', 'HOLD')
            price = signal_data.get('price', 0)
            
            if signal in ['BUY', 'SELL'] and price > 0:
                color = 'green' if signal == 'BUY' else 'red'
                symbol_marker = 'triangle-up' if signal == 'BUY' else 'triangle-down'
                
                fig.add_trace(
                    go.Scatter(
                        x=[data['Datetime'].iloc[-1] if 'Datetime' in data.columns else data.index[-1]],
                        y=[price],
                        mode='markers',
                        name=f'{signal} Signal',
                        marker=dict(
                            symbol=symbol_marker,
                            size=15,
                            color=color
                        )
                    ),
                    row=row, col=1
                )
        except Exception as e:
            print(f"Error adding signal markers: {str(e)}")
    
    def _update_chart_layout(self, fig: go.Figure, symbol: str):
        """Update chart layout and styling"""
        try:
            fig.update_layout(
                title=f'{symbol} - Professional Trading Analysis',
                height=800,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.01
                ),
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            # Update x-axis
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                showspikes=True,
                spikecolor="orange",
                spikethickness=1
            )
            
            # Update y-axis
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                showspikes=True,
                spikecolor="orange",
                spikethickness=1
            )
            
        except Exception as e:
            print(f"Error updating chart layout: {str(e)}")
    
    def _create_empty_chart(self, symbol: str) -> go.Figure:
        """Create empty chart with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"No data available for {symbol}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            title=f'{symbol} - No Data Available',
            height=400
        )
        return fig
    
    def _create_error_chart(self, symbol: str, error_msg: str) -> go.Figure:
        """Create error chart with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading chart for {symbol}: {error_msg}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title=f'{symbol} - Chart Error',
            height=400
        )
        return fig
    
    def create_heatmap_chart(self, data: Dict[str, float], title: str) -> go.Figure:
        """Create heatmap for sector performance"""
        try:
            if not data:
                return self._create_empty_heatmap(title)
            
            sectors = list(data.keys())
            values = list(data.values())
            
            # Create color scale based on values
            colors = []
            for val in values:
                if val > 2:
                    colors.append('darkgreen')
                elif val > 0:
                    colors.append('lightgreen')
                elif val > -2:
                    colors.append('lightcoral')
                else:
                    colors.append('darkred')
            
            fig = go.Figure(data=go.Bar(
                x=sectors,
                y=values,
                marker_color=colors,
                text=[f'{val:.1f}%' for val in values],
                textposition='auto'
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Sectors',
                yaxis_title='Change %',
                height=400
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating heatmap: {str(e)}")
            return self._create_empty_heatmap(title)
    
    def _create_empty_heatmap(self, title: str) -> go.Figure:
        """Create empty heatmap"""
        fig = go.Figure()
        fig.add_annotation(
            text="No sector data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title=title, height=400)
        return fig
