"""
Alert management system for trading notifications
"""

import streamlit as st
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Any, Optional
import requests
import asyncio
from dataclasses import dataclass
import threading
import time

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    symbol: str
    signal: str
    price: float
    confidence: int
    timestamp: datetime
    alert_type: str  # 'signal', 'price', 'volume', 'breakout'
    message: str
    sent_channels: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'signal': self.signal,
            'price': self.price,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'alert_type': self.alert_type,
            'message': self.message,
            'sent_channels': self.sent_channels
        }

class AlertManager:
    def __init__(self):
        self.alerts_history = []
        self.alert_cooldown = {}  # To prevent spam
        self.cooldown_period = 300  # 5 minutes
        self.max_alerts_per_hour = 20
        self.telegram_bot_token = "7248457164:AAF-IAycn_9fGcJtm4IifjA68QaDPnvwivg"
        self.telegram_chat_id = "6253409461"
        
        # Initialize session state for alerts
        if 'alerts_history' not in st.session_state:
            st.session_state.alerts_history = []
        
        if 'alert_settings' not in st.session_state:
            st.session_state.alert_settings = {
                'telegram_enabled': True,
                'email_enabled': False,
                'sound_enabled': False,
                'price_alerts': True,
                'volume_alerts': True,
                'signal_alerts': True
            }
    
    def send_signal_alert(self, symbol: str, signal_data: Dict[str, Any]) -> bool:
        """Send trading signal alert"""
        try:
            # Check cooldown
            if not self._check_cooldown(symbol, 'signal'):
                return False
            
            # Create alert
            alert = Alert(
                id=f"signal_{symbol}_{int(time.time())}",
                symbol=symbol,
                signal=signal_data.get('signal', 'HOLD'),
                price=signal_data.get('price', 0),
                confidence=signal_data.get('confidence', 50),
                timestamp=datetime.now(),
                alert_type='signal',
                message=self._format_signal_message(symbol, signal_data),
                sent_channels=[]
            )
            
            # Send through enabled channels
            success = False
            
            if st.session_state.alert_settings.get('telegram_enabled', False):
                if self._send_telegram_alert(alert):
                    alert.sent_channels.append('telegram')
                    success = True
            
            if st.session_state.alert_settings.get('email_enabled', False):
                if self._send_email_alert(alert):
                    alert.sent_channels.append('email')
                    success = True
            
            if st.session_state.alert_settings.get('sound_enabled', False):
                self._play_sound_alert(alert)
                alert.sent_channels.append('sound')
                success = True
            
            # Browser notification
            self._send_browser_notification(alert)
            alert.sent_channels.append('browser')
            
            # Store alert
            self._store_alert(alert)
            
            return success
            
        except Exception as e:
            print(f"Error sending signal alert: {str(e)}")
            return False
    
    def send_price_alert(self, symbol: str, current_price: float, 
                        target_price: float, alert_type: str) -> bool:
        """Send price alert (breakout, breakdown, target hit)"""
        try:
            if not self._check_cooldown(symbol, 'price'):
                return False
            
            message = f"🎯 {alert_type.upper()} Alert for {symbol}\n"
            message += f"Current Price: ₹{current_price:.2f}\n"
            message += f"Target Price: ₹{target_price:.2f}\n"
            message += f"Time: {datetime.now().strftime('%H:%M:%S')}"
            
            alert = Alert(
                id=f"price_{symbol}_{int(time.time())}",
                symbol=symbol,
                signal=alert_type.upper(),
                price=current_price,
                confidence=90,
                timestamp=datetime.now(),
                alert_type='price',
                message=message,
                sent_channels=[]
            )
            
            return self._send_alert_through_channels(alert)
            
        except Exception as e:
            print(f"Error sending price alert: {str(e)}")
            return False
    
    def send_volume_alert(self, symbol: str, volume_data: Dict[str, Any]) -> bool:
        """Send volume surge alert"""
        try:
            if not self._check_cooldown(symbol, 'volume'):
                return False
            
            volume_ratio = volume_data.get('volume_ratio', 1.0)
            current_volume = volume_data.get('current_volume', 0)
            
            message = f"📊 Volume Surge Alert for {symbol}\n"
            message += f"Volume: {volume_ratio:.1f}x average\n"
            message += f"Current Volume: {current_volume:,}\n"
            message += f"Time: {datetime.now().strftime('%H:%M:%S')}"
            
            alert = Alert(
                id=f"volume_{symbol}_{int(time.time())}",
                symbol=symbol,
                signal='VOLUME_SURGE',
                price=volume_data.get('price', 0),
                confidence=80,
                timestamp=datetime.now(),
                alert_type='volume',
                message=message,
                sent_channels=[]
            )
            
            return self._send_alert_through_channels(alert)
            
        except Exception as e:
            print(f"Error sending volume alert: {str(e)}")
            return False
    
    def send_test_alert(self) -> bool:
        """Send test alert to verify setup"""
        try:
            test_data = {
                'signal': 'BUY',
                'price': 2500.00,
                'confidence': 85,
                'rsi': 65,
                'macd': 0.5,
                'analysis': ['Test alert - all systems working']
            }
            
            return self.send_signal_alert('RELIANCE', test_data)
            
        except Exception as e:
            print(f"Error sending test alert: {str(e)}")
            return False
    
    def _send_telegram_alert(self, alert: Alert) -> bool:
        """Send alert via Telegram"""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            
            # Format message for Telegram
            if alert.signal in ['BUY', 'SELL']:
                emoji = '🟢' if alert.signal == 'BUY' else '🔴'
                message = f"{emoji} *{alert.signal} Signal*\n\n"
                message += f"📈 *Stock:* {alert.symbol}\n"
                message += f"💰 *Price:* ₹{alert.price:.2f}\n"
                message += f"🎯 *Confidence:* {alert.confidence}%\n"
                message += f"⏰ *Time:* {alert.timestamp.strftime('%H:%M:%S')}\n\n"
                message += f"📊 *Analysis:* Professional AI Signal\n"
                message += f"🤖 *Bot:* @MyStockSentryBot"
            else:
                message = alert.message
            
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            print(f"Error sending Telegram alert: {str(e)}")
            return False
    
    def _send_email_alert(self, alert: Alert) -> bool:
        """Send alert via email"""
        try:
            # Email configuration (would be set in environment variables)
            smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
            smtp_port = int(os.getenv('SMTP_PORT', '587'))
            email_user = os.getenv('EMAIL_USER', '')
            email_password = os.getenv('EMAIL_PASSWORD', '')
            recipient_email = os.getenv('RECIPIENT_EMAIL', '')
            
            if not all([email_user, email_password, recipient_email]):
                print("Email configuration not complete")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = email_user
            msg['To'] = recipient_email
            msg['Subject'] = f"Trading Alert: {alert.symbol} - {alert.signal}"
            
            body = alert.message
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(email_user, email_password)
            text = msg.as_string()
            server.sendmail(email_user, recipient_email, text)
            server.quit()
            
            return True
            
        except Exception as e:
            print(f"Error sending email alert: {str(e)}")
            return False
    
    def _play_sound_alert(self, alert: Alert):
        """Play sound alert"""
        try:
            # This would require additional sound libraries
            # For web deployment, we'll use browser notification sounds instead
            pass
        except Exception as e:
            print(f"Error playing sound alert: {str(e)}")
    
    def _send_browser_notification(self, alert: Alert):
        """Send browser notification using Streamlit"""
        try:
            if alert.signal == 'BUY':
                st.success(f"🟢 BUY Signal: {alert.symbol} at ₹{alert.price:.2f}")
            elif alert.signal == 'SELL':
                st.error(f"🔴 SELL Signal: {alert.symbol} at ₹{alert.price:.2f}")
            else:
                st.info(f"📊 Alert: {alert.symbol} - {alert.message}")
            
        except Exception as e:
            print(f"Error sending browser notification: {str(e)}")
    
    def _send_alert_through_channels(self, alert: Alert) -> bool:
        """Send alert through all enabled channels"""
        try:
            success = False
            
            if st.session_state.alert_settings.get('telegram_enabled', False):
                if self._send_telegram_alert(alert):
                    alert.sent_channels.append('telegram')
                    success = True
            
            if st.session_state.alert_settings.get('email_enabled', False):
                if self._send_email_alert(alert):
                    alert.sent_channels.append('email')
                    success = True
            
            # Browser notification
            self._send_browser_notification(alert)
            alert.sent_channels.append('browser')
            success = True
            
            # Store alert
            self._store_alert(alert)
            
            return success
            
        except Exception as e:
            print(f"Error sending alert through channels: {str(e)}")
            return False
    
    def _format_signal_message(self, symbol: str, signal_data: Dict[str, Any]) -> str:
        """Format signal message"""
        try:
            signal = signal_data.get('signal', 'HOLD')
            price = signal_data.get('price', 0)
            confidence = signal_data.get('confidence', 50)
            rsi = signal_data.get('rsi', 50)
            macd = signal_data.get('macd', 0)
            
            if signal == 'BUY':
                message = f"🟢 BUY SIGNAL for {symbol}\n"
                message += f"💰 Price: ₹{price:.2f}\n"
                message += f"🎯 Confidence: {confidence}%\n"
                message += f"📊 RSI: {rsi:.1f}\n"
                message += f"📈 MACD: {macd:.3f}\n"
                message += f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}\n"
                message += f"✅ Action: BUY KARO AB!"
            elif signal == 'SELL':
                message = f"🔴 SELL SIGNAL for {symbol}\n"
                message += f"💰 Price: ₹{price:.2f}\n"
                message += f"🎯 Confidence: {confidence}%\n"
                message += f"📊 RSI: {rsi:.1f}\n"
                message += f"📈 MACD: {macd:.3f}\n"
                message += f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}\n"
                message += f"❌ Action: SELL KARO AB!"
            else:
                message = f"🔵 HOLD for {symbol}\n"
                message += f"💰 Price: ₹{price:.2f}\n"
                message += f"📊 Wait for better setup\n"
                message += f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}"
            
            return message
            
        except Exception as e:
            print(f"Error formatting signal message: {str(e)}")
            return f"Alert for {symbol} at {datetime.now().strftime('%H:%M:%S')}"
    
    def _check_cooldown(self, symbol: str, alert_type: str) -> bool:
        """Check if alert is within cooldown period"""
        try:
            key = f"{symbol}_{alert_type}"
            current_time = datetime.now()
            
            if key in self.alert_cooldown:
                last_alert_time = self.alert_cooldown[key]
                if (current_time - last_alert_time).seconds < self.cooldown_period:
                    return False
            
            # Check hourly limit
            hour_ago = current_time - timedelta(hours=1)
            recent_alerts = [
                alert for alert in st.session_state.alerts_history
                if alert.get('timestamp', '') > hour_ago.isoformat()
            ]
            
            if len(recent_alerts) >= self.max_alerts_per_hour:
                return False
            
            # Update cooldown
            self.alert_cooldown[key] = current_time
            return True
            
        except Exception as e:
            print(f"Error checking cooldown: {str(e)}")
            return True  # Allow alert if check fails
    
    def _store_alert(self, alert: Alert):
        """Store alert in session state"""
        try:
            st.session_state.alerts_history.append(alert.to_dict())
            
            # Keep only last 100 alerts
            if len(st.session_state.alerts_history) > 100:
                st.session_state.alerts_history = st.session_state.alerts_history[-100:]
                
        except Exception as e:
            print(f"Error storing alert: {str(e)}")
    
    def get_recent_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        try:
            return st.session_state.alerts_history[-limit:]
        except Exception as e:
            print(f"Error getting recent alerts: {str(e)}")
            return []
    
    def clear_alerts_history(self):
        """Clear alerts history"""
        try:
            st.session_state.alerts_history = []
            st.success("✅ Alerts history cleared")
        except Exception as e:
            print(f"Error clearing alerts history: {str(e)}")
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        try:
            alerts = st.session_state.alerts_history
            
            if not alerts:
                return {'total': 0, 'buy': 0, 'sell': 0, 'hold': 0, 'success_rate': 0}
            
            total = len(alerts)
            buy_count = len([a for a in alerts if a.get('signal') == 'BUY'])
            sell_count = len([a for a in alerts if a.get('signal') == 'SELL'])
            hold_count = len([a for a in alerts if a.get('signal') == 'HOLD'])
            
            # Calculate success rate (simplified)
            high_confidence_alerts = [a for a in alerts if a.get('confidence', 0) > 70]
            success_rate = (len(high_confidence_alerts) / total * 100) if total > 0 else 0
            
            return {
                'total': total,
                'buy': buy_count,
                'sell': sell_count,
                'hold': hold_count,
                'success_rate': success_rate,
                'high_confidence': len(high_confidence_alerts)
            }
            
        except Exception as e:
            print(f"Error getting alert statistics: {str(e)}")
            return {'total': 0, 'buy': 0, 'sell': 0, 'hold': 0, 'success_rate': 0}
    
    def test_telegram_connection(self) -> bool:
        """Test Telegram bot connection"""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/getMe"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('ok'):
                    st.success(f"✅ Telegram bot connected: {data['result']['username']}")
                    return True
            
            st.error("❌ Telegram bot connection failed")
            return False
            
        except Exception as e:
            st.error(f"❌ Telegram connection error: {str(e)}")
            return False
    
    def update_alert_settings(self, settings: Dict[str, bool]):
        """Update alert settings"""
        try:
            st.session_state.alert_settings.update(settings)
            st.success("✅ Alert settings updated")
        except Exception as e:
            print(f"Error updating alert settings: {str(e)}")
