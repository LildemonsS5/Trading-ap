from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pytz
import warnings
from dataclasses import dataclass
import logging
import traceback

# Configuración de logging para una mejor visibilidad de los eventos
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ================================================================
# CLASES DE LÓGICA DE TRADING - COPIADAS DE TU SCRIPT FUNCIONAL
# ================================================================

@dataclass
class TradingConfig:
    """Clase para una configuración de parámetros más flexible y controlada."""
    swing_period: int = 5
    liquidity_tolerance: float = 0.00005
    max_distance_pips: float = 50.0
    min_confluence_score: int = 70
    risk_per_trade: float = 0.02
    price_multiplier: int = 100000
    max_freshness_minutes: int = 120
    preferred_pairs: List[str] = None
    trading_sessions: List[str] = None

    def __post_init__(self):
        if self.preferred_pairs is None:
            self.preferred_pairs = ["EURUSD", "GBPUSD", "USDJPY"]
        if self.trading_sessions is None:
            self.trading_sessions = ["London", "New York"]

@dataclass
class KillZoneInfo:
    """Información de Kill Zone"""
    name: Optional[str]
    priority: str
    remaining_minutes: int
    is_active: bool

@dataclass
class PremiumDiscountZones:
    """Zonas Premium y Discount"""
    equilibrium: Optional[float]
    premium_start: Optional[float]
    discount_end: Optional[float]
    range_high: Optional[float]
    range_low: Optional[float]
    current_zone: str

class IntegratedSMCStrategy:
    """
    Estrategia de trading integrada con análisis SMC e ICT.
    
    Esta clase implementa la lógica de análisis técnico para detectar
    patrones de mercado como Order Blocks, FVG y liquidez, combinándolos
    con el contexto de Kill Zones y zonas Premium/Discount para generar
    recomendaciones de trading de alta probabilidad.
    """

    def __init__(self, api_key: str, config: TradingConfig = None):
        self.api_key = api_key
        self.config = config or TradingConfig()
        self.session = requests.Session()
        self.session.trust_env = False
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.PRICE_MULTIPLIER = self.config.price_multiplier
        logger.info("Estrategia SMC/ICT inicializada.")

    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str] = None) -> bool:
        """Valida que el DataFrame tenga la estructura correcta."""
        if required_columns is None:
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        
        if df.empty:
            logger.warning("DataFrame está vacío")
            return False
            
        if not all(col in df.columns for col in required_columns):
            logger.warning(f"Columnas faltantes: {set(required_columns) - set(df.columns)}")
            return False
            
        return True

    def get_market_data(self, symbol: str, timeframe: str = "1min", limit: int = 200) -> pd.DataFrame:
        """Obtiene datos de mercado con manejo mejorado de errores."""
        url = f"https://financialmodelingprep.com/api/v3/historical-chart/{timeframe}/{symbol}?apikey={self.api_key}"
        try:
            response = self.session.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, list) or len(data) == 0:
                logger.error(f"❌ No se obtuvieron datos válidos para {symbol} en {timeframe}")
                return pd.DataFrame()
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            if not self._validate_dataframe(df):
                logger.error("Datos de mercado inválidos tras la obtención.")
                return pd.DataFrame()
            return df.tail(limit).reset_index(drop=True)
        except requests.exceptions.Timeout:
            logger.error(f"❌ Timeout obteniendo datos para {timeframe}")
        except requests.exceptions.ConnectionError:
            logger.error("❌ Error de conexión obteniendo datos.")
        except requests.exceptions.HTTPError as e:
            logger.error(f"❌ Error HTTP {e.response.status_code} para {symbol}")
        except Exception as e:
            logger.error(f"❌ Error inesperado: {str(e)}")
        return pd.DataFrame()

    def get_current_price(self, symbol: str = "EURUSD") -> Dict:
        """Obtiene el precio actual con manejo de errores."""
        url = f"https://financialmodelingprep.com/api/v3/fx/{symbol}?apikey={self.api_key}"
        try:
            response = self.session.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0]
            elif isinstance(data, dict):
                return data
            logger.error(f"❌ No se obtuvieron datos de precio válidos para {symbol}")
            return {}
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error obteniendo precio actual: {e}")
            return {}

    def detect_swing_points_vectorized(self, df: pd.DataFrame, period: int = None) -> Dict:
        """Detección vectorizada de swing points para mejor performance."""
        if period is None: period = self.config.swing_period
        if len(df) < period * 2 + 1:
            return {'swing_highs': [], 'swing_lows': [], 'all_swings': []}
        
        highs, lows, dates = df['high'].values, df['low'].values, df['date'].values
        
        high_rolling_max = pd.Series(highs).rolling(window=period*2+1, center=True).max()
        low_rolling_min = pd.Series(lows).rolling(window=period*2+1, center=True).min()
        
        swing_highs = [{'price': highs[i], 'time': dates[i], 'type': 'high', 'index': i}
                       for i in range(period, len(df) - period)
                       if highs[i] == high_rolling_max.iloc[i] and not pd.isna(high_rolling_max.iloc[i])]
        
        swing_lows = [{'price': lows[i], 'time': dates[i], 'type': 'low', 'index': i}
                      for i in range(period, len(df) - period)
                      if lows[i] == low_rolling_min.iloc[i] and not pd.isna(low_rolling_min.iloc[i])]
        
        all_swings = sorted(swing_highs + swing_lows, key=lambda x: x['time'])
        return {'swing_highs': swing_highs, 'swing_lows': swing_lows, 'all_swings': all_swings}

    def find_liquidity_levels(self, all_swings: List[Dict]) -> List[Dict]:
        """Versión mejorada de detección de niveles de liquidez."""
        if not all_swings: return []
        grouped_levels = {}
        tolerance = self.config.liquidity_tolerance
        current_time = datetime.now()
        for swing in all_swings:
            matched = False
            for level_price_key in list(grouped_levels.keys()):
                if abs(swing['price'] - level_price_key) < tolerance:
                    level_data = grouped_levels[level_price_key]
                    level_data['touches'].append(swing)
                    level_data['sum_price'] += swing['price']
                    level_data['count'] += 1
                    level_data['avg_price'] = level_data['sum_price'] / level_data['count']
                    latest_touch_time = max([pd.to_datetime(t['time']) for t in level_data['touches']])
                    level_data['freshness'] = (current_time - latest_touch_time).total_seconds() / 60
                    matched = True
                    break
            if not matched:
                swing_time = pd.to_datetime(swing['time'])
                freshness = (current_time - swing_time).total_seconds() / 60
                grouped_levels[swing['price']] = {
                    'avg_price': swing['price'], 'touches': [swing], 'count': 1, 'sum_price': swing['price'],
                    'freshness': freshness, 'is_swept': False, 'strength': 1
                }
        liquidity_levels = []
        for _, data in grouped_levels.items():
            high_touches = sum(1 for t in data['touches'] if t['type'] == 'high')
            level_type = 'high' if high_touches >= len(data['touches']) / 2 else 'low'
            strength = min(data['count'] * 10, 100)
            liquidity_levels.append({
                'price': data['avg_price'], 'type': level_type, 'touches_count': data['count'],
                'strength': strength, 'freshness': data['freshness'],
                'last_touch_time': max([t['time'] for t in data['touches']]), 'is_swept': data['is_swept']
            })
        fresh_levels = [lvl for lvl in liquidity_levels if lvl['freshness'] <= self.config.max_freshness_minutes]
        return sorted(fresh_levels, key=lambda x: (x['strength'], -x['freshness']), reverse=True)

    def detect_order_blocks(self, df: pd.DataFrame) -> List[Dict]:
        """Detección mejorada de Order Blocks."""
        order_blocks = []
        if len(df) < 3: return order_blocks
        current_time = datetime.now()
        for i in range(1, len(df) - 1):
            current_candle, next_candle = df.iloc[i], df.iloc[i+1]
            if not all(isinstance(current_candle[col], (int, float)) for col in ['open', 'high', 'low', 'close']): continue
            ob_size = abs(current_candle['open'] - current_candle['close']) * self.PRICE_MULTIPLIER
            if current_candle['close'] < current_candle['open'] and next_candle['close'] > current_candle['high']:
                if ob_size >= 2 or abs(next_candle['close'] - current_candle['high']) * self.PRICE_MULTIPLIER >= 3:
                    freshness = (current_time - pd.to_datetime(current_candle['date'])).total_seconds() / 60
                    if freshness <= self.config.max_freshness_minutes:
                        order_blocks.append({'type': 'bullish', 'price': current_candle['low'], 'zone_min': current_candle['low'], 'zone_max': current_candle['open'], 'time': current_candle['date'], 'size_pips': ob_size, 'freshness': freshness, 'validated_by_sweep': False, 'strength': min(ob_size * 5, 100)})
            elif current_candle['close'] > current_candle['open'] and next_candle['close'] < current_candle['low']:
                if ob_size >= 2 or abs(current_candle['low'] - next_candle['close']) * self.PRICE_MULTIPLIER >= 3:
                    freshness = (current_time - pd.to_datetime(current_candle['date'])).total_seconds() / 60
                    if freshness <= self.config.max_freshness_minutes:
                        order_blocks.append({'type': 'bearish', 'price': current_candle['high'], 'zone_min': current_candle['close'], 'zone_max': current_candle['high'], 'time': current_candle['date'], 'size_pips': ob_size, 'freshness': freshness, 'validated_by_sweep': False, 'strength': min(ob_size * 5, 100)})
        return sorted(order_blocks, key=lambda x: (x['strength'], -x['freshness']), reverse=True)

    def detect_fair_value_gaps(self, df: pd.DataFrame) -> List[Dict]:
        """Detección mejorada de Fair Value Gaps."""
        fvg_zones = []
        if len(df) < 3: return fvg_zones
        current_time = datetime.now()
        for i in range(2, len(df)):
            candle1, candle3 = df.iloc[i-2], df.iloc[i]
            time = df.iloc[i-1]['date']
            freshness = (current_time - pd.to_datetime(time)).total_seconds() / 60
            if freshness > self.config.max_freshness_minutes: continue
            if candle1['low'] > candle3['high']:
                gap_size = (candle1['low'] - candle3['high']) * self.PRICE_MULTIPLIER
                fvg_zones.append({'type': 'bullish', 'zone_min': candle3['high'], 'zone_max': candle1['low'], 'time': time, 'freshness': freshness, 'gap_size_pips': gap_size, 'strength': min(gap_size * 10, 100)})
            elif candle1['high'] < candle3['low']:
                gap_size = (candle3['low'] - candle1['high']) * self.PRICE_MULTIPLIER
                fvg_zones.append({'type': 'bearish', 'zone_min': candle1['high'], 'zone_max': candle3['low'], 'time': time, 'freshness': freshness, 'gap_size_pips': gap_size, 'strength': min(gap_size * 10, 100)})
        return sorted(fvg_zones, key=lambda x: (x['strength'], -x['freshness']), reverse=True)

    def detect_liquidity_sweeps(self, df: pd.DataFrame, liquidity_levels: List[Dict]) -> List[Dict]:
        """Detecta barridas de liquidez y marca los niveles."""
        if len(df) < 2 or not liquidity_levels: return []
        current_time = datetime.now()
        sweeps = []
        for level in liquidity_levels:
            for i in range(len(df) - 1, max(-1, len(df) - 20), -1):
                candle = df.iloc[i]
                if (level['type'] == 'low' and candle['low'] < level['price'] and candle['close'] > level['price']) or \
                   (level['type'] == 'high' and candle['high'] > level['price'] and candle['close'] < level['price']):
                    sweep_freshness = (current_time - pd.to_datetime(candle['date'])).total_seconds() / 60
                    if sweep_freshness < self.config.max_freshness_minutes:
                        level['is_swept'] = True
                        sweeps.append({'type': 'bullish_sweep' if level['type'] == 'low' else 'bearish_sweep', 
                                       'level_price': level['price'], 'time': candle['date'], 'freshness': sweep_freshness})
                        break
        return sorted(sweeps, key=lambda x: x['freshness'])

    def detect_bos_choch_improved(self, df: pd.DataFrame, swings: Dict) -> Dict:
        """Detección mejorada de BOS/CHoCH y tendencia."""
        default_response = {'bos': False, 'choch': False, 'signal': None, 'trend': 'N/A'}
        if not swings['all_swings'] or len(swings['all_swings']) < 4: return {**default_response, 'trend': 'INSUFFICIENT_DATA'}
        
        last_highs = [s for s in swings['all_swings'] if s['type'] == 'high']
        last_lows = [s for s in swings['all_swings'] if s['type'] == 'low']

        if len(last_highs) < 2 or len(last_lows) < 2: return {**default_response, 'trend': 'FORMING'}

        last_high, prev_high = last_highs[-1]['price'], last_highs[-2]['price']
        last_low, prev_low = last_lows[-1]['price'], last_lows[-2]['price']

        trend = 'sideways'
        if last_high > prev_high and last_low > prev_low: trend = 'bullish'
        elif last_high < prev_high and last_low < prev_low: trend = 'bearish'

        bos_detected, choch_detected, signal = False, False, None
        current_price = df.iloc[-1]['close']

        if trend == 'bullish' and current_price > last_high:
            bos_detected, signal = True, 'BUY'
        elif trend == 'bearish' and current_price < last_low:
            bos_detected, signal = True, 'SELL'
        elif trend == 'bullish' and current_price < last_low:
            choch_detected, signal = True, 'SELL'
        elif trend == 'bearish' and current_price > last_high:
            choch_detected, signal = True, 'BUY'
        
        return {'bos': bos_detected, 'choch': choch_detected, 'signal': signal, 'trend': trend}
    
    def detect_kill_zones(self) -> KillZoneInfo:
        """Detección mejorada de Kill Zones con prioridades."""
        utc_now = datetime.now(pytz.utc)
        current_hour, current_minute = utc_now.hour, utc_now.minute
        zones = {'Asia': (0, 4, 'medium'), 'London': (7, 10, 'high'), 'New York': (12, 15, 'high'), 'London Close': (15, 17, 'medium')}
        for zone_name, (start, end, priority) in zones.items():
            if start <= current_hour < end:
                remaining_minutes = (end - current_hour - 1) * 60 + (60 - current_minute) if current_minute > 0 else 0
                return KillZoneInfo(name=zone_name, priority=priority, remaining_minutes=remaining_minutes, is_active=True)
        next_zone_minutes = min([(start * 60 - (current_hour * 60 + current_minute)) for start, _, _ in zones.values() if start * 60 > current_hour * 60 + current_minute], default=(24*60) - (current_hour * 60 + current_minute))
        return KillZoneInfo(name=None, priority='low', remaining_minutes=next_zone_minutes, is_active=False)

    def calculate_premium_discount_zones(self, swings: Dict) -> PremiumDiscountZones:
        """Cálculo avanzado de zonas Premium/Discount."""
        if not swings['swing_highs'] or not swings['swing_lows'] or len(swings['all_swings']) < 4:
            return PremiumDiscountZones(None, None, None, None, None, 'UNKNOWN')
        recent_swings = sorted(swings['all_swings'], key=lambda x: x['time'], reverse=True)[:20]
        if not recent_swings:
            return PremiumDiscountZones(None, None, None, None, None, 'UNKNOWN')
        high_swings = [s['price'] for s in recent_swings if s['type'] == 'high']
        low_swings = [s['price'] for s in recent_swings if s['type'] == 'low']
        if not high_swings or not low_swings:
            return PremiumDiscountZones(None, None, None, None, None, 'UNKNOWN')
        range_high, range_low = max(high_swings), min(low_swings)
        equilibrium = (range_high + range_low) / 2
        range_size = range_high - range_low
        premium_start, discount_end = equilibrium + (range_size * 0.25), equilibrium - (range_size * 0.25)
        return PremiumDiscountZones(equilibrium, premium_start, discount_end, range_high, range_low, 'EQUILIBRIUM')
    
    def calculate_confluence_score(self, level: Dict, context: Dict) -> int:
        """Sistema avanzado de cálculo de confluencia."""
        score = 40  # Puntuación base
        
        kill_zone = context.get('kill_zone')
        if kill_zone and kill_zone.is_active:
            if kill_zone.priority == 'high': score += 30
            elif kill_zone.priority == 'medium': score += 20
        
        pd_zones = context.get('premium_discount_zones')
        current_price = context.get('current_price', 0)
        if pd_zones and pd_zones.equilibrium:
            is_bullish_buy = level.get('type') in ['bullish', 'low']
            is_bearish_sell = level.get('type') in ['bearish', 'high']
            is_in_discount = current_price <= pd_zones.discount_end
            is_in_premium = current_price >= pd_zones.premium_start
            if (is_bullish_buy and is_in_discount) or (is_bearish_sell and is_in_premium):
                score += 25
        
        touches = level.get('touches_count', 1)
        if touches >= 3: score += 20
        elif touches >= 2: score += 10
        
        freshness = level.get('freshness', 999)
        if freshness <= 15: score += 15
        elif freshness <= 30: score += 10
        elif freshness <= 60: score += 5
        
        strength = level.get('strength', 0)
        if strength >= 80: score += 10
        elif strength >= 60: score += 7
        elif strength >= 40: score += 5
        
        if level.get('is_swept', False) and level.get('source') != 'Liquidity':
            score += 10
        
        return min(score, 100)

    def find_reaction_levels(self, df: pd.DataFrame, liquidity_levels: List[Dict], 
                             current_price: float, order_blocks: List[Dict],
                             fvg_zones: List[Dict], sweeps: List[Dict],
                             kill_zone: KillZoneInfo, premium_discount: PremiumDiscountZones) -> List[Dict]:
        """Encuentra y puntúa los niveles de reacción con confluencia."""
        reaction_levels = []
        all_sources = []
        
        for ob in order_blocks:
            ob['source'] = 'Order Block'
            ob['price'] = ob['zone_min'] if ob['type'] == 'bullish' else ob['zone_max']
            all_sources.append(ob)
        
        for fvg in fvg_zones:
            fvg['source'] = 'Fair Value Gap'
            fvg['price'] = (fvg['zone_min'] + fvg['zone_max']) / 2
            all_sources.append(fvg)
        
        for lvl in liquidity_levels:
            lvl['source'] = 'Liquidity'
            all_sources.append(lvl)
        
        context = {
            'kill_zone': kill_zone,
            'premium_discount_zones': premium_discount,
            'current_price': current_price
        }

        for level in all_sources:
            distance_pips = abs(level['price'] - current_price) * self.PRICE_MULTIPLIER
            if distance_pips > self.config.max_distance_pips: continue
            
            level_type = level.get('type')
            if level['source'] == 'Liquidity':
                level['type'] = 'low' if level_type == 'low' else 'high'
            elif level['source'] == 'Order Block':
                level['type'] = 'bullish' if level_type == 'bullish' else 'bearish'
            elif level['source'] == 'Fair Value Gap':
                level['type'] = 'bullish' if level_type == 'bullish' else 'bearish'
                
            level['confluence_score'] = self.calculate_confluence_score(level, context)
            
            reason = f"{level['type'].capitalize()} {level['source']}"
            if level.get('is_swept'):
                reason += " (Validado por Sweep)"

            if level['confluence_score'] >= 70:
                reason += " | CONFLUENCIA ALTA"
                if kill_zone.is_active and kill_zone.priority == 'high':
                    reason += f" en {kill_zone.name}"
                if premium_discount.equilibrium:
                    is_bullish_buy = level['type'] in ['bullish', 'low']
                    pos = "Discount" if is_bullish_buy else "Premium"
                    reason += f" en zona {pos}"
            
            reaction_levels.append({
                'action': 'BUY' if level['type'] in ['bullish', 'low'] else 'SELL',
                'price': level['price'],
                'entry_zone_min': level.get('zone_min', level['price'] - self.config.liquidity_tolerance),
                'entry_zone_max': level.get('zone_max', level['price'] + self.config.liquidity_tolerance),
                'distance_pips': distance_pips,
                'confidence': level['confluence_score'],
                'source': level['source'],
                'freshness': level['freshness'],
                'reason': reason
            })

        return sorted([lvl for lvl in reaction_levels if lvl['confidence'] >= self.config.min_confluence_score], key=lambda x: x['confidence'], reverse=True)

    def analyze_symbol(self, symbol: str) -> Dict:
        """Análisis completo del símbolo con lógica SMC e ICT."""
        logger.info(f"Iniciando análisis SMC + ICT de {symbol}...")
        
        df_1min = self.get_market_data(symbol, "1min", 200)
        df_5min = self.get_market_data(symbol, "5min", 100)
        df_15min = self.get_market_data(symbol, "15min", 50)
        
        if df_1min.empty or df_5min.empty or df_15min.empty:
            return {'error': 'No se pudieron obtener datos suficientes de todos los timeframes'}
        
        current_data = self.get_current_price(symbol)
        current_price = current_data.get('ask', df_1min.iloc[-1]['close']) 
        
        active_kill_zone = self.detect_kill_zones()
        swings_15min = self.detect_swing_points_vectorized(df_15min)
        premium_discount_zones = self.calculate_premium_discount_zones(swings_15min)
        
        swings_1min = self.detect_swing_points_vectorized(df_1min)
        liquidity_1min = self.find_liquidity_levels(swings_1min['all_swings'])
        sweeps_1min = self.detect_liquidity_sweeps(df_1min, liquidity_1min)
        structure_1min = self.detect_bos_choch_improved(df_1min, swings_1min)
        order_blocks_1min = self.detect_order_blocks(df_1min)
        fvg_zones_1min = self.detect_fair_value_gaps(df_1min)
        
        reaction_levels = self.find_reaction_levels(
            df_1min, liquidity_1min, current_price, order_blocks_1min, 
            fvg_zones_1min, sweeps_1min, active_kill_zone, premium_discount_zones
        )

        return {
            'symbol': symbol, 'current_price': current_price,
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'structure_1min': structure_1min,
            'reaction_levels': reaction_levels,
            'active_kill_zone': active_kill_zone,
            'premium_discount_zones': premium_discount_zones,
            'recommendation': self.generate_recommendation(reaction_levels, structure_1min)
        }

    def generate_recommendation(self, reaction_levels: List[Dict], structure_1min: Dict) -> Dict:
        """Genera una recomendación de trading basada en los niveles de confluencia."""
        recommendation = {'action': 'HOLD', 'confidence': 0, 'reason': 'Esperando confluencia de alta probabilidad.'}
        if not reaction_levels: return recommendation

        best_level = reaction_levels[0]
        if best_level['confidence'] < self.config.min_confluence_score:
            recommendation['reason'] = f"Confianza por debajo del umbral mínimo ({self.config.min_confluence_score}%) para el nivel más cercano."
            return recommendation

        recommendation['action'] = f"STRONG {best_level['action']}" if best_level['confidence'] >= 90 else best_level['action']
        recommendation['entry_zone'] = f"{best_level['entry_zone_min']:.5f} - {best_level['entry_zone_max']:.5f}"
        recommendation['confidence'] = best_level['confidence']
        recommendation['primary_source'] = best_level['source']
        recommendation['reason'] = best_level['reason']
        
        # Agregamos target_price al diccionario de recomendación
        recommendation['target_price'] = best_level['price']
        
        return recommendation


# ================================================================
# ENDPOINTS DE LA API
# ================================================================

@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SMC + ICT Trading Analyzer</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; color: #333; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; color: white; }
        .header h1 { font-size: 2.5rem; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        .header p { font-size: 1.1rem; opacity: 0.9; }
        .input-section { background: white; border-radius: 15px; padding: 30px; margin-bottom: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        .input-group { display: flex; gap: 15px; align-items: end; flex-wrap: wrap; }
        .form-group { flex: 1; min-width: 200px; }
        .form-group label { display: block; margin-bottom: 8px; font-weight: 600; color: #555; }
        .form-group input { width: 100%; padding: 12px 15px; border: 2px solid #e1e5e9; border-radius: 8px; font-size: 16px; transition: border-color 0.3s; }
        .form-group input:focus { outline: none; border-color: #667eea; }
        .analyze-btn { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 12px 30px; border-radius: 8px; font-size: 16px; font-weight: 600; cursor: pointer; transition: transform 0.2s; min-width: 150px; }
        .analyze-btn:hover { transform: translateY(-2px); }
        .analyze-btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .loading { display: none; text-align: center; padding: 20px; color: white; }
        .loading i { font-size: 2rem; animation: spin 1s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .results { display: none; }
        .results-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .card { background: white; border-radius: 15px; padding: 25px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        .card h3 { color: #667eea; margin-bottom: 15px; font-size: 1.3rem; display: flex; align-items: center; gap: 10px; }
        .price-info { font-size: 1.8rem; font-weight: bold; color: #2c3e50; margin-bottom: 10px; }
        .status-badge { display: inline-block; padding: 5px 12px; border-radius: 20px; font-size: 0.9rem; font-weight: 600; margin: 5px 5px 5px 0; }
        .status-bullish { background: #d4edda; color: #155724; }
        .status-bearish { background: #f8d7da; color: #721c24; }
        .status-neutral { background: #fff3cd; color: #856404; }
        .kill-zone { background: linear-gradient(135deg, #ff6b6b, #ee5a24); color: white; }
        .kill-zone.inactive { background: #95a5a6; }
        .levels-list { list-style: none; }
        .level-item { background: #f8f9fa; margin: 10px 0; padding: 15px; border-radius: 8px; border-left: 4px solid #667eea; }
        .level-item.buy { border-left-color: #28a745; }
        .level-item.sell { border-left-color: #dc3545; }
        .level-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
        .level-action { font-weight: bold; font-size: 1.1rem; }
        .level-action.buy { color: #28a745; }
        .level-action.sell { color: #dc3545; }
        .confidence-bar { width: 100%; height: 8px; background: #e9ecef; border-radius: 4px; overflow: hidden; margin: 8px 0; }
        .confidence-fill { height: 100%; background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb, #0abde3); transition: width 0.3s; }
        .recommendation { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; grid-column: 1 / -1; }
        .recommendation h3 { color: white; }
        .rec-action { font-size: 2rem; font-weight: bold; margin: 15px 0; }
        .error { background: #f8d7da; color: #721c24; padding: 20px; border-radius: 8px; margin: 20px 0; border: 1px solid #f5c6cb; }
        .info-badge { background: #d1ecf1; color: #0c5460; padding: 8px 12px; border-radius: 6px; font-size: 0.9rem; margin-top: 10px; }
        @media (max-width: 768px) { .input-group { flex-direction: column; } .form-group { min-width: 100%; } .header h1 { font-size: 2rem; } .results-grid { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-chart-line"></i> SMC + ICT Trading Analyzer</h1>
            <p>Análisis avanzado de Smart Money Concepts e Inner Circle Trader para Forex</p>
        </div>
        <div class="input-section">
            <div class="input-group">
                <div class="form-group">
                    <label for="symbol">Par de Divisas</label>
                    <input type="text" id="symbol" placeholder="Ej: EURUSD, GBPJPY" value="EURUSD">
                </div>
                <div class="form-group">
                    <label for="apiKey">API Key (Opcional)</label>
                    <input type="text" id="apiKey" placeholder="Tu API Key de Financial Modeling Prep" value="1OFGTIDh9osWhsdERKSn6lL7Q9lUgeNH">
                </div>
                <button class="analyze-btn" onclick="analyzeSymbol()">
                    <i class="fas fa-search"></i> Analizar
                </button>
            </div>
            <div class="info-badge">
                <i class="fas fa-info-circle"></i> Esta aplicación usa la misma lógica que tu código base local, con manejo de errores mejorado para el entorno web.
            </div>
        </div>
        <div class="loading" id="loading">
            <i class="fas fa-spinner"></i>
            <p>Analizando mercado...</p>
        </div>
        <div class="results" id="results">
            <div class="results-grid" id="resultsGrid"></div>
        </div>
    </div>
    <script>
        async function analyzeSymbol() {
            const symbol = document.getElementById('symbol').value.trim().toUpperCase();
            const apiKey = document.getElementById('apiKey').value.trim();
            if (!symbol) { alert('Por favor ingresa un símbolo de par de divisas'); return; }
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            document.querySelector('.analyze-btn').disabled = true;
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ symbol: symbol, api_key: apiKey || '1OFGTIDh9osWhsdERKSn6lL7Q9lUgeNH' })
                });
                const data = await response.json();
                if (data.error) { showError(data.error); } else { displayResults(data); }
            } catch (error) { showError('Error de conexión: ' + error.message); }
            finally { document.getElementById('loading').style.display = 'none'; document.querySelector('.analyze-btn').disabled = false; }
        }
        function displayResults(data) {
            const resultsGrid = document.getElementById('resultsGrid');
            resultsGrid.innerHTML = '';
            
            // Current Price Card
            const priceCard = createCard('<i class="fas fa-dollar-sign"></i> Precio Actual', `<div class="price-info">${data.current_price.toFixed(5)}</div><small>Símbolo: ${data.symbol} | ${data.analysis_time}</small>`);
            resultsGrid.appendChild(priceCard);

            // Kill Zone Card
            const kz = data.active_kill_zone;
            const kzContent = `<div class="status-badge kill-zone ${kz.is_active ? '' : 'inactive'}">${kz.name || 'Ninguna sesión activa'}</div>`;
            const kzCard = createCard('<i class="fas fa-clock"></i> Kill Zone', kzContent);
            resultsGrid.appendChild(kzCard);

            // Premium/Discount Card
            const pdz = data.premium_discount_zones;
            let pdzContent = '';
            if (pdz.equilibrium) {
                const isInPremium = data.current_price > pdz.equilibrium;
                pdzContent = `
                    <div>Rango: ${pdz.range_low.toFixed(5)} - ${pdz.range_high.toFixed(5)}</div>
                    <div>Equilibrio: ${pdz.equilibrium.toFixed(5)}</div>
                    <div class="status-badge ${isInPremium ? 'status-bearish' : 'status-bullish'}">
                        ${isInPremium ? 'Zona PREMIUM (Ventas)' : 'Zona DISCOUNT (Compras)'}
                    </div>`;
            } else { pdzContent = '<div>No se pudo determinar el rango</div>'; }
            const pdzCard = createCard('<i class="fas fa-balance-scale"></i> Premium/Discount', pdzContent);
            resultsGrid.appendChild(pdzCard);

            // Structure Card
            const structure = data.structure_1min;
            const structureContent = `
                <div>Tendencia: <span class="status-badge status-${getTrendClass(structure.trend)}">${structure.trend.toUpperCase()}</span></div>
                <div>BOS: ${structure.bos ? '✅' : '❌'} | CHOCH: ${structure.choch ? '✅' : '❌'}</div>
                <div>Señal: <strong>${structure.signal || 'N/A'}</strong></div>
            `;
            const structureCard = createCard('<i class="fas fa-chart-area"></i> Estructura SMC', structureContent);
            resultsGrid.appendChild(structureCard);

            // Reaction Levels Card
            const levelsContent = data.reaction_levels.length > 0
                ? `<ul class="levels-list">${data.reaction_levels.map(level => createLevelItem(level)).join('')}</ul>`
                : '<p>No se encontraron niveles de reacción cercanos</p>';
            const levelsCard = createCard('<i class="fas fa-crosshairs"></i> Niveles de Reacción', levelsContent);
            resultsGrid.appendChild(levelsCard);

            // Final Recommendation Card
            const rec = data.recommendation;
            const recContent = `
                <div class="rec-action">${rec.action}</div>
                ${rec.target_price ? `<div>Precio Objetivo: ${rec.target_price.toFixed(5)}</div>` : ''}
                ${rec.entry_zone ? `<div>Zona de Entrada: ${rec.entry_zone}</div>` : ''}
                <div>Confianza: ${rec.confidence}%</div>
                <div class="confidence-bar"><div class="confidence-fill" style="width: ${rec.confidence}%"></div></div>
                <div style="margin-top: 15px;">${rec.reason}</div>
            `;
            const recCard = createCard('<i class="fas fa-lightbulb"></i> Recomendación Final', recContent, 'recommendation');
            resultsGrid.appendChild(recCard);
            
            document.getElementById('results').style.display = 'block';
        }

        function createCard(title, content, extraClass = '') {
            const card = document.createElement('div');
            card.className = `card ${extraClass}`;
            card.innerHTML = `<h3>${title}</h3>${content}`;
            return card;
        }

        function createLevelItem(level) {
            return `<li class="level-item ${level.action.toLowerCase()}">
                        <div class="level-header">
                            <span class="level-action ${level.action.toLowerCase()}">${level.action} @ ${level.price.toFixed(5)}</span>
                            <span>${level.confidence}%</span>
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${level.confidence}%"></div>
                        </div>
                        <div>Distancia: ${level.distance_pips.toFixed(1)} pips | Frescura: ${level.freshness.toFixed(1)} min</div>
                        <div><small>${level.reason}</small></div>
                    </li>`;
        }

        function getTrendClass(trend) {
            switch(trend.toLowerCase()) {
                case 'bullish': return 'bullish';
                case 'bearish': return 'bearish';
                default: return 'neutral';
            }
        }
        function showError(message) {
            const resultsGrid = document.getElementById('resultsGrid');
            resultsGrid.innerHTML = `<div class="error"><i class="fas fa-exclamation-triangle"></i> ${message}</div>`;
            document.getElementById('results').style.display = 'block';
        }
        document.getElementById('symbol').addEventListener('keypress', function(e) { if (e.key === 'Enter') { analyzeSymbol(); } });
        document.getElementById('apiKey').addEventListener('keypress', function(e) { if (e.key === 'Enter') { analyzeSymbol(); } });
    </script>
</body>
</html>
'''

@app.route('/api/analyze', methods=['POST'])
def analyze_trading():
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'EURUSD').upper()
        api_key = data.get('api_key', '1OFGTIDh9osWhsdERKSn6lL7Q9lUgeNH')
        
        logger.info(f"🚀 Iniciando análisis para {symbol} con API key: {api_key[:10]}...")
        
        if not symbol:
            return jsonify({'error': 'Símbolo requerido'}), 400
        
        # Uso de TradingConfig y la clase actualizada
        config = TradingConfig(
            risk_per_trade=0.02,
            min_confluence_score=75,
            preferred_pairs=['EURUSD', 'GBPUSD', 'USDJPY'],
            trading_sessions=['London', 'New York']
        )
        strategy = IntegratedSMCStrategy(api_key, config)
        result = strategy.analyze_symbol(symbol)
        
        if 'error' in result:
            logger.error(f"❌ Error en el análisis: {result['error']}")
            return jsonify(result), 500
            
        # Para que el objeto dataclass KillZoneInfo y PremiumDiscountZones sean serializables
        result['active_kill_zone'] = result['active_kill_zone'].__dict__
        result['premium_discount_zones'] = result['premium_discount_zones'].__dict__
        
        logger.info(f"✅ Análisis completado exitosamente para {symbol}")
        return jsonify(result)
        
    except Exception as e:
        error_msg = f"Error en el análisis: {str(e)}"
        logger.error(f"❌ Error detallado: {traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'OK', 'message': 'Trading API funcionando correctamente'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004, debug=True)

