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

# Configuración de logging para una mejor visibilidad de los eventos
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suprimir warnings de pandas para una salida más limpia
warnings.filterwarnings('ignore')

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
        
        # Kill Zone activa
        kill_zone = context.get('kill_zone')
        if kill_zone and kill_zone.is_active:
            if kill_zone.priority == 'high': score += 30
            elif kill_zone.priority == 'medium': score += 20
        
        # Premium/Discount
        pd_zones = context.get('premium_discount_zones')
        current_price = context.get('current_price', 0)
        if pd_zones and pd_zones.equilibrium:
            is_bullish_buy = level.get('type') in ['bullish', 'low']
            is_bearish_sell = level.get('type') in ['bearish', 'high']
            is_in_discount = current_price <= pd_zones.discount_end
            is_in_premium = current_price >= pd_zones.premium_start
            if (is_bullish_buy and is_in_discount) or (is_bearish_sell and is_in_premium):
                score += 25
        
        # Múltiples toques
        touches = level.get('touches_count', 1)
        if touches >= 3: score += 20
        elif touches >= 2: score += 10
        
        # Frescura del nivel
        freshness = level.get('freshness', 999)
        if freshness <= 15: score += 15
        elif freshness <= 30: score += 10
        elif freshness <= 60: score += 5
        
        # Fuerza del nivel (tamaño de OB/FVG o toques de liquidez)
        strength = level.get('strength', 0)
        if strength >= 80: score += 10
        elif strength >= 60: score += 7
        elif strength >= 40: score += 5
        
        # Validación por sweep
        if level.get('is_swept', False) and level.get('source') != 'Liquidity': # Solo OB/FVG
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
            # Asegurarse de que 'price' está presente
            ob['price'] = ob['zone_min'] if ob['type'] == 'bullish' else ob['zone_max']
            all_sources.append(ob)
        
        for fvg in fvg_zones:
            fvg['source'] = 'Fair Value Gap'
            # Calcular y asignar 'price' para FVG
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
            # Ahora, todos los 'level' tienen una clave 'price'
            distance_pips = abs(level['price'] - current_price) * self.PRICE_MULTIPLIER
            if distance_pips > self.config.max_distance_pips: continue
            
            # Ajustar el tipo para el scoring
            level_type = level.get('type')
            if level['source'] == 'Liquidity':
                level['type'] = 'low' if level_type == 'low' else 'high'
            elif level['source'] == 'Order Block':
                level['type'] = 'bullish' if level_type == 'bullish' else 'bearish'
            elif level['source'] == 'Fair Value Gap':
                level['type'] = 'bullish' if level_type == 'bullish' else 'bearish'
                
            level['confluence_score'] = self.calculate_confluence_score(level, context)
            
            # Agregar información de confluencia para la razón
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
        
        # Análisis ICT
        active_kill_zone = self.detect_kill_zones()
        swings_15min = self.detect_swing_points_vectorized(df_15min)
        premium_discount_zones = self.calculate_premium_discount_zones(swings_15min)
        
        # Análisis SMC en 1min
        swings_1min = self.detect_swing_points_vectorized(df_1min)
        liquidity_1min = self.find_liquidity_levels(swings_1min['all_swings'])
        sweeps_1min = self.detect_liquidity_sweeps(df_1min, liquidity_1min)
        structure_1min = self.detect_bos_choch_improved(df_1min, swings_1min)
        order_blocks_1min = self.detect_order_blocks(df_1min)
        fvg_zones_1min = self.detect_fair_value_gaps(df_1min)
        
        # Generar niveles de reacción y puntuarlos
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
        return recommendation

    def print_analysis_results(self, result: Dict):
        """Formatea e imprime los resultados del análisis de forma clara."""
        if result.get('error'):
            logger.error(f"ERROR: {result['error']}")
            return

        print("\n" + "="*50)
        print(f"📊 ANÁLISIS SMC + ICT - SCALPING - {result['symbol']}")
        print(f"🕒 {result['analysis_time']}")
        print("="*50)
        print(f"💰 Precio Actual: {result['current_price']:.5f}")
        
        print("\n" + "-"*20 + " CONTEXTO ICT " + "-"*20)
        kz = result['active_kill_zone']
        print(f"🕒 Kill Zone Activa: {kz.name if kz.is_active else 'Ninguna'} (Prioridad: {kz.priority.upper()})")
        pdz = result['premium_discount_zones']
        if pdz.equilibrium:
            print(f"📈 Rango de Trading (15min): {pdz.range_low:.5f} - {pdz.range_high:.5f}")
            print(f"⚖️ Equilibrio (50%): {pdz.equilibrium:.5f}")
            pos = "PREMIUM (Ventas)" if result['current_price'] > pdz.premium_start else "DISCOUNT (Compras)" if result['current_price'] < pdz.discount_end else "EQUILIBRIUM"
            print(f"📍 Posición Actual: Zona {pos}")
        else:
            print("📉 Zonas Premium/Discount: No se pudo determinar el rango.")

        print("\n" + "-"*20 + " ESTRUCTURA SMC (1min) " + "-"*20)
        s1 = result['structure_1min']
        print(f"  Tendencia: {s1['trend'].upper()} | BOS: {s1['bos']} | CHOCH: {s1['choch']} | Señal: {s1['signal']}")

        print("\n" + "-"*20 + " NIVELES DE REACCIÓN (1min) " + "-"*20)
        if not result['reaction_levels']:
            print("  No se encontraron niveles de reacción cercanos con alta confluencia.")
        for i, level in enumerate(result['reaction_levels']):
            print(f"  {i+1}. {level['action']} @ {level['price']:.5f} (Confianza: {level['confidence']:.0f}%)")
            print(f"     Distancia: {level['distance_pips']:.1f} pips | Frescura: {level['freshness']:.1f} min")
            print(f"     Razón: {level['reason']}")
            if i < len(result['reaction_levels']) - 1: print("-" * 10)

        print("\n" + "="*20 + " RECOMENDACIÓN FINAL " + "="*20)
        rec = result['recommendation']
        print(f"  Acción: {rec['action']}")
        if rec.get('entry_zone'):
            print(f"  Zona de Entrada: {rec.get('entry_zone', 'N/A')}")
        print(f"  Confianza: {rec['confidence']:.0f}%")
        print(f"  Razón: {rec['reason']}")
        print("="*50 + "\n")

def analyze_forex_smc(api_key: str):
    """Función principal para ejecutar el análisis."""
    config = TradingConfig(
        risk_per_trade=0.02,
        min_confluence_score=75,
        preferred_pairs=['EURUSD', 'GBPUSD', 'USDJPY'],
        trading_sessions=['London', 'New York']
    )
    strategy = IntegratedSMCStrategy(api_key, config)
    
    symbol_input = input("Por favor, introduce el par de divisas (ej. EURUSD, GBPJPY): ").upper()
    if symbol_input not in config.preferred_pairs:
        logger.warning(f"El par '{symbol_input}' no está en tu lista de pares preferidos. Continuando de todas formas.")
    
    result = strategy.analyze_symbol(symbol_input)
    if 'error' not in result:
        strategy.print_analysis_results(result)

if __name__ == "__main__":
    # La API key proporcionada en el prompt
    API_KEY = "1OFGTIDh9osWhsdERKSn6lL7Q9lUgeNH" 
    analyze_forex_smc(API_KEY)
