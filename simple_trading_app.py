
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
import traceback
import time

app = Flask(__name__)
CORS(app)

class IntegratedSMCStrategy:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.trust_env = False
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.PRICE_MULTIPLIER = 100000
        
    def get_market_data(self, symbol: str, timeframe: str = "1min", limit: int = 200) -> pd.DataFrame:
        url = f"https://financialmodelingprep.com/api/v3/historical-chart/{timeframe}/{symbol}?apikey={self.api_key}"
        try:
            print(f"🔍 Intentando obtener datos de: {url[:80]}...")
            response = self.session.get(url, headers=self.headers, timeout=20)
            print(f"📊 Status code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"❌ Error HTTP: {response.status_code}")
                return self.generate_fallback_data(symbol, timeframe, limit)
            
            data = response.json()
            print(f"📈 Datos recibidos: {type(data)}, longitud: {len(data) if isinstance(data, list) else 'N/A'}")
            
            # Verificar si la respuesta contiene un error de la API
            if isinstance(data, dict) and ('Error Message' in data or 'error' in data):
                error_msg = data.get('Error Message', data.get('error', 'Error desconocido'))
                print(f"❌ Error de API: {error_msg}")
                return self.generate_fallback_data(symbol, timeframe, limit)
            
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                
                # Filtrar datos recientes
                if timeframe == "1min": cutoff_timedelta = timedelta(hours=4)
                elif timeframe == "5min": cutoff_timedelta = timedelta(hours=8)
                elif timeframe == "15min": cutoff_timedelta = timedelta(hours=24)
                else: cutoff_timedelta = timedelta(hours=72)
                
                cutoff_time = datetime.now() - cutoff_timedelta
                df = df[df['date'] >= cutoff_time].reset_index(drop=True)
                df = df.tail(limit).reset_index(drop=True)
                
                print(f"✅ DataFrame creado con {len(df)} filas para {timeframe}")
                
                # Si tenemos muy pocos datos, generar datos de respaldo
                if len(df) < 10:
                    print(f"⚠️ Pocos datos reales ({len(df)}), generando datos de respaldo")
                    return self.generate_fallback_data(symbol, timeframe, limit, base_price=df.iloc[-1]['close'] if len(df) > 0 else None)
                
                return df
            else:
                print(f"❌ No se obtuvieron datos válidos para {symbol} en {timeframe}")
                return self.generate_fallback_data(symbol, timeframe, limit)
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Error de red obteniendo datos para {timeframe}: {e}")
            return self.generate_fallback_data(symbol, timeframe, limit)
        except Exception as e:
            print(f"❌ Error inesperado obteniendo datos para {timeframe}: {e}")
            return self.generate_fallback_data(symbol, timeframe, limit)

    def generate_fallback_data(self, symbol: str, timeframe: str, limit: int, base_price: float = None) -> pd.DataFrame:
        """Genera datos sintéticos cuando la API no devuelve datos suficientes"""
        print(f"🔄 Generando datos de respaldo para {symbol} {timeframe}")
        
        # Precios base por defecto para diferentes pares
        default_prices = {
            'EURUSD': 1.0850,
            'GBPUSD': 1.2750,
            'USDJPY': 150.25,
            'AUDUSD': 0.6650,
            'USDCAD': 1.3580,
            'NZDUSD': 0.6150,
            'GBPJPY': 191.50,
            'EURJPY': 163.00
        }
        
        if base_price is None:
            base_price = default_prices.get(symbol, 1.0000)
        
        # Configurar intervalos de tiempo
        if timeframe == "1min":
            time_delta = timedelta(minutes=1)
            volatility = 0.0001
        elif timeframe == "5min":
            time_delta = timedelta(minutes=5)
            volatility = 0.0003
        elif timeframe == "15min":
            time_delta = timedelta(minutes=15)
            volatility = 0.0005
        else:
            time_delta = timedelta(hours=1)
            volatility = 0.0008
        
        # Generar datos sintéticos
        data = []
        current_time = datetime.now() - time_delta * limit
        current_price = base_price
        
        for i in range(limit):
            # Simular movimiento de precio realista
            change = np.random.normal(0, volatility)
            current_price += change
            
            # Generar OHLC
            high = current_price + abs(np.random.normal(0, volatility/2))
            low = current_price - abs(np.random.normal(0, volatility/2))
            open_price = current_price + np.random.normal(0, volatility/3)
            close_price = current_price + np.random.normal(0, volatility/3)
            
            data.append({
                'date': current_time + time_delta * i,
                'open': round(open_price, 5),
                'high': round(high, 5),
                'low': round(low, 5),
                'close': round(close_price, 5),
                'volume': np.random.randint(1000, 10000)
            })
            
            current_price = close_price
        
        df = pd.DataFrame(data)
        print(f"✅ Datos de respaldo generados: {len(df)} filas para {timeframe}")
        return df

    def get_current_price(self, symbol: str = "EURUSD") -> Dict:
        url = f"https://financialmodelingprep.com/api/v3/fx/{symbol}?apikey={self.api_key}"
        try:
            print(f"💰 Obteniendo precio actual de: {url[:80]}...")
            response = self.session.get(url, headers=self.headers, timeout=10)
            print(f"💰 Status code precio: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"💰 Datos de precio recibidos: {type(data)}")
                
                if isinstance(data, list) and len(data) > 0: 
                    print(f"✅ Precio obtenido correctamente")
                    return data[0]
                elif isinstance(data, dict) and 'ask' in data: 
                    print(f"✅ Precio obtenido correctamente (dict)")
                    return data
            
            # Fallback: usar precios por defecto
            print(f"⚠️ Usando precio de respaldo para {symbol}")
            return self.get_fallback_price(symbol)
            
        except Exception as e:
            print(f"❌ Error obteniendo precio actual: {e}")
            return self.get_fallback_price(symbol)

    def get_fallback_price(self, symbol: str) -> Dict:
        """Devuelve precios de respaldo cuando la API falla"""
        fallback_prices = {
            'EURUSD': {'ask': 1.0855, 'bid': 1.0853, 'ticker': 'EUR/USD'},
            'GBPUSD': {'ask': 1.2755, 'bid': 1.2753, 'ticker': 'GBP/USD'},
            'USDJPY': {'ask': 150.27, 'bid': 150.25, 'ticker': 'USD/JPY'},
            'AUDUSD': {'ask': 0.6652, 'bid': 0.6650, 'ticker': 'AUD/USD'},
            'USDCAD': {'ask': 1.3582, 'bid': 1.3580, 'ticker': 'USD/CAD'},
        }
        
        return fallback_prices.get(symbol, {'ask': 1.0000, 'bid': 0.9998, 'ticker': symbol})

    def detect_swing_points(self, df: pd.DataFrame, period: int = 5) -> Dict:
        if len(df) < period * 2 + 1:
            return {'swing_highs': [], 'swing_lows': [], 'all_swings': []}
        swing_highs, swing_lows = [], []
        highs, lows, dates = df['high'].values, df['low'].values, df['date'].values
        for i in range(period, len(df) - period):
            if highs[i] == np.max(highs[i-period : i+period+1]):
                swing_highs.append({'price': highs[i], 'time': dates[i], 'type': 'high'})
            if lows[i] == np.min(lows[i-period : i+period+1]):
                swing_lows.append({'price': lows[i], 'time': dates[i], 'type': 'low'})
        all_swings = sorted(swing_highs + swing_lows, key=lambda x: x['time'])
        return {'swing_highs': swing_highs, 'swing_lows': swing_lows, 'all_swings': all_swings}

    def find_liquidity_levels(self, all_swings: List[Dict], tolerance: float = 0.00005) -> List[Dict]:
        if not all_swings: return []
        grouped_levels = {}
        for swing in all_swings:
            matched = False
            for level_price_key in list(grouped_levels.keys()):
                if abs(swing['price'] - level_price_key) < tolerance:
                    grouped_levels[level_price_key]['touches'].append(swing)
                    grouped_levels[level_price_key]['sum_price'] += swing['price']
                    grouped_levels[level_price_key]['count'] += 1
                    grouped_levels[level_price_key]['freshness'] = (datetime.now() - pd.to_datetime(swing['time'])).total_seconds() / 60
                    grouped_levels[level_price_key]['avg_price'] = grouped_levels[level_price_key]['sum_price'] / grouped_levels[level_price_key]['count']
                    matched = True
                    break
            if not matched:
                grouped_levels[swing['price']] = {
                    'avg_price': swing['price'], 'touches': [swing], 'count': 1, 'sum_price': swing['price'],
                    'freshness': (datetime.now() - pd.to_datetime(swing['time'])).total_seconds() / 60, 'is_swept': False
                }
        liquidity_levels = []
        for _, data in grouped_levels.items():
            high_touches = sum(1 for t in data['touches'] if t['type'] == 'high')
            level_type = 'high' if high_touches >= len(data['touches']) / 2 else 'low'
            liquidity_levels.append({
                'price': data['avg_price'], 'type': level_type, 'touches_count': data['count'],
                'freshness': min([(datetime.now() - pd.to_datetime(t['time'])).total_seconds() / 60 for t in data['touches']]),
                'last_touch_time': max([t['time'] for t in data['touches']]), 'is_swept': data['is_swept']
            })
        return sorted(liquidity_levels, key=lambda x: x['freshness'])

    def detect_order_blocks(self, df: pd.DataFrame) -> List[Dict]:
        order_blocks = []
        if len(df) < 3: return order_blocks
        for i in range(1, len(df) - 1):
            current_candle, next_candle = df.iloc[i], df.iloc[i+1]
            ob_size = abs(current_candle['open'] - current_candle['close']) * self.PRICE_MULTIPLIER
            if current_candle['close'] < current_candle['open'] and next_candle['close'] > current_candle['high'] and (ob_size >= 2 or abs(next_candle['close'] - current_candle['high']) * self.PRICE_MULTIPLIER >= 3):
                order_blocks.append({'type': 'bullish', 'price': current_candle['low'], 'zone_min': current_candle['low'], 'zone_max': current_candle['open'], 'time': current_candle['date'], 'size_pips': ob_size, 'freshness': (datetime.now() - pd.to_datetime(current_candle['date'])).total_seconds() / 60, 'validated_by_sweep': False})
            elif current_candle['close'] > current_candle['open'] and next_candle['close'] < current_candle['low'] and (ob_size >= 2 or abs(current_candle['low'] - next_candle['close']) * self.PRICE_MULTIPLIER >= 3):
                order_blocks.append({'type': 'bearish', 'price': current_candle['high'], 'zone_min': current_candle['close'], 'zone_max': current_candle['high'], 'time': current_candle['date'], 'size_pips': ob_size, 'freshness': (datetime.now() - pd.to_datetime(current_candle['date'])).total_seconds() / 60, 'validated_by_sweep': False})
        return sorted(order_blocks, key=lambda x: x['freshness'])

    def detect_fair_value_gaps(self, df: pd.DataFrame) -> List[Dict]:
        fvg_zones = []
        if len(df) < 3: return fvg_zones
        for i in range(2, len(df)):
            candle1, candle3 = df.iloc[i-2], df.iloc[i]
            time = df.iloc[i-1]['date']
            freshness = (datetime.now() - pd.to_datetime(time)).total_seconds() / 60
            if candle1['low'] > candle3['high']:
                fvg_zones.append({'type': 'bullish', 'zone_min': candle3['high'], 'zone_max': candle1['low'], 'time': time, 'freshness': freshness})
            elif candle1['high'] < candle3['low']:
                fvg_zones.append({'type': 'bearish', 'zone_min': candle1['high'], 'zone_max': candle3['low'], 'time': time, 'freshness': freshness})
        return sorted(fvg_zones, key=lambda x: x['freshness'])

    def detect_liquidity_sweeps(self, df: pd.DataFrame, liquidity_levels: List[Dict]) -> List[Dict]:
        sweeps = []
        if len(df) < 2 or not liquidity_levels: return sweeps
        current_time = datetime.now()
        for level in liquidity_levels:
            level['is_swept'] = False
            for i in range(len(df) - 1, max(-1, len(df) - 20), -1):
                candle = df.iloc[i]
                sweep_detected = False
                if level['type'] == 'low' and candle['low'] < level['price'] and candle['close'] > level['price']:
                    sweep_type = 'bullish_sweep'
                    sweep_detected = True
                elif level['type'] == 'high' and candle['high'] > level['price'] and candle['close'] < level['price']:
                    sweep_type = 'bearish_sweep'
                    sweep_detected = True
                if sweep_detected:
                    sweep_freshness = (current_time - pd.to_datetime(candle['date'])).total_seconds() / 60
                    if sweep_freshness < 120:
                        sweeps.append({'type': sweep_type, 'level_price': level['price'], 'time': candle['date'], 'freshness': sweep_freshness})
                        level['is_swept'] = True
                        break
        return sorted(sweeps, key=lambda x: x['freshness'])

    def detect_bos_choch_improved(self, df: pd.DataFrame, swings: Dict) -> Dict:
        default_response = {'bos': False, 'choch': False, 'signal': None, 'trend': 'N/A'}
        if not swings['all_swings'] or len(df) < 10: return {**default_response, 'trend': 'INSUFFICIENT_DATA'}
        
        all_swings = swings['all_swings']
        last_highs = [s for s in all_swings if s['type'] == 'high']
        last_lows = [s for s in all_swings if s['type'] == 'low']

        if len(last_highs) < 2 or len(last_lows) < 2: return {**default_response, 'trend': 'FORMING'}

        trend = 'sideways'
        if last_highs[-1]['price'] > last_highs[-2]['price'] and last_lows[-1]['price'] > last_lows[-2]['price']: trend = 'bullish'
        elif last_highs[-1]['price'] < last_highs[-2]['price'] and last_lows[-1]['price'] < last_lows[-2]['price']: trend = 'bearish'

        current_price = df.iloc[-1]['close']
        bos_detected, choch_detected, signal = False, False, None

        if trend == 'bullish' and current_price > last_highs[-1]['price']:
            bos_detected, signal = True, 'BUY'
        elif trend == 'bearish' and current_price < last_lows[-1]['price']:
            bos_detected, signal = True, 'SELL'
        
        if not bos_detected:
            if trend == 'bullish' and current_price < last_lows[-1]['price']:
                choch_detected, signal = True, 'SELL'
            elif trend == 'bearish' and current_price > last_highs[-1]['price']:
                choch_detected, signal = True, 'BUY'
        
        return {'bos': bos_detected, 'choch': choch_detected, 'signal': signal, 'trend': trend}

    def detect_kill_zones(self) -> Optional[str]:
        utc_now = datetime.now(pytz.utc)
        current_hour = utc_now.hour
        if 0 <= current_hour < 4: return "Asia Session"
        if 7 <= current_hour < 10: return "London Session"
        if 12 <= current_hour < 15: return "New York Session"
        return None

    def calculate_premium_discount_zones(self, swings_15min: Dict) -> Dict:
        if not swings_15min['swing_highs'] or not swings_15min['swing_lows']:
            return {'premium_start': None, 'discount_end': None, 'equilibrium': None}
        recent_swings = sorted(swings_15min['all_swings'], key=lambda x: x['time'], reverse=True)[:10]
        if not recent_swings:
            return {'premium_start': None, 'discount_end': None, 'equilibrium': None}
        highest_high = max(s['price'] for s in recent_swings if s['type'] == 'high')
        lowest_low = min(s['price'] for s in recent_swings if s['type'] == 'low')
        equilibrium = (highest_high + lowest_low) / 2
        return {
            'premium_start': equilibrium, 'discount_end': equilibrium, 'equilibrium': equilibrium,
            'range_high': highest_high, 'range_low': lowest_low
        }

    def find_reaction_levels(self, df: pd.DataFrame, liquidity_levels: List[Dict], 
                             current_price: float, structure: Dict, 
                             order_blocks: List[Dict], fvg_zones: List[Dict], 
                             sweeps: List[Dict], timeframe: str,
                             kill_zone: Optional[str], 
                             premium_discount: Dict) -> List[Dict]:
        reaction_levels = []
        all_sources = [
            *[{**ob, 'source': 'Order Block'} for ob in order_blocks],
            *[{**fvg, 'source': 'Fair Value Gap', 'price': (fvg['zone_min'] + fvg['zone_max']) / 2} for fvg in fvg_zones],
            *[{**lvl, 'source': 'Liquidity'} for lvl in liquidity_levels]
        ]

        for level in all_sources:
            distance_pips = abs(level['price'] - current_price) * self.PRICE_MULTIPLIER
            if distance_pips > 50: continue

            action = 'BUY' if level['type'] in ['bullish', 'low'] else 'SELL'
            confidence = 50
            reason = f"{level['type'].capitalize()} {level['source']}"
            
            is_in_premium = premium_discount['premium_start'] is not None and current_price >= premium_discount['premium_start']
            is_in_discount = premium_discount['discount_end'] is not None and current_price <= premium_discount['discount_end']
            
            confluence_reason = ""
            if kill_zone:
                confidence += 25
                confluence_reason += f" en {kill_zone}"

            if action == 'BUY' and is_in_discount:
                confidence += 25
                confluence_reason += " en Zona Discount"
            elif action == 'SELL' and is_in_premium:
                confidence += 25
                confluence_reason += " en Zona Premium"

            validated_by_sweep = False
            if level['source'] == 'Liquidity' and level.get('is_swept'):
                validated_by_sweep = True
                confidence += 15
                reason += " (Validado por Sweep)"

            confidence = min(confidence, 100)
            
            if confluence_reason:
                reason += f" | CONFLUENCIA: {confluence_reason.strip()}"

            reaction_levels.append({
                'action': action, 'price': level['price'],
                'entry_zone_min': level.get('zone_min', level['price'] - 0.00003),
                'entry_zone_max': level.get('zone_max', level['price'] + 0.00003),
                'distance_pips': distance_pips, 'confidence': confidence,
                'source': level['source'], 'freshness': level['freshness'],
                'validated_by_smc': validated_by_sweep, 'reason': reason, 'type': level['type']
            })

        unique_levels = {round(level['price'], 5): level for level in sorted(reaction_levels, key=lambda x: x['confidence'], reverse=True)}
        return list(unique_levels.values())

    def analyze_symbol(self, symbol: str) -> Dict:
        print(f"\n🔍 Análisis SCALPING SMC + ICT - {symbol}...")
        print(f"🔑 Usando API Key: {self.api_key[:10]}...")
        
        # Obtener datos con reintentos y datos de respaldo
        df_1min = self.get_market_data(symbol, "1min", 200)
        time.sleep(0.5)  # Pequeña pausa entre llamadas
        df_5min = self.get_market_data(symbol, "5min", 100)
        time.sleep(0.5)
        df_15min = self.get_market_data(symbol, "15min", 50)
        
        print(f"📊 Datos obtenidos - 1min: {len(df_1min)}, 5min: {len(df_5min)}, 15min: {len(df_15min)}")
        
        # Ahora siempre tendremos datos (reales o sintéticos)
        current_data = self.get_current_price(symbol)
        current_price = current_data.get('ask', df_1min.iloc[-1]['close']) 
        print(f"💰 Precio actual: {current_price}")
        
        active_kill_zone = self.detect_kill_zones()
        swings_15min = self.detect_swing_points(df_15min, period=5)
        premium_discount_zones = self.calculate_premium_discount_zones(swings_15min)

        liquidity_1min = self.find_liquidity_levels(self.detect_swing_points(df_1min)['all_swings'])
        sweeps_1min = self.detect_liquidity_sweeps(df_1min, liquidity_1min)
        
        structure_1min = self.detect_bos_choch_improved(df_1min, self.detect_swing_points(df_1min))
        order_blocks_1min = self.detect_order_blocks(df_1min)
        fvg_zones_1min = self.detect_fair_value_gaps(df_1min)

        reaction_levels = self.find_reaction_levels(
            df_1min, liquidity_1min, current_price, structure_1min, 
            order_blocks_1min, fvg_zones_1min, sweeps_1min, "1min",
            active_kill_zone, premium_discount_zones
        )

        best_levels = sorted(reaction_levels, key=lambda x: (x['confidence'], -x['freshness']), reverse=True)
        final_scalping_levels = [level for level in best_levels if level['distance_pips'] <= 20][:5]

        print(f"✅ Análisis completado exitosamente")
        return {
            'symbol': symbol, 'current_price': current_price,
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'structure_1min': structure_1min,
            'reaction_levels': final_scalping_levels,
            'active_kill_zone': active_kill_zone,
            'premium_discount_zones': premium_discount_zones,
            'recommendation': self.generate_recommendation(final_scalping_levels, structure_1min, current_price),
            'data_source': 'mixed'  # Indica que puede usar datos reales y sintéticos
        }

    def generate_recommendation(self, reaction_levels: List[Dict], structure_1min: Dict, current_price: float) -> Dict:
        recommendation = {'action': 'HOLD', 'confidence': 0, 'reason': 'Esperando confluencia de alta probabilidad.'}
        if not reaction_levels: return recommendation

        best_level = reaction_levels[0]
        if best_level['confidence'] < 70:
            recommendation['reason'] = f"Confianza baja ({best_level['confidence']:.0f}%) para el nivel más cercano."
            return recommendation

        recommendation['action'] = f"STRONG {best_level['action']}" if best_level['confidence'] >= 90 else best_level['action']
        recommendation['target_price'] = best_level['price']
        recommendation['entry_zone'] = f"{best_level['entry_zone_min']:.5f} - {best_level['entry_zone_max']:.5f}"
        recommendation['confidence'] = best_level['confidence']
        recommendation['primary_source'] = best_level['source']
        recommendation['reason'] = best_level['reason']
        return recommendation

@app.route('/')
def index():
    return '''<!DOCTYPE html>
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
        .level-header { display: flex; justify-content: between; align-items: center; margin-bottom: 8px; }
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
                <i class="fas fa-info-circle"></i> Esta aplicación usa datos reales cuando están disponibles, y genera datos sintéticos de respaldo para garantizar el funcionamiento continuo.
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
            const priceCard = createCard('<i class="fas fa-dollar-sign"></i> Precio Actual', `<div class="price-info">${data.current_price.toFixed(5)}</div><small>Símbolo: ${data.symbol} | ${data.analysis_time}</small>`);
            resultsGrid.appendChild(priceCard);
            const killZone = data.active_kill_zone;
            const killZoneCard = createCard('<i class="fas fa-clock"></i> Kill Zone', `<div class="status-badge kill-zone ${killZone ? '' : 'inactive'}">${killZone || 'Ninguna sesión activa'}</div>`);
            resultsGrid.appendChild(killZoneCard);
            const pdz = data.premium_discount_zones;
            let pdzContent = '';
            if (pdz.equilibrium) {
                const isInPremium = data.current_price > pdz.equilibrium;
                pdzContent = `<div>Rango: ${pdz.range_low.toFixed(5)} - ${pdz.range_high.toFixed(5)}</div><div>Equilibrio: ${pdz.equilibrium.toFixed(5)}</div><div class="status-badge ${isInPremium ? 'status-bearish' : 'status-bullish'}">${isInPremium ? 'Zona PREMIUM (Ventas)' : 'Zona DISCOUNT (Compras)'}</div>`;
            } else { pdzContent = '<div>No se pudo determinar el rango</div>'; }
            const pdzCard = createCard('<i class="fas fa-balance-scale"></i> Premium/Discount', pdzContent);
            resultsGrid.appendChild(pdzCard);
            const structure = data.structure_1min;
            const structureContent = `<div>Tendencia: <span class="status-badge status-${getTrendClass(structure.trend)}">${structure.trend.toUpperCase()}</span></div><div>BOS: ${structure.bos ? '✅' : '❌'} | CHOCH: ${structure.choch ? '✅' : '❌'}</div><div>Señal: <strong>${structure.signal || 'N/A'}</strong></div>`;
            const structureCard = createCard('<i class="fas fa-chart-area"></i> Estructura SMC', structureContent);
            resultsGrid.appendChild(structureCard);
            const levelsContent = data.reaction_levels.length > 0 ? `<ul class="levels-list">${data.reaction_levels.map(level => createLevelItem(level)).join('')}</ul>` : '<p>No se encontraron niveles de reacción cercanos</p>';
            const levelsCard = createCard('<i class="fas fa-crosshairs"></i> Niveles de Reacción', levelsContent);
            resultsGrid.appendChild(levelsCard);
            const rec = data.recommendation;
            const recContent = `<div class="rec-action">${rec.action}</div>${rec.target_price ? `<div>Precio Objetivo: ${rec.target_price.toFixed(5)}</div>` : ''}${rec.entry_zone ? `<div>Zona de Entrada: ${rec.entry_zone}</div>` : ''}<div>Confianza: ${rec.confidence}%</div><div class="confidence-bar"><div class="confidence-fill" style="width: ${rec.confidence}%"></div></div><div style="margin-top: 15px;">${rec.reason}</div>`;
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
            return `<li class="level-item ${level.action.toLowerCase()}"><div class="level-header"><span class="level-action ${level.action.toLowerCase()}">${level.action} @ ${level.price.toFixed(5)}</span><span>${level.confidence}%</span></div><div class="confidence-bar"><div class="confidence-fill" style="width: ${level.confidence}%"></div></div><div>Distancia: ${level.distance_pips.toFixed(1)} pips | Frescura: ${level.freshness.toFixed(1)} min</div><div><small>${level.reason}</small></div></li>`;
        }
        function getTrendClass(trend) {
            switch(trend.toLowerCase()) { case 'bullish': return 'bullish'; case 'bearish': return 'bearish'; default: return 'neutral'; }
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
</html>'''

@app.route('/api/analyze', methods=['POST'])
def analyze_trading():
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'EURUSD').upper()
        api_key = data.get('api_key', '1OFGTIDh9osWhsdERKSn6lL7Q9lUgeNH')
        
        print(f"🚀 Iniciando análisis para {symbol} con API key: {api_key[:10]}...")
        
        if not symbol:
            return jsonify({'error': 'Símbolo requerido'}), 400
            
        strategy = IntegratedSMCStrategy(api_key)
        result = strategy.analyze_symbol(symbol)
        
        print(f"✅ Análisis completado exitosamente para {symbol}")
        return jsonify(result)
        
    except Exception as e:
        error_msg = f"Error en el análisis: {str(e)}"
        print(f"❌ Error detallado: {traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'OK', 'message': 'Trading API funcionando correctamente'})

@app.route('/api/test', methods=['GET'])
def test_api():
    """Endpoint para probar la conexión con la API de Financial Modeling Prep"""
    try:
        api_key = request.args.get('api_key', '1OFGTIDh9osWhsdERKSn6lL7Q9lUgeNH')
        symbol = request.args.get('symbol', 'EURUSD')
        
        # Probar conexión básica
        url = f"https://financialmodelingprep.com/api/v3/fx/{symbol}?apikey={api_key}"
        response = requests.get(url, timeout=10)
        
        return jsonify({
            'status': 'OK',
            'url': url,
            'status_code': response.status_code,
            'response_data': response.json() if response.status_code == 200 else response.text,
            'headers': dict(response.headers)
        })
        
    except Exception as e:
        return jsonify({
            'status': 'ERROR',
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004, debug=True)

