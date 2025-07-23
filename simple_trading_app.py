from flask import Flask, render_template, request, jsonify
import json
import os
from smc_strategy import IntegratedSMCStrategy, TradingConfig  # <<-- Línea de importación corregida

app = Flask(__name__)

# Configuración de la clave de API
# Recomendado: Usa variables de entorno para las claves de API
API_KEY = os.environ.get("FMP_API_KEY", "1OFGTIDh9osWhsdERKSn6lL7Q9lUgeNH") 

@app.route('/')
def home():
    # Asume que tienes un archivo index.html en la carpeta 'templates'
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    symbol = data.get('symbol', 'EURUSD').upper()
    
    try:
        # Crea una instancia de la clase de la lógica de negocio
        config = TradingConfig()
        strategy = IntegratedSMCStrategy(api_key=API_KEY, config=config)
        
        # Ejecuta el análisis. Toda la lógica está en el otro archivo
        analysis_result = strategy.analyze_symbol(symbol)
        
        # Manejo de errores si la API falla
        if analysis_result.get('error'):
            return jsonify({'status': 'error', 'message': analysis_result['error']}), 500

        # Para que los objetos dataclass se puedan convertir a JSON
        analysis_result['active_kill_zone'] = analysis_result['active_kill_zone'].__dict__
        analysis_result['premium_discount_zones'] = analysis_result['premium_discount_zones'].__dict__
        
        # Devuelve el resultado en formato JSON
        return jsonify(analysis_result)
        
    except Exception as e:
        import traceback
        return jsonify({'status': 'error', 'message': str(e), 'trace': traceback.format_exc()}), 500

if __name__ == '__main__':
    # Para producción, usa un servidor como Gunicorn o Waitress
    app.run(debug=True)
