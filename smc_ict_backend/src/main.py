import os
import sys
# DON'T CHANGE THIS !!!
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, send_from_directory, request, jsonify
from src.models.user import db
from src.routes.user import user_bp
from src.smc_ict_analysis import IntegratedSMCStrategy, TradingConfig # Importar las clases necesarias
import logging
from flask_cors import CORS

# Configuración de logging para una mejor visibilidad de los eventos
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
CORS(app)  # Habilitar CORS para todas las rutas
app.config['SECRET_KEY'] = 'asdf#FGSgvasgf$5$WGT'

app.register_blueprint(user_bp, url_prefix='/api')

# uncomment if you need to use database
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(os.path.dirname(__file__), 'database', 'app.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)
with app.app_context():
    db.create_all()

# Inicializar la estrategia de trading con una API_KEY de ejemplo
# TODO: Considerar cómo manejar la API_KEY de forma segura en un entorno de producción
API_KEY = "1OFGTIDh9osWhsdERKSn6lL7Q9lUgeNH" 
config = TradingConfig(
    risk_per_trade=0.02,
    min_confluence_score=75,
    preferred_pairs=['EURUSD', 'GBPUSD', 'USDJPY', 'AUDCAD'], # Añadir AUDCAD
    trading_sessions=['London', 'New York']
)
strategy = IntegratedSMCStrategy(API_KEY, config)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    symbol = data.get('symbol')
    if not symbol:
        return jsonify({'error': 'Símbolo no proporcionado'}), 400
    
    logger.info(f"Solicitud de análisis para el símbolo: {symbol}")
    result = strategy.analyze_symbol(symbol.upper())
    
    if 'error' in result:
        return jsonify(result), 500
    
    # Convertir objetos dataclass a diccionarios para jsonify
    if 'active_kill_zone' in result and result['active_kill_zone']:
        result['active_kill_zone'] = result['active_kill_zone'].__dict__
    if 'premium_discount_zones' in result and result['premium_discount_zones']:
        result['premium_discount_zones'] = result['premium_discount_zones'].__dict__

    return jsonify(result)

@app.route('/', defaults={'path': ''}) 
@app.route('/<path:path>') 
def serve(path):
    static_folder_path = app.static_folder
    if static_folder_path is None:
            return "Static folder not configured", 404

    if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    else:
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, 'index.html')
        else:
            return "index.html not found", 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


