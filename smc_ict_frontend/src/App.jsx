import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [symbol, setSymbol] = useState('');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);
    setAnalysisResult(null);
    try {
      const response = await axios.post('/api/analyze', { symbol });
      setAnalysisResult(response.data);
    } catch (err) {
      console.error('Error during analysis:', err);
      setError('Error al realizar el análisis. Por favor, intente de nuevo.');
      if (err.response && err.response.data && err.response.data.error) {
        setError(err.response.data.error);
      }
    } finally {
      setLoading(false);
    }
  };

  const formatPrice = (price) => {
    return price ? parseFloat(price).toFixed(5) : 'N/A';
  };

  const formatTime = (timeString) => {
    if (!timeString) return 'N/A';
    const date = new Date(timeString);
    return date.toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };

  const formatDate = (timeString) => {
    if (!timeString) return 'N/A';
    const date = new Date(timeString);
    return date.toLocaleDateString('es-ES', { year: 'numeric', month: '2-digit', day: '2-digit' });
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Análisis de Trading SMC + ICT</h1>
      </header>
      <main className="app-main">
        <div className="input-section">
          <p>Ingresa un par de divisas (ej. EURUSD, AUDCAD) para analizar.</p>
          <div className="input-group">
            <input
              type="text"
              placeholder="Audcad"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              className="symbol-input"
            />
            <button onClick={handleAnalyze} disabled={loading} className="analyze-button">
              {loading ? 'Analizando...' : 'Analizar'}
            </button>
          </div>
        </div>

        {error && <div className="error-message">{error}</div>}

        {analysisResult && (
          <div className="results-section">
            <div className="card">
              <h2>Resumen del Análisis</h2>
              <p><strong>Símbolo:</strong> {analysisResult.symbol}</p>
              <p><strong>Hora del Análisis:</strong> {formatTime(analysisResult.analysis_time)}</p>
              <p><strong>Fecha del Análisis:</strong> {formatDate(analysisResult.analysis_time)}</p>
              <p><strong>Precio Actual:</strong> {formatPrice(analysisResult.current_price)}</p>
            </div>

            <div className="card recommendation-card">
              <h2>Recomendación Final</h2>
              <p><strong>Acción:</strong> {analysisResult.recommendation.action}</p>
              <p><strong>Confianza:</strong> {analysisResult.recommendation.confidence}%</p>
              <p><strong>Motivo:</strong> {analysisResult.recommendation.reason}</p>
            </div>

            <div className="card">
              <h2>Contexto SMC + ICT</h2>
              <p><strong>Tendencia (1min):</strong> {analysisResult.structure_1min.trend.toUpperCase()}</p>
              <p><strong>BOS:</strong> {analysisResult.structure_1min.bos ? 'Sí' : 'No'}</p>
              <p><strong>CHoCH:</strong> {analysisResult.structure_1min.choch ? 'Sí' : 'No'}</p>
              <p><strong>Kill Zone Activa:</strong> {analysisResult.active_kill_zone.name || 'Ninguna'} (Prioridad: {analysisResult.active_kill_zone.priority.toUpperCase()})</p>
              <p><strong>Zona P/D:</strong> {analysisResult.premium_discount_zones.current_zone}</p>
              {analysisResult.premium_discount_zones.range_low && analysisResult.premium_discount_zones.range_high && (
                <p>(Rango: {formatPrice(analysisResult.premium_discount_zones.range_low)} - {formatPrice(analysisResult.premium_discount_zones.range_high)})</p>
              )}
            </div>

            {analysisResult.reaction_levels && analysisResult.reaction_levels.length > 0 && (
              <div className="card reaction-levels-card">
                <h2>Niveles de Reacción (1min)</h2>
                {analysisResult.reaction_levels.map((level, index) => (
                  <div key={index} className="reaction-level-item">
                    <p><strong>{index + 1}. {level.action} @ {formatPrice(level.price)}</strong> (Confianza: {level.confidence}%)</p>
                    <p>Distancia: {level.distance_pips.toFixed(1)} pips | Frescura: {level.freshness.toFixed(1)} min</p>
                    <p>Razón: {level.reason}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;


