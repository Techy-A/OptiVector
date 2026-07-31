import React, { useState, useRef } from 'react';
import './App.css';
import { InteractiveSVG } from './components/InteractiveSVG';

function App() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [metrics, setMetrics] = useState({ step: 0, loss: 0, curves: 0, layer: '' });
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [svgResult, setSvgResult] = useState<string | null>(null);
  const [proxyImage, setProxyImage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // Settings State
  const [numCurves, setNumCurves] = useState(400);
  const [steps, setSteps] = useState(500);
  const [prompts, setPrompts] = useState("person, background");

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setPreviewUrl(URL.createObjectURL(selectedFile));
      setSvgResult(null);
      setProxyImage(null);
    }
  };

  const handleFastPreview = async () => {
    if (!file) return;
    setIsProcessing(true);
    setSvgResult(null);
    setProxyImage(null);
    setMetrics({ step: 0, loss: 0, curves: numCurves, layer: 'Neural Proxy...' });

    const formData = new FormData();
    formData.append("file", file);
    formData.append("num_curves", numCurves.toString());

    try {
      const response = await fetch("http://localhost:8000/preview", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Preview request failed");
      
      const blob = await response.blob();
      setProxyImage(URL.createObjectURL(blob));
      setIsProcessing(false);
    } catch (err) {
      console.error(err);
      alert("Failed to connect to the backend server or preview failed.");
      setIsProcessing(false);
    }
  };

  const handleVectorize = async () => {
    if (!file) return;
    setIsProcessing(true);
    setSvgResult(null);
    setProxyImage(null);
    setMetrics({ step: 0, loss: 0, curves: numCurves, layer: 'Starting...' });

    const formData = new FormData();
    formData.append("file", file);
    formData.append("num_curves", numCurves.toString());
    formData.append("steps", steps.toString());
    formData.append("segment_prompts", prompts);

    try {
      // Connect to the FastAPI backend (assuming SSH tunnel on port 8000)
      const response = await fetch("http://localhost:8000/vectorize", {
        method: "POST",
        body: formData,
      });

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || ""; // Keep the incomplete line in the buffer

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const dataStr = line.replace("data: ", "").trim();
            if (!dataStr) continue;
            
            try {
              const data = JSON.parse(dataStr);
              if (data.status === 'metrics') {
                setMetrics(prev => ({
                  ...prev,
                  step: data.step,
                  loss: data.loss,
                  layer: data.layer
                }));
                if (data.svg) {
                  setSvgResult(data.svg);
                }
              } else if (data.status === 'complete') {
                setSvgResult(data.svg);
                setIsProcessing(false);
              } else if (data.status === 'error') {
                alert("Error: " + data.message);
                setIsProcessing(false);
              }
            } catch (err) {
              console.error("Failed to parse SSE data", dataStr);
            }
          }
        }
      }
    } catch (err) {
      console.error(err);
      alert("Failed to connect to the backend server. Is it running on port 8000?");
      setIsProcessing(false);
    }
  };

  return (
    <div className="app-container">
      <input 
        type="file" 
        accept="image/*" 
        style={{ display: 'none' }} 
        ref={fileInputRef}
        onChange={handleFileChange}
      />
      
      {/* LEFT COLUMN: Control Deck */}
      <aside className="control-deck">
        <header className="deck-header hairline-border-bottom">
          <h1>CurveSplat</h1>
        </header>

        <div className="deck-content hairline-border-right">
          
          <section className="control-section hairline-border-bottom">
            <h2>Source Image</h2>
            <div 
              className={`upload-area ${file ? 'active' : ''}`}
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="upload-icon">⇪</div>
              <div className="upload-text">
                {file ? file.name : "Click to upload image"}
              </div>
            </div>
          </section>

          <section className="control-section hairline-border-bottom">
            <h2>Geometry Budget</h2>
            
            <div className="input-group">
              <label>Total Curves</label>
              <div className="slider-container">
                <input 
                  type="range" 
                  min="50" max="1000" step="50"
                  value={numCurves} 
                  onChange={(e) => setNumCurves(Number(e.target.value))}
                />
                <span className="slider-value">{numCurves}</span>
              </div>
            </div>

            <div className="input-group">
              <label>Optimization Steps</label>
              <div className="slider-container">
                <input 
                  type="range" 
                  min="100" max="2000" step="100"
                  value={steps} 
                  onChange={(e) => setSteps(Number(e.target.value))}
                />
                <span className="slider-value">{steps}</span>
              </div>
            </div>
          </section>

          <section className="control-section">
            <h2>Semantic Layering</h2>
            <div className="input-group">
              <label>Layer Prompts (comma separated)</label>
              <input 
                type="text" 
                value={prompts}
                onChange={(e) => setPrompts(e.target.value)}
                placeholder="e.g. person, background" 
              />
            </div>
          </section>
        </div>

        <footer className="deck-footer hairline-border-right hairline-border-top" style={{display: 'flex', gap: '12px'}}>
          <button 
            className="vectorize-btn"
            style={{flex: 1, background: 'var(--color-surface-slate)', border: '1px solid var(--color-accent-cyan)', color: 'var(--color-accent-cyan)'}}
            disabled={!file || isProcessing}
            onClick={handleFastPreview}
          >
            Fast Preview
          </button>
          <button 
            className="vectorize-btn"
            style={{flex: 1}}
            disabled={!file || isProcessing}
            onClick={handleVectorize}
          >
            {isProcessing && !proxyImage ? "Processing..." : "Vectorize"}
          </button>
        </footer>
      </aside>

      {/* RIGHT COLUMN: Canvas Viewport */}
      <main className="canvas-viewport">
        <div className="canvas-container">
          {previewUrl ? (
            <>
              {/* The underlying photo */}
              <img 
                src={previewUrl} 
                className={`preview-image ${(isProcessing && !proxyImage) || svgResult || proxyImage ? 'dimmed' : ''}`} 
                alt="Source"
              />

              {/* Proxy Preview Image */}
              {proxyImage && !svgResult && (
                <img 
                  src={proxyImage} 
                  className="svg-overlay" 
                  style={{width: '100%', height: '100%', objectFit: 'contain', zIndex: 6}} 
                  alt="Proxy Preview"
                />
              )}

              {/* The final SVG Result overlay - Interactive! */}
              {svgResult && (
                <div className="svg-overlay" style={{ pointerEvents: 'auto', zIndex: 10 }}>
                  <InteractiveSVG svgString={svgResult} />
                </div>
              )}
            </>
          ) : (
            <div style={{color: 'var(--color-border-grid)', margin: 'auto'}}>Awaiting Input</div>
          )}
        </div>

        {/* Floating Metrics Dashboard */}
        {(isProcessing || svgResult) && (
          <div className="metrics-ticker hairline-border">
            {metrics.layer && <div className="metric-row"><span className="metric-label">PHASE</span><span style={{color: 'var(--color-text-primary)'}}>{metrics.layer}</span></div>}
            <div className="metric-row">
              <span className="metric-label">STEP</span>
              <span>{metrics.step} / {steps}</span>
            </div>
            <div className="metric-row">
              <span className="metric-label">LOSS</span>
              <span>{metrics.loss > 0 ? metrics.loss.toFixed(5) : "---"}</span>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
