import { useState, useEffect, useRef, useCallback } from 'react'

// ─── API Base URL ──────────────────────────────────────────────────────────────
const API = 'http://localhost:8000'

// ─── Utility: format bytes ─────────────────────────────────────────────────────
function fmtBytes(b) {
  if (b < 1024) return `${b} B`
  if (b < 1048576) return `${(b / 1024).toFixed(1)} KB`
  return `${(b / 1048576).toFixed(1)} MB`
}

// ─── Tumor type metadata ──────────────────────────────────────────────────────
const NO_TUMOR_CLASSES = new Set(['no', 'no_tumor'])
const TUMOR_META = {
  glioma: { name: 'Glioma', emoji: '🔴' },
  meningioma: { name: 'Meningioma', emoji: '🟠' },
  pituitary: { name: 'Pituitary Tumor', emoji: '🟡' },
  yes: { name: 'Tumor', emoji: '⚠️' },
}

// ─── Probability Bar ──────────────────────────────────────────────────────────
function ProbBar({ label, value }) {
  const [width, setWidth] = useState(0)
  const isTumor = !NO_TUMOR_CLASSES.has(label)
  const meta = TUMOR_META[label]
  const displayLabel = meta ? `${meta.emoji} ${meta.name}` : label === 'no' || label === 'no_tumor' ? '🟢 No Tumor' : label

  useEffect(() => {
    const t = setTimeout(() => setWidth(value * 100), 100)
    return () => clearTimeout(t)
  }, [value])
  return (
    <div className="prob-row">
      <span className="prob-label" style={{ width: '130px' }}>{displayLabel}</span>
      <div className="prob-bar-bg">
        <div
          className={`prob-bar-fill ${isTumor ? 'tumor' : 'safe'}`}
          style={{ width: `${width}%` }}
        />
      </div>
      <span className={`prob-pct verdict-label ${isTumor ? 'tumor' : 'safe'}`}>
        {(value * 100).toFixed(1)}%
      </span>
    </div>
  )
}

// ─── Result Card ──────────────────────────────────────────────────────────────
function ResultCard({ result }) {
  const hasTumor = result.has_tumor
  const confidence = (result.confidence * 100).toFixed(1)
  const cardClass = hasTumor ? 'tumor' : 'safe'
  const tumorMeta = TUMOR_META[result.prediction]

  return (
    <div className={`result-card ${cardClass}`}>
      <div className="result-header">
        <div className="result-verdict">
          <div className={`verdict-icon ${cardClass}`}>
            {hasTumor ? (tumorMeta?.emoji || '⚠️') : '✅'}
          </div>
          <div>
            <div className={`verdict-label ${cardClass}`}>
              {hasTumor
                ? (tumorMeta ? `${tumorMeta.name} Detected` : 'Tumor Detected')
                : 'No Tumor Found'}
            </div>
            <div className="verdict-sub">
              {hasTumor
                ? `Class: ${result.prediction} — Abnormal patterns identified in scan`
                : 'MRI scan appears within normal parameters'}
            </div>
          </div>
        </div>
        <div className="confidence-badge">
          <div className={`conf-value ${cardClass}`}>{confidence}%</div>
          <div className="conf-label">Confidence</div>
        </div>
      </div>

      {/* Probability Bars */}
      <div className="prob-section">
        <div className="prob-section-title">Class Probabilities</div>
        {Object.entries(result.probabilities).map(([cls, val]) => (
          <ProbBar key={cls} label={cls} value={val} />
        ))}
      </div>

      {/* Meta chips */}
      <div className="meta-row">
        {result.filename && (
          <div className="meta-chip">📄 <strong>{result.filename}</strong></div>
        )}
        {result.inference_time_seconds != null && (
          <div className="meta-chip">⚡ Inference: <strong>{result.inference_time_seconds}s</strong></div>
        )}
        <div className="meta-chip">🤖 Model: <strong>ResNet18</strong></div>
      </div>

      <div className="warning-banner">
        <span className="warning-banner-icon">⚕️</span>
        <span>
          <strong>Medical Disclaimer:</strong> This AI analysis is for research and educational purposes only.
          It does not constitute medical advice. Always consult a qualified medical professional for diagnosis and treatment.
        </span>
      </div>
    </div>
  )
}

// ─── Model Info Card ──────────────────────────────────────────────────────────
function ModelInfoCard({ info }) {
  if (!info) return null
  const items = [
    { key: 'Architecture', value: info.architecture || 'ResNet18' },
    { key: 'Input Size', value: `${info.input_size || 224}×${info.input_size || 224}` },
    { key: 'Device', value: info.device || '—' },
    { key: 'Val Accuracy', value: info.val_accuracy != null ? `${(info.val_accuracy * 100).toFixed(2)}%` : '—' },
    { key: 'Classes', value: (info.classes || []).join(', ') },
  ]
  return (
    <div className="model-card">
      <div className="model-card-header">
        <div className="model-card-icon">🧬</div>
        <div>
          <div className="model-card-title">Model Information</div>
          <div className="model-card-sub">Loaded ResNet18 checkpoint details</div>
        </div>
      </div>
      <div className="model-grid">
        {items.map(it => (
          <div key={it.key} className="model-item">
            <div className="model-item-key">{it.key}</div>
            <div className="model-item-value">{it.value}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [health, setHealth] = useState(null)   // null | 'loading' | 'healthy' | 'unhealthy'
  const [modelInfo, setModelInfo] = useState(null)
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const fileInputRef = useRef(null)
  const resultRef = useRef(null)

  // ─── Health check on mount ──────────────────────────────────────────────────
  useEffect(() => {
    setHealth('loading')
    fetch(`${API}/health`)
      .then(r => r.json())
      .then(d => setHealth(d.model_loaded ? 'healthy' : 'unhealthy'))
      .catch(() => setHealth('unhealthy'))

    fetch(`${API}/model/info`)
      .then(r => r.json())
      .then(setModelInfo)
      .catch(() => { })
  }, [])

  // ─── File select ────────────────────────────────────────────────────────────
  const handleFile = useCallback((f) => {
    if (!f) return
    if (!['image/jpeg', 'image/png', 'image/jpg'].includes(f.type)) {
      setError('Please upload a JPG or PNG image.')
      return
    }
    setFile(f)
    setResult(null)
    setError(null)
    const reader = new FileReader()
    reader.onload = e => setPreview(e.target.result)
    reader.readAsDataURL(f)
  }, [])

  // ─── Drag & Drop ────────────────────────────────────────────────────────────
  const onDragOver = e => { e.preventDefault(); setIsDragOver(true) }
  const onDragLeave = () => setIsDragOver(false)
  const onDrop = e => {
    e.preventDefault()
    setIsDragOver(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }

  // ─── Predict ────────────────────────────────────────────────────────────────
  const handlePredict = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    const formData = new FormData()
    formData.append('file', file)
    try {
      const res = await fetch(`${API}/predict`, { method: 'POST', body: formData })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Prediction failed')
      setResult(data)
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  // ─── Status badge ───────────────────────────────────────────────────────────
  const statusMap = {
    loading: { cls: 'loading', icon: '⏳', text: 'Connecting…' },
    healthy: { cls: 'healthy', icon: '🟢', text: 'API Online' },
    unhealthy: { cls: 'unhealthy', icon: '🔴', text: 'API Offline' },
  }
  const statusInfo = statusMap[health] || statusMap.loading

  return (
    <div className="app-wrapper">
      {/* Animated background */}
      <div className="bg-canvas">
        <div className="bg-orb bg-orb-1" />
        <div className="bg-orb bg-orb-2" />
        <div className="bg-orb bg-orb-3" />
      </div>

      {/* ─── Navbar ──────────────────────────────────────────────────────────── */}
      <nav className="navbar">
        <div className="navbar-brand">
          <div className="navbar-icon">🧠</div>
          <div>
            <div className="navbar-title">NeuroScan AI</div>
            <div className="navbar-subtitle">Brain Tumor Detection</div>
          </div>
        </div>
        <div className={`status-badge ${statusInfo.cls}`}>
          <span className="status-dot" />
          {statusInfo.icon} {statusInfo.text}
        </div>
      </nav>

      {/* ─── Main Content ────────────────────────────────────────────────────── */}
      <main className="main-content">

        {/* Hero */}
        <section className="hero">
          <div className="hero-tag">✨ Powered by ResNet18 Deep Learning</div>
          <h1 className="hero-title">
            Detect Brain Tumors<br />
            <span>with AI Precision</span>
          </h1>
          <p className="hero-desc">
            Upload an MRI brain scan and our AI model will analyze it in milliseconds,
            providing detailed confidence scores for tumor presence.
          </p>
        </section>

        {/* Stats */}
        <div className="stats-row">
          <div className="stat-card">
            <div className="stat-value">ResNet18</div>
            <div className="stat-label">Architecture</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">224×224</div>
            <div className="stat-label">Input Resolution</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">&lt;1s</div>
            <div className="stat-label">Inference Time</div>
          </div>
        </div>

        {/* Model Info */}
        <ModelInfoCard info={modelInfo} />

        {/* Upload + Preview */}
        <div className="upload-section">
          {/* Upload Zone */}
          <div
            className={`upload-zone ${isDragOver ? 'drag-over' : ''} ${file ? 'has-file' : ''}`}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
            onClick={() => fileInputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={e => e.key === 'Enter' && fileInputRef.current?.click()}
            aria-label="Upload MRI image"
          >
            <input
              ref={fileInputRef}
              className="upload-input"
              type="file"
              accept="image/jpeg,image/png"
              onChange={e => handleFile(e.target.files[0])}
            />
            <div className="upload-icon">🫁</div>
            <div className="upload-title">
              {isDragOver ? 'Drop your MRI scan here' : 'Upload MRI Scan'}
            </div>
            <div className="upload-subtitle">
              Drag & drop or click to browse<br />
              Supports JPG, PNG
            </div>
            <button
              className="btn-select"
              onClick={e => { e.stopPropagation(); fileInputRef.current?.click() }}
            >
              📂 Choose File
            </button>
          </div>

          {/* Preview Panel */}
          <div className="preview-panel">
            <div className="preview-panel-title">Image Preview</div>
            <div className="preview-img-box">
              {preview
                ? <img src={preview} alt="MRI preview" />
                : (
                  <div className="preview-placeholder">
                    <span>🧠</span>
                    <span>Your MRI will appear here</span>
                  </div>
                )
              }
            </div>
            {file && (
              <div className="file-info">
                <span className="file-info-icon">🗂️</span>
                <span className="file-info-name">{file.name}</span>
                <span className="file-info-size">{fmtBytes(file.size)}</span>
              </div>
            )}
          </div>
        </div>

        {/* Analyze Button */}
        <button
          className="analyze-btn"
          onClick={handlePredict}
          disabled={!file || loading || health === 'unhealthy'}
        >
          {loading
            ? <><div className="spinner" /> Analyzing MRI Scan…</>
            : <><span>🔬</span> Analyze Brain Scan</>
          }
        </button>

        {/* Error */}
        {error && (
          <div className="result-card error" style={{ marginTop: '28px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <span style={{ fontSize: '28px' }}>❌</span>
              <div>
                <div style={{ fontWeight: 600, color: '#F6AD55', marginBottom: '4px' }}>Analysis Failed</div>
                <div style={{ fontSize: '0.88rem', color: 'var(--text-secondary)' }}>{error}</div>
              </div>
            </div>
          </div>
        )}

        {/* Result */}
        {result && (
          <div ref={resultRef} style={{ marginTop: '32px' }}>
            <ResultCard result={result} />
          </div>
        )}

      </main>

      {/* Footer */}
      <footer className="footer">
        🧠 NeuroScan AI · Built with React + FastAPI · ResNet18 Model · For Research Purposes Only
      </footer>
    </div>
  )
}
