import { useState } from 'react'

const API_BASE = ''  // proxied to localhost:8000 by vite.config.js in dev

export default function App() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [isMock, setIsMock] = useState(null)
  const [latency, setLatency] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleSearch(e) {
    e.preventDefault()
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, top_k: 5 }),
      })
      if (!res.ok) throw new Error(`Server responded ${res.status}`)
      const data = await res.json()
      setResults(data.results)
      setIsMock(data.mock)
      setLatency(data.latency_ms)
    } catch (err) {
      setError(err.message || 'Search failed — is the backend running on :8000?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page">
      <header className="header">
        <h1>Lumina</h1>
        <p className="subtitle">Multimodal semantic search — Phase 4 serving demo</p>
      </header>

      {isMock === true && (
        <div className="mock-banner">
          ⚠ MOCK MODE — results below are placeholders, not from a trained model.
          Phase 1 pretraining is still in progress.
        </div>
      )}

      <form className="search-form" onSubmit={handleSearch}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Describe an image... e.g. 'a dog in a park'"
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Searching…' : 'Search'}
        </button>
      </form>

      {error && <div className="error">{error}</div>}

      {latency !== null && !error && (
        <p className="latency">Latency: {latency.toFixed(1)} ms</p>
      )}

      <ul className="results">
        {results.map((r) => (
          <li key={r.image_id} className="result-card">
            <div className="result-id">{r.image_id}</div>
            <div className="result-caption">{r.caption}</div>
            <div className="result-score">score: {r.score.toFixed(4)}</div>
          </li>
        ))}
      </ul>
    </div>
  )
}
