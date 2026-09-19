import { useState, useEffect, useRef } from 'react'

const API_BASE = ''

const EXAMPLE_QUERIES = [
  'a dog in a park',
  'people cooking in a kitchen',
  'children playing outside',
  'a city street at night',
  'a mountain landscape',
]

function ResultSkeleton() {
  return (
    <li className="result-card skeleton">
      <div className="skeleton-line skeleton-id" />
      <div className="skeleton-line skeleton-caption" />
      <div className="skeleton-line skeleton-score" />
    </li>
  )
}

export default function App() {
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(5)
  const [results, setResults] = useState([])
  const [isMock, setIsMock] = useState(null)
  const [latency, setLatency] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [copiedId, setCopiedId] = useState(null)
  const [recentSearches, setRecentSearches] = useState([])
  const [theme, setTheme] = useState('dark')
  const [lastQuery, setLastQuery] = useState(null)
  const inputRef = useRef(null)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

  useEffect(() => {
    function handleKeydown(e) {
      if (e.key === '/' && document.activeElement !== inputRef.current) {
        e.preventDefault()
        inputRef.current?.focus()
      }
    }
    window.addEventListener('keydown', handleKeydown)
    return () => window.removeEventListener('keydown', handleKeydown)
  }, [])

  function toggleTheme() {
    setTheme((t) => (t === 'dark' ? 'light' : 'dark'))
  }

  async function runSearch(q, k) {
    if (!q.trim()) return
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, top_k: k }),
      })
      if (!res.ok) throw new Error(`Server responded ${res.status}`)
      const data = await res.json()
      setResults(data.results)
      setIsMock(data.mock)
      setLatency(data.latency_ms)
      setLastQuery(q)
      setRecentSearches((prev) => {
        const next = [q, ...prev.filter((item) => item !== q)]
        return next.slice(0, 5)
      })
    } catch (err) {
      setError(err.message || 'Search failed - is the backend running on :8000?')
    } finally {
      setLoading(false)
    }
  }

  function handleSearch(e) {
    e.preventDefault()
    runSearch(query, topK)
  }

  function handleExampleClick(example) {
    setQuery(example)
    runSearch(example, topK)
  }

  function handleRecentClick(q) {
    setQuery(q)
    runSearch(q, topK)
  }

  function handleRetry() {
    if (lastQuery) runSearch(lastQuery, topK)
    else if (query) runSearch(query, topK)
  }

  function handleClear() {
    setQuery('')
    setResults([])
    setIsMock(null)
    setLatency(null)
    setError(null)
    setLastQuery(null)
  }

  async function handleCopy(result) {
    const text = `${result.image_id}: ${result.caption} (score: ${result.score.toFixed(4)})`
    try {
      await navigator.clipboard.writeText(text)
      setCopiedId(result.image_id)
      setTimeout(() => setCopiedId(null), 1500)
    } catch {
    }
  }

  return (
    <div className="page">
      <header className="header">
        <div className="header-top">
          <h1>Lumina</h1>
          <button type="button" className="theme-toggle" onClick={toggleTheme} title="Toggle theme">
            {theme === 'dark' ? 'Light' : 'Dark'}
          </button>
        </div>
        <p className="subtitle">Multimodal semantic search - Phase 4 serving demo</p>
      </header>

      {isMock === true && (
        <div className="mock-banner">
          MOCK MODE - results below are placeholders, not from a trained model.
          Phase 1 pretraining is still in progress.
        </div>
      )}

      <div className="examples">
        {EXAMPLE_QUERIES.map((ex) => (
          <button
            key={ex}
            type="button"
            className="example-chip"
            onClick={() => handleExampleClick(ex)}
            disabled={loading}
          >
            {ex}
          </button>
        ))}
      </div>

      <form className="search-form" onSubmit={handleSearch}>
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Describe an image... e.g. 'a dog in a park'  (press / to focus)"
        />
        <select
          className="topk-select"
          value={topK}
          onChange={(e) => setTopK(Number(e.target.value))}
          disabled={loading}
        >
          <option value={3}>Top 3</option>
          <option value={5}>Top 5</option>
          <option value={10}>Top 10</option>
        </select>
        <button type="submit" disabled={loading}>
          {loading ? 'Searching...' : 'Search'}
        </button>
        {(query || results.length > 0) && (
          <button type="button" className="clear-btn" onClick={handleClear} disabled={loading}>
            Clear
          </button>
        )}
      </form>

      {recentSearches.length > 0 && (
        <div className="recent-searches">
          <span className="recent-label">Recent:</span>
          {recentSearches.map((q) => (
            <button
              key={q}
              type="button"
              className="recent-chip"
              onClick={() => handleRecentClick(q)}
              disabled={loading}
            >
              {q}
            </button>
          ))}
        </div>
      )}

      {error && (
        <div className="error">
          {error}
          <button type="button" className="retry-btn" onClick={handleRetry}>
            Retry
          </button>
        </div>
      )}

      {!loading && !error && lastQuery && (
        <p className="results-count">
          {results.length} result{results.length !== 1 ? 's' : ''} for "{lastQuery}"
          {latency !== null && ` - ${latency.toFixed(1)} ms`}
        </p>
      )}

      <ul className="results">
        {loading &&
          Array.from({ length: topK }).map((_, i) => <ResultSkeleton key={i} />)}

        {!loading &&
          results.map((r) => (
            <li key={r.image_id} className="result-card">
              <div className="result-top">
                <div className="result-id">{r.image_id}</div>
                <button
                  type="button"
                  className="copy-btn"
                  onClick={() => handleCopy(r)}
                  title="Copy result"
                >
                  {copiedId === r.image_id ? 'Copied!' : 'Copy'}
                </button>
              </div>
              <div className="result-caption">{r.caption}</div>
              <div className="result-score">score: {r.score.toFixed(4)}</div>
            </li>
          ))}

        {!loading && results.length === 0 && !error && (
          <li className="empty-state">
            No results yet - try an example above or type your own query.
          </li>
        )}
      </ul>
    </div>
  )
}
