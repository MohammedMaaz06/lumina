import { useState, useEffect, useRef, useCallback } from 'react'

const API_BASE = ''
const LS_THEME_KEY = 'lumina_theme'
const LS_RECENT_KEY = 'lumina_recent_searches'
const LS_LATENCIES_KEY = 'lumina_latencies'
const MAX_LATENCY_SAMPLES = 50

const EXAMPLE_QUERIES = [
  'a dog in a park',
  'people cooking in a kitchen',
  'children playing outside',
  'a city street at night',
  'a mountain landscape',
]

const searchCache = new Map()

function ResultSkeleton() {
  return (
    <li className="result-card skeleton">
      <div className="skeleton-line skeleton-id" />
      <div className="skeleton-line skeleton-caption" />
      <div className="skeleton-line skeleton-score" />
    </li>
  )
}

function getUrlParams() {
  const params = new URLSearchParams(window.location.search)
  return {
    q: params.get('q') || '',
    k: Number(params.get('k')) || 5,
  }
}

function setUrlParams(q, k) {
  const params = new URLSearchParams()
  if (q) params.set('q', q)
  params.set('k', String(k))
  const newUrl = `${window.location.pathname}?${params.toString()}`
  window.history.replaceState(null, '', newUrl)
}

function loadFromStorage(key, fallback) {
  try {
    const raw = localStorage.getItem(key)
    return raw ? JSON.parse(raw) : fallback
  } catch {
    return fallback
  }
}

function saveToStorage(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value))
  } catch {
  }
}

function percentile(sorted, p) {
  if (sorted.length === 0) return null
  const idx = Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length))
  return sorted[idx]
}

export default function App() {
  const initial = getUrlParams()
  const [query, setQuery] = useState(initial.q)
  const [topK, setTopK] = useState(initial.k)
  const [results, setResults] = useState([])
  const [isMock, setIsMock] = useState(null)
  const [latency, setLatency] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [copiedId, setCopiedId] = useState(null)
  const [recentSearches, setRecentSearches] = useState(() => loadFromStorage(LS_RECENT_KEY, []))
  const [theme, setTheme] = useState(() => loadFromStorage(LS_THEME_KEY, 'dark'))
  const [lastQuery, setLastQuery] = useState(null)
  const [fromCache, setFromCache] = useState(false)
  const [latencies, setLatencies] = useState(() => loadFromStorage(LS_LATENCIES_KEY, []))
  const [showStats, setShowStats] = useState(false)
  const inputRef = useRef(null)
  const abortRef = useRef(null)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    saveToStorage(LS_THEME_KEY, theme)
  }, [theme])

  useEffect(() => {
    saveToStorage(LS_RECENT_KEY, recentSearches)
  }, [recentSearches])

  useEffect(() => {
    saveToStorage(LS_LATENCIES_KEY, latencies)
  }, [latencies])

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

  useEffect(() => {
    if (initial.q) {
      runSearch(initial.q, initial.k)
    }
    return () => {
      abortRef.current?.abort()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function toggleTheme() {
    setTheme((t) => (t === 'dark' ? 'light' : 'dark'))
  }

  const runSearch = useCallback(async (q, k) => {
    if (!q.trim()) return

    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setLoading(true)
    setError(null)
    setFromCache(false)
    setUrlParams(q, k)

    const cacheKey = `${q.toLowerCase().trim()}::${k}`
    if (searchCache.has(cacheKey)) {
      const cached = searchCache.get(cacheKey)
      setResults(cached.results)
      setIsMock(cached.mock)
      setLatency(cached.latency_ms)
      setLastQuery(q)
      setFromCache(true)
      setLoading(false)
      setRecentSearches((prev) => [q, ...prev.filter((item) => item !== q)].slice(0, 5))
      return
    }

    try {
      const res = await fetch(`${API_BASE}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, top_k: k }),
        signal: controller.signal,
      })
      if (!res.ok) throw new Error(`Server responded ${res.status}`)
      const data = await res.json()
      searchCache.set(cacheKey, data)
      setResults(data.results)
      setIsMock(data.mock)
      setLatency(data.latency_ms)
      setLastQuery(q)
      setRecentSearches((prev) => [q, ...prev.filter((item) => item !== q)].slice(0, 5))
      setLatencies((prev) => [...prev, data.latency_ms].slice(-MAX_LATENCY_SAMPLES))
    } catch (err) {
      if (err.name === 'AbortError') return
      setError(err.message || 'Search failed - is the backend running on :8000?')
    } finally {
      if (abortRef.current === controller) {
        setLoading(false)
      }
    }
  }, [])

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
    setUrlParams('', topK)
  }

  function handleClearHistory() {
    setRecentSearches([])
    setLatencies([])
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

  function handleExportJson() {
    if (results.length === 0) return
    const payload = {
      query: lastQuery,
      mock: isMock,
      latency_ms: latency,
      results,
      exported_at: new Date().toISOString(),
    }
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `lumina-search-${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  function handleShare() {
    setUrlParams(query, topK)
    navigator.clipboard?.writeText(window.location.href).catch(() => {})
    setCopiedId('__share__')
    setTimeout(() => setCopiedId(null), 1500)
  }

  const sortedLatencies = [...latencies].sort((a, b) => a - b)
  const p50 = percentile(sortedLatencies, 50)
  const p95 = percentile(sortedLatencies, 95)

  return (
    <div className="page">
      <header className="header">
        <div className="header-top">
          <h1>Lumina</h1>
          <div className="header-actions">
            <button
              type="button"
              className="theme-toggle"
              onClick={() => setShowStats((s) => !s)}
              title="Session stats"
            >
              Stats
            </button>
            <button type="button" className="theme-toggle" onClick={toggleTheme} title="Toggle theme">
              {theme === 'dark' ? 'Light' : 'Dark'}
            </button>
          </div>
        </div>
        <p className="subtitle">Multimodal semantic search - Phase 4 serving demo</p>
      </header>

      {showStats && (
        <div className="stats-panel">
          <div className="stats-row">
            <span>Searches this session</span>
            <strong>{latencies.length}</strong>
          </div>
          <div className="stats-row">
            <span>p50 latency</span>
            <strong>{p50 !== null ? `${p50.toFixed(1)} ms` : '-'}</strong>
          </div>
          <div className="stats-row">
            <span>p95 latency</span>
            <strong>{p95 !== null ? `${p95.toFixed(1)} ms` : '-'}</strong>
          </div>
          <button type="button" className="toolbar-btn" onClick={handleClearHistory}>
            Clear history
          </button>
        </div>
      )}

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
        <div className="results-toolbar">
          <p className="results-count">
            {results.length} result{results.length !== 1 ? 's' : ''} for "{lastQuery}"
            {latency !== null && ` - ${latency.toFixed(1)} ms`}
            {fromCache && ' (cached)'}
          </p>
          <div className="toolbar-actions">
            <button type="button" className="toolbar-btn" onClick={handleShare}>
              {copiedId === '__share__' ? 'Link copied!' : 'Share'}
            </button>
            <button type="button" className="toolbar-btn" onClick={handleExportJson}>
              Export JSON
            </button>
          </div>
        </div>
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
