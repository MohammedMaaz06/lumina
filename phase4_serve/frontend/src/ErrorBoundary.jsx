import { Component } from 'react'

export class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }

  componentDidCatch(error, info) {
    console.error('Lumina crashed:', error, info)
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null })
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="crash-boundary">
          <h2>Something went wrong</h2>
          <p>The app hit an unexpected error and stopped rendering to avoid showing broken UI.</p>
          <pre className="crash-detail">{String(this.state.error)}</pre>
          <button type="button" onClick={this.handleReset} className="crash-retry">
            Try again
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
