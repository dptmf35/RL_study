import { useState, useEffect, useRef } from 'react'
import { useEpisodes, useRecordingControls } from '../hooks/useApi'
import { WsPayload } from '../hooks/useWebSocket'

interface Props {
  state: WsPayload | null
  recording: boolean
  episodeCount: number
  recordingDuration: number
  fileSizeBytes: number
  connectedToSim: boolean
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

function formatDuration(secs: number): string {
  const m = Math.floor(secs / 60)
  const s = Math.floor(secs % 60)
  const ms = Math.floor((secs % 1) * 10)
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${ms}`
}

function formatTime(isoStr: string): string {
  try {
    const d = new Date(isoStr)
    return d.toLocaleTimeString('en-US', { hour12: false })
  } catch {
    return isoStr
  }
}

export default function DataCollection({
  state,
  recording,
  episodeCount,
  recordingDuration,
  fileSizeBytes,
  connectedToSim,
}: Props) {
  const { episodes, loading } = useEpisodes(3000)
  const { startRecording, stopRecording, busy, error } = useRecordingControls()

  // Live timer that ticks up during recording
  const [liveTimer, setLiveTimer] = useState(0)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const recordingRef = useRef(recording)
  recordingRef.current = recording

  useEffect(() => {
    if (recording) {
      setLiveTimer(recordingDuration)
      timerRef.current = setInterval(() => {
        setLiveTimer(t => t + 0.1)
      }, 100)
    } else {
      if (timerRef.current) clearInterval(timerRef.current)
      setLiveTimer(recordingDuration)
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [recording, recordingDuration])

  const handleToggle = async () => {
    if (recording) {
      await stopRecording()
    } else {
      await startRecording()
    }
  }

  // Use state for real-time recording status if ws gives fresh data
  const isRecording = state?.recording ?? recording

  return (
    <div className="tab-content">
      {/* Recording Controls */}
      <section className="panel panel--centered">
        <h2 className="panel-title">Recording Control</h2>

        <div className="record-main">
          <button
            className={`record-btn ${isRecording ? 'record-btn--stop' : 'record-btn--start'}`}
            onClick={handleToggle}
            disabled={busy}
          >
            {busy ? (
              <span className="spinner" />
            ) : isRecording ? (
              <>
                <span className="record-icon-stop" />
                Stop Recording
              </>
            ) : (
              <>
                <span className="record-icon-start" />
                Start Recording
              </>
            )}
          </button>

          {isRecording && (
            <div className="record-timer">
              <span className="rec-dot" />
              <span className="rec-label">REC</span>
              <span className="rec-time">{formatDuration(liveTimer)}</span>
            </div>
          )}
        </div>

        {error && <div className="error-banner">{error}</div>}
      </section>

      {/* Stats Row */}
      <section className="panel">
        <h2 className="panel-title">Session Statistics</h2>
        <div className="stats-row">
          <div className="stat-card">
            <div className="stat-value">{episodeCount}</div>
            <div className="stat-label">Episodes</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{formatBytes(fileSizeBytes)}</div>
            <div className="stat-label">File Size</div>
          </div>
          <div className="stat-card">
            <div
              className="stat-value"
              style={{ color: connectedToSim ? '#66bb6a' : '#ef5350' }}
            >
              {connectedToSim ? 'Online' : 'Offline'}
            </div>
            <div className="stat-label">Simulator</div>
          </div>
          <div className="stat-card">
            <div
              className="stat-value"
              style={{ color: isRecording ? '#ef5350' : '#7a8fa6' }}
            >
              {isRecording ? 'Active' : 'Idle'}
            </div>
            <div className="stat-label">Status</div>
          </div>
        </div>
      </section>

      {/* Episode List */}
      <section className="panel panel--wide">
        <div className="panel-header">
          <h2 className="panel-title">Episode Log</h2>
          <span className="badge">{episodes.length} episodes</span>
        </div>

        {loading ? (
          <div className="loading-text">Loading episodes...</div>
        ) : episodes.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">○</div>
            <div className="empty-msg">No episodes recorded yet.</div>
            <div className="empty-sub">Start recording to capture your first episode.</div>
          </div>
        ) : (
          <div className="table-wrapper">
            <table className="episode-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Name</th>
                  <th>Duration</th>
                  <th>Steps</th>
                  <th>Start Time</th>
                </tr>
              </thead>
              <tbody>
                {[...episodes].reverse().map((ep, idx) => (
                  <tr key={ep.id} className={idx === 0 ? 'row--latest' : ''}>
                    <td className="cell--id">{ep.id}</td>
                    <td className="cell--name">{ep.name}</td>
                    <td className="cell--mono">{ep.duration_sec.toFixed(2)}s</td>
                    <td className="cell--mono">{ep.n_steps.toLocaleString()}</td>
                    <td className="cell--time">{formatTime(ep.start_time)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  )
}
