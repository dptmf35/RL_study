import { useState } from 'react'
import { useWebSocket } from './hooks/useWebSocket'
import { useStatus } from './hooks/useApi'
import RobotState from './tabs/RobotState'
import DataCollection from './tabs/DataCollection'
import TrainingMonitor from './tabs/TrainingMonitor'

type Tab = 'robot' | 'data' | 'training'

const TABS: { id: Tab; label: string }[] = [
  { id: 'robot', label: '🤖 Robot State' },
  { id: 'data', label: '📹 Data Collection' },
  { id: 'training', label: '📊 Training Monitor' },
]

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('robot')
  const { state, connected, history } = useWebSocket()
  const { status } = useStatus(2000)

  // Prefer WebSocket data for real-time values, fall back to REST status
  const recording = state?.recording ?? status?.recording ?? false
  const episodeCount = state?.episode_count ?? status?.episode_count ?? 0
  const recordingDuration = state?.recording_duration ?? status?.recording_duration ?? 0
  const fileSizeBytes = status?.file_size_bytes ?? 0
  const connectedToSim = status?.connected_to_sim ?? false

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <div className="header-logo">
            <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
              <rect x="4" y="10" width="20" height="12" rx="3" stroke="#4fc3f7" strokeWidth="1.5" />
              <circle cx="10" cy="16" r="2.5" fill="#4fc3f7" />
              <circle cx="18" cy="16" r="2.5" fill="#4fc3f7" />
              <path d="M10 10V7" stroke="#4fc3f7" strokeWidth="1.5" strokeLinecap="round" />
              <path d="M18 10V7" stroke="#4fc3f7" strokeWidth="1.5" strokeLinecap="round" />
              <path d="M14 10V6" stroke="#4fc3f7" strokeWidth="1.5" strokeLinecap="round" />
              <path d="M4 19H2" stroke="#4fc3f7" strokeWidth="1.5" strokeLinecap="round" />
              <path d="M26 19H24" stroke="#4fc3f7" strokeWidth="1.5" strokeLinecap="round" />
              <path d="M6 22H4v2" stroke="#4fc3f7" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M22 22H24v2" stroke="#4fc3f7" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
          <div>
            <div className="header-title">Spot Data Collection</div>
            <div className="header-sub">IsaacSim Control Dashboard</div>
          </div>
        </div>

        <div className="header-right">
          {recording && (
            <div className="rec-badge">
              <span className="rec-badge-dot" />
              REC
            </div>
          )}

          <div className="episode-chip">
            <span className="episode-chip-label">EP</span>
            <span className="episode-chip-count">{episodeCount}</span>
          </div>

          <div className={`conn-status ${connected ? 'conn-status--on' : 'conn-status--off'}`}>
            <span className="conn-dot" />
            <span className="conn-label">{connected ? 'Live' : 'Offline'}</span>
          </div>
        </div>
      </header>

      {/* Tab Navigation */}
      <nav className="tab-nav">
        {TABS.map(tab => (
          <button
            key={tab.id}
            className={`tab-btn ${activeTab === tab.id ? 'tab-btn--active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
        <div className="tab-nav-rule" />
      </nav>

      {/* Tab Content */}
      <main className="main">
        {activeTab === 'robot' && (
          <RobotState state={state} history={history} />
        )}
        {activeTab === 'data' && (
          <DataCollection
            state={state}
            recording={recording}
            episodeCount={episodeCount}
            recordingDuration={recordingDuration}
            fileSizeBytes={fileSizeBytes}
            connectedToSim={connectedToSim}
          />
        )}
        {activeTab === 'training' && (
          <TrainingMonitor />
        )}
      </main>
    </div>
  )
}
