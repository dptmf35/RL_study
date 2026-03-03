import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts'

// Placeholder data for visual demo
const placeholderRewards = Array.from({ length: 20 }, (_, i) => ({
  epoch: i + 1,
  reward: Math.max(0.1, Math.sin(i * 0.4) * 0.4 + 0.5 + Math.random() * 0.1),
}))

export default function TrainingMonitor() {
  return (
    <div className="tab-content">
      {/* Coming Soon Banner */}
      <section className="panel panel--centered panel--coming-soon">
        <div className="coming-soon-icon">
          <svg width="64" height="64" viewBox="0 0 64 64" fill="none">
            <circle cx="32" cy="32" r="30" stroke="#4fc3f7" strokeWidth="1.5" strokeDasharray="4 3" />
            <path
              d="M32 18v14l8 8"
              stroke="#4fc3f7"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <circle cx="32" cy="32" r="2" fill="#4fc3f7" />
          </svg>
        </div>
        <h2 className="coming-soon-title">Training Monitor</h2>
        <p className="coming-soon-sub">
          Real-time training metrics, reward curves, and policy performance visualization
          will be available in the next release.
        </p>
        <div className="coming-soon-tags">
          <span className="feature-tag">Reward Curves</span>
          <span className="feature-tag">Policy Loss</span>
          <span className="feature-tag">Value Function</span>
          <span className="feature-tag">Episode Returns</span>
        </div>
      </section>

      {/* Placeholder chart - reward preview */}
      <section className="panel panel--wide">
        <div className="panel-header">
          <h2 className="panel-title">Reward History</h2>
          <span className="badge badge--muted">Preview</span>
        </div>
        <div className="placeholder-overlay-wrapper">
          <div className="chart-wrapper">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={placeholderRewards} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis
                  dataKey="epoch"
                  tick={{ fill: '#7a8fa6', fontSize: 11 }}
                  stroke="rgba(255,255,255,0.08)"
                  label={{ value: 'Epoch', position: 'insideBottom', offset: -4, fill: '#7a8fa6', fontSize: 11 }}
                />
                <YAxis
                  tick={{ fill: '#7a8fa6', fontSize: 11 }}
                  stroke="rgba(255,255,255,0.08)"
                  domain={[0, 1]}
                  width={42}
                />
                <Tooltip
                  contentStyle={{
                    background: '#0f1923',
                    border: '1px solid rgba(79,195,247,0.2)',
                    borderRadius: '6px',
                    fontSize: '11px',
                  }}
                  cursor={{ fill: 'rgba(79,195,247,0.05)' }}
                />
                <Bar dataKey="reward" name="Mean Reward" radius={[3, 3, 0, 0]}>
                  {placeholderRewards.map((entry, index) => (
                    <Cell
                      key={index}
                      fill={`rgba(79,195,247,${0.2 + entry.reward * 0.6})`}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="placeholder-overlay">
            <span>Sample Data — Connect training pipeline to enable</span>
          </div>
        </div>
      </section>

      {/* Metric cards placeholder */}
      <section className="panel">
        <h2 className="panel-title">Training Metrics</h2>
        <div className="stats-row">
          {[
            { label: 'Policy Loss', value: '—' },
            { label: 'Value Loss', value: '—' },
            { label: 'Mean Return', value: '—' },
            { label: 'KL Divergence', value: '—' },
          ].map(m => (
            <div key={m.label} className="stat-card stat-card--placeholder">
              <div className="stat-value stat-value--muted">{m.value}</div>
              <div className="stat-label">{m.label}</div>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}
