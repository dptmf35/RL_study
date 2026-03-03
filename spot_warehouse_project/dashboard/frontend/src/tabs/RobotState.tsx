import { useState, useMemo } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import { WsPayload } from '../hooks/useWebSocket'

interface Props {
  state: WsPayload | null
  history: WsPayload[]
}

// 19 distinct colors for joint lines
const JOINT_COLORS = [
  '#4fc3f7', '#ef5350', '#66bb6a', '#ffa726', '#ab47bc',
  '#26c6da', '#ff7043', '#9ccc65', '#ffca28', '#5c6bc0',
  '#ec407a', '#26a69a', '#8d6e63', '#78909c', '#42a5f5',
  '#d4e157', '#ff5722', '#00bcd4', '#9e9e9e',
]

const JOINT_NAMES = [
  'FL_hip', 'FL_thigh', 'FL_calf',
  'FR_hip', 'FR_thigh', 'FR_calf',
  'RL_hip', 'RL_thigh', 'RL_calf',
  'RR_hip', 'RR_thigh', 'RR_calf',
  'arm_sh0', 'arm_sh1', 'arm_el0', 'arm_el1',
  'arm_wr0', 'arm_wr1', 'gripper',
]

interface CommandBarProps {
  label: string
  value: number
  min: number
  max: number
  unit: string
  color: string
}

function CommandBar({ label, value, min, max, unit, color }: CommandBarProps) {
  const range = max - min
  const zeroPct = ((0 - min) / range) * 100
  const valPct = ((value - min) / range) * 100
  const left = Math.min(zeroPct, valPct)
  const width = Math.abs(valPct - zeroPct)

  return (
    <div className="command-bar-row">
      <div className="command-bar-label">
        <span className="command-bar-name">{label}</span>
        <span className="command-bar-value" style={{ color }}>
          {value.toFixed(3)} {unit}
        </span>
      </div>
      <div className="command-bar-track">
        <div className="command-bar-zero" style={{ left: `${zeroPct}%` }} />
        <div
          className="command-bar-fill"
          style={{ left: `${left}%`, width: `${width}%`, backgroundColor: color }}
        />
      </div>
      <div className="command-bar-limits">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  )
}

export default function RobotState({ state, history }: Props) {
  const [showVel, setShowVel] = useState(false)

  // Build chart data from history (last 100 points = ~10s at 10 Hz)
  const chartData = useMemo(() => {
    const slice = history.slice(-100)
    if (slice.length === 0) return []
    const latest = slice[slice.length - 1].timestamp
    return slice.map(s => {
      const row: Record<string, number> = {
        t: -(latest - s.timestamp),
      }
      const arr = showVel ? s.joint_vel : s.joint_pos
      arr.forEach((v, i) => {
        row[`j${i}`] = v
      })
      return row
    })
  }, [history, showVel])

  const cmd = state?.command ?? [0, 0, 0]
  const pose = state?.pose ?? [0, 0, 0, 1, 0, 0, 0]

  return (
    <div className="tab-content">
      {/* Command Panel */}
      <section className="panel">
        <h2 className="panel-title">Velocity Commands</h2>
        <div className="command-bars">
          <CommandBar label="v_x" value={cmd[0]} min={-2} max={3} unit="m/s" color="#4fc3f7" />
          <CommandBar label="v_y" value={cmd[1]} min={-1.5} max={1.5} unit="m/s" color="#66bb6a" />
          <CommandBar label="w_z" value={cmd[2]} min={-2} max={2} unit="rad/s" color="#ffa726" />
        </div>
      </section>

      {/* Joint Graph */}
      <section className="panel panel--wide">
        <div className="panel-header">
          <h2 className="panel-title">Joint Data</h2>
          <div className="toggle-group">
            <button
              className={`toggle-btn ${!showVel ? 'active' : ''}`}
              onClick={() => setShowVel(false)}
            >
              Position (rad)
            </button>
            <button
              className={`toggle-btn ${showVel ? 'active' : ''}`}
              onClick={() => setShowVel(true)}
            >
              Velocity (rad/s)
            </button>
          </div>
        </div>

        <div className="chart-wrapper">
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis
                dataKey="t"
                type="number"
                domain={['auto', 0]}
                tickFormatter={v => `${(v as number).toFixed(1)}s`}
                tick={{ fill: '#7a8fa6', fontSize: 11 }}
                stroke="rgba(255,255,255,0.1)"
              />
              <YAxis
                tick={{ fill: '#7a8fa6', fontSize: 11 }}
                stroke="rgba(255,255,255,0.1)"
                width={42}
              />
              <Tooltip
                contentStyle={{
                  background: '#0f1923',
                  border: '1px solid rgba(79,195,247,0.3)',
                  borderRadius: '6px',
                  fontSize: '11px',
                }}
                labelStyle={{ color: '#7a8fa6' }}
                itemStyle={{ color: '#c8d8e8' }}
                formatter={(value: number) => value.toFixed(4)}
                labelFormatter={v => `${(v as number).toFixed(2)}s`}
              />
              {JOINT_NAMES.map((name, i) => (
                <Line
                  key={name}
                  type="monotone"
                  dataKey={`j${i}`}
                  name={name}
                  stroke={JOINT_COLORS[i]}
                  strokeWidth={1}
                  dot={false}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Scrollable legend */}
        <div className="joint-legend">
          {JOINT_NAMES.map((name, i) => (
            <div key={name} className="legend-item">
              <span className="legend-dot" style={{ backgroundColor: JOINT_COLORS[i] }} />
              <span className="legend-name">{name}</span>
            </div>
          ))}
        </div>
      </section>

      {/* Robot Pose */}
      <section className="panel">
        <h2 className="panel-title">Robot Pose</h2>
        <div className="pose-grid">
          <div className="pose-group">
            <div className="pose-group-label">Position</div>
            <div className="pose-values">
              <div className="pose-item">
                <span className="pose-axis" style={{ color: '#ef5350' }}>X</span>
                <span className="pose-val">{pose[0].toFixed(4)}</span>
                <span className="pose-unit">m</span>
              </div>
              <div className="pose-item">
                <span className="pose-axis" style={{ color: '#66bb6a' }}>Y</span>
                <span className="pose-val">{pose[1].toFixed(4)}</span>
                <span className="pose-unit">m</span>
              </div>
              <div className="pose-item">
                <span className="pose-axis" style={{ color: '#4fc3f7' }}>Z</span>
                <span className="pose-val">{pose[2].toFixed(4)}</span>
                <span className="pose-unit">m</span>
              </div>
            </div>
          </div>
          <div className="pose-group">
            <div className="pose-group-label">Quaternion</div>
            <div className="pose-values">
              {(['qw', 'qx', 'qy', 'qz'] as const).map((q, i) => (
                <div key={q} className="pose-item">
                  <span className="pose-axis" style={{ color: '#ab47bc' }}>{q}</span>
                  <span className="pose-val">{pose[3 + i].toFixed(4)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}
