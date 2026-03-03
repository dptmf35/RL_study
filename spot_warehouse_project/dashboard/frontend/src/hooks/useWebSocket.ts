import { useState, useEffect, useRef, useCallback } from 'react'

export interface WsPayload {
  timestamp: number
  command: number[]       // [v_x, v_y, w_z]
  pose: number[]          // [x, y, z, qw, qx, qy, qz]
  joint_pos: number[]     // 19 floats
  joint_vel: number[]     // 19 floats
  recording: boolean
  episode_count: number
  recording_duration: number
}

const HISTORY_SIZE = 200
const RECONNECT_DELAY = 2000

function getWsUrl(): string {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  return `${proto}://${location.host}/ws`
}

export function useWebSocket() {
  const [state, setState] = useState<WsPayload | null>(null)
  const [connected, setConnected] = useState(false)
  const [history, setHistory] = useState<WsPayload[]>([])
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const mountedRef = useRef(true)

  const connect = useCallback(() => {
    if (!mountedRef.current) return
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return

    try {
      const ws = new WebSocket(getWsUrl())
      wsRef.current = ws

      ws.onopen = () => {
        if (!mountedRef.current) return
        setConnected(true)
      }

      ws.onmessage = (event: MessageEvent) => {
        if (!mountedRef.current) return
        try {
          const payload: WsPayload = JSON.parse(event.data as string)
          setState(payload)
          setHistory(prev => {
            const next = [...prev, payload]
            if (next.length > HISTORY_SIZE) {
              return next.slice(next.length - HISTORY_SIZE)
            }
            return next
          })
        } catch {
          // ignore malformed frames
        }
      }

      ws.onerror = () => {
        if (!mountedRef.current) return
        setConnected(false)
      }

      ws.onclose = () => {
        if (!mountedRef.current) return
        setConnected(false)
        wsRef.current = null
        reconnectTimer.current = setTimeout(() => {
          if (mountedRef.current) connect()
        }, RECONNECT_DELAY)
      }
    } catch {
      setConnected(false)
      reconnectTimer.current = setTimeout(() => {
        if (mountedRef.current) connect()
      }, RECONNECT_DELAY)
    }
  }, [])

  useEffect(() => {
    mountedRef.current = true
    connect()
    return () => {
      mountedRef.current = false
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
      if (wsRef.current) {
        wsRef.current.onclose = null
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, [connect])

  return { state, connected, history }
}
