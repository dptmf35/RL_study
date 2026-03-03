import { useState, useEffect, useCallback } from 'react'

export interface ApiStatus {
  recording: boolean
  episode_count: number
  recording_duration: number
  file_size_bytes: number
  connected_to_sim: boolean
}

export interface Episode {
  id: string
  name: string
  n_steps: number
  duration_sec: number
  start_time: string
}

export interface StartRecordingResponse {
  episode: string
  status: string
}

export interface StopRecordingResponse {
  status: string
  episode: string
}

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(path, options)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json() as Promise<T>
}

export function useStatus(intervalMs = 2000) {
  const [status, setStatus] = useState<ApiStatus | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true

    const poll = async () => {
      try {
        const data = await apiFetch<ApiStatus>('/api/status')
        if (alive) {
          setStatus(data)
          setError(null)
        }
      } catch (e) {
        if (alive) setError(String(e))
      }
    }

    poll()
    const id = setInterval(poll, intervalMs)
    return () => {
      alive = false
      clearInterval(id)
    }
  }, [intervalMs])

  return { status, error }
}

export function useEpisodes(intervalMs = 3000) {
  const [episodes, setEpisodes] = useState<Episode[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let alive = true

    const poll = async () => {
      try {
        const data = await apiFetch<Episode[]>('/api/episodes')
        if (alive) {
          setEpisodes(data)
          setLoading(false)
        }
      } catch {
        if (alive) setLoading(false)
      }
    }

    poll()
    const id = setInterval(poll, intervalMs)
    return () => {
      alive = false
      clearInterval(id)
    }
  }, [intervalMs])

  return { episodes, loading }
}

export function useRecordingControls() {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const startRecording = useCallback(async (): Promise<StartRecordingResponse | null> => {
    setBusy(true)
    setError(null)
    try {
      const data = await apiFetch<StartRecordingResponse>('/api/record/start', { method: 'POST' })
      return data
    } catch (e) {
      setError(String(e))
      return null
    } finally {
      setBusy(false)
    }
  }, [])

  const stopRecording = useCallback(async (): Promise<StopRecordingResponse | null> => {
    setBusy(true)
    setError(null)
    try {
      const data = await apiFetch<StopRecordingResponse>('/api/record/stop', { method: 'POST' })
      return data
    } catch (e) {
      setError(String(e))
      return null
    } finally {
      setBusy(false)
    }
  }, [])

  return { startRecording, stopRecording, busy, error }
}
