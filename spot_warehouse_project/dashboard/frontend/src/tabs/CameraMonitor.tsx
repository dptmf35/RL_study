import { useState, useRef } from "react";

const CAMERAS = [
  { key: "frontleft",  label: "Front Left",  topic: "/scar/camera/frontleft/image" },
  { key: "frontright", label: "Front Right", topic: "/scar/camera/frontright/image" },
];

function CameraFeed({ camKey, label, topic }: { camKey: string; label: string; topic: string }) {
  const [status, setStatus] = useState<"loading" | "ok" | "error">("loading");
  const imgRef = useRef<HTMLImageElement>(null);

  const src = `/api/camera/${camKey}`;

  return (
    <div className="camera-card">
      <div className="camera-header">
        <span className="camera-label">{label}</span>
        <span className={`camera-status ${status}`}>
          {status === "loading" ? "⏳ connecting" : status === "ok" ? "● live" : "✕ no signal"}
        </span>
        <span className="camera-topic">{topic}</span>
      </div>
      <div className="camera-frame">
        <img
          ref={imgRef}
          src={src}
          alt={label}
          onLoad={() => setStatus("ok")}
          onError={() => setStatus("error")}
          style={{ width: "100%", display: status === "error" ? "none" : "block" }}
        />
        {status === "error" && (
          <div className="camera-placeholder">
            <div className="camera-no-signal">
              <span style={{ fontSize: "2rem" }}>📷</span>
              <p>No signal</p>
              <p style={{ fontSize: "0.75rem", opacity: 0.5 }}>{topic}</p>
              <button
                className="retry-btn"
                onClick={() => {
                  setStatus("loading");
                  if (imgRef.current) {
                    imgRef.current.src = src + "?t=" + Date.now();
                  }
                }}
              >
                Retry
              </button>
            </div>
          </div>
        )}
        {status === "loading" && (
          <div className="camera-placeholder">
            <div className="camera-no-signal">
              <div className="spinner" />
              <p style={{ marginTop: "0.75rem" }}>Waiting for stream…</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default function CameraMonitor() {
  return (
    <div className="camera-monitor">
      <div className="camera-grid">
        {CAMERAS.map((cam) => (
          <CameraFeed key={cam.key} camKey={cam.key} label={cam.label} topic={cam.topic} />
        ))}
      </div>
    </div>
  );
}
