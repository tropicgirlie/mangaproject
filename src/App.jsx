import { useEffect, useRef, useState, useCallback } from 'react';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { MangaWebGLRenderer } from './engine/mangaShader';
import {
  evaluateGazeLock,
  extractShaderEyeUniforms,
  renderMangaEyesOverlay,
  renderShuchuSenSpeedLines,
  renderDebugLandmarksOverlay,
} from './engine/mangaOverlay';

// Dwell timing specifications
const DWELL_FILL_DURATION_SEC = 1.2; // 0.0 -> 1.0 over 1.2 seconds
const DWELL_DECAY_RATE = 1.45;       // Smooth decay back to 0 when user looks away
const STAGE1_THRESHOLD = 0.15;       // Vector Manga Eyes trigger at progress > 0.15
const STAGE2_COOLDOWN_SEC = 3.0;     // 3-second cooldown before reverting when looking away in Stage 2

const COLOR_STYLES = [
  { id: 0, label: '01 Golden Hour', swatch: '#FF9E7D' },
  { id: 1, label: '02 Shinkai Sky', swatch: '#38BDF8' },
  { id: 2, label: '03 Neo-Tokyo Cel', swatch: '#FB7185' },
];

export default function App() {
  // Permission Gate State
  const [gateOpen, setGateOpen] = useState(true);
  const [isInitializing, setIsInitializing] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [modelStatus, setModelStatus] = useState('Standby');

  // Active Colorful Anime Style (0 = Golden Hour Nostalgia, 1 = Shinkai Vivid Sky, 2 = Neo-Tokyo Cel)
  const [colorStyle, setColorStyle] = useState(0);

  // UI & Interaction State Machine React Snapshots (throttled for 60fps performance)
  const [uiState, setUiState] = useState({
    stage: 'IDLE', // 'IDLE' | 'GAZE_LOCK' | 'STAGE_1' | 'STAGE_2'
    dwellProgress: 0,
    isGazeLocked: false,
    faceDetected: false,
    cooldownRemaining: 0,
    fps: 60,
    yaw: 0,
    pitch: 0,
    irisOffset: 0,
    eyeBlinkLeft: 0,
    eyeBlinkRight: 0,
  });

  // Screen shake state on Stage 2 snap (150ms)
  const [isShaking, setIsShaking] = useState(false);

  // Google Material Expressive "Mangaface Effect" Debug Drawer State
  const [debugOpen, setDebugOpen] = useState(false);
  const [showLandmarkMesh, setShowLandmarkMesh] = useState(false);

  // DOM & Engine Refs
  const webglCanvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const videoRef = useRef(null);
  const rendererRef = useRef(null);
  const landmarkerRef = useRef(null);
  const streamRef = useRef(null);

  // High-frequency Mutable State Machine Ref
  const machineRef = useRef({
    dwellProgress: 0.0,
    stage2Locked: false,
    stage2CooldownSec: 0.0,
    lastFrameTimeMs: performance.now(),
    lastVideoTime: -1,
    latestLandmarks: null,
    latestBlendshapes: null,
    gazeMetrics: {
      isGazeLocked: false,
      yaw: 0,
      pitch: 0,
      irisOffset: 0,
      eyeBlinkLeft: 0,
      eyeBlinkRight: 0,
      faceDetected: false,
    },
    fpsFrames: 0,
    fpsLastSampleMs: performance.now(),
    currentFps: 60,
    lastUiSyncMs: 0,
  });

  // Load MediaPipe FaceLandmarker with iris landmarks and face blendshapes enabled
  const ensureFaceLandmarker = useCallback(async () => {
    if (landmarkerRef.current) return landmarkerRef.current;

    const baseUrl = import.meta.env.BASE_URL || './';
    setModelStatus('Loading MediaPipe Vision WASM...');
    let vision;
    try {
      vision = await FilesetResolver.forVisionTasks(`${baseUrl}wasm`);
    } catch {
      vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm'
      );
    }

    const modelPath = `${baseUrl}models/face_landmarker.task`;
    setModelStatus('Initializing FaceLandmarker (Iris + Blendshapes)...');
    let landmarker;
    try {
      landmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: modelPath,
          delegate: 'GPU',
        },
        outputFaceBlendshapes: true,
        runningMode: 'VIDEO',
        numFaces: 1,
      });
    } catch {
      landmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: modelPath,
          delegate: 'CPU',
        },
        outputFaceBlendshapes: true,
        runningMode: 'VIDEO',
        numFaces: 1,
      });
    }

    landmarkerRef.current = landmarker;
    setModelStatus('FaceLandmarker Active');
    return landmarker;
  }, []);

  // Trigger 150ms Screen Shake on Stage 2 snap
  const triggerStage2Shake = useCallback(() => {
    setIsShaking(true);
    setTimeout(() => {
      setIsShaking(false);
    }, 160);
  }, []);

  // Start Live Webcam Feed — Calls getUserMedia synchronously on user click
  const handleEnableCamera = useCallback(async () => {
    setIsInitializing(true);
    setCameraError(null);

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Browser camera API is unavailable. Please ensure you are on HTTPS or localhost.');
      }

      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: 'user',
          },
          audio: false,
        });
      } catch {
        stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });
      }

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setGateOpen(false);
      await ensureFaceLandmarker();
    } catch (err) {
      const errName = err?.name ? `${err.name}: ` : '';
      const msg = err?.message || 'Camera permission was denied.';
      setCameraError(`${errName}${msg}`);
    } finally {
      setIsInitializing(false);
    }
  }, [ensureFaceLandmarker]);

  // Download Manga Frame Snapshot CTA in Stage 2
  const handleCaptureMangaPanel = useCallback(() => {
    const glCanvas = webglCanvasRef.current;
    const ovCanvas = overlayCanvasRef.current;
    if (!glCanvas || !ovCanvas) return;

    const exportCanvas = document.createElement('canvas');
    exportCanvas.width = glCanvas.width;
    exportCanvas.height = glCanvas.height;
    const ctx = exportCanvas.getContext('2d');

    ctx.drawImage(glCanvas, 0, 0);
    ctx.drawImage(ovCanvas, 0, 0);

    // Stamp Japanese editorial colophon on exported panel
    ctx.fillStyle = '#0A0A0A';
    ctx.fillRect(24, exportCanvas.height - 64, 290, 40);
    ctx.strokeStyle = '#F4F1EA';
    ctx.lineWidth = 2;
    ctx.strokeRect(24, exportCanvas.height - 64, 290, 40);
    ctx.fillStyle = '#F4F1EA';
    ctx.font = 'bold 14px "Space Mono", monospace';
    ctx.fillText('MANGAFACE EFFECT // STAGE 02', 38, exportCanvas.height - 39);

    const link = document.createElement('a');
    link.download = `mangaface-panel-${Date.now()}.png`;
    link.href = exportCanvas.toDataURL('image/png');
    link.click();
  }, []);

  // Main 60 FPS Render & State Machine Loop
  useEffect(() => {
    const glCanvas = webglCanvasRef.current;
    const ovCanvas = overlayCanvasRef.current;
    if (!glCanvas || !ovCanvas) return;

    const renderer = new MangaWebGLRenderer(glCanvas);
    rendererRef.current = renderer;
    const ovCtx = ovCanvas.getContext('2d');

    const handleResize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.floor(window.innerWidth * dpr);
      const h = Math.floor(window.innerHeight * dpr);
      renderer.resize(w, h);
      if (ovCanvas.width !== w || ovCanvas.height !== h) {
        ovCanvas.width = w;
        ovCanvas.height = h;
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);

    let animFrameId;

    const tick = (nowMs) => {
      const m = machineRef.current;
      const dtSec = Math.min(0.1, Math.max(0.001, (nowMs - m.lastFrameTimeMs) / 1000));
      m.lastFrameTimeMs = nowMs;

      // FPS counter calculation
      m.fpsFrames += 1;
      if (nowMs - m.fpsLastSampleMs >= 450) {
        m.currentFps = Math.round((m.fpsFrames * 1000) / (nowMs - m.fpsLastSampleMs));
        m.fpsFrames = 0;
        m.fpsLastSampleMs = nowMs;
      }

      const canvasW = ovCanvas.width;
      const canvasH = ovCanvas.height;

      // 1. Acquire live webcam frame & run MediaPipe FaceLandmarker detection
      const video = videoRef.current;
      let sourceW = 1280;
      let sourceH = 720;

      if (video && video.readyState >= 2) {
        sourceW = video.videoWidth || 1280;
        sourceH = video.videoHeight || 720;

        if (landmarkerRef.current && video.currentTime !== m.lastVideoTime) {
          m.lastVideoTime = video.currentTime;
          try {
            const result = landmarkerRef.current.detectForVideo(video, nowMs);
            m.latestLandmarks = result.faceLandmarks?.[0] || null;
            m.latestBlendshapes = result.faceBlendshapes || null;
          } catch {
            // Keep previous frame landmarks on transient timestamp collision
          }
        }
      }

      // 2. Evaluate Gaze Lock (head yaw/pitch + iris centering + eyeBlinkLeft/Right blendshapes)
      const gaze = evaluateGazeLock(m.latestLandmarks, m.latestBlendshapes);
      m.gazeMetrics = gaze;

      // 3. Advance Interaction State Machine (IDLE -> GAZE LOCK -> STAGE 1 -> STAGE 2)
      if (!gateOpen) {
        if (m.stage2Locked) {
          if (gaze.isGazeLocked) {
            m.dwellProgress = 1.0;
            m.stage2CooldownSec = STAGE2_COOLDOWN_SEC;
          } else {
            m.stage2CooldownSec = Math.max(0, m.stage2CooldownSec - dtSec);
            if (m.stage2CooldownSec <= 0) {
              m.stage2Locked = false;
              m.dwellProgress = Math.max(0, m.dwellProgress - dtSec * DWELL_DECAY_RATE);
            }
          }
        } else {
          if (gaze.isGazeLocked) {
            const nextProgress = Math.min(1.0, m.dwellProgress + dtSec / DWELL_FILL_DURATION_SEC);
            if (nextProgress >= 1.0 && m.dwellProgress < 1.0) {
              m.stage2Locked = true;
              m.stage2CooldownSec = STAGE2_COOLDOWN_SEC;
              triggerStage2Shake();
            }
            m.dwellProgress = nextProgress;
          } else {
            m.dwellProgress = Math.max(0.0, m.dwellProgress - dtSec * DWELL_DECAY_RATE);
          }
        }
      } else {
        m.dwellProgress = 0.0;
        m.stage2Locked = false;
        m.stage2CooldownSec = 0.0;
      }

      // Determine current discrete stage label
      let currentStage = 'IDLE';
      if (m.stage2Locked || m.dwellProgress >= 1.0) {
        currentStage = 'STAGE_2';
      } else if (m.dwellProgress > STAGE1_THRESHOLD) {
        currentStage = 'STAGE_1';
      } else if (gaze.isGazeLocked && m.dwellProgress > 0.01) {
        currentStage = 'GAZE_LOCK';
      }

      // 4. Render WebGL Video Feed with GPU Shader-Space Eye Relighting (Stage 1+) & Kuwahara Color Anime Paint (Stage 2)
      if (video && video.readyState >= 2) {
        const effectiveProgress = m.stage2Locked ? 1.0 : m.dwellProgress;
        const eyeShaderAmount =
          effectiveProgress > STAGE1_THRESHOLD
            ? Math.min(1.0, (effectiveProgress - STAGE1_THRESHOLD) / 0.25)
            : 0.0;
        const mangaShaderAmount = m.stage2Locked ? 1.0 : 0.0;
        const eyeUniforms = extractShaderEyeUniforms(
          m.latestLandmarks,
          gaze.eyeBlinkLeft,
          gaze.eyeBlinkRight,
          sourceW,
          sourceH
        );

        renderer.renderFrame(video, {
          eyeAmount: eyeShaderAmount,
          mangaAmount: mangaShaderAmount,
          mirror: true,
          edgeThreshold: 0.26,
          dotFrequency: 6.0,
          colorStyle,
          eyeUniforms,
        });
      }

      // 5. Render 2D Contour-Locked Anime Eyes Overlay (Stage 1 > 0.15) & Radial Speed Lines (Stage 2)
      ovCtx.clearRect(0, 0, canvasW, canvasH);

      if (!gateOpen && m.latestLandmarks) {
        renderMangaEyesOverlay(ovCtx, {
          landmarks: m.latestLandmarks,
          eyeBlinkLeft: gaze.eyeBlinkLeft,
          eyeBlinkRight: gaze.eyeBlinkRight,
          dwellProgress: m.dwellProgress,
          isStage2Locked: m.stage2Locked,
          canvasWidth: canvasW,
          canvasHeight: canvasH,
          videoWidth: sourceW,
          videoHeight: sourceH,
          mirror: true,
          colorStyle,
        });

        if (showLandmarkMesh) {
          renderDebugLandmarksOverlay(ovCtx, {
            landmarks: m.latestLandmarks,
            canvasWidth: canvasW,
            canvasHeight: canvasH,
            videoWidth: sourceW,
            videoHeight: sourceH,
            mirror: true,
            isGazeLocked: gaze.isGazeLocked,
          });
        }
      }

      if (!gateOpen && m.stage2Locked) {
        renderShuchuSenSpeedLines(ovCtx, canvasW, canvasH, nowMs, colorStyle);
      }

      // 6. Sync React UI state at ~24 Hz
      if (nowMs - m.lastUiSyncMs >= 42) {
        m.lastUiSyncMs = nowMs;
        setUiState({
          stage: currentStage,
          dwellProgress: m.dwellProgress,
          isGazeLocked: gaze.isGazeLocked,
          faceDetected: gaze.faceDetected,
          cooldownRemaining: m.stage2CooldownSec,
          fps: m.currentFps,
          yaw: gaze.yaw,
          pitch: gaze.pitch,
          irisOffset: gaze.irisOffset,
          eyeBlinkLeft: gaze.eyeBlinkLeft,
          eyeBlinkRight: gaze.eyeBlinkRight,
        });
      }

      animFrameId = requestAnimationFrame(tick);
    };

    animFrameId = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(animFrameId);
      window.removeEventListener('resize', handleResize);
      renderer.destroy();
    };
  }, [gateOpen, showLandmarkMesh, colorStyle, triggerStage2Shake]);

  // Clean up camera media tracks on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  // Force Stage 2 Lock toggle from Debug Panel for instant inspection
  const handleForceStage2Toggle = useCallback(() => {
    const m = machineRef.current;
    if (m.stage2Locked) {
      m.stage2Locked = false;
      m.dwellProgress = 0;
      m.stage2CooldownSec = 0;
    } else {
      m.dwellProgress = 1.0;
      m.stage2Locked = true;
      m.stage2CooldownSec = STAGE2_COOLDOWN_SEC;
      triggerStage2Shake();
    }
  }, [triggerStage2Shake]);

  // Derived UI progress metrics
  const dwellPct = Math.round(uiState.dwellProgress * 100);
  const ringRadius = 62;
  const ringCircumference = 2 * Math.PI * ringRadius;
  const ringDashOffset = ringCircumference * (1 - uiState.dwellProgress);
  const isStage2 = uiState.stage === 'STAGE_2';
  const isStage1 = uiState.stage === 'STAGE_1';

  return (
    <main
      id="mangaface-app"
      className={`manga-app ${isShaking ? 'is-shaking' : ''}`}
    >
      {/* Hidden source video element for webcam stream */}
      <video
        ref={videoRef}
        id="webcam-source-video"
        className="hidden-source-video"
        playsInline
        muted
      />

      {/* Thin Top Dwell Progress Bar (0 to 100%) */}
      <div
        id="top-dwell-progress-track"
        className="top-dwell-track"
        role="progressbar"
        aria-valuenow={dwellPct}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div
          className={`top-dwell-fill ${isStage2 ? 'stage-2' : ''}`}
          style={{ transform: `scaleX(${uiState.dwellProgress})` }}
        />
      </div>

      {/* Full-Bleed Dual-Canvas Stage (WebGL Colorful Anime Shader + 2D Contour-Locked Eye Overlay) */}
      <div id="viewport-stage" className="viewport-stage">
        <canvas ref={webglCanvasRef} id="manga-webgl-canvas" className="webgl-canvas" />
        <canvas ref={overlayCanvasRef} id="manga-overlay-canvas" className="overlay-canvas" />
      </div>

      {/* Brutalist Editorial Header */}
      <header id="editorial-header" className="editorial-header">
        <div className="brand-plaque">
          <span className="brand-hanko" aria-hidden="true">
            眼
          </span>
          <div className="brand-title-group">
            <h1 className="brand-title">Mangaface Effect</h1>
            <span className="brand-subtitle">Color Anime Gaze Transformation // Vol. 01</span>
          </div>
        </div>

        <div className="header-controls">
          {/* Colorful Palette Switcher Pills */}
          <div className="palette-switcher-group" style={{ display: 'flex', gap: '6px' }}>
            {COLOR_STYLES.map((st) => (
              <button
                key={st.id}
                id={`palette-style-btn-${st.id}`}
                type="button"
                className="mode-switch-btn"
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '6px',
                  background: colorStyle === st.id ? 'var(--ink-black)' : 'var(--paper-white)',
                  color: colorStyle === st.id ? 'var(--paper-white)' : 'var(--ink-black)',
                  borderColor: colorStyle === st.id ? 'var(--paper-white)' : 'var(--ink-black)',
                }}
                onClick={() => setColorStyle(st.id)}
              >
                <span
                  style={{
                    width: '10px',
                    height: '10px',
                    borderRadius: '50%',
                    background: st.swatch,
                    border: '1.5px solid currentColor',
                  }}
                />
                <span>{st.label}</span>
              </button>
            ))}
          </div>

          <div
            id="interaction-stage-badge"
            className={`state-badge-expressive ${
              isStage2 ? 'state-stage2' : isStage1 ? 'state-stage1' : ''
            }`}
          >
            <span
              className={`status-dot ${
                uiState.isGazeLocked || isStage2 ? 'pulsing' : ''
              }`}
            />
            <span>
              {isStage2
                ? 'STAGE 02 // FULL ANIME MODE'
                : isStage1
                ? `STAGE 01 // ANIME EYES (${dwellPct}%)`
                : uiState.isGazeLocked
                ? `GAZE LOCK // DWELL ${dwellPct}%`
                : 'IDLE // AWAITING GAZE'}
            </span>
          </div>
        </div>
      </header>

      {/* Stage 2 Vertical Japanese SFX Stamp & Panel Snapshot CTA */}
      <aside
        id="stage2-sfx-stamp"
        className={`stage2-editorial-stamp ${isStage2 && !gateOpen ? 'visible' : ''}`}
        aria-hidden={!isStage2}
      >
        <div className="vertical-sfx-box" title="Menacing / Intense Gaze SFX (Gogogogo)">
          ゴゴゴゴ
        </div>
        <button
          id="capture-manga-panel-btn"
          type="button"
          className="hanko-seal-cta"
          onClick={handleCaptureMangaPanel}
        >
          印 Save Panel
        </button>
      </aside>

      {/* Center Reticle Ring & Minimal Landing Page Overlay Copy ("Stare into the lens") */}
      {!gateOpen && (
        <section id="center-hud-layer" className="center-hud-layer">
          {/* Circular Dwell Progress Ring */}
          <div
            id="circular-dwell-ring"
            className={`dwell-ring-container ${isStage2 ? 'ring-locked' : ''}`}
          >
            <svg className="dwell-ring-svg" viewBox="0 0 148 148">
              <circle
                className="dwell-ring-track"
                cx="74"
                cy="74"
                r={ringRadius}
              />
              <circle
                className={`dwell-ring-progress ${isStage2 ? 'stage-2' : ''}`}
                cx="74"
                cy="74"
                r={ringRadius}
                strokeDasharray={ringCircumference}
                strokeDashoffset={ringDashOffset}
              />
            </svg>
            <div className="dwell-ring-readout">
              <span>{dwellPct}%</span>
            </div>
          </div>

          {/* Minimal Landing Page Copy Overlaid on Live Feed */}
          <div className="hero-prompt-bar">
            <div
              id="landing-copy-plaque"
              className={`stare-plaque ${
                isStage2
                  ? 'is-manga-locked'
                  : uiState.dwellProgress > 0.05
                  ? 'is-dwelling'
                  : ''
              }`}
            >
              <span className="stare-kicker">
                {isStage2
                  ? uiState.isGazeLocked
                    ? 'STAGE 02 LOCKED // COLOR CEL + SOBEL INK'
                    : `COOLDOWN HOLD // REVERTING IN ${uiState.cooldownRemaining.toFixed(1)}S`
                  : isStage1
                  ? 'STAGE 01 ACTIVE // CONTOUR-LOCKED ANIME EYES'
                  : 'INTERACTIVE GAZE EXPERIMENT'}
              </span>
              <h2 className="stare-headline">
                {isStage2 ? 'Awakened Anime State' : 'Stare into the lens'}
              </h2>
              <p className="stare-subcopy">
                {isStage2
                  ? 'Hold your gaze to sustain Full Anime Mode, or look away for 3s to release.'
                  : 'Align head forward & center irises for 1.2s to trigger colorful anime transformation.'}
              </p>
            </div>
          </div>
        </section>
      )}

      {/* Corner Debug Toggle & Panel — Google Material Expressive ("Mangaface Effect") */}
      <aside id="mangaface-debug-dock" className="material-expressive-dock">
        {debugOpen && (
          <div
            id="mangaface-debug-panel"
            className="mangaface-debug-card"
            role="region"
            aria-label="Mangaface Effect Telemetry"
          >
            <div className="debug-card-header">
              <div className="debug-card-title-wrap">
                <span className="debug-card-overline">Google Material Expressive</span>
                <h3 className="debug-card-title">Mangaface Effect</h3>
              </div>
              <span id="debug-fps-readout" className="debug-fps-chip">
                {uiState.fps} FPS
              </span>
            </div>

            <div className="debug-metrics-grid">
              <div className="debug-metric-box">
                <span className="metric-label">State Machine</span>
                <span id="debug-stage-value" className="metric-value">
                  {uiState.stage}
                </span>
              </div>

              <div className="debug-metric-box">
                <span className="metric-label">Raw Gaze State</span>
                <span id="debug-gaze-value" className="metric-value">
                  {uiState.isGazeLocked ? 'LOCKED' : 'UNLOCKED'}
                </span>
              </div>

              <div className="debug-metric-box span-2">
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span className="metric-label">Dwell Progress (1.2s ramp)</span>
                  <span id="debug-dwell-value" className="metric-value">
                    {uiState.dwellProgress.toFixed(2)} ({dwellPct}%)
                  </span>
                </div>
                <div className="debug-meter-bar">
                  <div
                    className={`debug-meter-fill ${isStage2 ? 'fill-red' : ''}`}
                    style={{ width: `${dwellPct}%` }}
                  />
                </div>
              </div>

              <div className="debug-metric-box">
                <span className="metric-label">Head Yaw / Pitch</span>
                <span className="metric-value">
                  {uiState.yaw.toFixed(2)} / {uiState.pitch.toFixed(2)}
                </span>
              </div>

              <div className="debug-metric-box">
                <span className="metric-label">Iris Offset</span>
                <span className="metric-value">
                  {uiState.irisOffset.toFixed(3)}
                </span>
              </div>

              <div className="debug-metric-box">
                <span className="metric-label">eyeBlinkLeft</span>
                <span className="metric-value">
                  {uiState.eyeBlinkLeft.toFixed(2)}
                </span>
              </div>

              <div className="debug-metric-box">
                <span className="metric-label">eyeBlinkRight</span>
                <span className="metric-value">
                  {uiState.eyeBlinkRight.toFixed(2)}
                </span>
              </div>
            </div>

            <div className="debug-controls-row">
              <button
                id="debug-toggle-landmarks-btn"
                type="button"
                className={`debug-action-btn ${showLandmarkMesh ? 'is-active' : ''}`}
                onClick={() => setShowLandmarkMesh((v) => !v)}
              >
                {showLandmarkMesh ? 'Hide Iris Mesh' : 'Show Iris Mesh'}
              </button>
              <button
                id="debug-force-stage2-btn"
                type="button"
                className={`debug-action-btn ${isStage2 ? 'is-active' : ''}`}
                onClick={handleForceStage2Toggle}
              >
                {isStage2 ? 'Reset Stage' : 'Snap Stage 2'}
              </button>
            </div>
          </div>
        )}

        <button
          id="mangaface-debug-toggle-btn"
          type="button"
          className={`mangaface-fab ${debugOpen ? 'is-open' : ''}`}
          onClick={() => setDebugOpen((prev) => !prev)}
          aria-expanded={debugOpen}
        >
          <span>Mangaface Effect</span>
          <span className="fab-pill-tag">{uiState.fps} FPS</span>
        </button>
      </aside>

      {/* Camera Permission Gate on Load ("Enable Camera to Enter") */}
      {gateOpen && (
        <div
          id="camera-permission-gate"
          className="permission-gate-backdrop"
          role="dialog"
          aria-modal="true"
          aria-labelledby="gate-main-heading"
        >
          <div className="permission-gate-card">
            <div className="gate-content-col">
              <div className="gate-issue-header">
                <span>MANGAFACE EFFECT // REAL-TIME VISION</span>
                <span className="gate-issue-tag">100% CLIENT-SIDE</span>
              </div>

              <div>
                <h2 id="gate-main-heading" className="gate-headline">
                  Stare Into The Lens. Become Anime.
                </h2>
              </div>

              <p className="gate-description">
                An interactive webcam experiment powered by MediaPipe{' '}
                <strong>FaceLandmarker</strong> (478 contour-locked iris landmarks
                &amp; blink blendshapes) and a custom <strong>Full-Color Cel-Shaded Sobel + Halftone</strong>{' '}
                WebGL shader. Lock your gaze for 1.2 seconds to awaken Full Anime Mode.
              </p>

              <div className="gate-spec-pills">
                <span className="gate-spec-pill">3 Color Palettes (Golden Hour / Shinkai / Neo-Tokyo)</span>
                <span className="gate-spec-pill">Stage 1: Contour-Locked Anime Irises</span>
                <span className="gate-spec-pill">Stage 2: Color Cel + Shuchu-Sen</span>
              </div>

              {cameraError && (
                <div id="camera-error-banner" className="gate-error-banner">
                  <strong>Camera Permission Required:</strong> {cameraError}
                </div>
              )}

              <div className="gate-cta-stack">
                <button
                  id="enable-camera-btn"
                  type="button"
                  className="enable-camera-btn"
                  onClick={handleEnableCamera}
                  disabled={isInitializing}
                >
                  <span>
                    {isInitializing ? 'Initializing Vision Engine...' : 'Enable Camera to Enter'}
                  </span>
                  <span aria-hidden="true">→</span>
                </button>
              </div>

              <div id="privacy-note-banner" className="gate-privacy-note">
                <span aria-hidden="true">🔒</span>
                <span>
                  <strong>PRIVACY GUARANTEE:</strong> All facial landmark detection,
                  iris tracking, and WebGL shader rendering stays 100% local in your
                  browser. Zero video frames ever leave your device. ({modelStatus})
                </span>
              </div>
            </div>

            <div className="gate-preview-col">
              <img
                src="./assets/hero-reference.png"
                alt="High-contrast manga transformation preview"
                className="gate-preview-img"
              />
              <div className="gate-preview-halftone" />
              <div className="gate-preview-caption">
                <div>FIG 01. GAZE LOCK THRESHOLD</div>
                <div>DWELL 0.15 → ANIME IRISES // DWELL 1.00 → FULL COLOR ANIME</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
