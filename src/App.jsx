import { useEffect, useRef, useState, useCallback } from 'react';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { MangaWebGLRenderer } from './engine/mangaShader';
import {
  evaluateGazeLock,
  renderMangaEyesOverlay,
  renderShuchuSenSpeedLines,
  renderDebugLandmarksOverlay,
} from './engine/mangaOverlay';
import { SyntheticStudioSource } from './engine/syntheticStudio';

// Dwell timing specifications
const DWELL_FILL_DURATION_SEC = 1.2; // 0.0 -> 1.0 over 1.2 seconds
const DWELL_DECAY_RATE = 1.45;       // Smooth decay back to 0 when user looks away
const STAGE1_THRESHOLD = 0.15;       // Vector Manga Eyes trigger at progress > 0.15
const STAGE2_COOLDOWN_SEC = 3.0;     // 3-second cooldown before reverting when looking away in Stage 2

export default function App() {
  // Permission Gate & Input Source State
  const [gateOpen, setGateOpen] = useState(true);
  const [isInitializing, setIsInitializing] = useState(false);
  const [sourceMode, setSourceMode] = useState('webcam'); // 'webcam' | 'studio'
  const [cameraError, setCameraError] = useState(null);
  const [modelStatus, setModelStatus] = useState('Standby');

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
  const [studioLookAway, setStudioLookAway] = useState(false);

  // DOM & Engine Refs
  const webglCanvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const videoRef = useRef(null);
  const rendererRef = useRef(null);
  const landmarkerRef = useRef(null);
  const syntheticStudioRef = useRef(null);
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
      // Fallback to CPU delegate if WebGL GPU delegate is restricted
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

  // Detect if running inside an embedded preview iframe that blocks camera permissions
  const [isEmbeddedIframe] = useState(() => {
    try {
      return window.self !== window.top;
    } catch {
      return true;
    }
  });

  const handleOpenInNewTab = useCallback(() => {
    window.open(window.location.href, '_blank', 'noopener,noreferrer');
  }, []);

  // Start Live Webcam Feed — IMPORTANT: Call getUserMedia FIRST synchronously on user click
  // so the browser preserves the transient user gesture activation and shows the permission prompt!
  const handleEnableCamera = useCallback(async () => {
    setIsInitializing(true);
    setCameraError(null);

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error(
          'Browser blocked camera API in this embedded frame. Click "Open in New Tab to Use Live Webcam" below.'
        );
      }

      // 1. Request camera immediately during the user click event!
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
        // Fallback to simplest possible constraint if specific resolution/facingMode is rejected
        stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });
      }

      // Stop any previous stream tracks
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      // 2. Open viewport immediately so user sees their live camera feed right away
      setSourceMode('webcam');
      setGateOpen(false);

      // 3. Initialize MediaPipe FaceLandmarker in parallel/after camera starts
      await ensureFaceLandmarker();
    } catch (err) {
      const errName = err?.name ? `${err.name}: ` : '';
      const msg =
        err?.message ||
        'Camera permission was denied or blocked by the embedded preview iframe.';
      setCameraError(
        `${errName}${msg} — Embedded preview iframes block webcam popups by browser security policy. Click the red "Open in New Browser Tab for Live Webcam ↗" button right below to pop this preview out into a full tab where your webcam permission prompt works!`
      );
    } finally {
      setIsInitializing(false);
    }
  }, [ensureFaceLandmarker]);

  // Start Interactive Studio Mode (100% client-side synthetic subject feeding the exact same pipeline)
  const handleStartStudioMode = useCallback(async () => {
    setIsInitializing(true);
    setCameraError(null);

    try {
      // Attempt to initialize MediaPipe in background so user can inspect status
      ensureFaceLandmarker().catch(() => {
        setModelStatus('Studio Landmark Stream Active');
      });

      if (!syntheticStudioRef.current) {
        syntheticStudioRef.current = new SyntheticStudioSource(1280, 720);
      }
      syntheticStudioRef.current.setLookAway(false);
      setStudioLookAway(false);

      // Reset dwell progress so user sees the 0.0 -> 1.0 Gaze Lock buildup immediately
      machineRef.current.dwellProgress = 0;
      machineRef.current.stage2Locked = false;
      machineRef.current.stage2CooldownSec = 0;

      setSourceMode('studio');
      setGateOpen(false);
    } finally {
      setIsInitializing(false);
    }
  }, [ensureFaceLandmarker]);

  // Toggle between Webcam and Studio Mode from top bar
  const handleToggleSourceMode = useCallback(() => {
    if (sourceMode === 'webcam') {
      handleStartStudioMode();
    } else {
      handleEnableCamera();
    }
  }, [sourceMode, handleStartStudioMode, handleEnableCamera]);

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

    if (!syntheticStudioRef.current) {
      syntheticStudioRef.current = new SyntheticStudioSource(1280, 720);
    }

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

      // 1. Acquire active video frame & run MediaPipe FaceLandmarker detection
      let activeSourceElement = null;
      let sourceW = 1280;
      let sourceH = 720;

      if (sourceMode === 'webcam' && videoRef.current && videoRef.current.readyState >= 2) {
        const video = videoRef.current;
        activeSourceElement = video;
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
      } else {
        // Synthetic Studio Subject fallback (also powers attractive background behind permission gate)
        const studio = syntheticStudioRef.current;
        const syntheticResult = studio.update(nowMs);
        activeSourceElement = studio.canvas;
        sourceW = studio.width;
        sourceH = studio.height;

        if (!gateOpen) {
          m.latestLandmarks = syntheticResult.faceLandmarks[0];
          m.latestBlendshapes = syntheticResult.faceBlendshapes;
        } else {
          m.latestLandmarks = null;
          m.latestBlendshapes = null;
        }
      }

      // 2. Evaluate Gaze Lock (head yaw/pitch + iris centering + eyeBlinkLeft/Right blendshapes)
      const gaze = evaluateGazeLock(m.latestLandmarks, m.latestBlendshapes);
      m.gazeMetrics = gaze;

      // 3. Advance Interaction State Machine (IDLE -> GAZE LOCK -> STAGE 1 -> STAGE 2)
      if (!gateOpen) {
        if (m.stage2Locked) {
          // In Stage 2 Full Manga Mode: lock it in while gaze continues,
          // or maintain a 3-second cooldown before reverting if user looks away
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
          // Building or decaying dwellProgress (0.0 -> 1.0 over 1.2 seconds)
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

      // 4. Render WebGL Video Feed & Stage 2 1-Bit Sobel + Diagonal Screentone Shader
      if (activeSourceElement) {
        const mangaShaderAmount = m.stage2Locked ? 1.0 : 0.0;
        renderer.renderFrame(activeSourceElement, {
          mangaAmount: mangaShaderAmount,
          mirror: true,
          edgeThreshold: 0.24,
          dotFrequency: 6.5,
        });
      }

      // 5. Render 2D Vector Overlay (Stage 1 Manga Eyes + Stage 2 Radial Shuchu-sen Speed Lines)
      ovCtx.clearRect(0, 0, canvasW, canvasH);

      if (!gateOpen && m.latestLandmarks) {
        // Stage 1 (> 0.15) & Stage 2 (== 1.0): Overlay stylized vector manga eyes with natural blink
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

      // Stage 2: Draw animated black radial speed lines (shuchu-sen) around outer 20% border
      if (!gateOpen && m.stage2Locked) {
        renderShuchuSenSpeedLines(ovCtx, canvasW, canvasH, nowMs);
      }

      // 6. Sync React UI state at ~24 Hz so DOM gauges stay silky without bottlenecking WebGL
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
  }, [sourceMode, gateOpen, showLandmarkMesh, triggerStage2Shake]);

  // Clean up camera media tracks on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  // Allow pointer movement in Studio Mode to steer synthetic subject's gaze interactively
  const handleViewportPointerMove = useCallback(
    (e) => {
      if (sourceMode !== 'studio' || gateOpen || studioLookAway) return;
      const studio = syntheticStudioRef.current;
      if (!studio) return;

      const normX = (e.clientX / window.innerWidth - 0.5) * 2;
      const normY = (e.clientY / window.innerHeight - 0.5) * 2;
      // Within central reticle zone, lock gaze dead-center into lens
      if (Math.hypot(normX, normY) < 0.42) {
        studio.setGazeTarget(0, 0);
      } else {
        studio.setGazeTarget(normX * 0.85, normY * 0.75);
      }
    },
    [sourceMode, gateOpen, studioLookAway]
  );

  // Toggle Look Away / Gaze Lock in Studio Mode
  const handleToggleStudioLookAway = useCallback(() => {
    const next = !studioLookAway;
    setStudioLookAway(next);
    if (syntheticStudioRef.current) {
      syntheticStudioRef.current.setLookAway(next);
    }
  }, [studioLookAway]);

  const handleTriggerStudioBlink = useCallback(() => {
    if (syntheticStudioRef.current) {
      syntheticStudioRef.current.triggerBlink();
    }
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
      onPointerMove={handleViewportPointerMove}
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

      {/* Full-Bleed Dual-Canvas Stage (WebGL 1-Bit Shader + 2D Vector Manga Overlay) */}
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
            <span className="brand-subtitle">1-Bit Gaze Transformation // Vol. 01</span>
          </div>
        </div>

        <div className="header-controls">
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
                ? 'STAGE 02 // FULL MANGA MODE'
                : isStage1
                ? `STAGE 01 // MANGA EYES (${dwellPct}%)`
                : uiState.isGazeLocked
                ? `GAZE LOCK // DWELL ${dwellPct}%`
                : 'IDLE // AWAITING GAZE'}
            </span>
          </div>

          {!gateOpen && (
            <>
              <button
                id="switch-input-mode-btn"
                type="button"
                className="mode-switch-btn"
                onClick={handleToggleSourceMode}
              >
                {sourceMode === 'webcam' ? 'Source: Live Camera' : 'Source: Studio Rig'}
              </button>
              {isEmbeddedIframe && (
                <button
                  id="header-open-new-tab-btn"
                  type="button"
                  className="mode-switch-btn"
                  style={{
                    background: 'var(--hanko-red)',
                    color: 'var(--paper-white)',
                  }}
                  onClick={handleOpenInNewTab}
                >
                  Pop Out Tab for Webcam ↗
                </button>
              )}
            </>
          )}
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
            {sourceMode === 'studio' && (
              <div id="studio-simulator-controls" className="simulator-pill">
                <span>STUDIO GAZE RIG:</span>
                <button
                  id="studio-gaze-lock-toggle"
                  type="button"
                  className={`sim-toggle-btn ${!studioLookAway ? 'active' : ''}`}
                  onClick={handleToggleStudioLookAway}
                >
                  {!studioLookAway ? 'Gaze Locked (Look Away)' : 'Looking Away (Lock Gaze)'}
                </button>
                <button
                  id="studio-trigger-blink-btn"
                  type="button"
                  className="sim-toggle-btn"
                  onClick={handleTriggerStudioBlink}
                >
                  Blink Eyes
                </button>
              </div>
            )}

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
                    ? 'STAGE 02 LOCKED // 1-BIT SOBEL + HALFTONE'
                    : `COOLDOWN HOLD // REVERTING IN ${uiState.cooldownRemaining.toFixed(1)}S`
                  : isStage1
                  ? 'STAGE 01 ACTIVE // BLINK-SYNCED SEINEN EYES'
                  : 'INTERACTIVE GAZE EXPERIMENT'}
              </span>
              <h2 className="stare-headline">
                {isStage2 ? 'Awakened Manga State' : 'Stare into the lens'}
              </h2>
              <p className="stare-subcopy">
                {isStage2
                  ? 'Hold your gaze to sustain Full Manga Mode, or look away for 3s to release.'
                  : 'Align head forward & center irises for 1.2s to trigger 1-bit ink transformation.'}
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
                  Stare Into The Lens. Become Ink.
                </h2>
              </div>

              <p className="gate-description">
                An interactive webcam experiment powered by MediaPipe{' '}
                <strong>FaceLandmarker</strong> (478 iris landmarks &amp; blink
                blendshapes) and a custom <strong>1-Bit Sobel + Diagonal Screentone</strong>{' '}
                WebGL shader. Lock your gaze for 1.2 seconds to awaken Full Manga Mode.
              </p>

              <div className="gate-spec-pills">
                <span className="gate-spec-pill">#0A0A0A Ink / #F4F1EA Paper</span>
                <span className="gate-spec-pill">Stage 1: Blink-Synced Vector Eyes</span>
                <span className="gate-spec-pill">Stage 2: Shuchu-Sen Speed Lines</span>
              </div>

              {cameraError && (
                <div id="camera-error-banner" className="gate-error-banner">
                  <strong>Why Camera Was Blocked:</strong> {cameraError}
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

                {(isEmbeddedIframe || cameraError) && (
                  <button
                    id="open-new-tab-camera-btn"
                    type="button"
                    className="enable-camera-btn"
                    style={{
                      background: 'var(--hanko-red)',
                      color: 'var(--paper-white)',
                      boxShadow: '6px 6px 0px var(--ink-black)',
                    }}
                    onClick={handleOpenInNewTab}
                  >
                    <span>Open in New Browser Tab for Live Webcam</span>
                    <span aria-hidden="true">↗</span>
                  </button>
                )}

                <button
                  id="launch-studio-fallback-btn"
                  type="button"
                  className="synthetic-studio-btn"
                  onClick={handleStartStudioMode}
                  disabled={isInitializing}
                >
                  Or Test Immediately with Interactive 60 FPS Studio Gaze Rig →
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
                <div>DWELL 0.15 → VECTOR EYES // DWELL 1.00 → FULL 1-BIT MANGA</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
