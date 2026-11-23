/**
 * MANGAFACE EFFECT — Gaze Lock Evaluator, Vector Manga Eyes & Shuchu-Sen Renderer
 *
 * Implements:
 * 1. GAZE LOCK DETECTION:
 *    - Head yaw & pitch check via facial symmetry (nose tip #1, left outer eye #33, right outer eye #263, forehead #10, chin #152)
 *    - Iris centering check in eye sockets (Left iris #468 between #33/#133 & #159/#145; Right iris #473 between #362/#263 & #386/#374)
 * 2. STAGE 1 (dwellProgress > 0.15):
 *    - Sharp, high-contrast seinen/shonen vector manga eyes overlaid directly onto left and right eye landmarks
 *    - Thick calligraphic upper lash line, dark radial iris with hatching spokes, two crisp white specular highlights
 *    - Dynamic vertical eye height scaling driven by MediaPipe's `eyeBlinkLeft` and `eyeBlinkRight` blendshapes
 * 3. STAGE 2 (dwellProgress == 1.0):
 *    - Animated black radial speed lines (shuchu-sen) around the outer 20% of the canvas border
 */

import { mapLandmarkToCanvas } from './mangaShader';

const INK_BLACK = '#0A0A0A';
const PAPER_WHITE = '#F4F1EA';
const HANKO_RED = '#D92B2B';

// MediaPipe FaceMesh Canonical Landmark Indices
export const LANDMARKS = {
  NOSE_TIP: 1,
  FOREHEAD: 10,
  CHIN: 152,

  // Subject's Right Eye (appears on Left side of unmirrored image, Right side of mirrored view)
  RIGHT_EYE_OUTER: 33,
  RIGHT_EYE_INNER: 133,
  RIGHT_EYE_TOP: 159,
  RIGHT_EYE_BOTTOM: 145,
  RIGHT_IRIS_CENTER: 468,

  // Subject's Left Eye (appears on Right side of unmirrored image, Left side of mirrored view)
  LEFT_EYE_INNER: 362,
  LEFT_EYE_OUTER: 263,
  LEFT_EYE_TOP: 386,
  LEFT_EYE_BOTTOM: 374,
  LEFT_IRIS_CENTER: 473,
};

/**
 * Extracts `eyeBlinkLeft` and `eyeBlinkRight` scores (0..1) from MediaPipe FaceLandmarker result
 */
export function extractBlinkBlendshapes(faceBlendshapes) {
  let eyeBlinkLeft = 0;
  let eyeBlinkRight = 0;

  if (faceBlendshapes && faceBlendshapes.length > 0 && faceBlendshapes[0].categories) {
    const categories = faceBlendshapes[0].categories;
    for (let i = 0; i < categories.length; i++) {
      const cat = categories[i];
      if (cat.categoryName === 'eyeBlinkLeft') {
        eyeBlinkLeft = cat.score;
      } else if (cat.categoryName === 'eyeBlinkRight') {
        eyeBlinkRight = cat.score;
      }
    }
  }

  return { eyeBlinkLeft, eyeBlinkRight };
}

/**
 * Evaluates whether the user's head faces forward (low yaw/pitch) and
 * whether the iris landmarks (#468, #473) are centered in the eye sockets.
 */
export function evaluateGazeLock(landmarks, blendshapes) {
  if (!landmarks || landmarks.length < 474) {
    return {
      isGazeLocked: false,
      yaw: 0,
      pitch: 0,
      irisOffset: 1,
      eyeBlinkLeft: 0,
      eyeBlinkRight: 0,
      faceDetected: false,
    };
  }

  const nose = landmarks[LANDMARKS.NOSE_TIP];
  const forehead = landmarks[LANDMARKS.FOREHEAD];
  const chin = landmarks[LANDMARKS.CHIN];

  const rOuter = landmarks[LANDMARKS.RIGHT_EYE_OUTER];
  const rInner = landmarks[LANDMARKS.RIGHT_EYE_INNER];
  const rTop = landmarks[LANDMARKS.RIGHT_EYE_TOP];
  const rBottom = landmarks[LANDMARKS.RIGHT_EYE_BOTTOM];
  const rIris = landmarks[LANDMARKS.RIGHT_IRIS_CENTER];

  const lInner = landmarks[LANDMARKS.LEFT_EYE_INNER];
  const lOuter = landmarks[LANDMARKS.LEFT_EYE_OUTER];
  const lTop = landmarks[LANDMARKS.LEFT_EYE_TOP];
  const lBottom = landmarks[LANDMARKS.LEFT_EYE_BOTTOM];
  const lIris = landmarks[LANDMARKS.LEFT_IRIS_CENTER];

  // 1. Head Yaw Estimate (-1..1, 0 = dead center)
  const eyeMidX = (rOuter.x + lOuter.x) * 0.5;
  const interOcularDist = Math.max(Math.hypot(lOuter.x - rOuter.x, lOuter.y - rOuter.y), 0.05);
  const yaw = (nose.x - eyeMidX) / (interOcularDist * 0.5);

  // 2. Head Pitch Estimate (-1..1, 0 = level forward)
  const faceHeight = Math.max(Math.hypot(chin.x - forehead.x, chin.y - forehead.y), 0.08);
  const eyeMidY = (rOuter.y + lOuter.y) * 0.5;
  const noseRelY = (nose.y - eyeMidY) / faceHeight;
  // Normal forward-facing noseRelY is ~0.27
  const pitch = (noseRelY - 0.27) * 2.4;

  // 3. Iris Centering within Eye Sockets (0 = dead center, >0.25 = looking sideways)
  const rWidth = Math.max(Math.hypot(rInner.x - rOuter.x, rInner.y - rOuter.y), 0.01);
  const rRatioX = Math.hypot(rIris.x - rOuter.x, rIris.y - rOuter.y) / rWidth;

  const lWidth = Math.max(Math.hypot(lOuter.x - lInner.x, lOuter.y - lInner.y), 0.01);
  const lRatioX = Math.hypot(lIris.x - lInner.x, lIris.y - lInner.y) / lWidth;

  // Vertical iris centering
  const rHeight = Math.max(Math.hypot(rBottom.x - rTop.x, rBottom.y - rTop.y), 0.005);
  const rRatioY = Math.hypot(rIris.x - rTop.x, rIris.y - rTop.y) / rHeight;

  const lHeight = Math.max(Math.hypot(lBottom.x - lTop.x, lBottom.y - lTop.y), 0.005);
  const lRatioY = Math.hypot(lIris.x - lTop.x, lIris.y - lTop.y) / lHeight;

  const irisOffsetX = (Math.abs(rRatioX - 0.5) + Math.abs(lRatioX - 0.5)) * 0.5;
  const irisOffsetY = (Math.abs(rRatioY - 0.5) + Math.abs(lRatioY - 0.5)) * 0.5;
  const irisOffset = irisOffsetX * 0.78 + irisOffsetY * 0.22;

  const { eyeBlinkLeft, eyeBlinkRight } = extractBlinkBlendshapes(blendshapes);

  // Gaze lock criteria: forward head pose + centered irises
  const lowYaw = Math.abs(yaw) < 0.26;
  const lowPitch = Math.abs(pitch) < 0.34;
  const centeredIris = irisOffsetX < 0.19;

  const isGazeLocked = lowYaw && lowPitch && centeredIris;

  return {
    isGazeLocked,
    yaw,
    pitch,
    irisOffset,
    eyeBlinkLeft,
    eyeBlinkRight,
    faceDetected: true,
  };
}

/**
 * Draws a single stylized vector Seinen/Shonen Manga Eye onto the 2D overlay canvas.
 * Scales vertical opening dynamically with MediaPipe's `blinkScore` blendshape (0.0 open -> 1.0 closed).
 */
function drawSingleMangaEye(ctx, {
  outerCorner,
  innerCorner,
  topLid,
  bottomLid,
  irisCenter,
  blinkScore,
  isLeftOnScreen,
  stageProgress,
}) {
  const dx = innerCorner.x - outerCorner.x;
  const dy = innerCorner.y - outerCorner.y;
  const rawEyeWidth = Math.hypot(dx, dy);

  // Expand slightly beyond physiological eye corners for dramatic manga proportions
  const eyeWidth = rawEyeWidth * 1.38;
  const eyeCenter = {
    x: (outerCorner.x + innerCorner.x) * 0.5,
    y: (topLid.y + bottomLid.y) * 0.5,
  };

  // Angle of eye axis
  let angle = Math.atan2(dy, dx);
  if (Math.abs(angle) > Math.PI * 0.5) {
    angle += angle > 0 ? -Math.PI : Math.PI;
  }

  // Convert MediaPipe blinkScore (0..1) into vertical openness (1.0 down to 0.06 when blinking)
  // Smooth amplification so natural blinks snap cleanly
  const clampedBlink = Math.min(1.0, Math.max(0.0, (blinkScore - 0.06) * 1.35));
  const openness = Math.max(0.05, 1.0 - clampedBlink);

  const maxEyeHeight = eyeWidth * 0.56;
  const currentHeight = maxEyeHeight * openness;

  // Iris relative offset inside the eye (-0.28..0.28)
  const irisRelX = Math.max(
    -eyeWidth * 0.22,
    Math.min(eyeWidth * 0.22, (irisCenter.x - eyeCenter.x) * 0.85)
  );
  const irisRelY = Math.max(
    -currentHeight * 0.18,
    Math.min(currentHeight * 0.18, (irisCenter.y - eyeCenter.y) * 0.7)
  );

  // Entrance scale animation as stageProgress goes from 0.15 -> 0.30
  const popScale = Math.min(1.0, Math.max(0.0, (stageProgress - 0.15) / 0.12));
  const easePop = 1 - Math.pow(1 - popScale, 3);

  ctx.save();
  ctx.translate(eyeCenter.x, eyeCenter.y);
  ctx.rotate(angle);
  ctx.scale(easePop, easePop);

  const halfW = eyeWidth * 0.5;
  const halfH = currentHeight * 0.5;
  // Direction toward outer corner (-1 for left-on-screen eye, +1 for right-on-screen eye)
  const outerDir = isLeftOnScreen ? -1 : 1;

  // 1. White paper backing buffer around the eye so it pops cleanly over raw video or halftone
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(-halfW * 1.12, 0);
  ctx.bezierCurveTo(
    -halfW * 0.55, -halfH * 1.5,
     halfW * 0.55, -halfH * 1.5,
     halfW * 1.12, -halfH * 0.08 * outerDir
  );
  ctx.bezierCurveTo(
     halfW * 0.55,  halfH * 1.35,
    -halfW * 0.55,  halfH * 1.35,
    -halfW * 1.12,  0
  );
  ctx.closePath();
  ctx.fillStyle = PAPER_WHITE;
  ctx.fill();
  ctx.restore();

  // If eye is open (> 0.14), draw sclera, detailed radial iris, pupil, and dual specular highlights
  if (openness > 0.14) {
    // 2. Almond/Angular Seinen Eye Clipping Path
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(-halfW, halfH * 0.06);
    ctx.bezierCurveTo(
      -halfW * 0.48, -halfH * 1.28,
       halfW * 0.48, -halfH * 1.28,
       halfW, -halfH * 0.12
    );
    ctx.bezierCurveTo(
       halfW * 0.52,  halfH * 1.18,
      -halfW * 0.52,  halfH * 1.18,
      -halfW, halfH * 0.06
    );
    ctx.closePath();

    // Sclera Fill (#F4F1EA)
    ctx.fillStyle = PAPER_WHITE;
    ctx.fill();
    ctx.clip();

    // Upper eyelid cast shadow line inside sclera
    ctx.fillStyle = INK_BLACK;
    ctx.beginPath();
    ctx.ellipse(0, -halfH * 0.85, halfW * 0.95, halfH * 0.35, 0, 0, Math.PI * 2);
    ctx.fill();

    // 3. Dark Radial Manga Iris
    const irisRadius = eyeWidth * 0.27;
    const ix = irisRelX;
    const iy = irisRelY * 0.5;

    // Outer thick ink iris ring
    ctx.fillStyle = INK_BLACK;
    ctx.beginPath();
    ctx.arc(ix, iy, irisRadius, 0, Math.PI * 2);
    ctx.fill();

    // Inner off-white paper ring for radial ink spokes
    ctx.fillStyle = PAPER_WHITE;
    ctx.beginPath();
    ctx.arc(ix, iy, irisRadius * 0.78, 0, Math.PI * 2);
    ctx.fill();

    // Upper iris ink hood (classic seinen intense gaze shadow)
    ctx.fillStyle = INK_BLACK;
    ctx.beginPath();
    ctx.arc(ix, iy - irisRadius * 0.32, irisRadius * 0.82, 0, Math.PI, false);
    ctx.fill();

    // Radial ink hatching spokes inside iris
    ctx.strokeStyle = INK_BLACK;
    ctx.lineWidth = Math.max(1.4, eyeWidth * 0.022);
    const spokeCount = 16;
    for (let s = 0; s < spokeCount; s++) {
      const a = (s / spokeCount) * Math.PI * 2;
      const rIn = irisRadius * 0.34;
      const rOut = irisRadius * 0.82;
      ctx.beginPath();
      ctx.moveTo(ix + Math.cos(a) * rIn, iy + Math.sin(a) * rIn);
      ctx.lineTo(ix + Math.cos(a) * rOut, iy + Math.sin(a) * rOut);
      ctx.stroke();
    }

    // Deep Ink Pupil Core
    ctx.fillStyle = INK_BLACK;
    ctx.beginPath();
    ctx.arc(ix, iy, irisRadius * 0.38, 0, Math.PI * 2);
    ctx.fill();

    // 4. Two Crisp White Specular Highlights
    ctx.fillStyle = PAPER_WHITE;
    // Primary top-outer highlight
    ctx.beginPath();
    ctx.arc(
      ix + outerDir * irisRadius * 0.34,
      iy - irisRadius * 0.28,
      irisRadius * 0.24,
      0,
      Math.PI * 2
    );
    ctx.fill();

    // Secondary bottom-inner pinpoint highlight
    ctx.beginPath();
    ctx.arc(
      ix - outerDir * irisRadius * 0.32,
      iy + irisRadius * 0.32,
      irisRadius * 0.13,
      0,
      Math.PI * 2
    );
    ctx.fill();

    ctx.restore();
  }

  // 5. Thick Calligraphic Upper Lash Line (#0A0A0A)
  ctx.fillStyle = INK_BLACK;
  ctx.strokeStyle = INK_BLACK;
  ctx.beginPath();
  ctx.moveTo(-halfW * 1.08, halfH * 0.1);
  ctx.bezierCurveTo(
    -halfW * 0.52, -halfH * 1.48,
     halfW * 0.52, -halfH * 1.48,
     halfW * 1.12, -halfH * 0.16
  );
  // Outer sharp wing tip
  ctx.lineTo(halfW * (1.12 + 0.14 * outerDir), -halfH * 0.38);
  ctx.bezierCurveTo(
     halfW * 0.5, -halfH * 1.05,
    -halfW * 0.5, -halfH * 1.05,
    -halfW * 1.14, -halfH * 0.12
  );
  ctx.closePath();
  ctx.fill();

  // Sharp upper lash spikes on outer edge
  const spikeBaseX = outerDir * halfW * 0.78;
  const spikeBaseY = -halfH * 0.65;
  ctx.beginPath();
  ctx.moveTo(spikeBaseX, spikeBaseY);
  ctx.lineTo(spikeBaseX + outerDir * eyeWidth * 0.16, spikeBaseY - eyeWidth * 0.11);
  ctx.lineTo(spikeBaseX - outerDir * eyeWidth * 0.06, spikeBaseY - eyeWidth * 0.02);
  ctx.closePath();
  ctx.fill();

  // 6. Crisp Lower Eyelid Line & Double-Eyelid Crease
  ctx.lineWidth = Math.max(2.0, eyeWidth * 0.035);
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(-halfW * 0.82, halfH * 0.72);
  ctx.quadraticCurveTo(0, halfH * 1.18, halfW * 0.86, halfH * 0.62);
  ctx.stroke();

  // Upper eyelid crease line (Mabuta)
  ctx.lineWidth = Math.max(1.5, eyeWidth * 0.024);
  ctx.beginPath();
  ctx.moveTo(-halfW * 0.75, -halfH * 1.25 - eyeWidth * 0.08);
  ctx.quadraticCurveTo(0, -halfH * 1.65 - eyeWidth * 0.09, halfW * 0.85, -halfH * 1.15 - eyeWidth * 0.07);
  ctx.stroke();

  ctx.restore();
}

/**
 * Renders both vector manga eyes onto the overlay canvas when dwellProgress > 0.15
 */
export function renderMangaEyesOverlay(ctx, {
  landmarks,
  eyeBlinkLeft,
  eyeBlinkRight,
  dwellProgress,
  isStage2Locked,
  canvasWidth,
  canvasHeight,
  videoWidth,
  videoHeight,
  mirror = true,
}) {
  if (!landmarks || landmarks.length < 474) return;
  const effectiveProgress = isStage2Locked ? 1.0 : dwellProgress;
  if (effectiveProgress <= 0.15) return;

  const project = (idx) =>
    mapLandmarkToCanvas(landmarks[idx], canvasWidth, canvasHeight, videoWidth, videoHeight, mirror);

  // Subject's Left Eye (appears on the LEFT side of mirrored canvas)
  const lOuter = project(LANDMARKS.LEFT_EYE_OUTER);
  const lInner = project(LANDMARKS.LEFT_EYE_INNER);
  const lTop = project(LANDMARKS.LEFT_EYE_TOP);
  const lBottom = project(LANDMARKS.LEFT_EYE_BOTTOM);
  const lIris = project(LANDMARKS.LEFT_IRIS_CENTER);

  // Subject's Right Eye (appears on the RIGHT side of mirrored canvas)
  const rOuter = project(LANDMARKS.RIGHT_EYE_OUTER);
  const rInner = project(LANDMARKS.RIGHT_EYE_INNER);
  const rTop = project(LANDMARKS.RIGHT_EYE_TOP);
  const rBottom = project(LANDMARKS.RIGHT_EYE_BOTTOM);
  const rIris = project(LANDMARKS.RIGHT_IRIS_CENTER);

  // Ensure left-to-right horizontal ordering for clean bezier orientation
  const screenLeftEye = {
    outerCorner: lOuter.x < lInner.x ? lOuter : lInner,
    innerCorner: lOuter.x < lInner.x ? lInner : lOuter,
    topLid: lTop,
    bottomLid: lBottom,
    irisCenter: lIris,
    // MediaPipe's eyeBlinkLeft/Right corresponds to subject's anatomical eyes
    blinkScore: eyeBlinkLeft,
    isLeftOnScreen: true,
    stageProgress: effectiveProgress,
  };

  const screenRightEye = {
    outerCorner: rOuter.x < rInner.x ? rOuter : rInner,
    innerCorner: rOuter.x < rInner.x ? rInner : rOuter,
    topLid: rTop,
    bottomLid: rBottom,
    irisCenter: rIris,
    blinkScore: eyeBlinkRight,
    isLeftOnScreen: false,
    stageProgress: effectiveProgress,
  };

  drawSingleMangaEye(ctx, screenLeftEye);
  drawSingleMangaEye(ctx, screenRightEye);
}

/**
 * STAGE 2 FRAMING:
 * Draws animated black radial speed lines (shuchu-sen) around the outer 20% of the canvas border,
 * leaving the central 80% clear for the subject's transformed manga portrait.
 */
export function renderShuchuSenSpeedLines(ctx, canvasWidth, canvasHeight, timestampMs) {
  const cx = canvasWidth * 0.5;
  const cy = canvasHeight * 0.5;
  const maxRadius = Math.hypot(cx, cy) * 1.05;

  // Inner clear ellipse radii (80% of canvas dimensions = outer 20% border filled with speed lines)
  const clearRx = canvasWidth * 0.39;
  const clearRy = canvasHeight * 0.39;

  // Animated jitter phase at ~18 FPS manga anime cadence
  const frameSeed = Math.floor(timestampMs / 55);
  const numSpokes = 132;

  ctx.save();
  ctx.fillStyle = INK_BLACK;

  for (let i = 0; i < numSpokes; i++) {
    // Deterministic pseudo-random per spoke + frameSeed
    const hash = Math.sin(i * 127.1 + frameSeed * 311.7) * 43758.5453;
    const rand1 = hash - Math.floor(hash);
    const hash2 = Math.sin(i * 269.5 + frameSeed * 183.3) * 15731.743;
    const rand2 = hash2 - Math.floor(hash2);

    const baseAngle = (i / numSpokes) * Math.PI * 2;
    const angle = baseAngle + (rand1 - 0.5) * 0.018;

    const cosA = Math.cos(angle);
    const sinA = Math.sin(angle);

    // Distance from center to the 80% inner boundary along this angle
    const ellipseRadiusAtAngle =
      (clearRx * clearRy) /
      Math.sqrt(Math.pow(clearRy * cosA, 2) + Math.pow(clearRx * sinA, 2));

    // Vary tip intrusion slightly around the outer 20% perimeter
    const tipRadius = ellipseRadiusAtAngle * (0.94 + rand2 * 0.16);
    const halfSpread = (0.004 + rand1 * 0.013);

    const tipX = cx + cosA * tipRadius;
    const tipY = cy + sinA * tipRadius;

    const outerX1 = cx + Math.cos(angle - halfSpread) * maxRadius;
    const outerY1 = cy + Math.sin(angle - halfSpread) * maxRadius;
    const outerX2 = cx + Math.cos(angle + halfSpread) * maxRadius;
    const outerY2 = cy + Math.sin(angle + halfSpread) * maxRadius;

    ctx.beginPath();
    ctx.moveTo(tipX, tipY);
    ctx.lineTo(outerX1, outerY1);
    ctx.lineTo(outerX2, outerY2);
    ctx.closePath();
    ctx.fill();
  }

  // Brutalist thick ink panel frame border
  ctx.strokeStyle = INK_BLACK;
  ctx.lineWidth = 14;
  ctx.strokeRect(7, 7, canvasWidth - 14, canvasHeight - 14);

  ctx.strokeStyle = PAPER_WHITE;
  ctx.lineWidth = 2;
  ctx.strokeRect(15, 15, canvasWidth - 30, canvasHeight - 30);

  ctx.restore();
}

/**
 * Optional Debug Mesh & Gaze Vector Overlay when debug landmarks toggle is active
 */
export function renderDebugLandmarksOverlay(ctx, {
  landmarks,
  canvasWidth,
  canvasHeight,
  videoWidth,
  videoHeight,
  mirror = true,
  isGazeLocked,
}) {
  if (!landmarks || landmarks.length < 474) return;

  const project = (idx) =>
    mapLandmarkToCanvas(landmarks[idx], canvasWidth, canvasHeight, videoWidth, videoHeight, mirror);

  ctx.save();
  const keyIndices = [
    LANDMARKS.NOSE_TIP,
    LANDMARKS.FOREHEAD,
    LANDMARKS.CHIN,
    LANDMARKS.RIGHT_EYE_OUTER,
    LANDMARKS.RIGHT_EYE_INNER,
    LANDMARKS.LEFT_EYE_INNER,
    LANDMARKS.LEFT_EYE_OUTER,
    LANDMARKS.RIGHT_IRIS_CENTER,
    LANDMARKS.LEFT_IRIS_CENTER,
  ];

  for (const idx of keyIndices) {
    const pt = project(idx);
    const isIris = idx === LANDMARKS.RIGHT_IRIS_CENTER || idx === LANDMARKS.LEFT_IRIS_CENTER;
    ctx.fillStyle = isIris ? HANKO_RED : PAPER_WHITE;
    ctx.strokeStyle = INK_BLACK;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, isIris ? 5.5 : 3.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }

  // Draw symmetry axis
  const f = project(LANDMARKS.FOREHEAD);
  const n = project(LANDMARKS.NOSE_TIP);
  const c = project(LANDMARKS.CHIN);
  ctx.strokeStyle = isGazeLocked ? HANKO_RED : PAPER_WHITE;
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(f.x, f.y);
  ctx.lineTo(n.x, n.y);
  ctx.lineTo(c.x, c.y);
  ctx.stroke();
  ctx.restore();
}
