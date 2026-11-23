/**
 * MANGAFACE EFFECT — Gaze Lock Evaluator, GPU Eye Uniform Extractor & Delicate Calligraphic Lash Accent
 *
 * Instead of drawing flat 2D cartoon vector eyes on top of the camera feed,
 * `extractShaderEyeUniforms` computes the exact normalized UV coordinates and radii
 * of the user's irises (`#468..#472` and `#473..#477`) plus `eyeBlinkLeft`/`eyeBlinkRight`
 * openness so the WebGL fragment shader optically magnifies and relights the user's REAL eye pixels.
 *
 * The 2D overlay canvas only adds ultra-delicate calligraphic lash ink accents along the user's
 * real upper eyelid landmarks and Stage 2 radial shuchu-sen speed lines.
 */

import { mapLandmarkToCanvas } from './mangaShader';

export const LANDMARKS = {
  NOSE_TIP: 1,
  FOREHEAD: 10,
  CHIN: 152,

  RIGHT_EYE_OUTER: 33,
  RIGHT_EYE_INNER: 133,
  RIGHT_EYE_TOP: 159,
  RIGHT_EYE_BOTTOM: 145,
  RIGHT_IRIS_CENTER: 468,
  RIGHT_IRIS_RING: [469, 470, 471, 472],
  RIGHT_UPPER_LID: [133, 173, 157, 158, 159, 160, 161, 246, 33],

  LEFT_EYE_INNER: 362,
  LEFT_EYE_OUTER: 263,
  LEFT_EYE_TOP: 386,
  LEFT_EYE_BOTTOM: 374,
  LEFT_IRIS_CENTER: 473,
  LEFT_IRIS_RING: [474, 475, 476, 477],
  LEFT_UPPER_LID: [362, 398, 384, 385, 386, 387, 388, 466, 263],
};

const PALETTES = [
  {
    ink: '#221518',
    blush: 'rgba(226, 95, 82, 0.50)',
    speedMain: '#221518',
    speedAccent: '#E06D58',
  },
  {
    ink: '#151336',
    blush: 'rgba(244, 114, 182, 0.50)',
    speedMain: '#151336',
    speedAccent: '#38BDF8',
  },
  {
    ink: '#0A0A0E',
    blush: 'rgba(244, 63, 94, 0.55)',
    speedMain: '#0A0A0E',
    speedAccent: '#2DD4BF',
  },
];

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

  const eyeMidX = (rOuter.x + lOuter.x) * 0.5;
  const interOcularDist = Math.max(Math.hypot(lOuter.x - rOuter.x, lOuter.y - rOuter.y), 0.05);
  const yaw = (nose.x - eyeMidX) / (interOcularDist * 0.5);

  const faceHeight = Math.max(Math.hypot(chin.x - forehead.x, chin.y - forehead.y), 0.08);
  const eyeMidY = (rOuter.y + lOuter.y) * 0.5;
  const noseRelY = (nose.y - eyeMidY) / faceHeight;
  const pitch = (noseRelY - 0.27) * 2.4;

  const rWidth = Math.max(Math.hypot(rInner.x - rOuter.x, rInner.y - rOuter.y), 0.01);
  const rRatioX = Math.hypot(rIris.x - rOuter.x, rIris.y - rOuter.y) / rWidth;

  const lWidth = Math.max(Math.hypot(lOuter.x - lInner.x, lOuter.y - lInner.y), 0.01);
  const lRatioX = Math.hypot(lIris.x - lInner.x, lIris.y - lInner.y) / lWidth;

  const rHeight = Math.max(Math.hypot(rBottom.x - rTop.x, rBottom.y - rTop.y), 0.005);
  const rRatioY = Math.hypot(rIris.x - rTop.x, rIris.y - rTop.y) / rHeight;

  const lHeight = Math.max(Math.hypot(lBottom.x - lTop.x, lBottom.y - lTop.y), 0.005);
  const lRatioY = Math.hypot(lIris.x - lTop.x, lIris.y - lTop.y) / lHeight;

  const irisOffsetX = (Math.abs(rRatioX - 0.5) + Math.abs(lRatioX - 0.5)) * 0.5;
  const irisOffsetY = (Math.abs(rRatioY - 0.5) + Math.abs(lRatioY - 0.5)) * 0.5;
  const irisOffset = irisOffsetX * 0.78 + irisOffsetY * 0.22;

  const { eyeBlinkLeft, eyeBlinkRight } = extractBlinkBlendshapes(blendshapes);

  const lowYaw = Math.abs(yaw) < 0.28;
  const lowPitch = Math.abs(pitch) < 0.36;
  const centeredIris = irisOffsetX < 0.20;

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
 * Extracts normalized source-UV iris positions, radii, and openness values
 * to pass directly into the WebGL fragment shader for GPU shader-space eye relighting & optical warp.
 */
export function extractShaderEyeUniforms(landmarks, eyeBlinkLeft, eyeBlinkRight, videoWidth, videoHeight) {
  if (!landmarks || landmarks.length < 478) return null;

  const aspect = (videoWidth || 1280) / Math.max(videoHeight || 720, 1);

  const getRadius = (center, ringIndices) => {
    let sum = 0;
    for (let i = 0; i < ringIndices.length; i++) {
      const pt = landmarks[ringIndices[i]];
      const dx = (pt.x - center.x) * aspect;
      const dy = pt.y - center.y;
      sum += Math.hypot(dx, dy);
    }
    return sum / ringIndices.length;
  };

  const lIris = landmarks[LANDMARKS.LEFT_IRIS_CENTER];
  const rIris = landmarks[LANDMARKS.RIGHT_IRIS_CENTER];

  const lOpen = Math.max(0.05, 1.0 - Math.min(1.0, Math.max(0.0, (eyeBlinkLeft - 0.06) * 1.3)));
  const rOpen = Math.max(0.05, 1.0 - Math.min(1.0, Math.max(0.0, (eyeBlinkRight - 0.06) * 1.3)));

  return {
    leftIrisUV: { x: lIris.x, y: lIris.y },
    rightIrisUV: { x: rIris.x, y: rIris.y },
    leftIrisRadius: getRadius(lIris, LANDMARKS.LEFT_IRIS_RING),
    rightIrisRadius: getRadius(rIris, LANDMARKS.RIGHT_IRIS_RING),
    leftOpenness: lOpen,
    rightOpenness: rOpen,
  };
}

function traceSmoothCurve(ctx, pts) {
  if (!pts || pts.length < 2) return;
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length - 1; i++) {
    const xc = (pts[i].x + pts[i + 1].x) * 0.5;
    const yc = (pts[i].y + pts[i + 1].y) * 0.5;
    ctx.quadraticCurveTo(pts[i].x, pts[i].y, xc, yc);
  }
  const last = pts[pts.length - 1];
  ctx.lineTo(last.x, last.y);
}

/**
 * Draws only a delicate calligraphic upper eyelash ink line & subtle cheek blush hatching
 * along the user's real upper eyelid landmarks (no cartoon white sticker!).
 */
function drawDelicateLashAccent(ctx, upperLidPts, palette, alpha) {
  const innerPt = upperLidPts[0];
  const outerPt = upperLidPts[upperLidPts.length - 1];
  const eyeWidth = Math.hypot(outerPt.x - innerPt.x, outerPt.y - innerPt.y);
  if (eyeWidth < 4) return;

  const outerDir = outerPt.x < innerPt.x ? -1 : 1;

  ctx.save();
  ctx.globalAlpha = alpha * 0.85;
  ctx.strokeStyle = palette.ink;
  ctx.lineWidth = Math.max(2.0, eyeWidth * 0.055);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  traceSmoothCurve(ctx, upperLidPts);
  ctx.stroke();

  // Subtle anime cheek blush hatch marks beneath the outer eye corner
  ctx.globalAlpha = alpha * 0.65;
  ctx.strokeStyle = palette.blush;
  ctx.lineWidth = Math.max(1.3, eyeWidth * 0.028);
  const blushBaseX = outerPt.x - outerDir * eyeWidth * 0.12;
  const blushBaseY = outerPt.y + eyeWidth * 0.30;
  for (let b = 0; b < 3; b++) {
    const bx = blushBaseX - outerDir * (b * eyeWidth * 0.10);
    ctx.beginPath();
    ctx.moveTo(bx + eyeWidth * 0.025, blushBaseY - eyeWidth * 0.03);
    ctx.lineTo(bx - eyeWidth * 0.025, blushBaseY + eyeWidth * 0.035);
    ctx.stroke();
  }

  ctx.restore();
}

export function renderMangaEyesOverlay(ctx, {
  landmarks,
  dwellProgress,
  isStage2Locked,
  canvasWidth,
  canvasHeight,
  videoWidth,
  videoHeight,
  mirror = true,
  colorStyle = 0,
}) {
  if (!landmarks || landmarks.length < 474) return;
  const effectiveProgress = isStage2Locked ? 1.0 : dwellProgress;
  if (effectiveProgress <= 0.15) return;

  const alpha = Math.min(1.0, Math.max(0.0, (effectiveProgress - 0.15) / 0.20));
  const palette = PALETTES[colorStyle] || PALETTES[0];
  const project = (idx) =>
    mapLandmarkToCanvas(landmarks[idx], canvasWidth, canvasHeight, videoWidth, videoHeight, mirror);

  drawDelicateLashAccent(ctx, LANDMARKS.LEFT_UPPER_LID.map(project), palette, alpha);
  drawDelicateLashAccent(ctx, LANDMARKS.RIGHT_UPPER_LID.map(project), palette, alpha);
}

/**
 * STAGE 2 FRAMING:
 * Colorful tapered Japanese radial speed lines (shuchu-sen) around the outer 20% border
 */
export function renderShuchuSenSpeedLines(ctx, canvasWidth, canvasHeight, timestampMs, colorStyle = 0) {
  const palette = PALETTES[colorStyle] || PALETTES[0];
  const cx = canvasWidth * 0.5;
  const cy = canvasHeight * 0.5;
  const maxRadius = Math.hypot(cx, cy) * 1.05;

  const clearRx = canvasWidth * 0.40;
  const clearRy = canvasHeight * 0.40;

  const frameSeed = Math.floor(timestampMs / 60);
  const numSpokes = 140;

  ctx.save();

  for (let i = 0; i < numSpokes; i++) {
    const hash = Math.sin(i * 127.1 + frameSeed * 311.7) * 43758.5453;
    const rand1 = hash - Math.floor(hash);
    const hash2 = Math.sin(i * 269.5 + frameSeed * 183.3) * 15731.743;
    const rand2 = hash2 - Math.floor(hash2);

    const baseAngle = (i / numSpokes) * Math.PI * 2;
    const angle = baseAngle + (rand1 - 0.5) * 0.014;

    const cosA = Math.cos(angle);
    const sinA = Math.sin(angle);

    const ellipseRadiusAtAngle =
      (clearRx * clearRy) /
      Math.sqrt(Math.pow(clearRy * cosA, 2) + Math.pow(clearRx * sinA, 2));

    const tipRadius = ellipseRadiusAtAngle * (0.95 + rand2 * 0.15);
    const halfSpread = 0.0025 + rand1 * 0.009;

    const tipX = cx + cosA * tipRadius;
    const tipY = cy + sinA * tipRadius;

    const outerX1 = cx + Math.cos(angle - halfSpread) * maxRadius;
    const outerY1 = cy + Math.sin(angle - halfSpread) * maxRadius;
    const outerX2 = cx + Math.cos(angle + halfSpread) * maxRadius;
    const outerY2 = cy + Math.sin(angle + halfSpread) * maxRadius;

    ctx.fillStyle = i % 6 === 0 ? palette.speedAccent : palette.speedMain;
    ctx.beginPath();
    ctx.moveTo(tipX, tipY);
    ctx.lineTo(outerX1, outerY1);
    ctx.lineTo(outerX2, outerY2);
    ctx.closePath();
    ctx.fill();
  }

  ctx.strokeStyle = palette.speedMain;
  ctx.lineWidth = 12;
  ctx.strokeRect(6, 6, canvasWidth - 12, canvasHeight - 12);

  ctx.strokeStyle = '#F4F1EA';
  ctx.lineWidth = 2;
  ctx.strokeRect(14, 14, canvasWidth - 28, canvasHeight - 28);

  ctx.restore();
}

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
    ctx.fillStyle = isIris ? '#D92B2B' : '#F4F1EA';
    ctx.strokeStyle = '#0A0A0A';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, isIris ? 5.5 : 3.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }

  const f = project(LANDMARKS.FOREHEAD);
  const n = project(LANDMARKS.NOSE_TIP);
  const c = project(LANDMARKS.CHIN);
  ctx.strokeStyle = isGazeLocked ? '#D92B2B' : '#F4F1EA';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(f.x, f.y);
  ctx.lineTo(n.x, n.y);
  ctx.lineTo(c.x, c.y);
  ctx.stroke();
  ctx.restore();
}
