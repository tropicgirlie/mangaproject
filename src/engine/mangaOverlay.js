/**
 * MANGAFACE EFFECT — Contour-Tracked Natural Anime Eye & Iris Renderer
 *
 * Fixes the "weird pasted sticker eye" look and guarantees 100% anatomical alignment:
 * 1. Traces the user's ACTUAL MediaPipe 16-point eye socket contour polygon
 *    and 5-point iris ring (#468..#472 and #473..#477) directly in canvas space.
 * 2. Never draws an artificial white oval sticker outside the user's real eye contour!
 * 3. Clips a jewel-toned colorful anime iris (Golden Hour Twilight, Shinkai Sapphire, or Neo-Tokyo Amber)
 *    with dual crisp specular catchlights INSIDE the user's real eyelid opening.
 * 4. Strokes a tapered calligraphic winged upper eyelash along the user's real upper eyelid landmarks
 *    plus delicate anime cheek-blush hatch marks below the outer eye corners.
 */

import { mapLandmarkToCanvas } from './mangaShader';

// Canonical MediaPipe 478 FaceLandmarker Contour Rings
export const LANDMARKS = {
  NOSE_TIP: 1,
  FOREHEAD: 10,
  CHIN: 152,

  // Subject's Right Eye
  RIGHT_EYE_OUTER: 33,
  RIGHT_EYE_INNER: 133,
  RIGHT_EYE_TOP: 159,
  RIGHT_EYE_BOTTOM: 145,
  RIGHT_IRIS_CENTER: 468,
  RIGHT_IRIS_RING: [469, 470, 471, 472],
  // Full 16-point anatomical eyelid loop
  RIGHT_EYE_LOOP: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
  // Upper lid contour from inner to outer corner
  RIGHT_UPPER_LID: [133, 173, 157, 158, 159, 160, 161, 246, 33],
  // Lower lid contour from outer to inner corner
  RIGHT_LOWER_LID: [33, 7, 163, 144, 145, 153, 154, 155, 133],

  // Subject's Left Eye
  LEFT_EYE_INNER: 362,
  LEFT_EYE_OUTER: 263,
  LEFT_EYE_TOP: 386,
  LEFT_EYE_BOTTOM: 374,
  LEFT_IRIS_CENTER: 473,
  LEFT_IRIS_RING: [474, 475, 476, 477],
  // Full 16-point anatomical eyelid loop
  LEFT_EYE_LOOP: [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
  // Upper lid contour from inner to outer corner
  LEFT_UPPER_LID: [362, 398, 384, 385, 386, 387, 388, 466, 263],
  // Lower lid contour from outer to inner corner
  LEFT_LOWER_LID: [263, 249, 390, 373, 374, 380, 381, 382, 362],
};

// Style palettes matching the 3 colorful WebGL shader styles
const EYE_PALETTES = [
  {
    // 0: Golden Hour Nostalgia (Reference Image Match)
    ink: '#221518',
    irisTop: '#1E1B24',
    irisMid: '#63536B',
    irisBot: '#E8B8B4',
    irisRim: '#1C1317',
    blush: 'rgba(226, 95, 82, 0.58)',
    speedMain: '#221518',
    speedAccent: '#E06D58',
  },
  {
    // 1: Shinkai Vivid Sky & Bloom
    ink: '#151336',
    irisTop: '#17153B',
    irisMid: '#2563EB',
    irisBot: '#A5F3FC',
    irisRim: '#0F172A',
    blush: 'rgba(244, 114, 182, 0.58)',
    speedMain: '#151336',
    speedAccent: '#38BDF8',
  },
  {
    // 2: Neo-Tokyo Pop Cel
    ink: '#0A0A0E',
    irisTop: '#7C2D12',
    irisMid: '#EA580C',
    irisBot: '#FDE047',
    irisRim: '#18181B',
    blush: 'rgba(244, 63, 94, 0.62)',
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
 * Helper to trace a smooth path through an array of 2D points
 */
function traceSmoothPolygon(ctx, pts, close = true) {
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
  if (close) ctx.closePath();
}

/**
 * Renders a natural, contour-locked Anime Eye directly onto the user's real MediaPipe eye & iris landmarks.
 * Scales vertically with `blinkScore` and never pastes an artificial white box outside the eye socket.
 */
function drawContourLockedAnimeEye(ctx, {
  loopPts,
  upperLidPts,
  lowerLidPts,
  irisCenter,
  irisRadius,
  blinkScore,
  stageProgress,
  palette,
}) {
  const innerPt = upperLidPts[0];
  const outerPt = upperLidPts[upperLidPts.length - 1];
  const eyeWidth = Math.hypot(outerPt.x - innerPt.x, outerPt.y - innerPt.y);
  if (eyeWidth < 4) return;

  // Determine screen-space direction toward the outer corner (-1 left, +1 right)
  const outerDir = outerPt.x < innerPt.x ? -1 : 1;

  // Smooth fade-in as dwellProgress crosses Stage 1 (0.15 -> 0.32)
  const alpha = Math.min(1.0, Math.max(0.0, (stageProgress - 0.15) / 0.18));

  // Blendshape blink factor (0.0 open -> 1.0 closed)
  const blinkClosed = Math.min(1.0, Math.max(0.0, (blinkScore - 0.08) * 1.35));
  const openness = 1.0 - blinkClosed;

  ctx.save();
  ctx.globalAlpha = alpha;

  // 1. If eye is open, render brightened anime sclera + jewel-toned iris + specular catchlights
  //    clipped strictly inside the user's real 16-point eyelid contour!
  if (openness > 0.16) {
    ctx.save();
    traceSmoothPolygon(ctx, loopPts, true);
    ctx.clip();

    // Gently brighten the user's real sclera so the anime iris pops cleanly
    ctx.fillStyle = 'rgba(252, 249, 244, 0.72)';
    ctx.fill();

    // Jewel-Toned Anime Iris centered on MediaPipe's exact iris landmark (#468 / #473)
    const r = Math.max(irisRadius * 1.12, eyeWidth * 0.24);
    const ix = irisCenter.x;
    const iy = irisCenter.y;

    // Outer dark limbal ring
    ctx.fillStyle = palette.irisRim;
    ctx.beginPath();
    ctx.arc(ix, iy, r, 0, Math.PI * 2);
    ctx.fill();

    // Vertical jewel gradient inside iris
    const irisGrad = ctx.createLinearGradient(ix, iy - r, ix, iy + r);
    irisGrad.addColorStop(0, palette.irisTop);
    irisGrad.addColorStop(0.48, palette.irisMid);
    irisGrad.addColorStop(1, palette.irisBot);

    ctx.fillStyle = irisGrad;
    ctx.beginPath();
    ctx.arc(ix, iy, r * 0.86, 0, Math.PI * 2);
    ctx.fill();

    // Delicate radial iris spokes
    ctx.strokeStyle = palette.ink;
    ctx.lineWidth = Math.max(1.0, r * 0.06);
    for (let s = 0; s < 14; s++) {
      const a = (s / 14) * Math.PI * 2;
      ctx.beginPath();
      ctx.moveTo(ix + Math.cos(a) * r * 0.35, iy + Math.sin(a) * r * 0.35);
      ctx.lineTo(ix + Math.cos(a) * r * 0.85, iy + Math.sin(a) * r * 0.85);
      ctx.stroke();
    }

    // Deep Ink Pupil Core
    ctx.fillStyle = palette.ink;
    ctx.beginPath();
    ctx.arc(ix, iy, r * 0.38, 0, Math.PI * 2);
    ctx.fill();

    // Glowing lower crescent reflection inside iris
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
    ctx.lineWidth = Math.max(1.2, r * 0.1);
    ctx.beginPath();
    ctx.arc(ix, iy, r * 0.58, 0.2 * Math.PI, 0.8 * Math.PI);
    ctx.stroke();

    // Upper eyelid cast shadow across top of iris & sclera
    ctx.fillStyle = 'rgba(22, 16, 24, 0.32)';
    ctx.beginPath();
    ctx.ellipse(ix, iy - r * 0.75, eyeWidth * 0.6, r * 0.48, 0, 0, Math.PI * 2);
    ctx.fill();

    // Two Crisp White Specular Catchlights (classic expressive anime look)
    ctx.fillStyle = '#FFFFFF';
    // Primary top-outer highlight
    ctx.beginPath();
    ctx.arc(ix + outerDir * r * 0.34, iy - r * 0.28, r * 0.25, 0, Math.PI * 2);
    ctx.fill();

    // Secondary bottom-inner pinpoint highlight
    ctx.beginPath();
    ctx.arc(ix - outerDir * r * 0.30, iy + r * 0.30, r * 0.13, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
  }

  // 2. Calligraphic Upper Eyelash Line following the user's real upper eyelid points
  ctx.save();
  ctx.strokeStyle = palette.ink;
  ctx.fillStyle = palette.ink;
  ctx.lineWidth = Math.max(2.6, eyeWidth * 0.075);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  traceSmoothPolygon(ctx, upperLidPts, false);
  ctx.stroke();

  // Elegant winged outer lash flick at the outer corner of the user's eye
  const wingLen = eyeWidth * 0.18;
  const wingUp = eyeWidth * 0.09;
  ctx.beginPath();
  ctx.moveTo(outerPt.x - outerDir * eyeWidth * 0.08, outerPt.y - eyeWidth * 0.02);
  ctx.quadraticCurveTo(
    outerPt.x + outerDir * wingLen * 0.5,
    outerPt.y + eyeWidth * 0.01,
    outerPt.x + outerDir * wingLen,
    outerPt.y - wingUp
  );
  ctx.lineTo(outerPt.x - outerDir * eyeWidth * 0.02, outerPt.y - eyeWidth * 0.06);
  ctx.closePath();
  ctx.fill();

  // Subtle lower lash accent along the outer half of the lower lid
  ctx.lineWidth = Math.max(1.3, eyeWidth * 0.03);
  ctx.globalAlpha = alpha * 0.72;
  traceSmoothPolygon(ctx, lowerLidPts.slice(0, 6), false);
  ctx.stroke();

  // Delicate double-eyelid crease line slightly above the real upper lid arch
  const midUpper = upperLidPts[4]; // #159 or #386 top arch
  ctx.lineWidth = Math.max(1.1, eyeWidth * 0.024);
  ctx.globalAlpha = alpha * 0.55;
  ctx.beginPath();
  ctx.moveTo(upperLidPts[2].x, upperLidPts[2].y - eyeWidth * 0.08);
  ctx.quadraticCurveTo(
    midUpper.x,
    midUpper.y - eyeWidth * 0.11,
    upperLidPts[6].x,
    upperLidPts[6].y - eyeWidth * 0.07
  );
  ctx.stroke();
  ctx.restore();

  // 3. Soft Diagonal Anime Cheek Blush Hatching below the outer eye corner (like reference image)
  ctx.save();
  ctx.globalAlpha = alpha * 0.78;
  ctx.strokeStyle = palette.blush;
  ctx.lineWidth = Math.max(1.4, eyeWidth * 0.032);
  ctx.lineCap = 'round';
  const blushBaseX = outerPt.x - outerDir * eyeWidth * 0.15;
  const blushBaseY = outerPt.y + eyeWidth * 0.28;
  for (let b = 0; b < 3; b++) {
    const bx = blushBaseX - outerDir * (b * eyeWidth * 0.11);
    ctx.beginPath();
    ctx.moveTo(bx + eyeWidth * 0.03, blushBaseY - eyeWidth * 0.035);
    ctx.lineTo(bx - eyeWidth * 0.03, blushBaseY + eyeWidth * 0.045);
    ctx.stroke();
  }
  ctx.restore();

  ctx.restore();
}

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
  colorStyle = 0,
}) {
  if (!landmarks || landmarks.length < 474) return;
  const effectiveProgress = isStage2Locked ? 1.0 : dwellProgress;
  if (effectiveProgress <= 0.15) return;

  const palette = EYE_PALETTES[colorStyle] || EYE_PALETTES[0];
  const project = (idx) =>
    mapLandmarkToCanvas(landmarks[idx], canvasWidth, canvasHeight, videoWidth, videoHeight, mirror);

  // Measure real projected iris radius from MediaPipe's 4-point iris perimeter ring
  const getIrisRadius = (centerPt, ringIndices) => {
    let sum = 0;
    for (let i = 0; i < ringIndices.length; i++) {
      const pt = project(ringIndices[i]);
      sum += Math.hypot(pt.x - centerPt.x, pt.y - centerPt.y);
    }
    return sum / ringIndices.length;
  };

  // Subject's Left Eye
  const lIrisCenter = project(LANDMARKS.LEFT_IRIS_CENTER);
  const lIrisRadius = getIrisRadius(lIrisCenter, LANDMARKS.LEFT_IRIS_RING);

  drawContourLockedAnimeEye(ctx, {
    loopPts: LANDMARKS.LEFT_EYE_LOOP.map(project),
    upperLidPts: LANDMARKS.LEFT_UPPER_LID.map(project),
    lowerLidPts: LANDMARKS.LEFT_LOWER_LID.map(project),
    irisCenter: lIrisCenter,
    irisRadius: lIrisRadius,
    blinkScore: eyeBlinkLeft,
    stageProgress: effectiveProgress,
    palette,
  });

  // Subject's Right Eye
  const rIrisCenter = project(LANDMARKS.RIGHT_IRIS_CENTER);
  const rIrisRadius = getIrisRadius(rIrisCenter, LANDMARKS.RIGHT_IRIS_RING);

  drawContourLockedAnimeEye(ctx, {
    loopPts: LANDMARKS.RIGHT_EYE_LOOP.map(project),
    upperLidPts: LANDMARKS.RIGHT_UPPER_LID.map(project),
    lowerLidPts: LANDMARKS.RIGHT_LOWER_LID.map(project),
    irisCenter: rIrisCenter,
    irisRadius: rIrisRadius,
    blinkScore: eyeBlinkRight,
    stageProgress: effectiveProgress,
    palette,
  });
}

/**
 * STAGE 2 FRAMING:
 * Colorful tapered Japanese radial speed lines (shuchu-sen) around the outer 20% border
 */
export function renderShuchuSenSpeedLines(ctx, canvasWidth, canvasHeight, timestampMs, colorStyle = 0) {
  const palette = EYE_PALETTES[colorStyle] || EYE_PALETTES[0];
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
