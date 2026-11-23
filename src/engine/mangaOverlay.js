/**
 * MANGAFACE EFFECT — Gaze Lock Evaluator, Refined Anime/Manga Eyes & Shuchu-Sen Renderer
 *
 * Designed after the user's reference illustration:
 * - Expressive anime/seinen eyes with layered slate-violet/amber-rose iris gradients,
 *   fine radial ink spokes, upper eyelid shadow cast, calligraphic winged upper lash,
 *   delicate cheek blush strokes, and dual specular highlights.
 * - Dynamic vertical openness driven by MediaPipe's `eyeBlinkLeft` and `eyeBlinkRight` blendshapes.
 * - Clean, razor-sharp radial shuchu-sen speed lines around the outer 20% border in Stage 2.
 */

import { mapLandmarkToCanvas } from './mangaShader';

const INK_BLACK = '#0A0A0A';
const PAPER_WHITE = '#F4F1EA';
const HANKO_RED = '#D92B2B';

export const LANDMARKS = {
  NOSE_TIP: 1,
  FOREHEAD: 10,
  CHIN: 152,

  RIGHT_EYE_OUTER: 33,
  RIGHT_EYE_INNER: 133,
  RIGHT_EYE_TOP: 159,
  RIGHT_EYE_BOTTOM: 145,
  RIGHT_IRIS_CENTER: 468,

  LEFT_EYE_INNER: 362,
  LEFT_EYE_OUTER: 263,
  LEFT_EYE_TOP: 386,
  LEFT_EYE_BOTTOM: 374,
  LEFT_IRIS_CENTER: 473,
};

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
 * Draws a polished anime/manga eye inspired by the attached reference artwork:
 * - Soft feathered skin/paper halo so the vector eye integrates organically with live webcam skin
 * - Deep calligraphic winged upper lash line with subtle warm brown-ink undertone
 * - Multi-layered radial iris (deep slate upper hood, warm twilight mid-ring, radial ink spokes, crisp pupil)
 * - Dual crisp `#F4F1EA` specular catchlights
 * - Subtle diagonal manga blush hatch marks beneath the outer eye corner
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
  isStage2Locked,
}) {
  const dx = innerCorner.x - outerCorner.x;
  const dy = innerCorner.y - outerCorner.y;
  const rawEyeWidth = Math.hypot(dx, dy);

  const eyeWidth = rawEyeWidth * 1.36;
  const eyeCenter = {
    x: (outerCorner.x + innerCorner.x) * 0.5,
    y: (topLid.y + bottomLid.y) * 0.5,
  };

  let angle = Math.atan2(dy, dx);
  if (Math.abs(angle) > Math.PI * 0.5) {
    angle += angle > 0 ? -Math.PI : Math.PI;
  }

  const clampedBlink = Math.min(1.0, Math.max(0.0, (blinkScore - 0.06) * 1.35));
  const openness = Math.max(0.05, 1.0 - clampedBlink);

  const maxEyeHeight = eyeWidth * 0.58;
  const currentHeight = maxEyeHeight * openness;

  const irisRelX = Math.max(
    -eyeWidth * 0.2,
    Math.min(eyeWidth * 0.2, (irisCenter.x - eyeCenter.x) * 0.82)
  );
  const irisRelY = Math.max(
    -currentHeight * 0.16,
    Math.min(currentHeight * 0.16, (irisCenter.y - eyeCenter.y) * 0.65)
  );

  const popScale = Math.min(1.0, Math.max(0.0, (stageProgress - 0.15) / 0.14));
  const easePop = 1 - Math.pow(1 - popScale, 3);

  ctx.save();
  ctx.translate(eyeCenter.x, eyeCenter.y);
  ctx.rotate(angle);
  ctx.scale(easePop, easePop);

  const halfW = eyeWidth * 0.5;
  const halfH = currentHeight * 0.5;
  const outerDir = isLeftOnScreen ? -1 : 1;

  // 1. Soft Feathered Backdrop Halo (prevents harsh cutout look over real webcam feed)
  const haloGrad = ctx.createRadialGradient(0, 0, halfW * 0.3, 0, 0, halfW * 1.28);
  haloGrad.addColorStop(0, isStage2Locked ? 'rgba(244, 241, 234, 0.96)' : 'rgba(244, 241, 234, 0.78)');
  haloGrad.addColorStop(0.7, isStage2Locked ? 'rgba(244, 241, 234, 0.65)' : 'rgba(244, 235, 226, 0.35)');
  haloGrad.addColorStop(1, 'rgba(244, 241, 234, 0.0)');
  ctx.fillStyle = haloGrad;
  ctx.beginPath();
  ctx.ellipse(0, 0, halfW * 1.28, Math.max(halfH * 1.55, eyeWidth * 0.26), 0, 0, Math.PI * 2);
  ctx.fill();

  // 2. Subtle Diagonal Anime Cheek Blush Hatch Marks below outer eye (like reference image!)
  ctx.save();
  const blushY = Math.max(halfH * 1.15, eyeWidth * 0.24) + eyeWidth * 0.08;
  const blushX = outerDir * halfW * 0.42;
  ctx.strokeStyle = isStage2Locked ? 'rgba(217, 43, 43, 0.65)' : 'rgba(224, 96, 88, 0.55)';
  ctx.lineWidth = Math.max(1.4, eyeWidth * 0.022);
  ctx.lineCap = 'round';
  for (let b = -1; b <= 2; b++) {
    const bx = blushX + b * eyeWidth * 0.085;
    ctx.beginPath();
    ctx.moveTo(bx + eyeWidth * 0.03, blushY - eyeWidth * 0.035);
    ctx.lineTo(bx - eyeWidth * 0.03, blushY + eyeWidth * 0.045);
    ctx.stroke();
  }
  ctx.restore();

  if (openness > 0.14) {
    // 3. Almond Eye Sclera Clipping Path
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(-halfW, halfH * 0.08);
    ctx.bezierCurveTo(
      -halfW * 0.5, -halfH * 1.32,
       halfW * 0.5, -halfH * 1.32,
       halfW, -halfH * 0.1
    );
    ctx.bezierCurveTo(
       halfW * 0.52,  halfH * 1.22,
      -halfW * 0.52,  halfH * 1.22,
      -halfW, halfH * 0.08
    );
    ctx.closePath();

    ctx.fillStyle = PAPER_WHITE;
    ctx.fill();
    ctx.clip();

    // Soft upper sclera lid shadow
    ctx.fillStyle = 'rgba(26, 24, 28, 0.18)';
    ctx.beginPath();
    ctx.ellipse(0, -halfH * 0.72, halfW * 1.02, halfH * 0.48, 0, 0, Math.PI * 2);
    ctx.fill();

    // 4. Layered Expressive Radial Iris
    const irisRadius = eyeWidth * 0.285;
    const ix = irisRelX;
    const iy = irisRelY * 0.45;

    // Outer dark ink limbal ring
    ctx.fillStyle = INK_BLACK;
    ctx.beginPath();
    ctx.arc(ix, iy, irisRadius, 0, Math.PI * 2);
    ctx.fill();

    // Inner iris gradient (deep charcoal top -> warm slate/amber bottom like reference image)
    const irisGrad = ctx.createLinearGradient(ix, iy - irisRadius, ix, iy + irisRadius);
    if (isStage2Locked) {
      irisGrad.addColorStop(0, '#0A0A0A');
      irisGrad.addColorStop(0.48, '#2C282A');
      irisGrad.addColorStop(0.85, '#EAE3D5');
      irisGrad.addColorStop(1, '#F4F1EA');
    } else {
      irisGrad.addColorStop(0, '#12131A');
      irisGrad.addColorStop(0.45, '#3E3A48');
      irisGrad.addColorStop(0.82, '#B8A8AC');
      irisGrad.addColorStop(1, '#F2E6DF');
    }

    ctx.fillStyle = irisGrad;
    ctx.beginPath();
    ctx.arc(ix, iy, irisRadius * 0.84, 0, Math.PI * 2);
    ctx.fill();

    // Radial ink hatching spokes inside iris
    ctx.strokeStyle = INK_BLACK;
    ctx.lineWidth = Math.max(1.2, eyeWidth * 0.019);
    const spokeCount = 16;
    for (let s = 0; s < spokeCount; s++) {
      const a = (s / spokeCount) * Math.PI * 2;
      const rIn = irisRadius * 0.34;
      const rOut = irisRadius * 0.84;
      ctx.beginPath();
      ctx.moveTo(ix + Math.cos(a) * rIn, iy + Math.sin(a) * rIn);
      ctx.lineTo(ix + Math.cos(a) * rOut, iy + Math.sin(a) * rOut);
      ctx.stroke();
    }

    // Deep Ink Pupil Core
    ctx.fillStyle = INK_BLACK;
    ctx.beginPath();
    ctx.arc(ix, iy, irisRadius * 0.37, 0, Math.PI * 2);
    ctx.fill();

    // Lower luminous crescent reflection inside iris
    ctx.strokeStyle = PAPER_WHITE;
    ctx.lineWidth = Math.max(1.5, irisRadius * 0.1);
    ctx.beginPath();
    ctx.arc(ix, iy, irisRadius * 0.58, 0.22 * Math.PI, 0.78 * Math.PI);
    ctx.stroke();

    // 5. Two Crisp White Specular Catchlights
    ctx.fillStyle = PAPER_WHITE;
    // Large primary upper-outer specular highlight
    ctx.beginPath();
    ctx.arc(
      ix + outerDir * irisRadius * 0.36,
      iy - irisRadius * 0.3,
      irisRadius * 0.25,
      0,
      Math.PI * 2
    );
    ctx.fill();

    // Secondary lower-inner pinpoint highlight
    ctx.beginPath();
    ctx.arc(
      ix - outerDir * irisRadius * 0.32,
      iy + irisRadius * 0.32,
      irisRadius * 0.135,
      0,
      Math.PI * 2
    );
    ctx.fill();

    ctx.restore();
  }

  // 6. Thick Calligraphic Upper Lash Line (#0A0A0A)
  ctx.fillStyle = INK_BLACK;
  ctx.strokeStyle = INK_BLACK;
  ctx.beginPath();
  ctx.moveTo(-halfW * 1.06, halfH * 0.1);
  ctx.bezierCurveTo(
    -halfW * 0.52, -halfH * 1.52,
     halfW * 0.52, -halfH * 1.52,
     halfW * 1.1, -halfH * 0.14
  );
  // Winged outer lash flick
  ctx.lineTo(halfW * (1.1 + 0.15 * outerDir), -halfH * 0.42);
  ctx.bezierCurveTo(
     halfW * 0.5, -halfH * 1.04,
    -halfW * 0.5, -halfH * 1.04,
    -halfW * 1.12, -halfH * 0.1
  );
  ctx.closePath();
  ctx.fill();

  // Delicate outer eyelash spikes
  const spikeBaseX = outerDir * halfW * 0.76;
  const spikeBaseY = -halfH * 0.68;
  ctx.beginPath();
  ctx.moveTo(spikeBaseX, spikeBaseY);
  ctx.lineTo(spikeBaseX + outerDir * eyeWidth * 0.15, spikeBaseY - eyeWidth * 0.1);
  ctx.lineTo(spikeBaseX - outerDir * eyeWidth * 0.05, spikeBaseY - eyeWidth * 0.02);
  ctx.closePath();
  ctx.fill();

  // 7. Lower Eyelid Line & Upper Double-Eyelid Fold (Mabuta)
  ctx.lineWidth = Math.max(1.8, eyeWidth * 0.03);
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(-halfW * 0.78, halfH * 0.76);
  ctx.quadraticCurveTo(0, halfH * 1.2, halfW * 0.82, halfH * 0.65);
  ctx.stroke();

  ctx.lineWidth = Math.max(1.3, eyeWidth * 0.02);
  ctx.beginPath();
  ctx.moveTo(-halfW * 0.72, -halfH * 1.25 - eyeWidth * 0.07);
  ctx.quadraticCurveTo(0, -halfH * 1.62 - eyeWidth * 0.08, halfW * 0.82, -halfH * 1.12 - eyeWidth * 0.06);
  ctx.stroke();

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
}) {
  if (!landmarks || landmarks.length < 474) return;
  const effectiveProgress = isStage2Locked ? 1.0 : dwellProgress;
  if (effectiveProgress <= 0.15) return;

  const project = (idx) =>
    mapLandmarkToCanvas(landmarks[idx], canvasWidth, canvasHeight, videoWidth, videoHeight, mirror);

  const lOuter = project(LANDMARKS.LEFT_EYE_OUTER);
  const lInner = project(LANDMARKS.LEFT_EYE_INNER);
  const lTop = project(LANDMARKS.LEFT_EYE_TOP);
  const lBottom = project(LANDMARKS.LEFT_EYE_BOTTOM);
  const lIris = project(LANDMARKS.LEFT_IRIS_CENTER);

  const rOuter = project(LANDMARKS.RIGHT_EYE_OUTER);
  const rInner = project(LANDMARKS.RIGHT_EYE_INNER);
  const rTop = project(LANDMARKS.RIGHT_EYE_TOP);
  const rBottom = project(LANDMARKS.RIGHT_EYE_BOTTOM);
  const rIris = project(LANDMARKS.RIGHT_IRIS_CENTER);

  drawSingleMangaEye(ctx, {
    outerCorner: lOuter.x < lInner.x ? lOuter : lInner,
    innerCorner: lOuter.x < lInner.x ? lInner : lOuter,
    topLid: lTop,
    bottomLid: lBottom,
    irisCenter: lIris,
    blinkScore: eyeBlinkLeft,
    isLeftOnScreen: true,
    stageProgress: effectiveProgress,
    isStage2Locked,
  });

  drawSingleMangaEye(ctx, {
    outerCorner: rOuter.x < rInner.x ? rOuter : rInner,
    innerCorner: rOuter.x < rInner.x ? rInner : rOuter,
    topLid: rTop,
    bottomLid: rBottom,
    irisCenter: rIris,
    blinkScore: eyeBlinkRight,
    isLeftOnScreen: false,
    stageProgress: effectiveProgress,
    isStage2Locked,
  });
}

/**
 * STAGE 2 FRAMING:
 * Crisp tapered Japanese radial speed lines (shuchu-sen) around the outer 20% border
 */
export function renderShuchuSenSpeedLines(ctx, canvasWidth, canvasHeight, timestampMs) {
  const cx = canvasWidth * 0.5;
  const cy = canvasHeight * 0.5;
  const maxRadius = Math.hypot(cx, cy) * 1.05;

  const clearRx = canvasWidth * 0.40;
  const clearRy = canvasHeight * 0.40;

  const frameSeed = Math.floor(timestampMs / 60);
  const numSpokes = 148;

  ctx.save();
  ctx.fillStyle = INK_BLACK;

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
    const halfSpread = 0.0025 + rand1 * 0.0095;

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

  // Editorial Manga Panel Double-Border
  ctx.strokeStyle = INK_BLACK;
  ctx.lineWidth = 12;
  ctx.strokeRect(6, 6, canvasWidth - 12, canvasHeight - 12);

  ctx.strokeStyle = PAPER_WHITE;
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
    ctx.fillStyle = isIris ? HANKO_RED : PAPER_WHITE;
    ctx.strokeStyle = INK_BLACK;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, isIris ? 5.5 : 3.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }

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
