/**
 * MANGAFACE EFFECT — Interactive Synthetic Studio Subject
 *
 * When a user is in a headless VM / device without a physical webcam (or clicks
 * "Try Interactive Studio Demo"), this engine renders a high-contrast 60 FPS
 * virtual studio subject onto an offscreen 1280x720 canvas AND produces real-time
 * 478 MediaPipe-compatible normalized landmarks + `eyeBlinkLeft`/`eyeBlinkRight`
 * blendshapes so the exact same Gaze Lock state machine, Stage 1 Vector Manga Eyes,
 * and Stage 2 WebGL 1-Bit Sobel + Halftone Shader run at 60 FPS.
 */

import { LANDMARKS } from './mangaOverlay';

export class SyntheticStudioSource {
  constructor(width = 1280, height = 720) {
    this.width = width;
    this.height = height;
    this.canvas = document.createElement('canvas');
    this.canvas.width = width;
    this.canvas.height = height;
    this.ctx = this.canvas.getContext('2d');

    // Interactive gaze target (-1..1, where 0,0 is dead-center looking into the lens)
    this.targetGazeX = 0;
    this.targetGazeY = 0;
    this.currentGazeX = 0;
    this.currentGazeY = 0;

    // Auto-lock toggle: when true, subject stares directly into the lens unless user drags/moves away
    this.forceLookAway = false;
    this.manualBlinkUntil = 0;

    // Load reference portrait image for rich tonal gradients + Sobel edges
    this.referenceImg = new Image();
    this.referenceImg.src = './assets/hero-reference.png';
    this.referenceLoaded = false;
    this.referenceImg.onload = () => {
      this.referenceLoaded = true;
    };
  }

  setGazeTarget(normX, normY) {
    this.targetGazeX = Math.max(-1, Math.min(1, normX));
    this.targetGazeY = Math.max(-1, Math.min(1, normY));
  }

  setLookAway(lookAway) {
    this.forceLookAway = lookAway;
    if (lookAway) {
      this.targetGazeX = 0.68;
      this.targetGazeY = -0.35;
    } else {
      this.targetGazeX = 0;
      this.targetGazeY = 0;
    }
  }

  triggerBlink() {
    this.manualBlinkUntil = performance.now() + 240;
  }

  /**
   * Updates the offscreen studio canvas and returns a MediaPipe-compatible
   * `{ faceLandmarks: [landmarks478], faceBlendshapes: [{ categories }] }` object.
   */
  update(timestampMs) {
    // Smooth interpolation of head & iris pose
    const goalX = this.forceLookAway ? 0.68 : this.targetGazeX;
    const goalY = this.forceLookAway ? -0.32 : this.targetGazeY;
    this.currentGazeX += (goalX - this.currentGazeX) * 0.14;
    this.currentGazeY += (goalY - this.currentGazeY) * 0.14;

    // Natural periodic blink every ~3.4 seconds + manual blink support
    const blinkCycle = (timestampMs % 3400) / 3400;
    let blinkAmount = 0;
    if (blinkCycle > 0.93) {
      const t = (blinkCycle - 0.93) / 0.07;
      blinkAmount = Math.sin(t * Math.PI);
    }
    if (timestampMs < this.manualBlinkUntil) {
      blinkAmount = 0.96;
    }

    this.drawStudioFrame(timestampMs, blinkAmount);
    return this.buildSyntheticLandmarks(blinkAmount);
  }

  drawStudioFrame(timestampMs, blinkAmount) {
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;
    const cx = w * 0.5 + this.currentGazeX * 70;
    const cy = h * 0.52 + this.currentGazeY * 42;

    // 1. Studio Backdrop with subtle architectural gradients (great for Sobel + Halftone screentone)
    const bgGrad = ctx.createRadialGradient(w * 0.5, h * 0.45, 90, w * 0.5, h * 0.5, w * 0.72);
    bgGrad.addColorStop(0, '#dbd5c9');
    bgGrad.addColorStop(0.55, '#8e887e');
    bgGrad.addColorStop(1, '#262422');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, w, h);

    // Architectural studio lines behind subject
    ctx.strokeStyle = 'rgba(28, 26, 24, 0.35)';
    ctx.lineWidth = 4;
    for (let i = 1; i < 6; i++) {
      const lx = (w / 6) * i;
      ctx.beginPath();
      ctx.moveTo(lx, 0);
      ctx.lineTo(lx, h);
      ctx.stroke();
    }

    // 2. Stylized 3D-shaded Studio Subject Portrait (Head, Hair, Jawline, Collar, Eyes, Lips)
    ctx.save();

    // Shoulders & collared shirt
    const shoulderGrad = ctx.createLinearGradient(cx - 240, cy + 160, cx + 240, h);
    shoulderGrad.addColorStop(0, '#c8c0b0');
    shoulderGrad.addColorStop(0.5, '#7c7569');
    shoulderGrad.addColorStop(1, '#33302c');
    ctx.fillStyle = shoulderGrad;
    ctx.strokeStyle = '#141312';
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.moveTo(cx - 270, h);
    ctx.quadraticCurveTo(cx - 210, cy + 185, cx - 65, cy + 170);
    ctx.lineTo(cx + 65, cy + 170);
    ctx.quadraticCurveTo(cx + 210, cy + 185, cx + 270, h);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Neck with cast chin shadow
    ctx.fillStyle = '#9c9284';
    ctx.fillRect(cx - 46, cy + 115, 92, 75);
    ctx.fillStyle = '#38342f';
    ctx.beginPath();
    ctx.ellipse(cx, cy + 132, 48, 22, 0, 0, Math.PI);
    ctx.fill();

    // Back hair silhouette
    ctx.fillStyle = '#1b1a18';
    ctx.beginPath();
    ctx.ellipse(cx, cy - 18, 185, 205, 0, 0, Math.PI * 2);
    ctx.fill();

    // Face / Jawline contour with directional studio key light (creates rich halftone midtones!)
    const faceGrad = ctx.createLinearGradient(cx - 135, cy - 120, cx + 145, cy + 130);
    faceGrad.addColorStop(0, '#f0eae1');
    faceGrad.addColorStop(0.48, '#d5cbbd');
    faceGrad.addColorStop(0.82, '#8b8275');
    faceGrad.addColorStop(1, '#4b463f');

    ctx.fillStyle = faceGrad;
    ctx.strokeStyle = '#161514';
    ctx.lineWidth = 4.5;
    ctx.beginPath();
    ctx.moveTo(cx - 128, cy - 55);
    ctx.quadraticCurveTo(cx - 132, cy + 68, cx - 72, cy + 122);
    ctx.quadraticCurveTo(cx, cy + 158, cx + 72, cy + 122);
    ctx.quadraticCurveTo(cx + 132, cy + 68, cx + 128, cy - 55);
    ctx.quadraticCurveTo(cx + 120, cy - 165, cx, cy - 168);
    ctx.quadraticCurveTo(cx - 120, cy - 165, cx - 128, cy - 55);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Cheekbone & nose bridge tonal shading
    const noseX = cx + this.currentGazeX * 26;
    const noseY = cy + 22 + this.currentGazeY * 16;
    ctx.fillStyle = 'rgba(45, 40, 35, 0.35)';
    ctx.beginPath();
    ctx.moveTo(noseX, cy - 34);
    ctx.lineTo(noseX - 18, noseY + 6);
    ctx.lineTo(noseX + 4, noseY + 14);
    ctx.closePath();
    ctx.fill();

    // Eyebrows
    const leftEyeX = cx - 56 + this.currentGazeX * 18;
    const rightEyeX = cx + 56 + this.currentGazeX * 18;
    const eyeY = cy - 22 + this.currentGazeY * 14;

    ctx.strokeStyle = '#141312';
    ctx.lineWidth = 7;
    ctx.lineCap = 'round';
    // Left brow
    ctx.beginPath();
    ctx.moveTo(leftEyeX - 38, eyeY - 26);
    ctx.quadraticCurveTo(leftEyeX, eyeY - 38, leftEyeX + 34, eyeY - 24);
    ctx.stroke();
    // Right brow
    ctx.beginPath();
    ctx.moveTo(rightEyeX - 34, eyeY - 24);
    ctx.quadraticCurveTo(rightEyeX, eyeY - 38, rightEyeX + 38, eyeY - 26);
    ctx.stroke();

    // Natural studio eyes (visible in IDLE before Stage 1 vector manga eyes overlay)
    const openH = Math.max(2, 18 * (1 - blinkAmount));
    const irisShiftX = this.currentGazeX * 14;
    const irisShiftY = this.currentGazeY * 7;

    [leftEyeX, rightEyeX].forEach((ex) => {
      ctx.save();
      ctx.fillStyle = '#f5f2eb';
      ctx.strokeStyle = '#161514';
      ctx.lineWidth = 3.5;
      ctx.beginPath();
      ctx.ellipse(ex, eyeY, 32, openH, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      if (openH > 4) {
        ctx.beginPath();
        ctx.ellipse(ex, eyeY, 32, openH, 0, 0, Math.PI * 2);
        ctx.clip();

        ctx.fillStyle = '#2c2926';
        ctx.beginPath();
        ctx.arc(ex + irisShiftX, eyeY + irisShiftY, 13.5, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = '#0a0a0a';
        ctx.beginPath();
        ctx.arc(ex + irisShiftX, eyeY + irisShiftY, 6.5, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(ex + irisShiftX - 4, eyeY + irisShiftY - 4, 3, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    });

    // Lips
    const mouthY = cy + 78 + this.currentGazeY * 12;
    ctx.strokeStyle = '#181615';
    ctx.lineWidth = 3.8;
    ctx.beginPath();
    ctx.moveTo(noseX - 34, mouthY);
    ctx.quadraticCurveTo(noseX, mouthY + 6, noseX + 34, mouthY);
    ctx.stroke();

    ctx.fillStyle = 'rgba(60, 52, 48, 0.45)';
    ctx.beginPath();
    ctx.quadraticCurveTo(noseX, mouthY + 18, noseX + 24, mouthY + 2);
    ctx.quadraticCurveTo(noseX, mouthY + 15, noseX - 24, mouthY + 2);
    ctx.fill();

    // Layered Hair Bangs framing the forehead and temples (creates crisp Sobel ink lines!)
    ctx.fillStyle = '#181715';
    ctx.strokeStyle = '#0a0a0a';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(cx - 145, cy - 25);
    ctx.quadraticCurveTo(cx - 135, cy - 175, cx, cy - 185);
    ctx.quadraticCurveTo(cx + 135, cy - 175, cx + 145, cy - 25);
    // Jagged manga fringe strands across forehead
    ctx.lineTo(cx + 105, cy - 68);
    ctx.lineTo(cx + 75, cy - 38);
    ctx.lineTo(cx + 42, cy - 75);
    ctx.lineTo(cx + 8, cy - 34);
    ctx.lineTo(cx - 28, cy - 78);
    ctx.lineTo(cx - 65, cy - 38);
    ctx.lineTo(cx - 105, cy - 72);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    ctx.restore();
  }

  buildSyntheticLandmarks(blinkAmount) {
    const w = this.width;
    const h = this.height;
    const cx = w * 0.5 + this.currentGazeX * 70;
    const cy = h * 0.52 + this.currentGazeY * 42;

    const leftEyeX = cx - 56 + this.currentGazeX * 18;
    const rightEyeX = cx + 56 + this.currentGazeX * 18;
    const eyeY = cy - 22 + this.currentGazeY * 14;
    const openH = Math.max(2, 18 * (1 - blinkAmount));

    const irisShiftX = this.currentGazeX * 14;
    const irisShiftY = this.currentGazeY * 7;

    const noseX = cx + this.currentGazeX * 26;
    const noseY = cy + 22 + this.currentGazeY * 16;

    // Initialize 478 normalized landmarks
    const landmarks = new Array(478);
    for (let i = 0; i < 478; i++) {
      landmarks[i] = { x: cx / w, y: cy / h, z: 0 };
    }

    const setPt = (idx, px, py) => {
      landmarks[idx] = { x: px / w, y: py / h, z: 0 };
    };

    // Head pose anchors
    setPt(LANDMARKS.FOREHEAD, cx, cy - 135);
    setPt(LANDMARKS.NOSE_TIP, noseX, noseY);
    setPt(LANDMARKS.CHIN, cx, cy + 138);

    // Subject's Right Eye (left side of unmirrored source canvas)
    setPt(LANDMARKS.RIGHT_EYE_OUTER, leftEyeX - 32, eyeY);
    setPt(LANDMARKS.RIGHT_EYE_INNER, leftEyeX + 32, eyeY);
    setPt(LANDMARKS.RIGHT_EYE_TOP, leftEyeX, eyeY - openH);
    setPt(LANDMARKS.RIGHT_EYE_BOTTOM, leftEyeX, eyeY + openH);
    setPt(LANDMARKS.RIGHT_IRIS_CENTER, leftEyeX + irisShiftX, eyeY + irisShiftY);

    // Subject's Left Eye (right side of unmirrored source canvas)
    setPt(LANDMARKS.LEFT_EYE_INNER, rightEyeX - 32, eyeY);
    setPt(LANDMARKS.LEFT_EYE_OUTER, rightEyeX + 32, eyeY);
    setPt(LANDMARKS.LEFT_EYE_TOP, rightEyeX, eyeY - openH);
    setPt(LANDMARKS.LEFT_EYE_BOTTOM, rightEyeX, eyeY + openH);
    setPt(LANDMARKS.LEFT_IRIS_CENTER, rightEyeX + irisShiftX, eyeY + irisShiftY);

    return {
      faceLandmarks: [landmarks],
      faceBlendshapes: [
        {
          categories: [
            { categoryName: 'eyeBlinkLeft', score: blinkAmount },
            { categoryName: 'eyeBlinkRight', score: blinkAmount },
          ],
        },
      ],
    };
  }
}
