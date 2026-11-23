/**
 * MANGAFACE EFFECT — WebGL 1-Bit Manga Shader Pipeline
 *
 * Implements:
 * - Mirrored live camera / video texture projection (cover fit)
 * - Strict 1-Bit Palette: Ink Black (#0A0A0A) and Off-White Paper (#F4F1EA)
 * - 3x3 Sobel Edge Detection pass for crisp ink outlines (jawline, hair, eyebrows, lips)
 * - 45-degree rotated diagonal halftone screentone dot matrix for midtone luminance
 * - Smooth/instant Stage 2 snap transition driven by `u_mangaAmount`
 */

const VERTEX_SHADER_SOURCE = `
  attribute vec2 a_position;
  attribute vec2 a_texCoord;
  varying vec2 v_texCoord;

  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = a_texCoord;
  }
`;

const FRAGMENT_SHADER_SOURCE = `
  precision highp float;

  varying vec2 v_texCoord;
  uniform sampler2D u_Texture;
  uniform vec2 u_resolution;
  uniform vec2 u_videoSize;
  uniform float u_mangaAmount; // 0.0 = raw mirrored feed, 1.0 = full 1-bit manga mode
  uniform float u_mirror;      // 1.0 = mirror horizontally
  uniform float u_edgeThreshold;
  uniform float u_dotFrequency;

  // Strict 1-bit Manga Palette
  const vec3 INK_BLACK = vec3(10.0 / 255.0, 10.0 / 255.0, 10.0 / 255.0);       // #0A0A0A
  const vec3 PAPER_WHITE = vec3(244.0 / 255.0, 241.0 / 255.0, 234.0 / 255.0);  // #F4F1EA

  float getLuminance(vec3 color) {
    // Rec. 709 luma with slight contrast S-curve for studio lighting resilience
    float rawLuma = dot(color, vec3(0.2126, 0.7152, 0.0722));
    return smoothstep(0.08, 0.92, rawLuma);
  }

  // Compute cover-fit UV coordinates so video fills canvas without distortion
  vec2 getCoverUV(vec2 uv) {
    float canvasAspect = u_resolution.x / max(u_resolution.y, 1.0);
    float videoAspect = u_videoSize.x / max(u_videoSize.y, 1.0);

    vec2 scale = vec2(1.0);
    if (canvasAspect > videoAspect) {
      scale.y = videoAspect / canvasAspect;
    } else {
      scale.x = canvasAspect / videoAspect;
    }

    vec2 centered = (uv - 0.5) * scale + 0.5;
    if (u_mirror > 0.5) {
      centered.x = 1.0 - centered.x;
    }
    return clamp(centered, 0.001, 0.999);
  }

  void main() {
    vec2 sampleUV = getCoverUV(v_texCoord);
    vec4 rawTex = texture2D(u_Texture, sampleUV);
    vec3 rawColor = rawTex.rgb;

    // Early exit when in pure camera mode
    if (u_mangaAmount <= 0.001) {
      gl_FragColor = vec4(rawColor, 1.0);
      return;
    }

    // 1. 3x3 Sobel Edge Detection Pass
    vec2 texel = 1.25 / max(u_videoSize, vec2(640.0, 480.0));
    if (u_mirror > 0.5) {
      texel.x = -texel.x;
    }

    float tl = getLuminance(texture2D(u_Texture, sampleUV + vec2(-texel.x, -texel.y)).rgb);
    float tm = getLuminance(texture2D(u_Texture, sampleUV + vec2(0.0,      -texel.y)).rgb);
    float tr = getLuminance(texture2D(u_Texture, sampleUV + vec2( texel.x, -texel.y)).rgb);

    float ml = getLuminance(texture2D(u_Texture, sampleUV + vec2(-texel.x,  0.0)).rgb);
    float mm = getLuminance(rawColor);
    float mr = getLuminance(texture2D(u_Texture, sampleUV + vec2( texel.x,  0.0)).rgb);

    float bl = getLuminance(texture2D(u_Texture, sampleUV + vec2(-texel.x,  texel.y)).rgb);
    float bm = getLuminance(texture2D(u_Texture, sampleUV + vec2(0.0,       texel.y)).rgb);
    float br = getLuminance(texture2D(u_Texture, sampleUV + vec2( texel.x,  texel.y)).rgb);

    float gx = -tl - 2.0 * ml - bl + tr + 2.0 * mr + br;
    float gy = -tl - 2.0 * tm - tr + bl + 2.0 * bm + br;
    float gradMagnitude = sqrt(gx * gx + gy * gy);

    bool isSobelEdge = gradMagnitude > u_edgeThreshold;

    // 2. Diagonal Halftone Screentone Dot Pattern (45 degrees rotated)
    float angle = 0.78539816339; // 45 deg in radians
    float s = sin(angle);
    float c = cos(angle);
    mat2 rot45 = mat2(c, -s, s, c);

    // Screen-space pixel coordinates scaled by dot cell size
    vec2 rotatedFrag = rot45 * gl_FragCoord.xy;
    vec2 cellUV = fract(rotatedFrag / u_dotFrequency) - 0.5;
    float distFromCellCenter = length(cellUV); // 0.0 at center to ~0.707 at corner

    // Strict 1-Bit Quantization:
    // - Deep shadow (< 0.22): Solid Ink Black
    // - Bright highlight (> 0.80): Solid Off-White Paper
    // - Midtones (0.22..0.80): 1-Bit Diagonal Halftone Screentone Dots
    bool isDeepShadow = mm < 0.22;
    bool isBrightHighlight = mm > 0.80;

    // Normalize midtone darkness: 1.0 near shadow, 0.0 near highlight
    float midtoneDarkness = clamp((0.80 - mm) / (0.80 - 0.22), 0.0, 1.0);
    // Radius threshold for binary dot
    float dotRadius = midtoneDarkness * 0.56;
    bool isHalftoneDot = (!isBrightHighlight) && (distFromCellCenter < dotRadius);

    // Final strict 1-bit decision: NO muddy grays!
    bool isInkPixel = isSobelEdge || isDeepShadow || isHalftoneDot;
    vec3 mangaColor = isInkPixel ? INK_BLACK : PAPER_WHITE;

    vec3 finalColor = mix(rawColor, mangaColor, clamp(u_mangaAmount, 0.0, 1.0));
    gl_FragColor = vec4(finalColor, 1.0);
  }
`;

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`WebGL shader compile error: ${info}`);
  }
  return shader;
}

export class MangaWebGLRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl =
      canvas.getContext('webgl', {
        alpha: false,
        antialias: false,
        depth: false,
        stencil: false,
        preserveDrawingBuffer: true,
      }) || canvas.getContext('experimental-webgl');

    if (!this.gl) {
      this.fallback2D = canvas.getContext('2d', { willReadFrequently: true });
      return;
    }

    this.initPipeline();
  }

  initPipeline() {
    const gl = this.gl;
    const vs = compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER_SOURCE);
    const fs = compileShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER_SOURCE);

    const program = gl.createProgram();
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(`WebGL program link error: ${gl.getProgramInfoLog(program)}`);
    }

    this.program = program;
    gl.useProgram(program);

    // Fullscreen quad: clip space positions (-1..1) and flipped Y texture coords (0..1)
    const vertices = new Float32Array([
      // x,   y,   u,   v
      -1.0, -1.0, 0.0, 1.0,
       1.0, -1.0, 1.0, 1.0,
      -1.0,  1.0, 0.0, 0.0,
      -1.0,  1.0, 0.0, 0.0,
       1.0, -1.0, 1.0, 1.0,
       1.0,  1.0, 1.0, 0.0,
    ]);

    this.vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

    const aPos = gl.getAttribLocation(program, 'a_position');
    const aTex = gl.getAttribLocation(program, 'a_texCoord');

    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 16, 0);

    gl.enableVertexAttribArray(aTex);
    gl.vertexAttribPointer(aTex, 2, gl.FLOAT, false, 16, 8);

    // Texture setup
    this.texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    // Uniform locations
    this.uniforms = {
      u_Texture: gl.getUniformLocation(program, 'u_Texture'),
      u_resolution: gl.getUniformLocation(program, 'u_resolution'),
      u_videoSize: gl.getUniformLocation(program, 'u_videoSize'),
      u_mangaAmount: gl.getUniformLocation(program, 'u_mangaAmount'),
      u_mirror: gl.getUniformLocation(program, 'u_mirror'),
      u_edgeThreshold: gl.getUniformLocation(program, 'u_edgeThreshold'),
      u_dotFrequency: gl.getUniformLocation(program, 'u_dotFrequency'),
    };
  }

  resize(width, height) {
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    if (this.gl) {
      this.gl.viewport(0, 0, width, height);
    }
  }

  /**
   * Renders a frame from either a live HTMLVideoElement or an HTMLCanvasElement (synthetic studio source)
   */
  renderFrame(sourceElement, {
    mangaAmount = 0.0,
    mirror = true,
    edgeThreshold = 0.24,
    dotFrequency = 6.5,
  } = {}) {
    const width = this.canvas.width;
    const height = this.canvas.height;
    const sourceW = sourceElement.videoWidth || sourceElement.width || 1280;
    const sourceH = sourceElement.videoHeight || sourceElement.height || 720;

    if (!this.gl) {
      this.renderFallback2D(sourceElement, { mangaAmount, mirror, sourceW, sourceH });
      return;
    }

    const gl = this.gl;
    gl.useProgram(this.program);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, sourceElement);

    gl.uniform1i(this.uniforms.u_Texture, 0);
    gl.uniform2f(this.uniforms.u_resolution, width, height);
    gl.uniform2f(this.uniforms.u_videoSize, sourceW, sourceH);
    gl.uniform1f(this.uniforms.u_mangaAmount, mangaAmount);
    gl.uniform1f(this.uniforms.u_mirror, mirror ? 1.0 : 0.0);
    gl.uniform1f(this.uniforms.u_edgeThreshold, edgeThreshold);
    gl.uniform1f(this.uniforms.u_dotFrequency, dotFrequency);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  renderFallback2D(sourceElement, { mangaAmount, mirror, sourceW, sourceH }) {
    const ctx = this.fallback2D;
    if (!ctx) return;
    const w = this.canvas.width;
    const h = this.canvas.height;

    ctx.save();
    if (mirror) {
      ctx.translate(w, 0);
      ctx.scale(-1, 1);
    }
    const scale = Math.max(w / sourceW, h / sourceH);
    const drawW = sourceW * scale;
    const drawH = sourceH * scale;
    const offsetX = (w - drawW) * 0.5;
    const offsetY = (h - drawH) * 0.5;
    ctx.drawImage(sourceElement, offsetX, offsetY, drawW, drawH);
    ctx.restore();

    if (mangaAmount > 0.5) {
      // CPU 1-bit Sobel + Diagonal Screentone fallback
      const imgData = ctx.getImageData(0, 0, w, h);
      const data = imgData.data;
      const luma = new Float32Array(w * h);
      for (let i = 0, p = 0; i < data.length; i += 4, p++) {
        luma[p] = (0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2]) / 255;
      }
      for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
          const idx = y * w + x;
          const gx =
            -luma[idx - w - 1] - 2 * luma[idx - 1] - luma[idx + w - 1] +
             luma[idx - w + 1] + 2 * luma[idx + 1] + luma[idx + w + 1];
          const gy =
            -luma[idx - w - 1] - 2 * luma[idx - w] - luma[idx - w + 1] +
             luma[idx + w - 1] + 2 * luma[idx + w] + luma[idx + w + 1];
          const edge = Math.sqrt(gx * gx + gy * gy) > 0.25;
          const lum = luma[idx];
          const rotX = (x * 0.7071 - y * 0.7071) % 6;
          const rotY = (x * 0.7071 + y * 0.7071) % 6;
          const d = Math.hypot(rotX - 3, rotY - 3) / 4.2;
          const isInk = edge || lum < 0.22 || (lum < 0.8 && d < (0.8 - lum) / 0.58);

          const p4 = idx * 4;
          data[p4] = isInk ? 10 : 244;
          data[p4 + 1] = isInk ? 10 : 241;
          data[p4 + 2] = isInk ? 10 : 234;
        }
      }
      ctx.putImageData(imgData, 0, 0);
    }
  }

  destroy() {
    if (this.gl && this.program) {
      this.gl.deleteTexture(this.texture);
      this.gl.deleteBuffer(this.vertexBuffer);
      this.gl.deleteProgram(this.program);
    }
  }
}

/**
 * Helper to project normalized MediaPipe landmarks (0..1 in source video space)
 * into the cover-fitted & mirrored canvas pixel space.
 */
export function mapLandmarkToCanvas(landmark, canvasWidth, canvasHeight, videoWidth, videoHeight, mirror = true) {
  const canvasAspect = canvasWidth / Math.max(canvasHeight, 1);
  const videoAspect = videoWidth / Math.max(videoHeight, 1);

  let scaleX = 1.0;
  let scaleY = 1.0;
  if (canvasAspect > videoAspect) {
    scaleY = videoAspect / canvasAspect;
  } else {
    scaleX = canvasAspect / videoAspect;
  }

  const rawX = mirror ? 1.0 - landmark.x : landmark.x;
  const rawY = landmark.y;

  const normX = (rawX - 0.5) / scaleX + 0.5;
  const normY = (rawY - 0.5) / scaleY + 0.5;

  return {
    x: normX * canvasWidth,
    y: normY * canvasHeight,
    z: landmark.z || 0,
  };
}
