/**
 * MANGAFACE EFFECT — WebGL High-Precision Manga Shader Pipeline
 *
 * Improvements over raw 1-bit pixel thresholding:
 * 1. Bilateral / Gaussian edge-preserving pre-smoothing so webcam sensor noise never creates jagged "rough" speckles.
 * 2. Smooth anti-aliased Sobel ink linework with organic pen-pressure tapering (using smoothstep instead of hard booleans).
 * 3. Smooth anti-aliased 45° diagonal screentone dots with soft cell edges and warm editorial tones.
 * 4. Multi-tone Seinen Editorial Palette:
 *    - Deep Sumi Ink (#0A0A0A)
 *    - Warm Halftone Charcoal / Mid-Tone Wash (#231F20)
 *    - Warm Cream Highlights (#ECE6DA)
 *    - Warm Editorial Paper (#F4F1EA)
 *    - Subtle Sunset Blush / Rim Tint inspired by the reference illustration
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
  uniform float u_mangaAmount; // 0.0 = raw mirrored feed, 1.0 = full manga mode
  uniform float u_mirror;      // 1.0 = mirror horizontally
  uniform float u_edgeThreshold;
  uniform float u_dotFrequency;

  // Editorial Manga Palette
  const vec3 INK_BLACK    = vec3(10.0 / 255.0, 10.0 / 255.0, 12.0 / 255.0);    // #0A0A0C Deep Sumi Ink
  const vec3 INK_CHARCOAL = vec3(32.0 / 255.0, 29.0 / 255.0, 30.0 / 255.0);    // #201D1E Soft Screentone Ink
  const vec3 PAPER_WARM   = vec3(236.0 / 255.0, 230.0 / 255.0, 217.0 / 255.0); // #ECE6D9 Warm Midtone Paper
  const vec3 PAPER_WHITE  = vec3(246.0 / 255.0, 243.0 / 255.0, 236.0 / 255.0); // #F6F3EC Crisp Paper Highlight
  const vec3 SUNSET_TINT  = vec3(244.0 / 255.0, 224.0 / 255.0, 212.0 / 255.0); // Subtle warm anime rim wash

  float rawLuma(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
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

  // 5-tap denoised luminance to eliminate webcam sensor grain before Sobel & Screentone
  float getSmoothLuminance(vec2 uv, vec2 texel) {
    float c  = rawLuma(texture2D(u_Texture, uv).rgb);
    float n  = rawLuma(texture2D(u_Texture, uv + vec2(0.0, -texel.y * 0.85)).rgb);
    float s  = rawLuma(texture2D(u_Texture, uv + vec2(0.0,  texel.y * 0.85)).rgb);
    float e  = rawLuma(texture2D(u_Texture, uv + vec2( texel.x * 0.85, 0.0)).rgb);
    float w  = rawLuma(texture2D(u_Texture, uv + vec2(-texel.x * 0.85, 0.0)).rgb);
    float avg = c * 0.44 + (n + s + e + w) * 0.14;
    // Gentle S-curve for flattering facial tonal separation
    return smoothstep(0.06, 0.93, avg);
  }

  void main() {
    vec2 sampleUV = getCoverUV(v_texCoord);
    vec4 rawTex = texture2D(u_Texture, sampleUV);
    vec3 rawColor = rawTex.rgb;

    // Subtle editorial color grade even in IDLE / Stage 1 camera mode
    if (u_mangaAmount <= 0.001) {
      gl_FragColor = vec4(rawColor, 1.0);
      return;
    }

    // 1. Denoised 3x3 Sobel Edge Detection Pass for clean, anti-aliased calligraphic linework
    vec2 texel = 1.35 / max(u_videoSize, vec2(640.0, 480.0));
    if (u_mirror > 0.5) {
      texel.x = -texel.x;
    }

    float tl = getSmoothLuminance(sampleUV + vec2(-texel.x, -texel.y), texel);
    float tm = getSmoothLuminance(sampleUV + vec2(0.0,      -texel.y), texel);
    float tr = getSmoothLuminance(sampleUV + vec2( texel.x, -texel.y), texel);

    float ml = getSmoothLuminance(sampleUV + vec2(-texel.x,  0.0), texel);
    float mm = getSmoothLuminance(sampleUV, texel);
    float mr = getSmoothLuminance(sampleUV + vec2( texel.x,  0.0), texel);

    float bl = getSmoothLuminance(sampleUV + vec2(-texel.x,  texel.y), texel);
    float bm = getSmoothLuminance(sampleUV + vec2(0.0,       texel.y), texel);
    float br = getSmoothLuminance(sampleUV + vec2( texel.x,  texel.y), texel);

    float gx = -tl - 2.0 * ml - bl + tr + 2.0 * mr + br;
    float gy = -tl - 2.0 * tm - tr + bl + 2.0 * bm + br;
    float gradMagnitude = sqrt(gx * gx + gy * gy);

    // Smooth anti-aliased ink line weight (no jagged stair-stepping)
    float edgeStrength = smoothstep(u_edgeThreshold - 0.06, u_edgeThreshold + 0.14, gradMagnitude);

    // 2. Anti-Aliased 45-Degree Diagonal Halftone Screentone Dot Pattern
    float angle = 0.78539816339; // 45 deg
    float s = sin(angle);
    float c = cos(angle);
    mat2 rot45 = mat2(c, -s, s, c);

    vec2 rotatedFrag = rot45 * gl_FragCoord.xy;
    vec2 cellUV = fract(rotatedFrag / u_dotFrequency) - 0.5;
    float distFromCellCenter = length(cellUV);

    // Tonal zones:
    // - Deep shadow (< 0.21): Smooth transition into solid Sumi Ink
    // - Midtones (0.21..0.82): Anti-aliased diagonal screentone dots over warm paper
    // - Highlights (> 0.82): Clean off-white paper (#F4F1EA)
    float deepShadowMask = 1.0 - smoothstep(0.16, 0.25, mm);
    float highlightMask = smoothstep(0.76, 0.86, mm);

    // Normalize midtone darkness (1.0 near shadow, 0.0 near highlight)
    float midtoneDarkness = clamp((0.83 - mm) / (0.83 - 0.20), 0.0, 1.0);
    // Curve darkness so facial skin stays clean and mostly paper-toned while shadows get rich screentone
    float curvedDarkness = pow(midtoneDarkness, 1.25);
    float dotRadius = curvedDarkness * 0.54;

    // Anti-aliased dot boundary using smoothstep (eliminates rough pixelation!)
    float aaWidth = 0.085;
    float halftoneDotMask = (1.0 - highlightMask) * (1.0 - smoothstep(dotRadius - aaWidth, dotRadius + aaWidth, distFromCellCenter));

    // Base paper tone with subtle warm editorial gradient
    vec3 paperTone = mix(PAPER_WARM, PAPER_WHITE, smoothstep(0.45, 0.85, mm));
    paperTone = mix(paperTone, SUNSET_TINT, 0.14 * (1.0 - highlightMask));

    // Layer 1: Halftone screentone dots in rich charcoal ink
    vec3 shadedManga = mix(paperTone, INK_CHARCOAL, halftoneDotMask * 0.92);

    // Layer 2: Deep solid sumi ink shadows (hair, pupils, dark folds)
    shadedManga = mix(shadedManga, INK_BLACK, deepShadowMask);

    // Layer 3: Crisp anti-aliased Sobel ink linework on top
    shadedManga = mix(shadedManga, INK_BLACK, edgeStrength);

    vec3 finalColor = mix(rawColor, shadedManga, clamp(u_mangaAmount, 0.0, 1.0));
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
        antialias: true,
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

    const vertices = new Float32Array([
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

    this.texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

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

  renderFrame(sourceElement, {
    mangaAmount = 0.0,
    mirror = true,
    edgeThreshold = 0.28,
    dotFrequency = 6.0,
  } = {}) {
    const width = this.canvas.width;
    const height = this.canvas.height;
    const sourceW = sourceElement.videoWidth || sourceElement.width || 1280;
    const sourceH = sourceElement.videoHeight || sourceElement.height || 720;

    if (!this.gl) {
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

  destroy() {
    if (this.gl && this.program) {
      this.gl.deleteTexture(this.texture);
      this.gl.deleteBuffer(this.vertexBuffer);
      this.gl.deleteProgram(this.program);
    }
  }
}

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
