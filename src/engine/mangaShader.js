/**
 * MANGAFACE EFFECT — WebGL Full-Color Cel-Shaded Anime & Manga Shader Pipeline
 *
 * Implements:
 * 1. Exact Cover-Fit UV projection matching `mapLandmarkToCanvas` 1:1 so MediaPipe
 *    478 eye/iris landmarks lock onto the user's real eyes with zero offset.
 * 2. 3 Colorful Anime Rendering Styles:
 *    - Style 0: GOLDEN HOUR NOSTALGIA (Reference Image Match: warm sunset peach/coral key light,
 *      cyan-teal street shadow grading, warm rose halftone dots, and espresso-brown ink lines)
 *    - Style 1: SHINKAI VIVID BLOOM (Cerulean sky highlights, sakura-pink rim light,
 *      lavender-indigo cel shadows, and deep indigo ink outlines)
 *    - Style 2: NEO-TOKYO POP CEL (Vibrant multi-band anime cel shading, warm marigold highlights,
 *      magenta/mint halftone accents, and crisp sumi ink lines)
 * 3. 5-Tap Bilateral / Edge-Preserving Spatial Smoothing + Anti-Aliased Sobel Calligraphic Outlines
 *    so real webcam feeds look like a hand-painted anime film frame rather than noisy thresholded video.
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
  uniform float u_mangaAmount; // 0.0 = live camera, 1.0 = full colorful anime/manga mode
  uniform float u_mirror;      // 1.0 = mirror horizontally
  uniform float u_edgeThreshold;
  uniform float u_dotFrequency;
  uniform float u_colorStyle;  // 0.0 = Golden Hour, 1.0 = Shinkai Bloom, 2.0 = Neo-Tokyo Cel

  float rawLuma(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
  }

  // Exact cover-fit UV projection (matches mapLandmarkToCanvas 1:1)
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

  // 5-tap denoised RGB + Luminance so skin & hair render as smooth painted anime cel regions
  vec4 getSmoothSample(vec2 uv, vec2 texel) {
    vec3 c = texture2D(u_Texture, uv).rgb;
    vec3 n = texture2D(u_Texture, uv + vec2(0.0, -texel.y * 0.9)).rgb;
    vec3 s = texture2D(u_Texture, uv + vec2(0.0,  texel.y * 0.9)).rgb;
    vec3 e = texture2D(u_Texture, uv + vec2( texel.x * 0.9, 0.0)).rgb;
    vec3 w = texture2D(u_Texture, uv + vec2(-texel.x * 0.9, 0.0)).rgb;
    vec3 avgRgb = c * 0.44 + (n + s + e + w) * 0.14;
    float lum = smoothstep(0.05, 0.95, rawLuma(avgRgb));
    return vec4(avgRgb, lum);
  }

  // Boost saturation and quantize into soft anime cel-shading bands while preserving natural hues
  vec3 computeColorfulAnimeCel(vec3 smoothRgb, float lum, vec2 screenUV, float styleIdx) {
    // 1. Gently boost chromaticity of the user's real colors
    vec3 gray = vec3(rawLuma(smoothRgb));
    vec3 vibrant = mix(gray, smoothRgb, 1.38);

    // 2. Soft 4-band anime cel quantization on luminance (smooth transitions, no harsh posterization)
    float band1 = smoothstep(0.18, 0.25, lum);
    float band2 = smoothstep(0.42, 0.50, lum);
    float band3 = smoothstep(0.70, 0.78, lum);
    float celLum = 0.16 + band1 * 0.24 + band2 * 0.28 + band3 * 0.28;

    vec3 baseCel = vibrant * (celLum / max(lum, 0.08));
    baseCel = clamp(baseCel, 0.0, 1.0);

    // Directional anime lighting gradient across screen (warm sunlight left, cool shadow right)
    float sunSide = 1.0 - smoothstep(0.15, 0.85, screenUV.x);
    float shadowSide = 1.0 - sunSide;

    if (styleIdx < 0.5) {
      // STYLE 0: GOLDEN HOUR NOSTALGIA (Reference Image Match)
      vec3 sunsetHighlight = vec3(1.00, 0.88, 0.79); // #FFE0CA Warm Peach Sunlight
      vec3 coralMidtone    = vec3(0.96, 0.69, 0.58); // #F5B094 Sunset Coral
      vec3 tealShadow      = vec3(0.14, 0.42, 0.49); // #246A7C Cyan-Teal Street Shadow
      vec3 deepEspresso    = vec3(0.15, 0.10, 0.12); // #261A1E Warm Dark Shadow

      vec3 paletteGrade = mix(tealShadow, coralMidtone, smoothstep(0.22, 0.62, celLum));
      paletteGrade = mix(paletteGrade, sunsetHighlight, smoothstep(0.60, 0.90, celLum));

      // Blend user's real cel-shaded colors (62%) with the Golden Hour illustration grade (38%)
      vec3 result = mix(baseCel, paletteGrade, 0.38);
      // Add warm sunlight wash on left & cool teal rim shadow on right
      result = mix(result, result * vec3(1.06, 0.96, 0.88), sunSide * 0.35);
      result = mix(result, mix(result, tealShadow, 0.32), shadowSide * (1.0 - band2) * 0.65);
      result = mix(deepEspresso, result, smoothstep(0.08, 0.22, lum));
      return result;
    } else if (styleIdx < 1.5) {
      // STYLE 1: SHINKAI VIVID SKY & BLOOM
      vec3 sakuraHighlight = vec3(1.00, 0.92, 0.96); // #FFEBF5 Sakura Glow
      vec3 ceruleanMid     = vec3(0.45, 0.78, 0.98); // #73C7FA Sky Blue
      vec3 twilightShadow  = vec3(0.24, 0.22, 0.52); // #3D3885 Twilight Indigo
      vec3 deepIndigo      = vec3(0.09, 0.08, 0.23); // #17153B Deep Indigo

      vec3 paletteGrade = mix(twilightShadow, ceruleanMid, smoothstep(0.22, 0.60, celLum));
      paletteGrade = mix(paletteGrade, sakuraHighlight, smoothstep(0.58, 0.88, celLum));

      vec3 result = mix(baseCel, paletteGrade, 0.36);
      result = mix(result, result * vec3(1.05, 0.94, 1.04), sunSide * 0.35);
      result = mix(deepIndigo, result, smoothstep(0.08, 0.22, lum));
      return result;
    } else {
      // STYLE 2: NEO-TOKYO POP CEL
      vec3 marigoldLight = vec3(1.00, 0.95, 0.76); // #FFF2C2 Warm Gold Highlight
      vec3 electricRose  = vec3(0.98, 0.44, 0.52); // #FB7185 Pop Rose
      vec3 cyberTeal     = vec3(0.08, 0.58, 0.53); // #149487 Cyber Mint Shadow
      vec3 sumiShadow    = vec3(0.07, 0.07, 0.09);

      vec3 paletteGrade = mix(cyberTeal, electricRose, smoothstep(0.22, 0.58, celLum));
      paletteGrade = mix(paletteGrade, marigoldLight, smoothstep(0.58, 0.88, celLum));

      vec3 result = mix(baseCel, paletteGrade, 0.38);
      result = mix(sumiShadow, result, smoothstep(0.08, 0.22, lum));
      return result;
    }
  }

  void main() {
    vec2 sampleUV = getCoverUV(v_texCoord);
    vec2 texel = 1.35 / max(u_videoSize, vec2(640.0, 480.0));
    if (u_mirror > 0.5) {
      texel.x = -texel.x;
    }

    vec4 centerSample = getSmoothSample(sampleUV, texel);
    vec3 rawColor = texture2D(u_Texture, sampleUV).rgb;

    if (u_mangaAmount <= 0.001) {
      gl_FragColor = vec4(rawColor, 1.0);
      return;
    }

    // 1. Denoised 3x3 Sobel Edge Detection for Calligraphic Anime Ink Linework
    float tl = getSmoothSample(sampleUV + vec2(-texel.x, -texel.y), texel).a;
    float tm = getSmoothSample(sampleUV + vec2(0.0,      -texel.y), texel).a;
    float tr = getSmoothSample(sampleUV + vec2( texel.x, -texel.y), texel).a;

    float ml = getSmoothSample(sampleUV + vec2(-texel.x,  0.0), texel).a;
    float mm = centerSample.a;
    float mr = getSmoothSample(sampleUV + vec2( texel.x,  0.0), texel).a;

    float bl = getSmoothSample(sampleUV + vec2(-texel.x,  texel.y), texel).a;
    float bm = getSmoothSample(sampleUV + vec2(0.0,       texel.y), texel).a;
    float br = getSmoothSample(sampleUV + vec2( texel.x,  texel.y), texel).a;

    float gx = -tl - 2.0 * ml - bl + tr + 2.0 * mr + br;
    float gy = -tl - 2.0 * tm - tr + bl + 2.0 * bm + br;
    float gradMagnitude = sqrt(gx * gx + gy * gy);

    // Smooth anti-aliased ink linework
    float edgeMask = smoothstep(u_edgeThreshold - 0.05, u_edgeThreshold + 0.16, gradMagnitude);

    // 2. Colorful Cel-Shaded Base
    vec3 celColor = computeColorfulAnimeCel(centerSample.rgb, mm, v_texCoord, u_colorStyle);

    // 3. Subtle Colorful Diagonal Halftone Screentone in Midtone Shadows
    float angle = 0.78539816339;
    float s = sin(angle);
    float c = cos(angle);
    mat2 rot45 = mat2(c, -s, s, c);

    vec2 rotatedFrag = rot45 * gl_FragCoord.xy;
    vec2 cellUV = fract(rotatedFrag / u_dotFrequency) - 0.5;
    float distFromCellCenter = length(cellUV);

    // Only place delicate screentone dots in shadow/midtone transitions (keeps facial skin super clean!)
    float shadowZone = clamp((0.62 - mm) / 0.45, 0.0, 1.0);
    float dotRadius = pow(shadowZone, 1.35) * 0.44;
    float halftoneMask = (1.0 - smoothstep(dotRadius - 0.08, dotRadius + 0.08, distFromCellCenter)) * 0.42;

    vec3 halftoneTint = (u_colorStyle < 0.5)
      ? vec3(0.48, 0.22, 0.24)  // Warm terracotta-rose halftone
      : (u_colorStyle < 1.5)
      ? vec3(0.18, 0.20, 0.48)  // Indigo-violet halftone
      : vec3(0.75, 0.15, 0.35); // Pop magenta halftone

    vec3 shadedAnime = mix(celColor, halftoneTint, halftoneMask * shadowZone);

    // 4. Composite Calligraphic Ink Linework (#231618 Espresso Ink / #17153B Indigo Ink)
    vec3 inkLineColor = (u_colorStyle < 0.5)
      ? vec3(0.14, 0.09, 0.10)  // #231618 Deep Espresso Ink (Reference Image style)
      : (u_colorStyle < 1.5)
      ? vec3(0.09, 0.08, 0.22)  // #17153B Deep Indigo Ink
      : vec3(0.05, 0.05, 0.07); // #0D0D12 Crisp Sumi Ink

    shadedAnime = mix(shadedAnime, inkLineColor, edgeMask * 0.92);

    vec3 finalColor = mix(rawColor, shadedAnime, clamp(u_mangaAmount, 0.0, 1.0));
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
      u_colorStyle: gl.getUniformLocation(program, 'u_colorStyle'),
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
    edgeThreshold = 0.26,
    dotFrequency = 6.0,
    colorStyle = 0,
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
    gl.uniform1f(this.uniforms.u_colorStyle, colorStyle);

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
