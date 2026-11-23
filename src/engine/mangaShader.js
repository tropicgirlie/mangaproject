/**
 * MANGAFACE EFFECT — WebGL Anisotropic Kuwahara + Shader-Space Optical Eye Relighting
 *
 * Replaces flat 2D cartoon vector eye stickers with 100% cohesive GPU shader-space processing:
 * 1. GPU Shader-Space Eye Relighting & Optical Magnification Warp:
 *    - Takes MediaPipe's exact normalized iris coordinates (`u_leftIris`, `u_rightIris`)
 *      and eye radii (`u_leftEyeRadius`, `u_rightEyeRadius`) directly into the fragment shader.
 *    - Applies a subtle optical lens magnification warp (14% expansion) around each eye socket
 *      starting in Stage 1 (`u_eyeAmount > 0.0`), naturally enlarging the user's real eyes
 *      while keeping real eyelashes, skin pores, and eyelids intact.
 *    - Relights the user's real iris pixels in shader space with a jewel-toned radial color grade
 *      (Golden Hour Amber-Violet, Shinkai Cerulean-Sapphire, or Neo-Tokyo Marigold) + crisp
 *      dual specular catchlights before the ink pass runs.
 * 2. Anisotropic Kuwahara / Multi-Sector Oil-Paint Smoothing Filter:
 *    - Evaluates 4 overlapping directional quadrants around each pixel and selects the quadrant
 *      with minimum variance. This turns noisy webcam video into flat, hand-painted anime
 *      brush planes while keeping edges (jawline, hair strands, collar, lips) razor-sharp.
 * 3. Smooth Calligraphic DoG / Sobel Ink Linework + Subtle Shadow Screentone:
 *    - Runs over the Kuwahara-painted and eye-relit pixels so the eyes and face belong to
 *      the exact same painted anime medium.
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
  uniform float u_eyeAmount;   // 0.0 -> 1.0 (Stage 1+ shader-space optical eye warp & iris relighting)
  uniform float u_mangaAmount; // 0.0 -> 1.0 (Stage 2 full Kuwahara anime paint + ink outlines)
  uniform float u_mirror;      // 1.0 = mirror horizontally
  uniform float u_edgeThreshold;
  uniform float u_dotFrequency;
  uniform float u_colorStyle;  // 0.0 = Golden Hour, 1.0 = Shinkai Bloom, 2.0 = Neo-Tokyo Cel

  // Live MediaPipe Iris & Eye Socket Uniforms (in source video UV space 0..1)
  uniform vec2 u_leftIrisUV;
  uniform vec2 u_rightIrisUV;
  uniform float u_leftIrisRadius;
  uniform float u_rightIrisRadius;
  uniform float u_leftOpenness;
  uniform float u_rightOpenness;

  float rawLuma(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
  }

  // Exact cover-fit UV projection
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

  // 1. Shader-Space Optical Eye Magnification Warp (subtle 15% anime eye enlargement on real pixels)
  vec2 applyOpticalEyeWarp(vec2 uv, float amount) {
    if (amount <= 0.001) return uv;

    float aspect = u_videoSize.x / max(u_videoSize.y, 1.0);
    vec2 warpedUV = uv;

    // Warp around Left Iris
    if (u_leftIrisRadius > 0.002) {
      vec2 deltaL = (uv - u_leftIrisUV) * vec2(aspect, 1.0);
      float distL = length(deltaL);
      float warpRadiusL = u_leftIrisRadius * 2.6;
      if (distL < warpRadiusL) {
        float pct = distL / warpRadiusL;
        float pinch = mix(0.84, 1.0, smoothstep(0.0, 1.0, pct));
        float factor = mix(1.0, pinch, amount * 0.85);
        warpedUV = u_leftIrisUV + (uv - u_leftIrisUV) * factor;
      }
    }

    // Warp around Right Iris
    if (u_rightIrisRadius > 0.002) {
      vec2 deltaR = (warpedUV - u_rightIrisUV) * vec2(aspect, 1.0);
      float distR = length(deltaR);
      float warpRadiusR = u_rightIrisRadius * 2.6;
      if (distR < warpRadiusR) {
        float pct = distR / warpRadiusR;
        float pinch = mix(0.84, 1.0, smoothstep(0.0, 1.0, pct));
        float factor = mix(1.0, pinch, amount * 0.85);
        warpedUV = u_rightIrisUV + (warpedUV - u_rightIrisUV) * factor;
      }
    }

    return clamp(warpedUV, 0.001, 0.999);
  }

  // 2. 4-Quadrant Kuwahara Oil-Paint Filter (eliminates webcam noise while keeping edges sharp)
  vec3 sampleKuwaharaPaint(vec2 uv, vec2 texel) {
    vec3 mean0 = vec3(0.0), mean1 = vec3(0.0), mean2 = vec3(0.0), mean3 = vec3(0.0);
    vec3 sq0   = vec3(0.0), sq1   = vec3(0.0), sq2   = vec3(0.0), sq3   = vec3(0.0);

    // Quadrant 0: Top-Left (4 taps)
    vec3 c00 = texture2D(u_Texture, uv + vec2(-texel.x, -texel.y)).rgb;
    vec3 c01 = texture2D(u_Texture, uv + vec2(0.0,      -texel.y)).rgb;
    vec3 c02 = texture2D(u_Texture, uv + vec2(-texel.x,  0.0)).rgb;
    vec3 c03 = texture2D(u_Texture, uv).rgb;
    mean0 = (c00 + c01 + c02 + c03) * 0.25;
    sq0   = (c00*c00 + c01*c01 + c02*c02 + c03*c03) * 0.25;

    // Quadrant 1: Top-Right (4 taps)
    vec3 c10 = texture2D(u_Texture, uv + vec2( texel.x, -texel.y)).rgb;
    mean1 = (c10 + c01 + c03 + texture2D(u_Texture, uv + vec2(texel.x, 0.0)).rgb) * 0.25;
    vec3 c13 = texture2D(u_Texture, uv + vec2(texel.x, 0.0)).rgb;
    sq1   = (c10*c10 + c01*c01 + c03*c03 + c13*c13) * 0.25;

    // Quadrant 2: Bottom-Left (4 taps)
    vec3 c20 = texture2D(u_Texture, uv + vec2(-texel.x, texel.y)).rgb;
    vec3 c21 = texture2D(u_Texture, uv + vec2(0.0,      texel.y)).rgb;
    mean2 = (c20 + c21 + c02 + c03) * 0.25;
    sq2   = (c20*c20 + c21*c21 + c02*c02 + c03*c03) * 0.25;

    // Quadrant 3: Bottom-Right (4 taps)
    vec3 c30 = texture2D(u_Texture, uv + vec2(texel.x, texel.y)).rgb;
    mean3 = (c30 + c21 + c13 + c03) * 0.25;
    sq3   = (c30*c30 + c21*c21 + c13*c13 + c03*c03) * 0.25;

    float var0 = dot(abs(sq0 - mean0 * mean0), vec3(1.0));
    float var1 = dot(abs(sq1 - mean1 * mean1), vec3(1.0));
    float var2 = dot(abs(sq2 - mean2 * mean2), vec3(1.0));
    float var3 = dot(abs(sq3 - mean3 * mean3), vec3(1.0));

    float minVar = var0;
    vec3 bestMean = mean0;
    if (var1 < minVar) { minVar = var1; bestMean = mean1; }
    if (var2 < minVar) { minVar = var2; bestMean = mean2; }
    if (var3 < minVar) { bestMean = mean3; }

    return bestMean;
  }

  // 3. Shader-Space Iris Relighting & Specular Catchlight (operates on the user's real iris pixels!)
  vec3 applyShaderEyeRelighting(vec3 baseColor, vec2 uv, vec2 irisUV, float irisRadius, float openness, float amount, float styleIdx) {
    if (amount <= 0.001 || irisRadius <= 0.002 || openness <= 0.18) return baseColor;

    float aspect = u_videoSize.x / max(u_videoSize.y, 1.0);
    vec2 rel = (uv - irisUV) * vec2(aspect, 1.0);

    // Elliptical vertical squash when blinking so iris relighting stays inside real eyelids
    vec2 lidRel = vec2(rel.x / (irisRadius * 1.45), rel.y / (irisRadius * max(0.18, 0.92 * openness)));
    float lidMask = 1.0 - smoothstep(0.78, 1.05, length(lidRel));
    if (lidMask <= 0.001) return baseColor;

    float rNorm = length(rel) / (irisRadius * 1.12);
    if (rNorm > 1.25) return baseColor;

    // Palette jewel tint for iris
    vec3 jewelTint = (styleIdx < 0.5)
      ? vec3(0.88, 0.62, 0.58)  // Golden Hour Sunset Amber-Rose
      : (styleIdx < 1.5)
      ? vec3(0.32, 0.72, 0.98)  // Shinkai Cerulean Sapphire
      : vec3(0.98, 0.72, 0.24); // Neo-Tokyo Marigold Gold

    vec3 color = baseColor;

    // Inside iris disc (rNorm < 1.0)
    float irisMask = (1.0 - smoothstep(0.88, 1.02, rNorm)) * lidMask;
    if (irisMask > 0.001) {
      // Darken outer limbal ring & pupil core for crisp anime depth
      float limbalRing = smoothstep(0.68, 0.94, rNorm);
      float pupilCore  = 1.0 - smoothstep(0.18, 0.36, rNorm);
      // Lower iris crescent glow
      float lowerGlow  = smoothstep(-0.1, 0.75, rel.y / irisRadius) * (1.0 - pupilCore) * (1.0 - limbalRing);

      vec3 relitIris = color * 1.18 + jewelTint * lowerGlow * 0.52;
      relitIris = mix(relitIris, relitIris * 0.25, limbalRing * 0.75);
      relitIris = mix(relitIris, vec3(0.04, 0.03, 0.05), pupilCore * 0.85);

      // Procedural Crisp Dual Specular Catchlights in Shader Space
      vec2 hl1Pos = vec2(-0.34, -0.30) * irisRadius;
      vec2 hl2Pos = vec2( 0.30,  0.28) * irisRadius;
      float hl1 = 1.0 - smoothstep(irisRadius * 0.16, irisRadius * 0.24, length(rel - hl1Pos));
      float hl2 = 1.0 - smoothstep(irisRadius * 0.08, irisRadius * 0.14, length(rel - hl2Pos));
      float catchlight = clamp(hl1 + hl2, 0.0, 1.0);

      relitIris = mix(relitIris, vec3(0.99, 0.98, 0.96), catchlight * 0.92);
      color = mix(color, relitIris, irisMask * amount);
    }

    return color;
  }

  // 4. Colorful Anime Cel-Shading Grading
  vec3 computeColorfulAnimeCel(vec3 kuwaharaRgb, float lum, vec2 screenUV, float styleIdx) {
    vec3 gray = vec3(rawLuma(kuwaharaRgb));
    vec3 vibrant = mix(gray, kuwaharaRgb, 1.36);

    // Smooth 4-band anime cel quantization
    float band1 = smoothstep(0.18, 0.25, lum);
    float band2 = smoothstep(0.42, 0.50, lum);
    float band3 = smoothstep(0.70, 0.78, lum);
    float celLum = 0.16 + band1 * 0.24 + band2 * 0.28 + band3 * 0.28;

    vec3 baseCel = clamp(vibrant * (celLum / max(lum, 0.08)), 0.0, 1.0);

    float sunSide = 1.0 - smoothstep(0.15, 0.85, screenUV.x);
    float shadowSide = 1.0 - sunSide;

    if (styleIdx < 0.5) {
      // STYLE 0: GOLDEN HOUR NOSTALGIA (Reference Image Match)
      vec3 sunsetHighlight = vec3(1.00, 0.88, 0.79);
      vec3 coralMidtone    = vec3(0.96, 0.69, 0.58);
      vec3 tealShadow      = vec3(0.14, 0.42, 0.49);
      vec3 deepEspresso    = vec3(0.15, 0.10, 0.12);

      vec3 paletteGrade = mix(tealShadow, coralMidtone, smoothstep(0.22, 0.62, celLum));
      paletteGrade = mix(paletteGrade, sunsetHighlight, smoothstep(0.60, 0.90, celLum));

      vec3 result = mix(baseCel, paletteGrade, 0.36);
      result = mix(result, result * vec3(1.06, 0.96, 0.88), sunSide * 0.35);
      result = mix(result, mix(result, tealShadow, 0.32), shadowSide * (1.0 - band2) * 0.65);
      result = mix(deepEspresso, result, smoothstep(0.08, 0.22, lum));
      return result;
    } else if (styleIdx < 1.5) {
      // STYLE 1: SHINKAI VIVID SKY & BLOOM
      vec3 sakuraHighlight = vec3(1.00, 0.92, 0.96);
      vec3 ceruleanMid     = vec3(0.45, 0.78, 0.98);
      vec3 twilightShadow  = vec3(0.24, 0.22, 0.52);
      vec3 deepIndigo      = vec3(0.09, 0.08, 0.23);

      vec3 paletteGrade = mix(twilightShadow, ceruleanMid, smoothstep(0.22, 0.60, celLum));
      paletteGrade = mix(paletteGrade, sakuraHighlight, smoothstep(0.58, 0.88, celLum));

      vec3 result = mix(baseCel, paletteGrade, 0.35);
      result = mix(result, result * vec3(1.05, 0.94, 1.04), sunSide * 0.35);
      result = mix(deepIndigo, result, smoothstep(0.08, 0.22, lum));
      return result;
    } else {
      // STYLE 2: NEO-TOKYO POP CEL
      vec3 marigoldLight = vec3(1.00, 0.95, 0.76);
      vec3 electricRose  = vec3(0.98, 0.44, 0.52);
      vec3 cyberTeal     = vec3(0.08, 0.58, 0.53);
      vec3 sumiShadow    = vec3(0.07, 0.07, 0.09);

      vec3 paletteGrade = mix(cyberTeal, electricRose, smoothstep(0.22, 0.58, celLum));
      paletteGrade = mix(paletteGrade, marigoldLight, smoothstep(0.58, 0.88, celLum));

      vec3 result = mix(baseCel, paletteGrade, 0.36);
      result = mix(sumiShadow, result, smoothstep(0.08, 0.22, lum));
      return result;
    }
  }

  void main() {
    vec2 baseUV = getCoverUV(v_texCoord);
    // Apply subtle optical eye magnification warp starting in Stage 1
    vec2 sampleUV = applyOpticalEyeWarp(baseUV, u_eyeAmount);

    vec2 texel = 1.4 / max(u_videoSize, vec2(640.0, 480.0));
    if (u_mirror > 0.5) {
      texel.x = -texel.x;
    }

    vec3 rawColor = texture2D(u_Texture, sampleUV).rgb;

    // Apply Shader-Space Iris Relighting & Catchlights directly onto the user's real eye pixels (Stage 1 & Stage 2)
    vec3 eyeRelitColor = rawColor;
    if (u_eyeAmount > 0.001) {
      eyeRelitColor = applyShaderEyeRelighting(
        eyeRelitColor,
        sampleUV,
        u_leftIrisUV,
        u_leftIrisRadius,
        u_leftOpenness,
        u_eyeAmount,
        u_colorStyle
      );
      eyeRelitColor = applyShaderEyeRelighting(
        eyeRelitColor,
        sampleUV,
        u_rightIrisUV,
        u_rightIrisRadius,
        u_rightOpenness,
        u_eyeAmount,
        u_colorStyle
      );
    }

    if (u_mangaAmount <= 0.001) {
      gl_FragColor = vec4(eyeRelitColor, 1.0);
      return;
    }

    // Stage 2: Anisotropic Kuwahara Oil-Paint Pass + Calligraphic Sobel Ink + Color Cel Shading
    vec3 paintedRgb = sampleKuwaharaPaint(sampleUV, texel);
    // Preserve the crisp shader-space iris catchlights inside the painted pass
    paintedRgb = mix(paintedRgb, eyeRelitColor, 0.55);
    float mm = smoothstep(0.05, 0.95, rawLuma(paintedRgb));

    // Calligraphic Sobel Edge Detection
    float tl = rawLuma(texture2D(u_Texture, sampleUV + vec2(-texel.x, -texel.y)).rgb);
    float tm = rawLuma(texture2D(u_Texture, sampleUV + vec2(0.0,      -texel.y)).rgb);
    float tr = rawLuma(texture2D(u_Texture, sampleUV + vec2( texel.x, -texel.y)).rgb);
    float ml = rawLuma(texture2D(u_Texture, sampleUV + vec2(-texel.x,  0.0)).rgb);
    float mr = rawLuma(texture2D(u_Texture, sampleUV + vec2( texel.x,  0.0)).rgb);
    float bl = rawLuma(texture2D(u_Texture, sampleUV + vec2(-texel.x,  texel.y)).rgb);
    float bm = rawLuma(texture2D(u_Texture, sampleUV + vec2(0.0,       texel.y)).rgb);
    float br = rawLuma(texture2D(u_Texture, sampleUV + vec2( texel.x,  texel.y)).rgb);

    float gx = -tl - 2.0 * ml - bl + tr + 2.0 * mr + br;
    float gy = -tl - 2.0 * tm - tr + bl + 2.0 * bm + br;
    float gradMagnitude = sqrt(gx * gx + gy * gy);
    float edgeMask = smoothstep(u_edgeThreshold - 0.04, u_edgeThreshold + 0.16, gradMagnitude);

    vec3 celColor = computeColorfulAnimeCel(paintedRgb, mm, v_texCoord, u_colorStyle);

    // Subtle Diagonal Screentone in Shadow Transitions
    float s = 0.70710678;
    mat2 rot45 = mat2(s, -s, s, s);
    vec2 cellUV = fract((rot45 * gl_FragCoord.xy) / u_dotFrequency) - 0.5;
    float distFromCellCenter = length(cellUV);

    float shadowZone = clamp((0.58 - mm) / 0.44, 0.0, 1.0);
    float dotRadius = pow(shadowZone, 1.4) * 0.42;
    float halftoneMask = (1.0 - smoothstep(dotRadius - 0.08, dotRadius + 0.08, distFromCellCenter)) * 0.38;

    vec3 halftoneTint = (u_colorStyle < 0.5)
      ? vec3(0.48, 0.22, 0.24)
      : (u_colorStyle < 1.5)
      ? vec3(0.18, 0.20, 0.48)
      : vec3(0.75, 0.15, 0.35);

    vec3 shadedAnime = mix(celColor, halftoneTint, halftoneMask * shadowZone);

    vec3 inkLineColor = (u_colorStyle < 0.5)
      ? vec3(0.14, 0.09, 0.10)
      : (u_colorStyle < 1.5)
      ? vec3(0.09, 0.08, 0.22)
      : vec3(0.05, 0.05, 0.07);

    shadedAnime = mix(shadedAnime, inkLineColor, edgeMask * 0.90);

    vec3 finalColor = mix(eyeRelitColor, shadedAnime, clamp(u_mangaAmount, 0.0, 1.0));
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
      u_eyeAmount: gl.getUniformLocation(program, 'u_eyeAmount'),
      u_mangaAmount: gl.getUniformLocation(program, 'u_mangaAmount'),
      u_mirror: gl.getUniformLocation(program, 'u_mirror'),
      u_edgeThreshold: gl.getUniformLocation(program, 'u_edgeThreshold'),
      u_dotFrequency: gl.getUniformLocation(program, 'u_dotFrequency'),
      u_colorStyle: gl.getUniformLocation(program, 'u_colorStyle'),
      u_leftIrisUV: gl.getUniformLocation(program, 'u_leftIrisUV'),
      u_rightIrisUV: gl.getUniformLocation(program, 'u_rightIrisUV'),
      u_leftIrisRadius: gl.getUniformLocation(program, 'u_leftIrisRadius'),
      u_rightIrisRadius: gl.getUniformLocation(program, 'u_rightIrisRadius'),
      u_leftOpenness: gl.getUniformLocation(program, 'u_leftOpenness'),
      u_rightOpenness: gl.getUniformLocation(program, 'u_rightOpenness'),
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
    eyeAmount = 0.0,
    mangaAmount = 0.0,
    mirror = true,
    edgeThreshold = 0.26,
    dotFrequency = 6.0,
    colorStyle = 0,
    eyeUniforms = null,
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
    gl.uniform1f(this.uniforms.u_eyeAmount, eyeAmount);
    gl.uniform1f(this.uniforms.u_mangaAmount, mangaAmount);
    gl.uniform1f(this.uniforms.u_mirror, mirror ? 1.0 : 0.0);
    gl.uniform1f(this.uniforms.u_edgeThreshold, edgeThreshold);
    gl.uniform1f(this.uniforms.u_dotFrequency, dotFrequency);
    gl.uniform1f(this.uniforms.u_colorStyle, colorStyle);

    if (eyeUniforms) {
      gl.uniform2f(this.uniforms.u_leftIrisUV, eyeUniforms.leftIrisUV.x, eyeUniforms.leftIrisUV.y);
      gl.uniform2f(this.uniforms.u_rightIrisUV, eyeUniforms.rightIrisUV.x, eyeUniforms.rightIrisUV.y);
      gl.uniform1f(this.uniforms.u_leftIrisRadius, eyeUniforms.leftIrisRadius);
      gl.uniform1f(this.uniforms.u_rightIrisRadius, eyeUniforms.rightIrisRadius);
      gl.uniform1f(this.uniforms.u_leftOpenness, eyeUniforms.leftOpenness);
      gl.uniform1f(this.uniforms.u_rightOpenness, eyeUniforms.rightOpenness);
    } else {
      gl.uniform1f(this.uniforms.u_leftIrisRadius, 0.0);
      gl.uniform1f(this.uniforms.u_rightIrisRadius, 0.0);
    }

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
