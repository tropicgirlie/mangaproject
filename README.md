# Mangaface Effect — Real-Time Gaze-Driven Anime & Manga Transformation

**Live Demo:** [https://tropicgirlie.github.io/mangaproject/](https://tropicgirlie.github.io/mangaproject/)

An interactive client-side webcam experiment that uses **MediaPipe FaceLandmarker** (478 iris-enabled landmarks and `eyeBlinkLeft` / `eyeBlinkRight` blendshapes) paired with a custom **WebGL Anisotropic Kuwahara & GPU Shader-Space Eye Relighting** pipeline to trigger a real-time manga/anime transformation when you lock your gaze into the lens.

## Interaction State Machine

1. **IDLE**: Live mirrored webcam viewport with brutalist editorial overlay (*"Stare into the lens"*).
2. **GAZE LOCK (`dwellProgress 0.0 -> 1.0`)**: Detects forward head orientation (low yaw/pitch) and centered iris landmarks (`#468` & `#473`). Holding gaze fills `dwellProgress` from `0.0` to `1.0` over `1.2s`; looking away smoothly decays it back to `0`.
3. **STAGE 1 (`progress > 0.15`)**: Triggers GPU shader-space optical eye magnification (`14%` lens warp) and jewel-toned iris relighting with dual specular catchlights and calligraphic upper lash accents synced to your real blink blendshapes.
4. **STAGE 2 (`progress == 1.0`)**: Snaps the canvas into Full Anime/Manga Mode with a `150ms` screen shake, 4-quadrant Kuwahara oil-paint smoothing, calligraphic Sobel ink outlines, diagonal halftone screentone shadows, animated radial speed lines (*shuchu-sen*), and a vertical `ゴゴゴゴ` SFX stamp + panel snapshot exporter.

## 3 Selectable Color Palettes
- **01 Golden Hour Nostalgia**: Warm sunset peach/coral key light, cyan-teal street shadows, and deep espresso ink.
- **02 Shinkai Vivid Sky**: Cerulean sky highlights, sakura-pink rim light, and twilight indigo ink.
- **03 Neo-Tokyo Pop Cel**: Marigold gold highlights, electric rose & cyber-mint halftone accents, and crisp sumi ink.

## Privacy
100% client-side in the browser. Zero video frames ever leave your device.

---

Designed & built by [luana.systems](https://www.luana.systems)
