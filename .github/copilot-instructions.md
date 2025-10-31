# Copilot instructions for AI coding agents

This repository contains TP1/TP2/TP3 exercises for an "AI for IoT" course. These notes focus on TP3 (Arduino + DHT22 + Logistic Regression) where most active development happens.

Keep guidance short, actionable and specific to this codebase.

Essentials
- Primary microproject: `TP3/` — an Arduino (PlatformIO) project. Main source: `TP3/src/main.cpp`.
- Build config: `TP3/platformio.ini` (env `uno`, atmelavr, framework `arduino`). Serial monitor speed is 9600.
- Wokwi simulation configured via `TP3/wokwi.toml` and wiring in `TP3/diagram.json`.

Big picture
- TP3 reads Temperature & Humidity from a DHT22 sensor (pin D2) and must run a pre-trained logistic regression model on the Arduino.
- Data flow: DHT sensor -> `loop()` reads into X[] -> standardize with exported MEAN/STD -> compute z = dot(W, x_scaled) + BIAS -> y = sigmoid(z) -> Serial output.
- Model parameters are supplied as constants in `TP3/src/main.cpp`. The file currently scaffolds N_FEATURES=12 and placeholders for MEAN, STD, WEIGHTS and BIAS.

What to change / implement
- Add the exported model parameters (MEAN, STD, WEIGHTS, BIAS) in `TP3/src/main.cpp`. Maintain feature order: feature 0 = Temperature, feature 1 = Humidity (TP3 docs use this order).
- Implement standardize(), sigmoid(), predict() helper functions and call them in `loop()` where TODO markers exist.
- Keep Serial at 9600 and preserve the existing logging format (humidity, temperature, predicted probability).

Build / run / debug (exact commands)
- Using PlatformIO in VS Code (recommended): open folder at `AIoT-2025/TP3`, then click Build / Upload / Serial Monitor in the PlatformIO sidebar.
- CLI (PowerShell):
  - Build: pio run -d TP3
  - Upload: pio run -d TP3 -t upload
  - Serial monitor: pio device monitor -d TP3 --baud 9600
- Wokwi simulation: open `TP3/diagram.json` or `TP3/wokwi.toml` in Wokwi to run the simulation. The `wokwi.toml` points to the PlatformIO build artifacts for simulation.

Project-specific conventions
- The Arduino code expects fixed-size arrays for features. `N_FEATURES` is set to 12: only the first two indices are overwritten by the sensor reads; other entries may be left for other features or historical values. Preserve this structure when modifying constants.
- Use floating-point `float` types (not double) because Arduino Uno's math is 32-bit float.
- Serial prints use Arduino `F()` macro sometimes; maintain that where present for memory efficiency.

Key files to reference (examples)
- `TP3/src/main.cpp` — sensor read loop, placeholders for model code (primary editing target).
- `TP3/platformio.ini` — dependencies and board config (Adafruit DHT library declared here).
- `TP3/TP3_Logistic_Arduino.md` — explicit student instructions for exporting μ, σ, W, and b and expected implementation details. Use it to produce exact parameter order and example functions.
- `TP3/install.md` — developer workflow: wiring, PlatformIO usage, serial monitor settings.
- `TP3/diagram.json` and `TP3/wokwi.toml` — wiring and simulation metadata (DHT -> UNO D2, VCC -> 5V, GND -> GND).

Edge cases and checks for AI edits
- Ensure `isnan()` checks remain (guard against sensor read failures). If early-returning after failed read, avoid printing predictions.
- When adding math functions, add the math header (math.h) only if needed; test compilation since the Arduino core provides exp().
- Keep `N_FEATURES` consistent with any arrays inserted. Mismatched sizes will cause compile-time issues.

Small code snippets to follow the project's style
- Standardize:
  float standardize(float x_raw, int idx) {
    return (x_raw - MEAN[idx]) / STD[idx];
  }
- Sigmoid:
  float sigmoid(float z) {
    return 1.0f / (1.0f + exp(-z));
  }
- Predict (dot product):
  float predict(float features[]) {
    float z = 0.0f;
    for (int i = 0; i < N_FEATURES; ++i) z += WEIGHTS[i] * features[i];
    z += BIAS;
    return sigmoid(z);
  }

What not to change
- Do not change board `env:uno` or `monitor_speed` in `platformio.ini` unless adding a new target. Many users will rely on the UNO configuration.
- Avoid increasing the serial baud rate. Course materials expect 9600 in the docs and serial monitor instructions.

If you need to add files
- Place code in `TP3/src/`. If adding test scaffolds, keep them out of `src/` (use `test/` or a separate folder).

When unsure, ask the user
- If model parameters (MEAN/STD/WEIGHTS/BIAS) are not provided, ask which Python model file produced them and request a numeric export in the TP3 docs format (arrays for mean/std, array for weights, scalar bias).

End: ask for clarification
- After applying changes, ask the student whether they want the AI to also implement and verify the logistic functions in `TP3/src/main.cpp`, or only place parameter values.