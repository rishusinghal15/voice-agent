// Voice Agent client application with real-time Web Audio, barge-in detection, and live captions.

// Barge-in Configuration Constants
const BARGE_IN_GRACE_PERIOD_MS = 600;       // Only monitor after playback started for >= 600 ms
const BARGE_IN_MIN_RMS = 0.06;              // Minimum fixed RMS threshold
const BARGE_IN_NOISE_MULTIPLIER = 3.0;      // Adaptive noise floor multiplier (noise floor x 3)
const BARGE_IN_SUSTAIN_MS = 250;            // Require RMS above threshold sustained for >= 250 ms

let sessionId = null;
let ws = null;
let audioContext = null;
let micStream = null;
let scriptProcessor = null;
let pingInterval = null;

let currentState = "idle";
let currentTurnId = null;
let currentSampleRate = 22050;
let nextPlayTime = 0;
let speechPlaybackStartTime = 0;
let activeAudioSources = [];
let sustainedSpeechMs = 0;
let noiseFloorRMS = 0.01;
let playbackDoneTimer = null;
let partialCaptionEl = null;

const btnStart = document.getElementById("btnStart");
const btnEnd = document.getElementById("btnEnd");
const statusBadge = document.getElementById("statusBadge");
const statusText = document.getElementById("statusText");
const captions = document.getElementById("captions");
const micMeterBar = document.getElementById("micMeterBar");
const micMeterValue = document.getElementById("micMeterValue");
const dbgSampleRate = document.getElementById("dbgSampleRate");
const dbgFps = document.getElementById("dbgFps");
const dbgBytesPerFrame = document.getElementById("dbgBytesPerFrame");

let biquadFilter = null;
let inputCarry = new Float32Array(0);
let resamplePos = 0.0;
let outputCarry = new Float32Array(0);
let framesSentCount = 0;
let lastFpsTimestamp = 0;

function setStatus(state) {
  currentState = state.toLowerCase();
  statusBadge.className = `status-badge status-${currentState}`;
  statusText.textContent = state.charAt(0).toUpperCase() + state.slice(1);
}

function logMessage(text, role = "system") {
  const el = document.createElement("div");
  el.className = `msg msg-${role}`;
  el.textContent = text;
  captions.appendChild(el);
  captions.scrollTop = captions.scrollHeight;
  return el;
}

function updatePartialCaption(text) {
  if (!partialCaptionEl) {
    partialCaptionEl = document.createElement("div");
    partialCaptionEl.className = "msg msg-user";
    partialCaptionEl.style.opacity = "0.7";
    captions.appendChild(partialCaptionEl);
  }
  partialCaptionEl.textContent = `[Live] ${text}`;
  captions.scrollTop = captions.scrollHeight;
}

function commitUserCaption(text) {
  if (partialCaptionEl) {
    partialCaptionEl.remove();
    partialCaptionEl = null;
  }
  logMessage(`You: ${text}`, "user");
}

function stopAllPlayback() {
  console.log(`[Playback] stopAllPlayback called, stopping ${activeAudioSources.length} active sources`);
  for (const src of activeAudioSources) {
    try {
      src.stop();
      src.disconnect();
    } catch (_) {}
  }
  activeAudioSources = [];
  if (audioContext) {
    nextPlayTime = audioContext.currentTime;
  }
  speechPlaybackStartTime = 0;
  if (playbackDoneTimer) {
    clearTimeout(playbackDoneTimer);
    playbackDoneTimer = null;
  }
}

function scheduleAudioChunk(arrayBuffer) {
  if (!audioContext || audioContext.state === "closed") return;

  const int16 = new Int16Array(arrayBuffer);
  if (int16.length === 0) return;

  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) {
    float32[i] = int16[i] / 32768.0;
  }

  const audioBuffer = audioContext.createBuffer(1, float32.length, currentSampleRate);
  audioBuffer.copyToChannel(float32, 0);

  const source = audioContext.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(audioContext.destination);

  const now = audioContext.currentTime;
  if (nextPlayTime < now) {
    nextPlayTime = now;
  }
  if (speechPlaybackStartTime === 0) {
    speechPlaybackStartTime = nextPlayTime;
    console.log(`[Playback] Speech playback starting at time=${speechPlaybackStartTime.toFixed(3)}s`);
  }

  console.log(`[Playback] Audio chunk scheduled: ${arrayBuffer.byteLength} bytes at ${nextPlayTime.toFixed(3)}s (audioContext.currentTime=${now.toFixed(3)}s, duration=${audioBuffer.duration.toFixed(3)}s)`);
  source.start(nextPlayTime);
  nextPlayTime += audioBuffer.duration;
  activeAudioSources.push(source);

  source.onended = () => {
    const idx = activeAudioSources.indexOf(source);
    if (idx !== -1) {
      activeAudioSources.splice(idx, 1);
    }
  };
}

function schedulePlaybackDone(turnId) {
  if (playbackDoneTimer) {
    clearTimeout(playbackDoneTimer);
  }

  if (!audioContext) return;

  const remainingSeconds = Math.max(0, nextPlayTime - audioContext.currentTime);
  const remainingMs = Math.round(remainingSeconds * 1000) + 50;

  playbackDoneTimer = setTimeout(() => {
    if (ws && ws.readyState === WebSocket.OPEN && currentState === "speaking") {
      console.log(`[Playback] Speech playback completed for turn ${turnId}`);
      speechPlaybackStartTime = 0;
      ws.send(JSON.stringify({
        type: "playback_done",
        turn_id: turnId,
      }));
    }
  }, remainingMs);
}

btnStart.addEventListener("click", async () => {
  try {
    btnStart.disabled = true;
    logMessage("Requesting microphone permission...", "system");

    // Request mic permission
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    logMessage("Creating session...", "system");
    const res = await fetch("/sessions", { method: "POST" });
    if (!res.ok) throw new Error("Failed to create session");
    const data = await res.json();
    sessionId = data.session_id;
    logMessage(`Session started: ${sessionId}`, "system");

    // Connect WebSocket
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${proto}//${window.location.host}/ws/voice?session_id=${sessionId}`;
    ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      btnEnd.disabled = false;
      logMessage("Connected to server.", "system");

      // Setup audio capture, anti-aliasing low-pass, 16kHz resampling, and live meter
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      if (audioContext.state === "suspended") {
        audioContext.resume().catch((err) => console.warn("AudioContext resume warning:", err));
      }
      const inSampleRate = audioContext.sampleRate;
      if (dbgSampleRate) dbgSampleRate.textContent = inSampleRate;
      if (dbgBytesPerFrame) dbgBytesPerFrame.textContent = "640";
      if (dbgFps) dbgFps.textContent = "0.0";

      inputCarry = new Float32Array(0);
      resamplePos = 0.0;
      outputCarry = new Float32Array(0);
      framesSentCount = 0;
      lastFpsTimestamp = performance.now();
      sustainedSpeechMs = 0;
      noiseFloorRMS = 0.01;

      // Create hardware anti-aliasing lowpass filter (cutoff 7500 Hz for 16 kHz Nyquist at 8000 Hz)
      biquadFilter = audioContext.createBiquadFilter();
      biquadFilter.type = "lowpass";
      biquadFilter.frequency.setValueAtTime(7500, audioContext.currentTime);
      biquadFilter.Q.setValueAtTime(0.707, audioContext.currentTime);

      const source = audioContext.createMediaStreamSource(micStream);
      source.connect(biquadFilter);

      // ScriptProcessor with buffer size 2048 to minimize callback overhead
      scriptProcessor = audioContext.createScriptProcessor(2048, 1, 1);
      biquadFilter.connect(scriptProcessor);
      scriptProcessor.connect(audioContext.destination);

      scriptProcessor.onaudioprocess = (e) => {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;

        const inputData = e.inputBuffer.getChannelData(0);
        const bufferDurationMs = (inputData.length / audioContext.sampleRate) * 1000;

        // 1. Calculate RMS frame energy for live meter & barge-in
        let sum = 0;
        for (let i = 0; i < inputData.length; i++) {
          sum += inputData[i] * inputData[i];
        }
        const rms = Math.sqrt(sum / inputData.length);

        // Update live mic meter bar
        const meterPct = Math.min(100, Math.round(rms * 100 * 3));
        if (micMeterBar) micMeterBar.style.width = `${meterPct}%`;
        if (micMeterValue) micMeterValue.textContent = `${meterPct}%`;

        // Measure background noise floor during LISTENING
        if (currentState === "listening") {
          noiseFloorRMS = 0.95 * noiseFloorRMS + 0.05 * rms;
        }

        // Adaptive barge-in threshold (noise floor x 3, minimum fixed value BARGE_IN_MIN_RMS)
        const currentBargeInThreshold = Math.max(BARGE_IN_MIN_RMS, noiseFloorRMS * BARGE_IN_NOISE_MULTIPLIER);

        // Barge-in detection:
        // (a) Only monitor while state is speaking AND at least 600 ms after playback started
        // (b) Require RMS above adaptive threshold sustained for at least 250 ms
        const playbackElapsedMs = speechPlaybackStartTime > 0 ? (audioContext.currentTime - speechPlaybackStartTime) * 1000 : 0;
        const isPlaybackActive = activeAudioSources.length > 0 && speechPlaybackStartTime > 0;

        if (currentState === "speaking" && isPlaybackActive && playbackElapsedMs >= BARGE_IN_GRACE_PERIOD_MS) {
          if (rms >= currentBargeInThreshold) {
            sustainedSpeechMs += bufferDurationMs;
            if (sustainedSpeechMs >= BARGE_IN_SUSTAIN_MS) {
              console.log(
                `[Barge-in] Triggered! RMS: ${rms.toFixed(4)}, Threshold: ${currentBargeInThreshold.toFixed(4)}, ` +
                `Sustained: ${sustainedSpeechMs.toFixed(0)}ms, PlaybackElapsed: ${playbackElapsedMs.toFixed(0)}ms`
              );
              stopAllPlayback();
              const playedMs = Math.max(0, (audioContext.currentTime - speechPlaybackStartTime) * 1000);
              ws.send(JSON.stringify({
                type: "interrupt",
                turn_id: currentTurnId,
                played_ms: Math.round(playedMs),
              }));
              sustainedSpeechMs = 0;
              logMessage("[Barge-in] Interrupted agent speech.", "system");
            }
          } else {
            sustainedSpeechMs = 0;
          }
        } else {
          sustainedSpeechMs = 0;
        }

        // 2. Resample from inSampleRate to 16,000 Hz with continuous phase tracking
        const outSampleRate = 16000;
        const ratio = inSampleRate / outSampleRate;

        // Concatenate previous unprocessed input samples with new buffer
        const totalInput = new Float32Array(inputCarry.length + inputData.length);
        totalInput.set(inputCarry);
        totalInput.set(inputData, inputCarry.length);

        // Linear interpolation with zero-drift carry-over
        const resampled = [];
        while (resamplePos + 1 < totalInput.length) {
          const idx = Math.floor(resamplePos);
          const frac = resamplePos - idx;
          const s0 = totalInput[idx];
          const s1 = totalInput[idx + 1];
          resampled.push(s0 + frac * (s1 - s0));
          resamplePos += ratio;
        }

        // Keep unprocessed samples and maintain fractional offset
        const lastProcessedIdx = Math.floor(resamplePos);
        inputCarry = totalInput.slice(lastProcessedIdx);
        resamplePos -= lastProcessedIdx;

        // 3. Accumulate into output carry-over buffer
        const combinedOutput = new Float32Array(outputCarry.length + resampled.length);
        combinedOutput.set(outputCarry);
        combinedOutput.set(resampled, outputCarry.length);
        outputCarry = combinedOutput;

        // 4. Emit strictly 320-sample (640-byte = 20ms) 16kHz mono s16le frames
        const FRAME_SAMPLES = 320; // 20 ms at 16 kHz
        while (outputCarry.length >= FRAME_SAMPLES) {
          const frameSlice = outputCarry.subarray(0, FRAME_SAMPLES);
          outputCarry = outputCarry.slice(FRAME_SAMPLES);

          const pcm16 = new Int16Array(FRAME_SAMPLES);
          for (let i = 0; i < FRAME_SAMPLES; i++) {
            // Clip to [-1.0, 1.0] and scale correctly by (x * 32767)
            const s = Math.max(-1.0, Math.min(1.0, frameSlice[i]));
            pcm16[i] = Math.round(s * 32767);
          }

          ws.send(pcm16.buffer);
          framesSentCount++;
        }

        // 5. Update debug line FPS
        const now = performance.now();
        if (now - lastFpsTimestamp >= 1000) {
          const fps = ((framesSentCount * 1000) / (now - lastFpsTimestamp)).toFixed(1);
          if (dbgFps) dbgFps.textContent = fps;
          if (dbgBytesPerFrame) dbgBytesPerFrame.textContent = "640";
          framesSentCount = 0;
          lastFpsTimestamp = now;
        }
      };

      // Ping keepalive every 15s
      pingInterval = setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "ping" }));
        }
      }, 15000);
    };

    ws.onmessage = (event) => {
      if (typeof event.data === "string") {
        const msg = JSON.parse(event.data);

        if (msg.type === "state") {
          setStatus(msg.state);
        } else if (msg.type === "transcript_partial") {
          updatePartialCaption(msg.text);
        } else if (msg.type === "transcript_final") {
          currentTurnId = msg.turn_id;
          commitUserCaption(msg.text);
        } else if (msg.type === "agent_text") {
          logMessage(`Agent: ${msg.text}`, "agent");
        } else if (msg.type === "agent_audio_start") {
          console.log("[Audio] agent_audio_start:", msg);
          currentTurnId = msg.turn_id;
          currentSampleRate = msg.sample_rate || 22050;
          speechPlaybackStartTime = 0;
        } else if (msg.type === "turn_done") {
          schedulePlaybackDone(msg.turn_id);
        } else if (msg.type === "tool_event") {
          logMessage(`[Tool] ${msg.name} -> ${msg.status}`, "tool");
        } else if (msg.type === "error") {
          logMessage(`[Error] (${msg.code}) ${msg.message}`, "error");
        } else if (msg.type === "call_ended") {
          logMessage(`Call ended (${msg.reason})`, "system");
          cleanup();
        }
      } else {
        // Binary PCM chunk received from agent TTS
        console.log(`[Audio] Received audio chunk: ${event.data.byteLength} bytes`);
        scheduleAudioChunk(event.data);
      }
    };

    ws.onclose = () => {
      logMessage("Disconnected from server.", "system");
      cleanup();
    };

    ws.onerror = (err) => {
      console.error(err);
      logMessage("WebSocket error encountered.", "error");
    };

  } catch (err) {
    console.error(err);
    logMessage(`Error: ${err.message}`, "error");
    cleanup();
  }
});

btnEnd.addEventListener("click", () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "end_call" }));
  }
});

function cleanup() {
  if (pingInterval) clearInterval(pingInterval);
  stopAllPlayback();
  if (micStream) micStream.getTracks().forEach((t) => t.stop());
  if (scriptProcessor) scriptProcessor.disconnect();
  if (biquadFilter) {
    try { biquadFilter.disconnect(); } catch (_) {}
    biquadFilter = null;
  }
  if (audioContext && audioContext.state !== "closed") audioContext.close();

  inputCarry = new Float32Array(0);
  outputCarry = new Float32Array(0);
  resamplePos = 0.0;
  framesSentCount = 0;
  sustainedSpeechMs = 0;
  noiseFloorRMS = 0.01;
  speechPlaybackStartTime = 0;

  if (micMeterBar) micMeterBar.style.width = "0%";
  if (micMeterValue) micMeterValue.textContent = "0%";
  if (dbgFps) dbgFps.textContent = "0.0";

  btnStart.disabled = false;
  btnEnd.disabled = true;
  setStatus("Ended");
}

async function loadHealthz() {
  const footerEl = document.getElementById("footerProviders");
  if (!footerEl) return;
  try {
    const res = await fetch("/healthz");
    if (res.ok) {
      const data = await res.json();
      if (data && data.providers) {
        footerEl.textContent = data.providers;
      }
    }
  } catch (err) {
    console.warn("Could not fetch /healthz:", err);
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", loadHealthz);
} else {
  loadHealthz();
}
