// Voice Agent client application with real-time Web Audio, barge-in detection, and live captions.

let sessionId = null;
let ws = null;
let audioContext = null;
let micStream = null;
let scriptProcessor = null;
let pingInterval = null;

let currentState = "idle";
let currentTurnId = null;
let currentSampleRate = 24000;
let nextPlayTime = 0;
let speechPlaybackStartTime = 0;
let activeAudioSources = [];
let consecutiveSpeechFrames = 0;
let playbackDoneTimer = null;
let partialCaptionEl = null;

const btnStart = document.getElementById("btnStart");
const btnEnd = document.getElementById("btnEnd");
const statusBadge = document.getElementById("statusBadge");
const statusText = document.getElementById("statusText");
const captions = document.getElementById("captions");

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
    speechPlaybackStartTime = now;
  }

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

      // Setup audio capture and barge-in energy monitor
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioContext.createMediaStreamSource(micStream);
      scriptProcessor = audioContext.createScriptProcessor(1024, 1, 1);

      scriptProcessor.onaudioprocess = (e) => {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;

        const input = e.inputBuffer.getChannelData(0);

        // Calculate RMS frame energy
        let sum = 0;
        for (let i = 0; i < input.length; i++) {
          sum += input[i] * input[i];
        }
        const rms = Math.sqrt(sum / input.length);

        // Barge-in detection while state is SPEAKING
        if (currentState === "speaking" && rms > 0.04) {
          consecutiveSpeechFrames++;
          if (consecutiveSpeechFrames >= 3) { // ~150ms speech detected
            stopAllPlayback();
            const playedMs = Math.max(0, (audioContext.currentTime - speechPlaybackStartTime) * 1000);
            ws.send(JSON.stringify({
              type: "interrupt",
              turn_id: currentTurnId,
              played_ms: Math.round(playedMs),
            }));
            consecutiveSpeechFrames = 0;
            logMessage("[Barge-in] Interrupted agent speech.", "system");
          }
        } else {
          consecutiveSpeechFrames = 0;
        }

        // Downsample/convert Float32 to Int16 PCM (16kHz mono)
        const pcm16 = new Int16Array(input.length);
        for (let i = 0; i < input.length; i++) {
          let s = Math.max(-1, Math.min(1, input[i]));
          pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        ws.send(pcm16.buffer);
      };

      source.connect(scriptProcessor);
      scriptProcessor.connect(audioContext.destination);

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
          currentTurnId = msg.turn_id;
          currentSampleRate = msg.sample_rate || 24000;
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
  if (audioContext && audioContext.state !== "closed") audioContext.close();
  btnStart.disabled = false;
  btnEnd.disabled = true;
  setStatus("Ended");
}
