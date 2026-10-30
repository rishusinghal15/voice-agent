// Voice Agent client application
let sessionId = null;
let ws = null;
let audioContext = null;
let micStream = null;
let pingInterval = null;

const btnStart = document.getElementById("btnStart");
const btnEnd = document.getElementById("btnEnd");
const statusBadge = document.getElementById("statusBadge");
const statusText = document.getElementById("statusText");
const captions = document.getElementById("captions");

function setStatus(state) {
  statusBadge.className = `status-badge status-${state.toLowerCase()}`;
  statusText.textContent = state.charAt(0).toUpperCase() + state.slice(1);
}

function logMessage(text, role = "system") {
  const el = document.createElement("div");
  el.className = `msg msg-${role}`;
  el.textContent = text;
  captions.appendChild(el);
  captions.scrollTop = captions.scrollHeight;
}

btnStart.addEventListener("click", async () => {
  try {
    btnStart.disabled = true;
    logMessage("Requesting microphone permission...", "system");

    // Request mic permission
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });

    // Request session creation
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

      // Setup audio capture
      audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      const source = audioContext.createMediaStreamSource(micStream);
      const processor = audioContext.createScriptProcessor(1024, 1, 1);

      processor.onaudioprocess = (e) => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          const input = e.inputBuffer.getChannelData(0);
          const pcm16 = new Int16Array(input.length);
          for (let i = 0; i < input.length; i++) {
            let s = Math.max(-1, Math.min(1, input[i]));
            pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
          }
          ws.send(pcm16.buffer);
        }
      };

      source.connect(processor);
      processor.connect(audioContext.destination);

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
          logMessage(`[Partial] ${msg.text}`, "user");
        } else if (msg.type === "transcript_final") {
          logMessage(msg.text, "user");
        } else if (msg.type === "agent_text") {
          logMessage(msg.text, "agent");
        } else if (msg.type === "call_ended") {
          logMessage(`Call ended (${msg.reason})`, "system");
          cleanup();
        }
      } else {
        // Binary audio chunk received (ECHO)
        logMessage(`Received audio chunk (${event.data.byteLength} bytes)`, "agent");
      }
    };

    ws.onclose = () => {
      logMessage("Disconnected.", "system");
      cleanup();
    };

    ws.onerror = (err) => {
      console.error(err);
      logMessage("WebSocket error encountered.", "system");
    };

  } catch (err) {
    console.error(err);
    logMessage(`Error: ${err.message}`, "system");
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
  if (micStream) micStream.getTracks().forEach((t) => t.stop());
  if (audioContext) audioContext.close();
  btnStart.disabled = false;
  btnEnd.disabled = true;
  setStatus("Ended");
}

