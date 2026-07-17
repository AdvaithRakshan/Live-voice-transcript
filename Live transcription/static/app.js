let socket, audioContext, worklet;
let finalText = "";
let languageSent = false;

document.getElementById("start").onclick = start;
document.getElementById("stop").onclick = stop;

function start() {
  finalText = "";
  languageSent = false;
  document.getElementById("output").textContent = "";

  const selectedLang = document.getElementById("language").value;

  socket = new WebSocket("ws://127.0.0.1:8000/ws");

  socket.onopen = () => {
    console.log("WebSocket connected");

    socket.send(JSON.stringify({ language: selectedLang }));
    languageSent = true;

    startAudio();
  };

  socket.onmessage = e => {
    const data = JSON.parse(e.data);

    if (!data.text) return;

    if (data.final) finalText += data.text + " ";

    document.getElementById("output").textContent =
      finalText + (data.final ? "" : data.text);
  };

  socket.onclose = () => {
    console.log("WebSocket closed");
  };

  socket.onerror = (err) => {
    console.error("WebSocket error:", err);
  };
}

async function startAudio() {
  if (!languageSent) return;

  audioContext = new AudioContext({ sampleRate: 16000 });
  await audioContext.audioWorklet.addModule("/static/processor.js");

  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const source = audioContext.createMediaStreamSource(stream);

  worklet = new AudioWorkletNode(audioContext, "pcm-processor");
  worklet.port.onmessage = e => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(e.data);
    }
  };

  source.connect(worklet);
}

function stop() {
  if (audioContext) audioContext.close();
  if (socket) socket.close();
}
