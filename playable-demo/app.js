const messages = document.querySelector("#messages");
const form = document.querySelector("#chatForm");
const input = document.querySelector("#messageInput");
const apiUrl = document.querySelector("#apiUrl");
const contextPath = document.querySelector("#contextPath");
const provider = document.querySelector("#provider");
const temperature = document.querySelector("#temperature");
const tempValue = document.querySelector("#tempValue");
const replyTokens = document.querySelector("#replyTokens");
const resetBtn = document.querySelector("#resetBtn");
const suggestions = document.querySelector("#suggestions");
const trace = document.querySelector("#trace");

const prompts = [
  "I need a light weapon for the marsh.",
  "How much is the Short Sword?",
  "Which weapon hits hardest?",
  "Tell me about Avis Wind.",
  "I prefer to fight from a distance.",
  "I'll take the Marsh Pike.",
];

let sessionId = localStorage.getItem("npc-demo-session") || makeSessionId();
localStorage.setItem("npc-demo-session", sessionId);
apiUrl.value =
  location.port === "5173"
    ? `${location.protocol}//${location.hostname || "localhost"}:8120/api/chat`
    : `${location.origin}/api/chat`;

for (const prompt of prompts) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = prompt;
  button.addEventListener("click", () => {
    input.value = prompt;
    input.focus();
  });
  suggestions.appendChild(button);
}

temperature.addEventListener("input", () => {
  tempValue.textContent = temperature.value;
});

resetBtn.addEventListener("click", () => {
  sessionId = makeSessionId();
  localStorage.setItem("npc-demo-session", sessionId);
  messages.innerHTML = "";
  trace.textContent = "No tool calls yet.";
  addMessage("npc", "Garrick", "Fresh start. The racks are full and the marsh is still hungry. What do you need?");
  input.focus();
});

function makeSessionId() {
  if (globalThis.crypto && typeof globalThis.crypto.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }
  const random = Math.random().toString(16).slice(2);
  return `session-${Date.now().toString(16)}-${random}`;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  addMessage("player", "You", text);
  await send(text);
});

function addMessage(kind, speaker, text) {
  const node = document.createElement("article");
  node.className = `message ${kind}`;
  node.innerHTML = `<span></span><p></p>`;
  node.querySelector("span").textContent = speaker;
  node.querySelector("p").textContent = text;
  messages.appendChild(node);
  messages.scrollTop = messages.scrollHeight;
}

function renderTrace(tools) {
  if (!tools || tools.length === 0) {
    trace.textContent = "No tool call for this turn.";
    return;
  }
  trace.textContent = tools
    .map((tool) => `${tool.name}(${JSON.stringify(tool.parameters, null, 0)})\n→ ${JSON.stringify(tool.result, null, 2)}`)
    .join("\n\n");
}

async function send(text) {
  const button = form.querySelector("button");
  button.disabled = true;
  button.textContent = "Thinking";
  try {
    const response = await fetch(apiUrl.value, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        session_id: sessionId,
        message: text,
        context_path: contextPath.value,
        provider: provider.value,
        temperature: Number(temperature.value),
        reply_max_tokens: Number(replyTokens.value),
      }),
    });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || `HTTP ${response.status}`);
    }
    const data = await response.json();
    sessionId = data.session_id;
    localStorage.setItem("npc-demo-session", sessionId);
    renderTrace(data.tools);
    addMessage("npc", data.npc || "NPC", data.reply);
  } catch (error) {
    renderTrace([]);
    addMessage("error", "Service", error.message);
  } finally {
    button.disabled = false;
    button.textContent = "Send";
    input.focus();
  }
}
