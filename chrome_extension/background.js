const MODEL_ENDPOINT = "http://localhost:8000/classify";

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type !== "CLASSIFY_COMMENT") return;

  classifyComment(message.text)
    .then(sendResponse)
    .catch((error) => {
      console.error("Classification failed:", error);
      sendResponse({
        ok: false,
        error: String(error),
      });
    });

  return true;
});

async function classifyComment(text) {
  const res = await fetch(MODEL_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text }),
  });

  if (!res.ok) {
    throw new Error(`Model API returned ${res.status}`);
  }

  const data = await res.json();

  return {
    ok: true,
    score: Number(data.score ?? 0),
    label: data.label ?? "unknown",
    reason: data.reason ?? "",
  };
}
