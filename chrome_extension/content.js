const MIN_CHARS = 120;
const AI_THRESHOLD = 0.85;
const POSSIBLE_THRESHOLD = 0.5;

const seen = new WeakSet();

function getCommentNodes() {
  return [
    ...document.querySelectorAll("shreddit-comment"),
    ...document.querySelectorAll(".thing.comment"),
  ];
}

function extractText(commentNode) {
  const candidates = [
    commentNode.querySelector("[slot='comment']"),
    commentNode.querySelector(".md"),
    commentNode,
  ];

  for (const node of candidates) {
    const text = node?.innerText?.trim();
    if (text && text.length >= MIN_CHARS) return text;
  }

  return "";
}

function hashText(text) {
  let hash = 0;
  for (let i = 0; i < text.length; i++) {
    hash = (hash << 5) - hash + text.charCodeAt(i);
    hash |= 0;
  }
  return String(hash);
}

async function getCachedOrClassify(text) {
  const key = `ai_comment_${hashText(text)}`;
  const cached = await chrome.storage.local.get(key);

  if (cached[key]) return cached[key];

  const result = await chrome.runtime.sendMessage({
    type: "CLASSIFY_COMMENT",
    text,
  });

  if (result?.ok) {
    await chrome.storage.local.set({ [key]: result });
  }

  return result;
}

function addBadge(commentNode, result) {
  if (!result?.ok) return;

  const score = result.score;

  console.log(result);

  console.log(result.label);
  console.log(score);

  if (result.label == "HUMAN") {
    return;
  }

  let label = null;
  if (score >= AI_THRESHOLD) {
    label = `AI-suspected ${(score * 100).toFixed(0)}%`;
  } else if (score >= POSSIBLE_THRESHOLD) {
    label = `Possibly AI-like ${(score * 100).toFixed(0)}%`;
  }

  if (!label) return;

  const existing = commentNode.querySelector(".ai-flagger-badge");
  if (existing) return;

  const badge = document.createElement("span");
  badge.className =
    score >= AI_THRESHOLD
      ? "ai-flagger-badge ai-flagger-high"
      : "ai-flagger-badge ai-flagger-medium";

  badge.textContent = label;

  if (result.reason) {
    badge.title = result.reason;
  }

  commentNode.prepend(badge);
}

async function scanComments() {
  const comments = getCommentNodes();

  for (const comment of comments) {
    if (seen.has(comment)) continue;
    seen.add(comment);

    const text = extractText(comment);
    if (!text) continue;

    try {
      const result = await getCachedOrClassify(text);
      addBadge(comment, result);
    } catch (err) {
      console.warn("Could not classify comment:", err);
    }

    await sleep(250);
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

const observer = new MutationObserver(() => {
  scanComments();
});

observer.observe(document.body, {
  childList: true,
  subtree: true,
});

scanComments();
