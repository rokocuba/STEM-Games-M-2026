# Reddit AI Comment Flagger

Chrome extension and local API for flagging Reddit comments that appear AI-generated.

## Install the Chrome Extension

The extension is loaded as an unpacked Chrome extension and calls the local model API at `http://localhost:8000/classify`.

### 1. Start the local API

Install the Python dependencies:

```bash
uv sync
```

Start the classifier API from the repository root:

```bash
uv run uvicorn app:app --host 127.0.0.1 --port 8000
```

Keep this process running while you use the extension. The API requires the model file at `tf-logreg/reddit_ai_detector.joblib`.

### 2. Load the extension in Chrome

1. Open Chrome and go to `chrome://extensions/`.
2. Enable `Developer mode` in the top-right corner.
3. Click `Load unpacked`.
4. Select the `chrome_extension` folder in this repository.
5. Confirm that `Reddit AI Comment Flagger` appears in the extensions list.

### 3. Use the extension

1. Visit `https://www.reddit.com/` or `https://old.reddit.com/`.
2. Open a Reddit comments page.
3. The extension will scan comments and display AI-detection labels using the local API.

If labels do not appear, make sure the API is still running on port `8000`, then refresh the Reddit page.
