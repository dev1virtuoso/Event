# Clef Vision Tools — Tag Videos Frame by Frame (Beginner Guide)

Welcome! This is a hands-on toolkit from the **PyCon HK 2026 Sprint**.

You will learn how to use a **decision model** (clef-flash) to look at images or video frames and quickly answer simple questions such as:

- Is there a person?
- Is this a screen recording?
- What kind of scene is this?
- Should we run OCR?

The model gives short structured answers (yes/no, choice, or score) with confidence scores — not long paragraphs.

---

## What You Need First

### 1. Install Ollama

Ollama lets you run the model on your own computer.

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Download the Model (about 12 GB)

```bash
ollama pull clef-flash
```

Wait until it finishes. This only needs to be done once.

### 3. Check that Ollama is running

```bash
ollama list
```

You should see `clef-flash` in the list.

---

## Python Setup

We recommend using `uv` (fast and simple).

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Inside the project folder
uv init
uv add ollama pygame opencv-python flask pyyaml requests
```

(You can also use a normal `pip install` if you prefer.)

---

## The Three Main Tools

### 1. Image Decision Tool (Desktop – easiest for beginners)

**File:** `image_clef.py`

A simple window where you can drag & drop an image or click to choose one.

```bash
uv run image_clef.py
```

**How to use:**
1. Drag an image onto the left panel (or click the panel to open a file dialog).
2. Wait a few seconds.
3. On the right you will see the model’s answers (Choice / Noul / Score) with probabilities.

Great for learning what the model returns.

---

### 2. Image Decision Tool (Web version)

**File:** `image_clef_web.py`

Same idea, but in your browser. Looks nicer and has a sample image gallery.

```bash
uv run image_clef_web.py
```

Then open: [http://127.0.0.1:5000](http://127.0.0.1:5000)

- Drag & drop an image, or click to browse
- Or click any sample image in the gallery at the bottom
- Results appear on the right

---

### 3. Video Frame Annotation Studio (the main sprint tool)

**File:** `annotation_studio.py`  
**Questions file:** `questions.yaml`

This tool samples frames from a video and asks many useful questions about each frame (people, vehicles, screen content, camera motion, etc.).

```bash
uv run annotation_studio.py
```

Then open the URL it prints (usually [http://127.0.0.1:5000](http://127.0.0.1:5000)).

**How to use:**
1. Click the big area and choose a video (mp4, mov, webm, etc.) or an image.
2. Choose how many frames you want to annotate (default is 5).
3. Wait while it extracts frames and sends them to the model.
4. Click the thumbnails to see the tags for each frame.
5. Look at the **Clip summary** on the right for an overview.

You can change the questions the model is asked by editing `questions.yaml` and clicking **Reload questions** in the UI. No code changes needed.

---

## Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| `ffmpeg` missing | Install it (`brew install ffmpeg` on macOS) |
| Port 5000 already in use | The script will automatically try the next free port, or run with `--port 8000` |
| Model not found | Run `ollama pull clef-flash` again |
| Ollama not reachable | Make sure the Ollama app/service is running |

---

## Project Structure (simple view)

```
├── image_clef.py              ← Desktop image tool (pygame)
├── image_clef_web.py          ← Browser image tool
├── annotation_studio.py       ← Video frame annotation studio
├── questions.yaml             ← All questions for the video tool
├── app.py                     ← Older / experimental version
└── slides-inital-draft.*      ← Sprint slides
```

---

## Tips for Beginners

1. Start with `image_clef.py` or the web version. Drop a few different images and read the answers.
2. Then try the video tool with a short clip (10–30 seconds is perfect).
3. Look at `questions.yaml` — it is written in plain English. You can change the wording or add new questions later.
4. The model works best when you give it a clear image. Very blurry or extremely dark frames are harder.

---

## What is a Decision Model? (one-sentence version)

You give the model a state (an image + some text) and a list of bounded questions. It returns structured answers (yes/no, choice, or score) with confidence scores — fast and ready to use for tagging.

Happy tagging!  
If something is unclear, just ask in the sprint.
