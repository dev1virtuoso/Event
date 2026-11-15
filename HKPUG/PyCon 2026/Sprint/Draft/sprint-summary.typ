// PyCon HK 2026 Sprint — Final Summary Deck
// Compile: typst compile sprint-summary.typ
// Place the screenshot next to this file as: annotation-studio-demo.png
// Audience: people outside the group who want a clear overview

#let paper  = rgb("#F6F3F9")
#let ink    = rgb("#1A1420")
#let mute   = rgb("#5A5266")
#let violet = rgb("#5B2C86")
#let wash   = rgb("#EDE6F4")
#let hair   = rgb("#C9BFD6")
#let accent = rgb("#50C878")

#set page(
  width: 16cm,
  height: 9cm,
  margin: (left: 0.95cm, top: 0.58cm, bottom: 0.5cm, right: 0.65cm),
  fill: paper,
  background: place(left + top, rect(width: 0.22cm, height: 100%, fill: violet)),
)

#set text(
  font: ("Nimbus Sans", "IBM Plex Sans", "Noto Sans CJK TC", "Noto Sans CJK JP", "Noto Sans CJK SC"),
  size: 12.5pt,
  fill: ink,
  tracking: 0.1pt,
)

#set table(stroke: hair + 0.5pt)
#set strong(delta: 200)

#let slide(title, body) = {
  block(breakable: false, width: 100%)[
    #text(size: 17pt, weight: "bold", fill: ink, tracking: 0.2pt)[#title]
    #v(0.08em)
    #rect(width: 2.2cm, height: 0.1cm, fill: violet, stroke: none)
    #v(0.2em)
    #body
  ]
  pagebreak(weak: true)
}

#let bullet-list(items) = [
  #for item in items [
    #grid(
      columns: (0.45cm, 1fr),
      column-gutter: 0.12cm,
      align: (top, top),
      [#pad(top: 0.28em)[#box(width: 0.16cm, height: 0.16cm, fill: violet)]],
      [#text(fill: ink)[#item]],
    )
    #v(0.18em)
  ]
]

#let code-block(lines) = [
  #block(
    fill: wash,
    inset: (x: 10pt, y: 9pt),
    width: 100%,
    stroke: (left: 2pt + violet),
  )[
    #for line in lines [
      #text(font: ("IBM Plex Mono", "DejaVu Sans Mono"), size: 10.5pt, fill: violet)[#line]
      #linebreak()
    ]
  ]
]

#let swiss-table(..args) = {
  table(
    inset: (x: 5pt, y: 3.2pt),
    stroke: (x: none, y: hair + 0.5pt),
    fill: (_, row) => if row == 0 { wash } else { paper },
    ..args,
  )
}

// ============================================================
// SLIDE 1 — Title
// ============================================================

#align(left + horizon)[
  #text(size: 8.5pt, weight: "bold", fill: violet, tracking: 1.6pt)[PYCON HK 2026 SPRINT]
  #v(0.55em)
  #text(size: 24pt, weight: "bold", fill: ink, tracking: -0.3pt)[
    Tagging Videos Frame by Frame
  ]
  #v(0.18em)
  #text(size: 17pt, weight: "bold", fill: violet)[
    Faster with Decision Models
  ]
  #v(0.4em)
  #rect(width: 2.4cm, height: 0.12cm, fill: violet, stroke: none)
  #v(0.4em)
  #text(size: 11pt, fill: mute)[
    Sprint summary · What we built · What we learned
  ]
  #v(0.25em)
  #text(size: 10pt, fill: mute)[
    Track: AI, Data & Models · Local clef-flash · Image + Video
  ]
]

#pagebreak()

// ============================================================
// SLIDE 2 — Original Goal
// ============================================================

#slide("Original Goal")[
  #bullet-list((
    [Let participants try a *decision model* for video tagging on their own machines.],
    [Use the model to tag objects / scenes frame by frame at high speed.],
    [Show a practical alternative to “ask an LLM to describe every frame”.],
    [Leave people able to run the tools and change the questions themselves.],
  ))
  #v(0.25em)
  #text(size: 10pt, fill: mute)[
    Proposal language: zh-Hant · Track: AI, Data & Models
  ]
]

// ============================================================
// SLIDE 3 — What We Actually Shipped
// ============================================================

#slide("What We Actually Shipped")[
  #set text(size: 10.2pt)
  #swiss-table(
    columns: (4.3cm, 1fr),
    align: (left, left),
    [*Deliverable*], [*What it does*],
    [`image_clef.py`], [Desktop drag-and-drop image tool (pygame)],
    [`image_clef_web.py`], [Browser version + sample gallery (Flask + Tailwind)],
    [`annotation_studio.py`], [Video frame annotation studio — main tool],
    [`questions.yaml`], [All questions live here; edit & reload, no code change],
    [`readme.md`], [Beginner-friendly setup & usage guide],
  )
  #v(0.12em)
  #text(size: 9.5pt, fill: mute)[
    Model: *clef-flash* (Cloudflare, open, image-capable) via local Ollama `/v1/systemone`.
  ]
]

// ============================================================
// SLIDE 4 — What Is a Decision Model
// ============================================================

#slide("What Is a Decision Model?")[
  #bullet-list((
    [You give the model a *state* (today: an image + short context) and a few *bounded* questions.],
    [It returns a short structured judgment — not a paragraph or a chat reply.],
    [Three question types: *Choice* (pick one), *Noul* (yes/no probability), *Score* (levels).],
    [Every answer carries a confidence / probability. Use it to *decide* (tag, filter, route).],
  ))
]

// ============================================================
// SLIDE 5 — System 1 vs System 2
// ============================================================

#slide("System 1 vs System 2")[
  #text(size: 9.5pt, fill: mute)[
    Framing from Kahneman’s *Thinking, Fast and Slow*. Decision-native models aim at System 1 style.
  ]
  #v(0.1em)
  #set text(size: 9.6pt)
  #swiss-table(
    columns: (2.4cm, 1fr, 1fr),
    align: (left, left, left),
    [ ], [*System 1 — fast*], [*System 2 — deliberate*],
    [Typical model], [Decision-native / decision model], [Common LLM],
    [What it does], [One pass → multiple structured answers], [Writes token by token],
    [Output], [Choice / Noul / Score + confidence], [Full sentences, explanations],
    [Good for], [Tagging, filtering, deciding], [Chat, long reasoning],
  )
]

// ============================================================
// SLIDE 6 — Why Decision Models Fit Video Tagging
// ============================================================

#slide("Why Decision Models Fit Video Tagging")[
  #bullet-list((
    [Video tagging = look at frames (images) and attach compact labels.],
    [We only need tags and confidence — not paragraphs.],
    [One forward pass per frame → much faster than generating text token-by-token.],
    [Easy to run locally on a laptop with Ollama + clef-flash.],
    [Can later be combined with traditional detectors / OCR in a hybrid pipeline.],
  ))
]

// ============================================================
// SLIDE 7 — The Main Tool: Annotation Studio
// ============================================================

#slide("Main Tool — Annotation Studio")[
  #bullet-list((
    [Samples frames from a video (or accepts a single image).],
    [Sends each frame + the question set to clef-flash.],
    [Returns structured tags: scene type, people, vehicles, screen content, camera motion, recommended next action, etc.],
    [Shows per-frame tags + a clip-level summary.],
    [Questions live in `questions.yaml` — edit and click “Reload questions”.],
  ))
]

// ============================================================
// SLIDE 8 — Live Demo Snapshot (with the screenshot)
// ============================================================

#slide("Live Demo Snapshot")[
  #text(size: 9.3pt, fill: mute)[
    Real run of the Annotation Studio on a short video clip (15 frames analysed).
  ]
  #v(0.08em)
  #align(center)[
    #image("annotation-studio-demo.png", width: 60%)
  ]
]

// ============================================================
// SLIDE 9 — Question Design
// ============================================================

#slide("Question Design (questions.yaml)")[
  #bullet-list((
    [Questions are external YAML — change them without touching Python.],
    [Types: `noul` (P(true)), `choice` (one of criteria), `score` (levels).],
    [Conditional questions with `when:` keep cost down (only ask when relevant).],
    [Examples we ship: scene type, people presence & prominence, vehicles, screen content, livestream layout, camera motion, information density, recommended action.],
  ))
]

// ============================================================
// SLIDE 10 — Pipeline Overview
// ============================================================

#slide("How the Pipeline Works")[
  #set text(size: 10.5pt)
  #code-block((
    "1. User drops a video or image",
    "2. ffmpeg samples frames (default every 2.5 s)",
    "3. For each frame:",
    "   - build state text (media facts + local metrics)",
    "   - send image + questions to clef-flash",
    "   - resolve conditional questions in waves",
    "4. Flatten answers → displayable tags",
    "5. Roll up into a clip-level summary",
  ))
]

// ============================================================
// SLIDE 11 — Concrete Takeaways
// ============================================================

#slide("Three Concrete Takeaways")[
  #bullet-list((
    [How to run a decision model *locally* (Ollama + clef-flash + System One API).],
    [How to tag video frames with structured questions (Choice / Noul / Score) instead of free text.],
    [How to combine decision models with traditional tools (detectors, OCR) for a more efficient hybrid pipeline.],
  ))
]

// ============================================================
// SLIDE 12 — Why It Matters for Python Users
// ============================================================

#slide("Why This Matters for Python Users")[
  #bullet-list((
    [Faster alternative to “ask an LLM to describe every frame”.],
    [Structured output is easy to pipe into the rest of a Python pipeline.],
    [Questions live in YAML — domain experts can tune them without deep ML knowledge.],
    [Everything runs locally — good for privacy, cost, and offline workshops.],
  ))
]

// ============================================================
// SLIDE 13 — What Worked Well / What Is Harder
// ============================================================

#slide("What Worked Well / What Is Harder")[
  #bullet-list((
    [Works well: clear frames, people, UI screens, game HUDs, text-heavy slides.],
    [Harder: very blurry / dark frames, extreme motion blur, tiny distant objects.],
    [Conditional questions help a lot — we only ask “people prominence” when people are actually present.],
    [Local metrics (brightness, edge density, motion) enrich the state text and improve consistency.],
  ))
]

// ============================================================
// SLIDE 14 — Hands-on Experience
// ============================================================

#slide("Hands-on Experience (What Participants Did)")[
  #bullet-list((
    [Morning: image tools — drop different images (UI, photo, chart, code) and read Choice / Noul / Score answers.],
    [Afternoon: video annotation studio — upload a short clip, inspect per-frame tags and the clip summary.],
    [Optional: edit `questions.yaml`, reload, and re-annotate to see the effect immediately.],
  ))
]

// ============================================================
// SLIDE 15 — Project Layout
// ============================================================

#slide("Project Layout (simple view)")[
  #set text(size: 10.5pt)
  #code-block((
    "image_clef.py              ← Desktop image tool",
    "image_clef_web.py          ← Browser image tool",
    "annotation_studio.py       ← Video frame annotation studio",
    "questions.yaml             ← All questions for the video tool",
    "app.py                     ← Older / experimental version",
    "readme.md                  ← Beginner setup & usage guide",
  ))
]

// ============================================================
// SLIDE 16 — Progress Snapshot
// ============================================================

#slide("Progress Snapshot")[
  #bullet-list((
    [Working local demo for both images and video frames.],
    [Questions fully externalised in YAML with conditional logic.],
    [Clean UI for both desktop and browser paths.],
    [Beginner-friendly readme so newcomers can start quickly.],
    [Real experience of the speed / quality trade-offs of decision models vs full LLMs.],
  ))
]

// ============================================================
// SLIDE 17 — Possible Next Steps
// ============================================================

#slide("Possible Next Steps")[
  #bullet-list((
    [Add more domain-specific question packs (e.g. traffic, education, accessibility).],
    [Hybrid pipeline: decision model for routing + traditional detectors / OCR for precision.],
    [Batch export of tags (JSON / CSV) for downstream use.],
    [Lightweight evaluation set so people can measure accuracy on their own data.],
  ))
]

// ============================================================
// SLIDE 18 — Key Message
// ============================================================

#slide("Key Message")[
  #bullet-list((
    [Decision models give short, structured answers (Choice / Noul / Score) + confidence.],
    [They are a natural fit for video frame tagging when you need speed and structure, not essays.],
    [Today everything runs locally with open tools (Ollama + clef-flash).],
    [Questions are editable YAML — the same studio can be reused after the sprint.],
  ))
]

// ============================================================
// SLIDE 19 — Wrap-up
// ============================================================

#slide("Wrap-up & Questions")[
  #bullet-list((
    [We set out to let people try decision models for video tagging locally — and we delivered working tools.],
    [Participants left with runnable demos and the ability to change the questions themselves.],
    [The approach is practical, local-first, and easy to combine with existing Python pipelines.],
    [Questions welcome — thanks for listening!],
  ))
]