// deck.typ — Decision Models for Efficient Video Tagging
// Compile: typst compile deck.typ

#set page(
  width: 16cm,
  height: 9cm,
  margin: 0.7cm,
  fill: rgb("#0f172a"),
)

#set text(
  font: ("DejaVu Sans", "Arial", "Helvetica"),
  size: 14pt,
  fill: rgb("#e6eef8"),
)

// ---- Reusable helpers ----

// Title in normal flow (no `place`), so body sits BELOW it.
#let slide(title, body) = [
  #text(size: 22pt, weight: "bold", fill: rgb("#ffffff"))[#title]
  #v(0.9em)
  #body
  #pagebreak()
]

#let bullet-list(items) = [
  #for item in items [
    #grid(
      columns: (0.45em, 1fr),
      align: (left, left),
      [•],
      [#text(fill: rgb("#d6e2f0"))[#item]],
    )
    #v(0.35em)
  ]
]

#let code-block(lines) = [
  #rect(
    fill: rgb("#1e293b"),
    inset: 8pt,
    radius: 4pt,
    width: 100%,
  )[
    #for line in lines [
      #text(font: "DejaVu Sans Mono", size: 11pt, fill: rgb("#4fc3f7"))[#line]
      #linebreak()
    ]
  ]
]

// ---- Slide 1: Title ----

#align(center + horizon)[
  #text(size: 28pt, weight: "bold", fill: rgb("#ffffff"))[
    Tagging Videos Frame by Frame\
    — Faster with Decision Models
  ]
  #v(1em)
  #text(size: 13pt, fill: rgb("#9fb3c8"))[
    A hands-on sprint · 15 Nov 2026 · 10:00–18:00
  ]
]

#pagebreak()

// ---- Slide 2: Goal ----

#slide("Today's Goal")[
  #bullet-list((
    "Can we label video frames quickly and reliably —",
    "without asking a big language model to write sentences?",
    "You'll leave with: what a decision model is, what video tagging means,",
    "and a working local demo.",
  ))
]

// ---- Slide 3: What is a decision model ----

#slide("What Is a Decision Model?")[
  #bullet-list((
    "Think of it as a 'yes / no + score' machine.",
    "It gives short, structured answers: TRUE, FALSE, or a number like 0.87.",
    "It does NOT write full sentences — no essays, no small talk.",
    "Simple picture: input → one quick pass → short answer.",
  ))
]

// ---- Slide 4: What is video tagging ----

#slide("What Is Video Tagging?")[
  #bullet-list((
    "Looking at each frame (or a short clip) and attaching labels.",
    "Examples: a box around an object, a scene name ('kitchen', 'street'),",
    "or keywords ('person', 'car', 'rain').",
    "The output is compact and abstract — tags, not paragraphs.",
  ))
]

// ---- Slide 5: Why faster ----

#slide("Why Decision Models Are Faster")[
  #bullet-list((
    "Decision model: one forward pass (formally Forward Propagation) → one short answer. Done.",
    "Large language model (LLM): generates word by word → many passes → slower.",
    "We only need tags, not sentences — so a decision model fits better.",
    "Result: less compute, faster tagging, easy to run on a laptop.",
  ))
]

// ---- Slide 6: Comparison table ----

#slide("Side-by-Side Comparison")[
  #table(
    columns: (auto, 1fr, 1fr),
    inset: 7pt,
    stroke: rgb("#334155"),
    fill: (_, row) => if row == 0 { rgb("#1e293b") } else { rgb("#0f172a") },
    align: (left, left, left),
    [ ], [*Decision Model*], [*LLM*],
    [Output],   [TRUE / score],       [Full sentences],
    [Speed],    [Very fast],          [Slower],
    [Good for], [Tagging, filtering], [Explanation, chat],
  )
]

// ---- Slide 7: Agenda ----

#slide("Agenda at a Glance")[
  #bullet-list((
    "10:00  Check-in  ·  10:15  Intro",
    "10:30  Intro talk → Hands-on Session 1 → Wrap-up",
    "13:00  Lunch  ·  15:00  Team sharing",
    "15:10  Hands-on Session 2 → Wrap-up",
    "17:45  Closing",
  ))
]

// ---- Slide 8: Setup 1 ----

#slide("Setup 1 — Install Ollama")[
  Ollama runs models locally on your machine.
  #v(0.6em)
  #code-block((
    "curl -fsSL https://ollama.com/install.sh | sh",
  ))
]

// ---- Slide 9: Setup 2 ----

#slide("Setup 2 — Get the Model")[
  About 12 GB. Give it time.
  #v(0.6em)
  #code-block((
    "ollama pull clef-flash",
  ))
]

// ---- Slide 10: Setup 3 ----

#slide("Setup 3 — Get the Repo")[
  #code-block((
    "git pull https://github.com/dev1virtuoso/Event.git",
    "cd path/to/Event/folder",
    "cd \"./HKPUG/PyCon 2026/Sprint\"",
  ))
]

// ---- Slide 11: Setup 4 ----

#slide("Setup 4 — Python Environment")[
  #code-block((
    "uv init",
    "uv add ollama pygame cv2",
  ))
]

// ---- Slide 12: Setup 5 ----

#slide("Setup 5 — Verify")[
  If it runs, your environment is ready.
  #v(0.6em)
  #code-block((
    "uv run snake_clef.py",
  ))
]

// ---- Slide 13: Hands-on 1 ----

#slide("Hands-on Session 1 (Morning)")[
  #bullet-list((
    "Goal: tag a few frames with the decision model.",
    "Focus: understand the input → output shape.",
  ))
]

// ---- Slide 14: Hands-on 2 ----

#slide("Hands-on Session 2 (Afternoon)")[
  #bullet-list((
    "Goal: tag a short video frame by frame.",
    "Focus: speed, batching, and simple keyword output.",
  ))
]

// ---- Slide 15: Wrap-up ----

#slide("Wrap-up & Questions")[
  #bullet-list((
    "Decision model = short answers, fast.",
    "Video tagging = labels per frame.",
    "Together = efficient tagging.",
    "Reach out anytime with questions or ideas. Thanks!",
  ))
]