// Visual fork of deck-intro-draft.typ — Swiss typographic style, violet tint.
// Content unchanged. Local draft only; do not push unless NX asks.
// Compile: typst compile deck-intro-swiss.typ

#let paper = rgb("#F6F3F9")   // paper with a light purple tint
#let ink = rgb("#1A1420")
#let mute = rgb("#5A5266")
#let violet = rgb("#5B2C86")  // Swiss-style accent / tint
#let wash = rgb("#EDE6F4")
#let hair = rgb("#C9BFD6")

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
    #v(0.2em)
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
      #text(font: ("IBM Plex Mono", "DejaVu Sans Mono"), size: 11pt, fill: violet)[#line]
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

// ---- Slide 1: Title ----

#align(left + horizon)[
  #text(size: 8.5pt, weight: "bold", fill: violet, tracking: 1.6pt)[PYCON HK 2026 SPRINT]
  #v(0.55em)
  #text(size: 26pt, weight: "bold", fill: ink, tracking: -0.3pt)[
    Tagging Videos Frame by Frame
  ]
  #v(0.2em)
  #text(size: 18pt, weight: "bold", fill: violet)[
    Faster with Decision Models
  ]
  #v(0.45em)
  #rect(width: 2.4cm, height: 0.12cm, fill: violet, stroke: none)
  #v(0.45em)
  #text(size: 11pt, fill: mute)[
    A hands-on sprint · 15 Nov 2026 · 10:00–18:00
  ]
]

#pagebreak()

// ---- Slide 2: Goal ----

#slide("Today's Goal")[
  #bullet-list((
    [Know what a *decision model* is. Today's focus: *multimodal* decision-native models — in this sprint, that means at least *image*.],
    [Recognize the many names people use for this same kind of idea.],
    [See why it can tag video frames quickly, without writing sentences.],
    [Leave with a working local demo.],
  ))
]

// ---- Slide 3: What is a decision model ----

#slide("What Is a Decision Model?")[
  #bullet-list((
    [You give the model a current *state* — today that includes at least an *image* — then ask a few bounded questions.],
    [It returns a short, structured judgment — not a paragraph or a chat reply.],
    [Question types: *Choice* (pick one option), *Noul* (yes / no), *Score* (a number).],
    [Each answer also comes with a *confidence* score or probability. Use it to *decide* (tag, filter, choose), not to write essays.],
  ))
]

// ---- Slide 4: Evidence — labs and companies ----

#slide("Already Being Built")[
  #text(size: 9.5pt, fill: mute)[
    Still new, but large companies and research teams are already shipping it — a few highlights, not the full list.
  ]
  #v(0.08em)
  #set text(size: 9.2pt)
  #swiss-table(
    columns: (3.7cm, 1fr),
    align: (left, left),
    [*Who*], [*What they shipped*],
    [TypeSafe], [System One Models, and the flagship hosted model Jev],
    [Cloudflare], [Open-source Clef and Clef-flash — flash is fast, reads images; we run it today],
    [Google], [Gemini Decisions API, and Gemini Decision 1],
    [Nokia Applied Research], [AnyJev — typed decisions from an open LLM],
    [Ollama], [Local `/v1/systemone` runtime, plus a model library],
    [Also on Ollama], [Nimble (Bespoke Labs) · Tev1 (Together AI, also 0.8B)],
  )
  #v(0.1em)
  #text(size: 9.3pt, fill: mute)[
    Independent products: the same *kind* of idea, not the same model.
  ]
]

// ---- Slide 5: Names you may hear ----

#slide("Names You May Hear")[
  #text(size: 9.6pt, fill: mute)[
    This idea is still new. These names often point to the *same kind of thing* —
    a decision model — but they are *not* all the same product or the same weights.
  ]
  #v(0.08em)
  #set text(size: 9.4pt)
  #swiss-table(
    columns: (3.7cm, 1fr),
    align: (left, left),
    [*Kind*], [*Names you may hear*],
    [Category — not one model], [Decision Model · decision-native · Jev-like · 決策模型],
    [TypeSafe official], [TypeSafe (company) · System One Models (product family) · Jev (flagship, hosted)],
    [Interface — not a model], [`/v1/systemone` · `ollama.systemone` (pass a `model=`)],
    [Today's local model], [clef-flash (Cloudflare) — reads images; compatible, *not* Jev],
    [Other models], [Clef · Nimble · Tev1 · OpenJev · … — not Jev],
  )
]

// ---- Slide 6: System 1 vs System 2 ----

#slide("System 1 vs System 2")[
  #text(size: 9.6pt, fill: mute)[
    From Kahneman's *Thinking, Fast and Slow* (快思慢想). A Red Hat talk also used this framing for models.
  ]
  #v(0.08em)
  #set text(size: 9.6pt)
  #swiss-table(
    columns: (2.35cm, 1fr, 1fr),
    align: (left, left, left),
    [ ], [*System 1 — fast & automatic*], [*System 2 — slow & deliberate*],
    [Typical model], [Decision-native / decision model], [Common LLM],
    [What it does], [One pass → Multiple structured answers], [Writes token by token],
    [Output], [Choice / Noul / Score + confidence], [Full sentences, explanations, Reasonings],
  )
  #v(0.14em)
  #text(size: 9.5pt, fill: mute)[
    TypeSafe's product name *System One* is taken from this idea; it is not the name of every decision model.
  ]
]

// ---- Slide 7: What is video tagging ----

#slide("What Is Video Tagging?")[
  #bullet-list((
    [Looking at each frame (or a short clip) — an *image* — and attaching labels.],
    [Examples: a box around an object, a scene name ("kitchen", "street"), or keywords ("person", "car", "rain").],
    [The output is compact — tags, not paragraphs.],
    [A decision model can attach those tags as Choice / Noul / Score, with confidence.],
  ))
]

// ---- Slide 8: Why faster ----

#slide("Why Decision Models Are Faster")[
  #bullet-list((
    [System 1 style: one forward pass → structured answers plus confidence. Done.],
    [System 2 style (LLM): generates word by word → many steps → slower.],
    [We only need tags, not sentences — so a decision model fits better.],
    [Result: less compute, faster tagging, easier to run on a laptop.],
  ))
]

// ---- Slide 9: Comparison ----

#slide("Side-by-Side Comparison")[
  #set text(size: 10.4pt)
  #swiss-table(
    columns: (auto, 1fr, 1fr),
    align: (left, left, left),
    [ ], [*Decision Model*], [*LLM*],
    [Thinking style], [System 1 — fast & automatic], [System 2 — slow & deliberate],
    [Output], [Choice / Noul / Score], [Full sentences],
    [Certainty], [Confidence / probability], [Usually implicit in the text],
    [Speed], [Very fast (one pass)], [Slower (token by token)],
    [Good for], [Tagging, filtering, deciding], [Explanation, chat],
  )
]

// ---- Slide 10: Agenda ----

#slide("Agenda at a Glance")[
  #bullet-list((
    [10:00  Check-in  ·  10:15  Intro],
    [10:30  Intro talk → Hands-on Session 1 → Wrap-up],
    [13:00  Lunch  ·  15:00  Team sharing],
    [15:10  Hands-on Session 2 → Wrap-up],
    [17:45  Closing],
  ))
]

// ---- Slide 11: Setup 1 ----

#slide("Setup 1 — Install Ollama")[
  Ollama runs models locally on your machine.
  #v(0.5em)
  #code-block((
    "curl -fsSL https://ollama.com/install.sh | sh",
  ))
]

// ---- Slide 12: Setup 2 ----

#slide("Setup 2 — Get the Model")[
  About 12 GB. Give it time.
  #v(0.5em)
  #code-block((
    "ollama pull clef-flash",
  ))
]

// ---- Slide 13: Setup 3 ----

#slide("Setup 3 — Get the Repo")[
  #code-block((
    "git pull https://github.com/dev1virtuoso/Event.git",
    "cd path/to/Event/folder",
    "cd \"./HKPUG/PyCon 2026/Sprint\"",
  ))
]

// ---- Slide 14: Setup 4 ----

#slide("Setup 4 — Python Environment")[
  #code-block((
    "uv init",
    "uv add ollama pygame cv2",
  ))
]

// ---- Slide 15: Setup 5 ----

#slide("Setup 5 — Verify")[
  If it runs, your environment is ready.
  #v(0.5em)
  #code-block((
    "uv run snake_clef.py",
  ))
]

// ---- Slide 16: Hands-on 1 ----

#slide("Hands-on Session 1 (Morning)")[
  #bullet-list((
    [Goal: tag a few frames with the decision model.],
    [Focus: understand the input → output shape.],
  ))
]

// ---- Slide 17: Hands-on 2 ----

#slide("Hands-on Session 2 (Afternoon)")[
  #bullet-list((
    [Goal: tag a short video frame by frame.],
    [Focus: speed, batching, and simple keyword output.],
  ))
]

// ---- Slide 18: Wrap-up ----

#slide("Wrap-up & Questions")[
  #bullet-list((
    [Decision model = short answers (Choice / Noul / Score) plus confidence.],
    [Many names, one kind of idea — but not always the same model.],
    [From *Thinking, Fast and Slow* (快思慢想): decision-native ≈ System 1; common LLMs ≈ System 2.],
    [Today we run clef-flash locally — multimodal (at least image) — through a System One–compatible API. Questions welcome — thanks!],
  ))
]
