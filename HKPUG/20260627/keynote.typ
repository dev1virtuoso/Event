#import "@preview/mmdr:0.2.2": mermaid

#set page(
  paper: "presentation-16-9",
  margin: (x: 2cm, top: 2.2cm, bottom: 1.8cm),
  fill: rgb("#fcfdfd"),
  header: context {
    let page_num = counter(page).get().first()
    if page_num > 1 {
      grid(
        columns: (1fr, auto),
        align(left + horizon)[
          #text(fill: rgb("#1e3a8a"), size: 0.85em, weight: "bold", tracking: 0.5pt)[
            L.E.P.A.U.T.E. Framework
          ]
        ],
        align(right + horizon)[
          #text(fill: rgb("#64748b"), size: 0.75em)[Hong Kong Python User Group (HKPUG)]
        ]
      )
      v(-0.3em)
      line(length: 100%, stroke: 0.5pt + rgb("#cbd5e1"))
    }
  },
  footer: context {
    let page_num = counter(page).get().first()
    let total_pages = counter(page).final().first()
    if page_num > 1 {
      line(length: 100%, stroke: 0.5pt + rgb("#cbd5e1"))
      v(-0.5em)
      grid(
        columns: (1fr, auto),
        text(fill: rgb("#94a3b8"), size: 0.7em)[June 27, 2026],
        text(fill: rgb("#1e3a8a"), size: 0.75em, weight: "bold")[#page_num / #total_pages]
      )
    }
  }
)

#set text(
  font: "Liberation Sans",
  size: 20pt,
  fill: rgb("#334155")
)

#show heading: set text(fill: rgb("#1e3a8a"))
#show raw: set text(font: "Fira Code", size: 0.85em)
#show raw.where(block: true): it => block(
  fill: rgb("#0f172a"),
  inset: 12pt,
  radius: 6pt,
  width: 100%,
  text(fill: rgb("#f8fafc"), it)
)

#let slide(title: "", body) = {
  pagebreak(weak: true)
  if title != "" {
    v(0.1em)
    text(size: 1.4em, weight: "bold", fill: rgb("#0f172a"))[#title]
    v(0.6em)
  }
  body
}

#page[
  #align(center + horizon)[
    #rect(fill: rgb("#1e3a8a"), radius: 4pt, inset: 12pt)[
      #text(fill: white, size: 1.1em, weight: "bold", tracking: 1.5pt)[HKPUG Meetup | June 2026]
    ]
    #v(0.8em)
    #text(size: 1.8em, weight: "bold", fill: rgb("#0f172a"))[
      Building a Spatial Vision Framework Completely Alone \ Handcrafting L.E.P.A.U.T.E. Geometry & Asynchronous Code
    ]
    #v(0.4em)
    #text(size: 1.2em, style: "italic", fill: rgb("#2563eb"))[A solo developer's journey hacking simulators, pipeline architecture, and multithreading locks]
    
    #v(2.5em)
    #grid(
      columns: (1fr, 1fr),
      align(left)[
        #text(size: 0.85em, weight: "semibold")[Speaker: Carson Wu] \
        #text(size: 0.7em, fill: rgb("#64748b"))[CV & NLP Solo Developer]
      ],
      align(right)[
        #text(size: 0.85em, weight: "semibold")[Hong Kong Python User Group] \
        #text(size: 0.7em, fill: rgb("#64748b"))[Personal Project Presentation]
      ]
    )
  ]
]

#slide(title: "Why did I start this? (I hate brute-forcing data)")[
  #text(weight: "medium")[Most autonomous vision frameworks are massively heavy. As a solo dev, I don't have a massive cluster to train giant models on millions of miles, so I thought: *Can we use geometric laws to force the network to take a shortcut?*]

  #v(0.5em)
  *My core design principles to make this possible:*
  - *Physics instead of memory:* Map continuous sequence frames directly onto $S E(3)$ Lie groups, so spatial awareness is baked into the math.
  - *Enforcing consistency:* Ensure that when the vehicle pitches or rolls in the simulator, the extracted features don't lose their spatial orientation.
  - *Stitching new tools together:* Along with optical flow, I threw in Transformer structural embeddings and a Zero-shot Open-Vocabulary classifier to see how far I could push it.
]

#slide(title: "Solo Architecture: Keeping things alive by breaking threads")[
  Full disclosure: *This framework hasn't touched a real vehicle yet.* But testing on my desktop simulator setup taught me early on that mixing ingestion with inference breaks everything immediately.

  #v(0.3em)
  - *Isolated Ingestion Ring:* Image capture runs on its own isolated loop. No matter how much the downstream models lag, the camera/simulator stream never drops a frame.
  - *Asynchronous Inference Pool:* Heavyweights like Depth-Anything and SigLIP are thrown into separate worker threads to crunch data at their own pace.
  - *Zero-Overhead Persistence:* Atomic operations dump telemetry directly into caches and local JSON files so I can review exactly where my tracking drifted post-run.
]

#slide(title: "How the Pipeline Loops in the Background")[
  #text(size: 0.8em)[This is the asynchronous topology running locally on my development machine:]
  
  #v(0.2em)
  #align(center)[
    #mermaid("
    flowchart TD
    %% Default nodes: Neutral light gray with dark gray text
    classDef default fill:#f1f5f9,stroke:#cbd5e1,stroke-width:1px,color:#334155;
      
    A[CameraIOStream / Mock] --> B[InferenceWorker Queue]
    B --> C[InferenceWorker Thread Pipeline]
    C --> D[\"DenseSE3Tracker\\nRAFT + Depth-Anything\"]
    D --> E[PnP RANSAC Solver]
    E --> F[SigLIP + Vision Transformer]
    F --> G[Pose Filtering & Fusion]
    G --> H[Main Control Stream]
    G --> I[DiskManager Cache]
    I --> J[Persistent JSON DB]
      
    %% Highlight Pipeline Engine: Strong Slate Blue
    style C fill:#e0f2fe,stroke:#0284c7,stroke-width:2px,color:#0369a1
      
    %% Subtle Storage Layer: Muted Steel Blue
    style I fill:#f8fafc,stroke:#64748b,stroke-width:1px,color:#475569
    style J fill:#f8fafc,stroke:#64748b,stroke-width:1px,color:#475569
  ",
  base-theme: "neutral"
)
  ]
]

#slide(title: "Hardcore Geometry: Why bother with SE(3)?")[
  Without geometric constraints, if your vehicle hits a massive bump in a virtual world and shakes violently, the model completely loses its bearings. I wanted my code to have an innate "sense of balance."

  #v(0.3em)
  #rect(fill: rgb("#f1f5f9"), inset: 15pt, radius: 6pt, width: 100%)[
    *Lie Algebra Spatial Mapping:*
    
    Sequential updates can be translated directly into explicit Lie vectors:
    $ xi = (v, omega) in frak("se")(3) $
    
    When a movement warps the input sequence by some transformation $g in S E(3)$, the feature extractor $f(x)$ naturally flows with it:
    
    $ f(g dot x) = g dot f(x) $
  ]
  #v(0.3em)
  - *The real-world benefit:* Forcing this mathematical constraint straight into the tensor layers gives the system geometric memory, keeping tracking stable even during erratic movement.
]

#slide(title: "The Joy of Custom Code: Differentiable Warping & Rings")[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      *1. Blind Track Tracking*
      - I wrote custom geometric differentiable layers to warp feature maps dynamically based on estimated physical transforms.
      - Even without any simulated GPS or pre-mapped environments, the system extracts a clean 6-DoF trajectory purely out of pixel changes.
    ],
    [
      *2. Squeezing Performance with Ring Buffers*
      - Running on a single GPU means inference speeds struggle to match raw video framerates.
      - I built a Thread Ring protected by atomic parameter locks, discarding old frames seamlessly so the control loop only reads the latest state.
    ]
  )
]

#slide(title: "Open-Vocabulary and Multi-Tasking (Where I lost my hair)")[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      *Handling Unexpected Obstacles*
      - I plugged SigLIP's zero-shot classification embeddings straight into the perception pipeline.
      - If an odd object—like a strange cardboard box or fallen debris—appears in the simulator, I don't have to retrain. I just pass a new label at runtime to detect it.
    ],
    [
      *The Balancing Act of Multi-Loss*
      - This was my absolute biggest pain point: ego-pose tracking requires Huber-loss regressions, but visual parsing needs $N T-X e n t$ contrastive loss.
      - Balancing these two entirely different objectives alone felt like walking a tightrope—but after countless nights, they finally converged together.
    ]
  )
]

#slide(title: "Defensive Programming: Handling Failures at Home")[
  #v(0.2em)
  - *Sim/Sensor Disconnect Recovery:* If a webcam drops or the simulator's webhook hits a lag spike, the ingestion layer automatically flips to a mathematical parametric generator. It extrapolates frames using the last known velocity to keep the system from throwing an exception and crashing.
  - *Hard Drive Protection:* Long testing sessions create massive amounts of telemetry. The `DiskManager` acts as a circular cache, holding onto the last 2,000 files and overwriting the rest so I don't wake up to a filled hard drive.
]

#slide(title: "What This Taught Me in Virtual Worlds")[
  Even though this code hasn't seen a single mile of asphalt yet, the results on my testbench show massive potential for solo developers:

  #v(0.5em)
  - *Math scales better than data:* It proves you don't need a massive team throwing datasets at a wall. Strict geometric priors allow lightweight setups to excel in spatial mapping.
  - *Isolation brings stability:* Separating the threads makes the pipeline incredibly robust against latency spikes and hardware stuttering.
  - *Dynamic adaptability:* Open-vocabulary setups make it trivial to introduce new object classes on the fly without running heavy training loops.
]

#slide(title: "How to Run It: Runtime Configurations")[
  When deploying this on my local PC or a standalone testboard, I use simple environment flags to keep configuration overhead to zero:
  ```bash
  export LEPAUTE_DEVICE="cuda"
  export LEPAUTE_MAX_DISK_FILES=2000
  export LEPAUTE_MODE="autonomous"
```

If you want to look at the raw matrix values spitting out of the geometric layers in real time, simply drop the logging level down to Debug.
]

#slide(title: "Code Execution: Three Lines to Fire Up the Engine")[
The underlying multithreading code and C++ geometric wrappers took months to figure out, but the final Python API is dead simple:

```python
import logging
from lepaute.core import CameraIOStream, InferenceWorker

logging.basicConfig(level=logging.DEBUG)

stream = CameraIOStream(device_index=0, mode="autonomous")

worker = InferenceWorker(source=stream)
worker.start()
```

]

#slide[
#align(center + horizon)[
#text(size: 2.4em, weight: "bold", fill: rgb("#1e3a8a"))[Q&A]

]
]

#slide[
#align(center + horizon)[
#text(size: 2.2em, weight: "bold", fill: rgb("#1e3a8a"))[Thank You!]
]

]