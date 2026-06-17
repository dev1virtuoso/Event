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
          #text(
            fill: rgb("#1e3a8a"),
            size: 0.85em,
            weight: "bold",
            tracking: 0.5pt,
          )[
            L.E.P.A.U.T.E. Framework
          ]
        ],
        align(right + horizon)[
          #text(
            fill: rgb("#64748b"),
            size: 0.75em,
          )[Hong Kong Python User Group (HKPUG)]
        ],
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
        text(
          fill: rgb("#1e3a8a"),
          size: 0.75em,
          weight: "bold",
        )[#page_num / #total_pages],
      )
    }
  },
)

#set text(
  font: "Liberation Sans",
  size: 20pt,
  fill: rgb("#334155"),
)

#show heading: set text(fill: rgb("#1e3a8a"))
#show raw: set text(font: "Fira Code", size: 0.85em)
#show raw.where(block: true): it => block(
  fill: rgb("#0f172a"),
  inset: 12pt,
  radius: 6pt,
  width: 100%,
  text(fill: rgb("#f8fafc"), it),
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
      #text(
        fill: white,
        size: 1.1em,
        weight: "bold",
        tracking: 1.5pt,
      )[HKPUG Meetup | June 2026]
    ]
    #v(0.8em)
    #text(size: 1.8em, weight: "bold", fill: rgb("#0f172a"))[
      Building a Spatial Vision Framework Completely Alone \ Handcrafting L.E.P.A.U.T.E. Geometry & Asynchronous Code
    ]
    #v(0.4em)
    #text(
      size: 1.2em,
      style: "italic",
      fill: rgb("#2563eb"),
    )[A solo developer's journey hacking simulators, pipeline architecture, and Lie groups]

    #v(2.5em)
    #grid(
      columns: (1fr, 1fr),
      align(left)[
        #text(size: 0.85em, weight: "semibold")[Speaker: Carson Wu] \
        #text(size: 0.7em, fill: rgb("#64748b"))[CV & NLP Developer]
      ],
      align(right)[
        #text(size: 0.85em, weight: "semibold")[Hong Kong Python User Group] \
        #text(size: 0.7em, fill: rgb("#64748b"))[27 June 2026]
      ],
    )
  ]
]

#slide(title: "Speaker Information & Project Repository")[
  #v(1em)
  #align(center)[
    #grid(
      columns: (1fr, 1fr),
      gutter: 2cm,
      [
        #align(center)[
          #image("contact.png", width: 220pt, height: 220pt)
          #v(0.5em)
          #text(size: 0.9em, weight: "medium")[Scan to Connect]
        ]
      ],
      [
        #align(center)[
          #image("repo.png", width: 220pt, height: 220pt)
          #v(0.5em)
          #text(size: 0.9em, weight: "medium")[L.E.P.A.U.T.E. Repository]
        ]
      ],
    )
  ]
]

#slide(title: "Part I: What is a Lie Group?")[
  Let's address the elephant in the room. Why do mathematicians use fancy words like *Lie Groups* when they just mean moving things around?

  #v(0.5em)
  - *The Intuition:* A Lie group is a group that is also a *smooth manifold*. It represents a continuous space of symmetries or transformations.
  - *Our Core Focus:* For spatial vision, we care about $ upright(S E)(3) $ (Special Euclidean Group). It encompasses all continuous 3D rotations and translations.
  - *The Real-World Reality:* When a car bumps, pitches, or rolls, its change in camera perspective is a continuous journey along this smooth geometric surface.
]

#slide(title: "Why Normal Neural Networks Hate Raw Matrices")[
  Why can't we just dump a standard $4 times 4$ transformation matrix into a linear layer and let the network figure it out?

  #v(0.3em)
  #rect(fill: rgb("#fee2e2"), inset: 12pt, radius: 6pt, width: 100%)[
    #text(fill: rgb("#991b1b"), weight: "bold")[The Non-Euclidean Trap:] \
    If your neural network updates a $4 times 4$ matrix via standard backpropagation, it will quickly distort the upper-left $3 times 3$ matrix. Suddenly, your rotation matrix scales or shears space. Your virtual camera is now an abstract painting.
  ]

  #v(0.3em)
  - *Euler Angles:* Suffer from *Gimbal Lock* and wrap-around discontinuities ($180 upright(°)$ suddenly flipping to $-180 upright(°)$).
  - *The Mathematical Cure:* We need a way to parameterize coordinates linearly without ever escaping the geometric boundaries of true physical space.
]

#slide(title: "The Tangent Space: Introducing Lie Algebra")[
  Since deep learning engines love flat, linear vector coordinates, we map the curved manifold surface onto its local flat tangent space at the Identity matrix.

  #v(0.5em)
  - *The Lie Algebra $ frak(s e)(3) $:* This acts as our local flat vector space.
  - Any 3D velocity or relative twist can be fully represented as a simple 6-element vector:
    $ xi = (v, omega) in frak(s e)(3) $
  - *The Bridge:* We map back and forth seamlessly using exact calculus conversions:
    - *Exponential Map* (`se3_exp_map`): Maps a flat 6D vector up onto the true curved 3D transformation manifold.
    - *Logarithmic Map* (`se3_log_map`): Flattens a 3D transformation matrix down into 6 linear numbers.
]

#slide(title: "Visualizing the Exponential and Logarithmic Maps")[
  #v(1em)
  #align(center)[
    #mermaid(
      "
    graph LR
      A[Lie Algebra Vector space] -- Exponential Map / se3_exp_map --> B((Curved Lie Group Manifold))
      B -- Logarithmic Map / se3_log_map --> A

      style A fill:#e0f2fe,stroke:#0284c7,stroke-width:2px
      style B fill:#fef08a,stroke:#ca8a04,stroke-width:2px
    ",
      base-theme: "neutral",
    )
  ]
  #v(1em)
  By keeping our parameters in $ frak(s e)(3) $ during neural processing, our model can optimize relative pose changes smoothly using standard optimization algorithms without ever corrupting the intrinsic laws of geometry.
]

#slide(title: "Part II: L.E.P.A.U.T.E. Core Technical Architecture")[
  #text(
    weight: "medium",
  )[Most autonomous vision frameworks are massively heavy. As a solo dev, I don't have a massive cluster to train giant models on millions of miles, so I thought: *Can we use geometric laws to force the network to take a shortcut?*]

  #v(0.5em)
  *My core design principles to make this possible:*
  - *Physics instead of memory:* Map continuous sequence frames directly onto $ upright(S E)(3) $ Lie groups, so spatial awareness is baked into the math.
  - *Enforcing consistency:* Ensure that when the vehicle pitches or rolls in the simulator, the extracted features don't lose their spatial orientation.
  - *Stitching new tools together:* Along with optical flow, I threw in Transformer structural embeddings and a Zero-shot Open-Vocabulary classifier to see how far I could push it.
]

#slide(title: "Solo Architecture: Keeping things alive by breaking threads")[
  Full disclosure: *This framework hasn't touched a real vehicle yet.* But testing on my desktop simulator setup taught me early on that mixing ingestion with inference breaks everything immediately.

  #v(0.3em)
  - *Isolated Ingestion Ring:* Image capture runs on its own isolated loop. No matter how much the downstream models lag, the camera/simulator stream never drops a frame.
  - *Asynchronous Inference Pool:* Heavyweights like Depth-Anything and SigLIP are thrown into separate worker threads to crunch data at their own pace.
  - *Zero-Overhead Persistence:* Atomic operations dump telemetry directly into caches and local SQLite files so I can review exactly where my tracking drifted post-run.
]

#slide(title: "How the Pipeline Loops in the Background")[
  #text(
    size: 0.8em,
  )[This is the asynchronous topology running locally on my development machine:]

  #v(0.2em)
  #align(center)[
    #mermaid(
      "
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
    G --> J[Persistent DB Layer]

    %% Highlight Pipeline Engine: Strong Slate Blue
    style C fill:#e0f2fe,stroke:#0284c7,stroke-width:2px,color:#0369a1

    %% Subtle Storage Layer: Muted Steel Blue
    style I fill:#f8fafc,stroke:#64748b,stroke-width:1px,color:#475569
    style J fill:#f8fafc,stroke:#64748b,stroke-width:1px,color:#475569
  ",
      base-theme: "neutral",
    )
  ]
]

#slide(title: "Hardcore Geometry: Why bother with SE(3)?")[
  Without geometric constraints, if your vehicle hits a massive bump in a virtual world and shakes violently, the model completely loses its bearings. I wanted my code to have an innate "sense of balance."

  #v(0.3em)
  #rect(fill: rgb("#f1f5f9"), inset: 15pt, radius: 6pt, width: 100%)[
    *Lie Algebra Equivariance Constraint:*

    When a physical movement warps the input image sequence by some transformation $ g in upright(S E)(3) $, the feature extractor $f(x)$ naturally flows equivariantly with it:

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
      *1. Differentiable Warping Layers*
      - Implemented inside `module.py` as `MonocularSE3Warping`.
      - It projects pixels into 3D coordinates using depth scale priors, applies the $ frak(s e)(3) $ twist, and samples the next frame via a fully differentiable `grid_sample` operator.
      - Enables direct gradient propagation from pixel discrepancies back into estimated geometric motion.
    ],
    [
      *2. Squeezing Latency with Buffers*
      - Designed an `InferenceWorker` queue thread loop inside `main.py`.
      - It uses thread-safe locks to isolate inference from capture loops.
      - Discards obsolete frames seamlessly under high loads, ensuring the main trajectory fusion system always reads the fresh spatial updates.
    ],
  )
]

#slide(title: "Open-Vocabulary and Multi-Tasking (Where I lost my hair)")[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      *Handling Unexpected Obstacles*
      - I plugged SigLIP's zero-shot classification embeddings straight into the perception pipeline inside `module.py`.
      - If an odd object, like a strange cardboard box or fallen debris, appears in the simulator, I don't have to retrain. I just pass a new text label string at runtime to evaluate its geometric scale prior.
    ],
    [
      *The Balancing Act of Multi-Loss*
      - This was my absolute biggest pain point: ego-pose tracking requires heteroscedastic uncertainty parameter optimization.
      - Balancing the visual spatial reconstruction against regression variance felt like walking a tightrope, but after countless nights, they finally converged together.
    ],
  )
]

#slide(title: "Defensive Programming: Handling Failures at Home")[
  #v(0.2em)
  - *Sensor Disconnect Recovery:* If a webcam drops or the simulator's webhook hits a lag spike, the ingestion layer seamlessly invokes `ManifoldKinematicForecaster`. It extrapolates motion using median historic velocity history entirely on the $ frak(s e)(3) $ manifold to prevent system execution crashes.
  - *Database & Memory Protection:* Long testing sessions generate massive telemetry tracking pairs. The pipeline leverages asynchronous SQLite WAL logging via `SequenceDataCollector` to write state packets without starving processing cycles.
]

#slide(title: "What This Taught Me in Virtual Worlds")[
  Even though this code hasn't seen a single mile of asphalt yet, the results on my testbench show massive potential for solo developers:

  #v(0.5em)
  - *Math scales better than data:* It proves you don't need a massive team throwing datasets at a wall. Strict geometric priors allow lightweight setups to excel in spatial mapping.
  - *Isolation brings stability:* Separating the threads makes the pipeline incredibly robust against latency spikes and hardware stuttering.
  - *Dynamic adaptability:* Open-vocabulary setups make it trivial to introduce new object classes on the fly without running heavy training loops.
]

#slide(title: "How to Run It: Runtime Configurations")[
  Dependencies are isolated minimal requirements listed cleanly in `requirements.txt`. When deploying this on my local PC or a standalone testboard, I use simple environment flags to keep configuration overhead to zero:

  ```bash
    export LEPAUTE_DEVICE="cuda"
    export LEPAUTE_MAX_DISK_FILES=2000
    export LEPAUTE_MODE="autonomous"

  ```

  If you want to look at the raw matrix values spitting out of the geometric layers in real time, simply drop the logging level down to Debug.
]

#slide(title: "Code Execution: Three Lines to Fire Up the Engine")[
  The underlying multithreading code and C++ geometric wrappers took months to figure out, but the final Python API exported by `__init__.py` is dead simple:

  ```python
  import logging
  from main import run_pipeline
  from module import LepauteConfig, DisplayMode

  logging.basicConfig(level=logging.INFO)

  config = LepauteConfig(device="cuda")
  processed_data = run_pipeline(
      config=config,
      display_mode=DisplayMode.JSON,
      mock=True
  )

  ```

  Offline training and data formatting routines can be easily evaluated using `convert_bop_to_lepaute.py` and `example.py`. Verification checks can be systematically run via `test_module.py`.
]

#slide[
  #align(center + horizon)[
    #text(size: 2.8em, weight: "bold", fill: rgb("#1e3a8a"))[Live Demo]

    #v(1.2em)
    #text(size: 1.4em, fill: rgb("#334155"))[
      Real-time Spatial Vision Pipeline
    ]

    #v(2em)
    #text(size: 1.1em, fill: rgb("#64748b"))[
      Watch the L.E.P.A.U.T.E. framework running live \
      with simulator input and SE(3) pose tracking
    ]
  ]
]

#slide[
  #align(center + horizon)[
    #text(size: 2.4em, weight: "bold", fill: rgb("#1e3a8a"))[Q&A]

    #v(0.5em)
    #text(
      size: 1.1em,
      fill: rgb("#64748b"),
    )[Feel free to ask about the math, architecture, or threading failures!]

  ]
]

#slide(title: "Share This Keynote Presentation")[
  #v(1em)
  #align(center)[
    #image("keynote.png", width: 220pt, height: 220pt)
    #v(0.8em)
    #text(
      size: 1em,
      weight: "medium",
      fill: rgb("#334155"),
    )[Scan to download keynote and abstract materials]
  ]
]

#slide[
  #align(center + horizon)[
    #text(size: 2.2em, weight: "bold", fill: rgb("#1e3a8a"))[Thank You!]

  ]
]
