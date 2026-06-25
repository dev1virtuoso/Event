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
      L.E.P.A.U.T.E. Framework: FSD-style Visual Perception with Lie Group Equivariant Attention
    ]
    #v(0.4em)
    #text(
      size: 1.2em,
      style: "italic",
      fill: rgb("#2563eb"),
    )[Building robust spatial understanding in Python]

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

#slide(title: "Speaker & Project")[
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
  #v(1em)
  *Note:* This is an independent technical project. No affiliation with or endorsement by Tesla.
]

#slide(title: "The Challenge of Spatial Perception")[
  Autonomous driving perception (FSD-style) requires more than 2D object detection.

  - Understand *rotation*, *translation*, and *viewpoint changes*.
  - Maintain consistent scene understanding despite camera motion.
  - Reason about *scale*, *depth*, and *relative pose*.

  Standard CNNs struggle here because they treat images as flat pixel grids without built-in geometry.
]

#slide(title: "Why Lie Groups?")[
  Lie Groups provide the mathematical language for continuous rigid motions.

  - *SE(3)*: Special Euclidean group in 3D: rotations + translations.
  - Smooth manifold + group structure.
  - *Lie Algebra* $ frak("se")(3) $: Flat 6D tangent space (3 translation + 3 rotation twist).

  Key advantage: All operations stay geometrically valid. No invalid poses.
]

#slide(title: "Core Mathematical Tools")[
  #text(weight: "medium")[Implemented in `module.py`]

  - *Skew-symmetric matrix*: $ [omega]_times = $ `skew_symmetric(omega)`
  - *Exponential Map*: $ exp: frak("se")(3) -> upright("SE")(3) $ (`se3_exp_map`)
    $ T = mat(R, t; 0, 1) $
    Closed-form using Rodrigues formula for rotation and translation.
  - *Logarithmic Map*: $ log: upright("SE")(3) -> frak("se")(3) $ (`se3_log_map`)
  - *Composition*: $ T_1 circle T_2 = T_1 T_2 $ (`compose_poses`)

  These maps are fully differentiable -> end-to-end training.
]

#slide(title: "Lie Groups vs. Standard CNNs")[
  Standard CNNs on raw 4x4 matrices or Euler angles fail because:

  - Gradients distort rotation matrices (lose orthogonality).
  - Gimbal lock and discontinuities in Euler angles.
  - Must learn geometry implicitly from millions of examples ("raw materials").

  *Why Lie Groups win:*
  - Enforce geometric constraints by design (equivariance: $f(g \cdot x) = g \cdot f(x)$).
  - Work with far less data.
  - Gradients respect manifold structure.
  - Robust under viewpoint changes, rotations, and translations.
]

#slide(title: "Why CNNs Need Massive Raw Data")[
  A plain CNN predicting poses must rediscover physics from scratch:

  - How pixels change under rotation/translation.
  - Parallax effects from depth.
  - Rigidity and continuity constraints.

  This requires enormous diverse datasets (lighting, textures, motions, viewpoints) to avoid overfitting and physically impossible outputs.

  Lie groups inject these rules mathematically. The network learns *residuals* only, not the entire geometry -> better generalization with smaller datasets.
]

#slide(title: "L.E.P.A.U.T.E. Architecture")[
  Lightweight solo-developed framework combining:

  - Lie group geometry.
  - Asynchronous multi-threaded pipeline.
  - Hybrid tracking (direct + neural + kinematic recovery).
  - Zero-shot open-vocabulary classification (SigLIP).
]

#slide(title: "Asynchronous Pipeline Design")[
  Mixing heavy inference with capture causes collapse under load.

  - *CameraIOStream*: Isolated ingestion (real camera or mock).
  - *InferenceWorker*: Dedicated thread + bounded queue + lock-protected history buffer.
  - *ManifoldKinematicForecaster*: SE(3) velocity-based recovery when neural results lag.
  - *SequenceDataCollector*: Async SQLite WAL logging.

  This concurrent design keeps the system stable.
]

#slide(title: "Hybrid Tracking Pipeline")[
  #mermaid(
    "
    flowchart TD
      A[CameraIOStream] --> B[MonocularDirectTracker (Gauss-Newton)]
      B --> C{Track Score > 0.1?}
      C -->|Yes| D[InferenceWorker (SigLIP + SE3ResidualRefiner)]
      C -->|No| E[ManifoldKinematicForecaster]
      D --> F[State Fusion]
      E --> F
      F --> G[Global SE(3) Pose via exp/log maps]
  ",
  )

  Differentiable SE(3) warping (`MonocularSE3Warping`) enables direct photometric gradients.
]

#slide(title: "Equivariant Attention & Residual Refiner")[
  - `SE3ResidualRefiner`: Learns corrections in Lie algebra space.
  - Heteroscedastic loss: $ 0.5 exp(-"unc") thin ||Delta xi||^2 + 0.5 "unc" $
  - Equivariant design ensures consistent understanding under transformations.
]

#slide(title: "Open-Vocabulary & Training")[
  - SigLIP for runtime object labels and scale priors.
  - Dataset from BOP (YCB-V) via relative SE(3) computation.
  - Training script: `train.py` with PyTorch compile support.
]

#slide(title: "Defensive Design & Lessons")[
  - Geometry provides robustness with less data.
  - Async isolation prevents pipeline collapse.
  - Sensor dropout -> manifold forecasting.
  - Memory safety via tensor detachment and WAL logging.
]


#slide[
  #align(center + horizon)[
    #text(
      size: 2.8em,
      weight: "bold",
      fill: rgb("#1e3a8a"),
    )[Hands-On and Live Demo]

    #v(1.2em)
    #text(size: 1.4em, fill: rgb("#334155"))[
      Real-time SE(3) Spatial Vision Pipeline
    ]
  ]
]

#slide[
  #align(center + horizon)[
    #text(size: 2.4em, weight: "bold", fill: rgb("#1e3a8a"))[Q&A]
    #v(1em)
    Questions welcome on math, architecture, or implementation details.
  ]
]

#slide[
  #align(center + horizon)[
    #text(size: 2.2em, weight: "bold", fill: rgb("#1e3a8a"))[Thank You!]
  ]
]
