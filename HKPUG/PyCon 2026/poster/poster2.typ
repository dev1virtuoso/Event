#set page(
  paper: "a0",
  flipped: false,
  margin: (x: 2cm, y: 1.5cm),
  fill: rgb("#0D1117"),
)

#set text(
  font: ("Liberation Sans", "DejaVu Sans"),
  fill: rgb("#C9D1D9"),
  size: 15pt,
)

// Brand Colors Definition
#let python-blue = rgb("#387EB8")
#let python-yellow = rgb("#FFD43B")
#let card-bg = rgb("#161B22")
#let card-border = rgb("#30363D")
#let editor-bg = rgb("#0D1117")
#let code-text-color = rgb("#E6EDE3")
#let line-number-color = rgb("#6E7681")   // Gray for line numbers

// Custom Card UI Helper
#let poster-card(title: "", content) = {
  block(
    fill: card-bg,
    stroke: 1.3pt + card-border,
    inset: 1.1em,
    radius: 9pt,
    width: 100%,
    stack(
      spacing: 0.75em,
      if title != "" [
        #block(
          width: 100%,
          stroke: (bottom: 1pt + rgb("#21262D")),
          inset: (bottom: 0.45em),
          [
            #box(width: 6pt, height: 18pt, fill: python-blue, radius: 2pt)
            #h(8pt)
            #text(size: 1.2em, weight: "bold", fill: white, title)
          ]
        )
      ],
      content
    )
  )
}

// VS Code Window Component
#let vscode-window(filename: "", code-rows: (), status-tags: ()) = {
  block(
    fill: editor-bg,
    stroke: 1.3pt + card-border,
    radius: 7pt,
    width: 100%,
    clip: true,
    [
      // Title Bar
      #block(
        fill: card-bg,
        inset: (x: 0.9em, y: 0.45em),
        stroke: (bottom: 1pt + card-border),
        width: 100%,
        grid(
          columns: (auto, 1fr, auto),
          align: horizon,
          stack(
            dir: ltr,
            spacing: 6pt,
            circle(radius: 4.5pt, fill: rgb("#FF5F56")),
            circle(radius: 4.5pt, fill: rgb("#FFBD2E")),
            circle(radius: 4.5pt, fill: rgb("#27C93F")),
          ),
          align(center, text(size: 0.75em, font: "DejaVu Sans Mono", fill: rgb("#8B949E"), filename)),
          none
        )
      )
      
      // Editor Tab
      #block(
        fill: rgb("#010409"),
        inset: (x: 0.9em, y: 0.35em),
        stroke: (bottom: 1pt + card-border),
        [
          #box(
            fill: editor-bg,
            inset: (x: 0.75em, y: 0.35em),
            stroke: (top: 2pt + python-yellow, right: 1pt + card-border),
            [#text(weight: "bold", fill: python-blue, "Py") #h(4pt) #text(size: 0.75em, font: "DejaVu Sans Mono", fill: white, filename)]
          )
        ]
      )

      // Code Content
      #block(
        inset: 0.95em,
        stack(
          spacing: 0.75em,
          ..code-rows
        )
      )

      // Status Bar
      #block(
        fill: card-bg,
        inset: (x: 0.9em, y: 0.35em),
        stroke: (top: 1pt + card-border),
        width: 100%,
        stack(
          dir: ltr,
          spacing: 1.4em,
          ..status-tags.map(tag => [
            #box(circle(radius: 3pt, fill: rgb("#2EA043")))
            #h(4pt)
            #text(size: 0.65em, font: "DejaVu Sans Mono", fill: rgb("#8B949E"), tag)
          ])
        )
      )
    ]
  )
}

// ================= HEADER SECTION =================
#grid(
  columns: (240pt, 1fr, auto),
  align: horizon,
  rect(
    width: 200%,
    height: 200pt,
    fill: white,
    radius: 7pt,
    inset: 7pt,
    align(center + horizon, image("pyconlogo.svg", width: 85%))
  ),
  none,
  rect(
    fill: card-bg,
    stroke: 1pt + card-border,
    radius: 7pt,
    inset: 0.8em,
    [
      #set text(font: "DejaVu Sans Mono", size: 0.75em)
      #text(fill: rgb("#6A737D"), "// Built with Python 3.13.12 & PyTorch 2.14.1") \
      #text(fill: rgb("#6A737D"), "// Core architecture by ") #text(fill: python-yellow, "Carson Wu (@dev1virtuoso)")
    ]
  )
)

#v(0.8em)

// ================= HERO TITLE SECTION =================
#block(
  fill: card-bg,
  stroke: (left: 5pt + python-blue, rest: 1.3pt + card-border),
  radius: 9pt,
  inset: 1.3em,
  width: 100%,
  [
    #text(size: 1.75em, weight: "bold", fill: white)[Beyond Brute-Force: Embedding Lie Group Theory into PyTorch Layers for Spatial Perception]
    #v(0.3em)
    #text(size: 1.05em, weight: "medium", fill: python-blue)[L.E.P.A.U.T.E. Framework — Making 3D Spatial Computing Fast & Smooth in PyTorch]
    #v(0.55em)
    #box(
      fill: rgb(255, 212, 59, 20),
      stroke: 1pt + rgb(255, 212, 59, 60),
      radius: 14pt,
      inset: (x: 0.9em, y: 0.35em),
      text(font: "DejaVu Sans Mono", size: 0.75em, fill: python-yellow, "Developed by Carson Wu (@dev1virtuoso)")
    )
  ]
)

#v(0.55em)

// ================= SYSTEM ARCHITECTURE =================
// Proportional scaling – only width is constrained, height follows original aspect ratio
#poster-card(
  title: "System Architecture & Pipeline Flow",
  align(center, image("liblepaute Visual Odometry Architecture-2026-10-28-120513.png", width: 35%))
)

#v(0.55em)

// ================= PROBLEM & MOTIVATION =================
#poster-card(
  title: "Problem & Motivation: Why Lie Groups for Robust Spatial Perception?",
  [
    #par(justify: true, leading: 0.65em)[
      *The Problem:* Autonomous driving and FSD-style 3D spatial perception require robust reasoning about continuous rigid-body transformations (3D rotation, translation, depth, and scale). Standard CNNs treat images as flat 2D pixel grids without built-in geometric constraints. As a result, gradients distort rotation matrices, Euler angles suffer from discontinuities and gimbal lock, and models must implicitly rediscover physical laws from scratch using massive datasets.
    ]
    #v(0.4em)
    #par(justify: true, leading: 0.65em)[
      *Why Lie Groups?* We chose a *Lie Group ("SE"(3)) approach* because it provides the natural mathematical language for continuous rigid motions. By operating on smooth manifolds and their flat 6D tangent spaces ($frak("se")(3)$), all transformations remain strictly valid by design. This injects rigorous geometric equivariance ($f(g dot x) = g dot f(x)$) directly into PyTorch layers, enabling networks to learn physical residuals rather than raw geometry.
    ]
    #v(0.6em)

    #vscode-window(
      filename: "geometry.py — Lie Group Equivariant Exponential Mapping",
      code-rows: (
        text(size: 0.8em, fill: rgb("#8B949E"), [Mapping 6D Lie algebra tangent vectors ($frak("se")(3)$) into smooth $"SE"(3)$ manifold transformation matrices via closed-form exponential maps.]),
        block(
          fill: editor-bg,
          stroke: 1pt + rgb("#21262D"),
          inset: 0.75em,
          radius: 5pt,
          width: 100%,
          text(font: "DejaVu Sans Mono", size: 0.75em, fill: code-text-color)[
            #text(fill: line-number-color)[01] #text(fill: rgb("#FF7B72"))[def] #text(fill: rgb("#D2A8FF"))[se3_exp_map(xi: torch.Tensor) -> torch.Tensor:]: \
            #text(fill: line-number-color)[02]     #text(fill: rgb("#8B949E"), "# Extract 3D translation (rho) and 3D rotation (phi) from tangent vector") \
            #text(fill: line-number-color)[03]     rho, phi = xi[:, :3], xi[:, 3:] \
            #text(fill: line-number-color)[04]     theta = torch.sqrt(torch.clamp(torch.sum(phi \*\* 2, dim=1), min=1e-20)) \
            #text(fill: line-number-color)[05]     #text(fill: rgb("#8B949E"), "# Rodrigues closed-form formula mapping tangent twist to orthogonal SE(3) matrix") \
            #text(fill: line-number-color)[06]     #text(fill: rgb("#FF7B72"), "return") rodrigues_exp_with_taylor_fallback(rho, phi, theta)
          ]
        )
      ),
      status-tags: ("Geometric Equivariance", "Geometric Consistency Enforced")
    )
  ]
)

#v(0.55em)

// ================= ASYNCHRONOUS ARCHITECTURE =================
#poster-card(
  title: "Asynchronous Subsystem Architecture ('main.py' & 'InferenceWorker')",
  [
    #par(justify: true, leading: 0.65em)[
      *GIL Bypass via Process Isolation:* To prevent heavy neural network inference (`SE3ResidualRefiner` and `YOLOClassifier`) from stalling the real-time direct visual tracker, L.E.P.A.U.T.E. uses Python's `multiprocessing.get_context('spawn')` to isolate deep learning pipelines in a dedicated background worker process.
    ]
    #v(0.55em)
    #vscode-window(
      filename: "main.py — Multi-Process Non-Blocking Async Queue",
      code-rows: (
        block(
          fill: editor-bg,
          stroke: 1pt + rgb("#21262D"),
          inset: 0.75em,
          radius: 5pt,
          width: 100%,
          text(font: "DejaVu Sans Mono", size: 0.75em, fill: code-text-color)[
            #text(fill: line-number-color)[01] class InferenceWorker: \
            #text(fill: line-number-color)[02]     def \_\_init\_\_(self, config, model_path): \
            #text(fill: line-number-color)[03]         self.ctx = mp.get_context('spawn')  \# Bypasses Python GIL lock \
            #text(fill: line-number-color)[04]         self.job_queue = self.ctx.Queue(maxsize=5) \
            #text(fill: line-number-color)[05]         self.result_queue = self.ctx.Queue() \
            #text(fill: line-number-color)[06]         self.process = self.ctx.Process(target=self.\_\_run\_process, daemon=True) \
            #text(fill: line-number-color)[07]         self.process.start()
          ]
        ),
      ),
      status-tags: ("Zero Frame Drop", "IPC Shared Memory", "GIL Isolated")
    )
  ]
)

#v(0.55em)

// ================= BENCHMARKS & DEMO GRID =================
#grid(
  columns: (1fr, 1fr),
  gutter: 1.5cm,
  
  // Left Column: Benchmarks
  poster-card(
    title: "Performance Leap & Verified Benchmarks",
    stack(
      spacing: 0.75em,
      rect(
        width: 100%,
        fill: rgb(56, 126, 184, 25),
        stroke: 1.3pt + python-blue,
        radius: 7pt,
        inset: 0.95em,
        align(center)[
          #text(size: 0.7em, font: "DejaVu Sans Mono", fill: rgb("#8B949E"), "PIPELINE END-TO-END SPEED") \
          #v(0.2em)
          #text(size: 1.55em, weight: "bold", font: "DejaVu Sans Mono", fill: python-yellow, "≈ 0.6 FPS → 12 FPS")
        ]
      ),
      
      rect(
        width: 100%, fill: editor-bg, stroke: 1pt + card-border, radius: 5pt, inset: 0.75em,
        [
          #text(size: 0.8em, weight: "bold", font: "DejaVu Sans Mono", fill: python-blue, "Environment") \
          #v(0.25em)
          #text(size: 0.7em, font: "DejaVu Sans Mono", "macOS ARM64 | 12-core CPU | Python 3.13.12 | PyTorch 2.14.1")
        ]
      ),
      rect(
        width: 100%, fill: editor-bg, stroke: 1pt + card-border, radius: 5pt, inset: 0.75em,
        [
          #text(size: 0.8em, weight: "bold", font: "DejaVu Sans Mono", fill: python-blue, "Refiner Inference Latency") \
          #v(0.25em)
          #text(size: 0.7em, font: "DejaVu Sans Mono")[
            • CPU Mode: Mean 21.72 ms (≈ 46.03 FPS, Median: 21.42 ms) \
            • MPS GPU Mode: Mean 7.10 ms (≈ 140.82 FPS, Median: 6.29 ms)
          ]
        ]
      ),
      rect(
        width: 100%, fill: editor-bg, stroke: 1pt + card-border, radius: 5pt, inset: 0.75em,
        [
          #text(size: 0.8em, weight: "bold", font: "DejaVu Sans Mono", fill: python-blue, "Stability & Precision Metrics") \
          #v(0.25em)
          #text(size: 0.7em, font: "DejaVu Sans Mono")[
            • Geometric MSE: 0.0025 ~ 0.0027 \
            • Convergence Success Rate (MSE < 0.01): 100% (1.0) \
            • Boundary Protection: Near-zero & near-π angle errors < 10⁻¹²
          ]
        ]
      )
    )
  ),

  // Right Column: Real-time Demo Slot
  poster-card(
    title: "Real-time Demo & Perception Screenshots",
    rect(
      width: 100%,
      height: 400pt,
      stroke: (dash: "dashed", paint: python-blue),
      fill: rgb(13, 17, 23, 50),
      radius: 7pt,
      align(center, image("Screenshot 2026-10-28 at 20.57.01.png", width: 50%))
    )
  )
)

#v(0.55em)

// ================= MANIFOLD GEOMETRY CODE-MATH ALIGNMENT =================
#poster-card(
  title: "Manifold Geometry (geometry.py)",
  vscode-window(
    filename: "geometry.py — SE(3)/SO(3) Manifold Geometry",
    code-rows: (
      text(size: 0.8em, fill: rgb("#8B949E"), [Dedicated module handling *SE(3) / SO(3) Lie group mappings* with robust boundary protection (near-zero & near-π singularities).]),
      
      // Alignment Row 1
      grid(
        columns: (1.3fr, 38pt, 1fr),
        align: horizon,
        rect(
          width: 100%, fill: editor-bg, stroke: 1pt + rgb("#21262D"), inset: 0.7em, radius: 5pt,
          text(font: "DejaVu Sans Mono", size: 0.65em, fill: code-text-color)[
            #text(fill: line-number-color)[01] A = (R_pi + torch.eye(3, device=T.device...).unsqueeze(0)) \* 0.5 \
            #text(fill: line-number-color)[02] diag_A = torch.diagonal(A, dim1=-2, dim2=-1) \
            #text(fill: line-number-color)[03] max_idx = torch.argmax(diag_A, dim=-1) \
            #text(fill: line-number-color)[04] batch_idx = torch.arange(R_pi.shape[0], device=T.device) \
            #text(fill: line-number-color)[05] v = A[batch_idx, :, max_idx]
          ]
        ),
        align(center, text(fill: python-blue, size: 1.35em, "➔")),
        rect(
          width: 100%, fill: rgb(255, 212, 59, 10), stroke: (left: 2.5pt + python-yellow, rest: 1pt + card-border), inset: 0.7em, radius: 5pt,
          [
            #text(size: 0.75em, weight: "bold", fill: python-yellow, "1. Symmetric Matrix Projection") \
            #text(size: 0.75em, weight: "bold", fill: python-yellow, "2. Dominant Axis Selection via Diagonal") \
            #v(0.25em)
            #text(size: 0.7em, "Diagonal Dominance Selection: ") $A = (R + I) / 2$
          ]
        )
      ),

      // Alignment Row 2
      grid(
        columns: (1.3fr, 38pt, 1fr),
        align: horizon,
        rect(
          width: 100%, fill: editor-bg, stroke: 1pt + rgb("#21262D"), inset: 0.7em, radius: 5pt,
          text(font: "DejaVu Sans Mono", size: 0.65em, fill: code-text-color)[
            #text(fill: line-number-color)[06] v = v / torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=1e-10) \
            #text(fill: line-number-color)[07] sign = torch.sign(torch.sum(v \* phi_raw[mask_pi], dim=-1...)) \
            #text(fill: line-number-color)[08] sign[sign == 0] = 1.0 \
            #text(fill: line-number-color)[09] v = v \* sign
          ]
        ),
        align(center, text(fill: python-blue, size: 1.35em, "➔")),
        rect(
          width: 100%, fill: rgb(255, 212, 59, 10), stroke: (left: 2.5pt + python-yellow, rest: 1pt + card-border), inset: 0.7em, radius: 5pt,
          text(size: 0.75em, weight: "bold", fill: python-yellow, "Sign Resolution using Raw Skew-Symmetric Components")
        )
      ),

      // Alignment Row 3
      grid(
        columns: (1.3fr, 38pt, 1fr),
        align: horizon,
        rect(
          width: 100%, fill: editor-bg, stroke: 1pt + rgb("#21262D"), inset: 0.7em, radius: 5pt,
          text(font: "DejaVu Sans Mono", size: 0.65em, fill: code-text-color)[
            #text(fill: line-number-color)[10] mask_small = (theta < 1e-4) \
            #text(fill: line-number-color)[11] mask_pi = (theta > (torch.pi - 1e-3)) \
            #text(fill: line-number-color)[12] mask_large = ~(mask_small | mask_pi)
          ]
        ),
        align(center, text(fill: python-blue, size: 1.35em, "➔")),
        rect(
          width: 100%, fill: rgb(255, 212, 59, 10), stroke: (left: 2.5pt + python-yellow, rest: 1pt + card-border), inset: 0.7em, radius: 5pt,
          text(size: 0.75em, weight: "bold", fill: python-yellow, "Dynamic Angle Stratification for Numerical Stability")
        )
      )
    ),
    status-tags: ("Python 3.13.12", "PyTorch 2.14.1", "MPS / CUDA Ready")
  )
)

#v(0.55em)

// ================= FOOTER / QR CODE =================
#poster-card(
  title: "Repository & Code Files",
  grid(
    columns: (120pt, 1fr),
    gutter: 1.2em,
    align: horizon,
    rect(
      fill: white, radius: 7pt, inset: 7pt,
      image("hkpug_pycon_qr.svg", width: 100%)
    ),
    text(size: 0.95em)[
      Scan to explore modular code files, architecture scripts, and reproducible benchmarks on GitHub.
    ]
  )
)