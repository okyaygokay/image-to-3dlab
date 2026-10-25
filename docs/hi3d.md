# Hi3D 1.0 — refinement, not generation

Hi3D takes a **picture and a coarse mesh of the same object** and samples a far finer mesh
from the pair. It is not an image-to-mesh model, and picking it expecting one leads to the
wrong result about what is wrong when a model comes out crude: the coarse mesh was.

So a full run is two stages:

```
picture ──> Hunyuan3D-2 mini turbo ──> coarse mesh ──> Hi3D ──> refined GLB
            (or --coarse-mesh, any triangle mesh in a roughly unit-scale frame)
```

The studio and the CLI both run the pair, so it still looks like one route.

## Setup

```bash
python scripts/bootstrap_hi3d.py --check    # exit 0 when ready, installs nothing
python scripts/bootstrap_hi3d.py            # names what it will fetch, then asks
```

Linux with an NVIDIA card, **compute capability 8.0 or newer** (Ampere onwards). There is
no CPU path and no Apple Silicon path. About 14.7 GB of weights, in its own venv under
`vendor/hi3d/`.

## Running it

```bash
vendor/hi3d/.venv/bin/python scripts/hi3d_generate.py in.png out.glb
vendor/hi3d/.venv/bin/python scripts/hi3d_generate.py in.png out.glb --resolution 1536
vendor/hi3d/.venv/bin/python scripts/hi3d_generate.py in.png out.glb --coarse-mesh c.glb
vendor/hi3d/.venv/bin/python scripts/hi3d_generate.py --check
```

Or through the shared front doors:

```bash
python pipeline.py --hi3d in.png --resolution 1024
./lab            # Create > Advanced: 3D engine > Hi3D
```

| Flag | Default | What it does |
|---|---|---|
| `--resolution` | 1024 | Sparse extraction grid: 512 (one pass), 1024 (two), 1536 (three) |
| `--steps` | 25 | Scheduler updates per pass |
| `--coarse-steps` | 30 | Hunyuan3D-2 mini sampler steps; 30 suits the turbo variant |
| `--coarse-octree` | 256 | Hunyuan's extraction octree. Hi3D re-voxelises to 64 anyway |
| `--coarse-mesh` | — | Reuse a coarse mesh and skip the coarse stage entirely |

`--resolution` is the **sparse extraction grid**, not a dense voxel count. A 1536 run does
not produce a 1536³ tensor; it walks three refinement passes, each extracting at a finer
grid than the last.

## The pinned stack, and why it is not Hi3D's

Hi3D documents Python 3.10 / torch 2.5.0+cu124 / spconv-cu120. **That combination has no
kernels for compute capability 12.0**, which is every RTX 50-series card and the RTX PRO
6000 (the Colab G4). On those cards it cannot run at all — the failure is inside a sparse
convolution or flash-attn, long after the download.

So this lab pins a stack that does run there:

| Component | Hi3D's docs | Here | Why |
|---|---|---|---|
| torch | 2.5.0+cu124 | **2.9.1+cu128** | cu124 has no sm_120 kernels |
| torchvision | 0.20.0+cu124 | **0.24.1+cu128** | matches torch |
| spconv | spconv-cu120 2.3.6 | **spconv-cu126 2.3.8** | the cu120 build has no Blackwell kernels |
| flash-attn | 2.7.4.post1 | **2.7.4.post1** (prebuilt) | same tag; built for torch 2.9.1/cp310/sm_120 |
| transformers | 4.46.3 | 4.46.3 | unchanged |
| numpy | 1.26.4 | 1.26.4 | Hi3D requires `<2` |
| Python | 3.10.12 | 3.10.22 | same minor |

Everything else is pinned to the exact version in Hi3D's verified stack. The flash-attn
wheel is fetched prebuilt because compiling it takes about an hour and needs the CUDA
toolkit; it is addressed by HF **dataset commit and sha256**, never by `main`.

**What was verified:** the pinned stack installs and imports on a live RTX PRO 6000
(sm_120), and a full image → coarse → refine → GLB run completes, producing a
1,000,000-triangle glTF 2.0 mesh. Not verified: the 1024 and 1536 rungs (the run above was
512, to keep testing cheap), and bitwise determinism — GPU sparse attention is not
deterministic, so re-runs with the same seed are geometrically close, not identical.

## Memory

Hi3D never published a validated minimum. The project measured 1536³ on a **96 GB H20** and
explicitly declined to establish a lower bound. The studio warns below these rough figures
and never refuses:

| Rung | Rough guide |
|---|---|
| 512³ | 16 GB |
| 1024³ | 24 GB |
| 1536³ | 48 GB |

Sequential CPU offload (`pipeline.enable_sequential_cpu_offload`) reduces GPU residency but
does not remove the decoded-feature cost, and is not wired into this wrapper.

## Output

Geometry only. **No textures, materials or vertex colours** — Hi3D's own scope note, not a
limitation of this integration. The mesh is recentred and normalised to unit maximum radius,
so absolute scale and world position are not preserved.

Each run writes three files:

- `<name>.glb`
- `<name>.json` — the run manifest: settings, timings, and which coarse mesh was refined
- `<name>.provenance.json` — the licence record, naming all four models involved

## Licence

**Hi3D 1.0: Hi3D Open License Agreement 1.0.** Commercial use is permitted, with two
obligations that travel with anything you ship:

1. A commercial product integrating it must display **"Powered by Hi3D 1.0"**, linking to
   <https://hi3d.ai/> where the display supports a link.
2. Passing **200,000 monthly active users** or **USD 1,000,000 monthly revenue** requires
   notifying the licensor and applying for a separate written licence within 60 days.

Standalone outputs — a mesh containing no Hi3D materials — are not themselves subject to
the branding or scale clauses; it is integrating the model into a product that is.

**The coarse stage is a separate licence.** Hunyuan3D-2 mini's weights are under the Tencent
Hunyuan 3D 2.0 Community License, which **does not apply in the EU, the UK or South Korea**.
Hi3D's own licence is broader, but a Hi3D run here always consumes a Hunyuan mesh, so the
pairing inherits that restriction. That is why `validate_run_policy` refuses a
`game`/`worldwide` run for this backend, and why the studio shows the warning unconditionally.

The two conditioners (DINOv2-large, Apache-2.0; CLIP-large, MIT) are commercial-clear; the
remover (BiRefNet-lite, MIT) is too. All of that is recorded in the sidecar.
