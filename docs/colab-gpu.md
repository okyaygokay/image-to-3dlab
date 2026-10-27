# Running on a rented Colab GPU

The studio is local. The GPU is not.

`scripts/colab_generate.py` runs any Linux/NVIDIA backend on a Google Colab session and
brings the GLB back. It exists for the machine that does not have the card: a Mac, a
laptop, a box with an AMD card.

```
this machine                            Google Colab
─────────────                           ─────────────
picture  ──upload──▶
                        scripts/bootstrap_<backend>.py  (once per VM)
                        scripts/<backend>_generate.py   (detached)
        ◀─download──    <name>.glb
```

## Setup

Once, on this machine:

```bash
uv tool install google-colab-cli
colab sessions          # signs in on first use; lists yours
```

That is the whole local requirement. Nothing about a backend is installed here — no torch,
no weights, no cuDNN. The remote VM builds its own venv.

## Use it

```bash
python scripts/colab_generate.py photo.png out.glb --backend hi3d --accelerator G4
```

The first job on a fresh VM installs the backend (15–30 minutes, ~14.7 GB for Hi3D). After
that the session is warm; a second job is just the generation.

| Flag | Default | What it does |
|---|---|---|
| `--backend` | `hi3d` | `hi3d`, `hunyuan_cuda`, `trellis_cuda`, `pixal3d`, `qwen_image` or `autorig` |
| `--accelerator` | `G4` | `G4`, `A100`, `L4`, `T4`, `H100` |
| `--session` | a warm one, else new | Use this VM by name (implies `--keep`) |
| `--new-session` | off | Rent a fresh VM and pay the install again |
| `--no-share` | off | Skip the lab's shared VM; a private session per job, stopped when it ends |
| `--keep` | off | Leave the VM running when the job finishes |
| `--skip-setup` | off | Trust an existing install on the VM |
| `--yes` | off | Skip the confirmation (the studio passes this after its own dialog) |
| `--exec` | off | Run one repo *script* on the VM instead of a generation (below) |

### Warm sessions, and the lab's shared VM

A fresh VM installs for 15–30 minutes before it can do anything. That is fine once and
unusable per job, so a run **reuses a session that is still inside its idle window** —
it already has the venv and the weights, and the job starts in seconds. Only sessions this
lab started for the *same backend* are candidates; a Hunyuan job will not be handed Hi3D's
venv. `--new-session` is for a clean-room run, and `--session NAME` pins one explicitly.

On top of that, the studio's runs land on **one shared VM** (`i2l-lab`): if none is live,
a job starts one, and every later job lands on it while it is live and idle. Every
backend's install serves every job there, so a picture model's GLB can be rigged the next
click without waiting for Autorig's 15-minute install. While a job runs there,
`scripts/colab_prefetch.py` installs the *other* backends onto it in the background (one
at a time, disk-guarded, and the big ones wait for the GPU to free up). The shared VM is
never stopped by a job — the 3-hour idle rule reaps it — and a job that finds it busy with
another job (one GPU, one job) rents a private session exactly as before. `--no-share`
opts out of the whole arrangement, and `--new-session` never touches it.

Idle sessions are reaped at the start of each job (that is the only long-lived process the
lab runs regularly), as well as by the 3-hour rule.

Anything after `--` is forwarded to the remote wrapper untouched:

```bash
python scripts/colab_generate.py photo.png out.glb --backend hi3d -- --resolution 1536
python scripts/colab_generate.py photo.png out.glb --backend pixal3d -- --seed 7
```

Pixal3D runs at its one resolution (1024), so it has no detail flag; its first job on a
fresh VM downloads 8.4 GB of weights (about ten minutes) instead of compiling anything.

### `--exec`: one repo script on the VM

The two non-generation backends need a way to run *part* of a route remotely:

```bash
# Hero's head-crop upscale, on the venv the pixal3d backend already installed
python scripts/colab_generate.py --backend pixal3d --exec scripts/upscale_esrgan.py \
    --exec-input crop.png --exec-output crop_up.png --yes -- {in} {out} --size 1024
```

`{in}` and `{out}` in the forwarded arguments are the remote paths of the uploaded input
and the downloaded output, so the script itself never learns it is running remotely. With
`--exec-bin`, the script runs as a raw binary from the backend's install (Hero's head
clean-up: `vendor/pixal3d-cpp/build/post-replay`) instead of under the venv's Python.
The setup, the detach, the waiter and the status polling are identical to a generation.

### The picture step and the auto-rigger ride along too

Two studio steps had no local engine on a Mac with nothing installed, and now fall back
to this route instead of refusing:

- **Qwen-Image** (Create's "start from a description"): `picture_route()` in
  `viewer/image_api.py` picks the local sd-cli when it exists, a rented GPU when it does
  not and the CLI is signed in, and only then refuses. The prompt and settings travel as
  one small JSON in the supervisor's input slot; the drawing itself is built by the *same*
  `build_command` the local route uses, imported on the VM by `scripts/qwen_image_colab.py`,
  so the two cannot drift. Upstream publishes no Linux CUDA build of stable-diffusion.cpp,
  so the VM compiles it from the pinned release commit (~10 minutes once).
- **Auto-rig / Animate** (the Animate tab): when SkinTokens is not installed locally, the
  rig runs as one supervisor job (`--backend autorig`), and the preset clips the same way.
  The commands are the local route's own, built by `scripts/autorig_colab.py` on the VM.

### A motion from a description

The Animate tab's prompt box ("…or describe a move") has no local engine at all: the video
model that acts the prompt out only exists on the rented GPU. One supervisor job
(`--backend prompt2rig`, `scripts/prompt2rig_colab.py` on the VM) runs the whole chain:

1. **MiniMax H3** (`fl2va`, int8) acts the description out — one person, full body, static
   camera, ~6 s at 960×544, anchored on a full-body still of your own rig rendered on the
   VM. The checkpoint is ~124 GB in bf16, more than a Colab disk
   holds comfortably, so the bootstrap downloads each big component alone, quantizes it to
   int8 with the torchao recipe from the model card, saves that and deletes the raw copy:
   peak disk stays near 100 GB, and a warm session loads the int8 weights straight back.
2. **GVHMR** lifts SMPL-X motion from the clip (its `static_camera` fast path — a generated
   clip never moves the camera).
3. **WiLoR** tracks both hands — MANO ships with the WiLoR authors' own weights, so
   nothing is registration-gated — with MediaPipe as the fallback, and the keypoints ride
   along as wrist-relative world tracks.
4. `scripts/pack_smplx_clip.py` packs the motion into the clip format `kimodo_retarget.py`
   plays — the same format the preset moves ship in — and the preset path's own retarget
   command plays it on the rig (the SkinTokens venv is installed on the VM too).

The clip and the packed motion come home beside the animated GLB (`<name>.clip.mp4`,
`<name>.motion.npz`), so a run keeps its evidence and the motion replays on any other rig.
The first run on a cold VM pays the install (~30–45 minutes, most of it the H3 download);
a warm session runs the whole chain in minutes. Everything is pinned: the H3, SMPL-X and
WiLoR revisions are commits with checksums, `gvhmr` and `diffusers` are versions.

### What comes back with a Colab Pixal3D model

A local Pixal3D run writes more than the GLB, and Finish needs exactly those files, so the
supervisor downloads them beside the GLB: the run record (`.json`), the clean-up dump
(`.post.bin`) and the camera views (`.svviews/`). Pixel Match works on Colab-made models
for the same reason: the cameras are here now.

### Hero face on any engine's body

The hero route's body can be any engine's mesh — Hi3D for the geometry with Pixal3D for
the head is a good pairing. Where the head's Pixal3D calls run is decided by availability,
not by the body's origin: this machine's trellis-cli when it is installed, else the rented
GPU (`--pixal colab`, and the throwaway head-finding rig rides the same VM when no local
SkinTokens exists). A Pixal3D body brings its cameras and dump and gets Pixel Match; a
body from another engine has neither and keeps its own texture through the classic
clean-up, which is usually why that engine was chosen for the mesh.

### Which accelerator

`G4` is an **RTX PRO 6000 Blackwell** (sm_120, ~96 GB). It is the only one this lab has
verified Hi3D and Hunyuan on, so it is the default. `A100`, `L4`, `T4` and `H100` are
offered because the CLI accepts them, not because anything was measured here.

Availability is tier-gated. A `400` when allocating means no quota or entitlement for that
card on your account; the supervisor says so instead of retrying forever.

## The bill

A rented GPU costs Colab compute units from your account, by the second.

- **The session stops when the job finishes**, unless you pass `--keep` — or it is the
  lab's shared VM, which is left for the idle rule so the next step starts instantly.
- **Every session stops after three hours without a heartbeat.** Every call that talks to
  Colab refreshes one; `stop_idle_sessions()` is what acts on it.
- A session whose heartbeat is *unknown* is never treated as idle. Not knowing is not
  evidence, and guessing wrong would kill a run someone is watching.

To stop one by hand: `colab stop -s <session>`, or `colab sessions` to see what is live.

## Why the job runs detached on the VM

`colab exec` blocks the VM's kernel. A refinement can run for an hour, and a websocket
hiccup in the middle would lose the whole thing. So the supervisor starts the real work
with `start_new_session=True`, writes a token file, and polls the VM's status file and log
over the REST contents API — which keeps working while the kernel is busy.

The contents API also has a size ceiling per call, so big files cross the link in
proven-sized pieces both ways: uploads are split locally and reassembled on the VM,
downloads are split on the VM and reassembled here — checksum-checked either way, because
the one file a later step needs (a head's ~460 MB clean-up dump, say) failing quietly
once cost a hero run an hour in. A *declared* artifact that exists on the VM but will not
come back fails the fetch, named; one the wrapper never wrote is still skipped quietly.

The supervisor itself is an ordinary local process printing ordinary stage lines, which is
why the studio can watch it exactly like a local run.

Detaching has one cost: nothing waits for the child, so a crash leaves a zombie and the
status file still says "running" — a hang with no error anywhere. `scripts/colab_supervisor.py`
is the waiter. It streams the output, writes a terminal status on a non-zero exit, and
records a clean exit too, because a job whose GLB exists but whose status was never updated
never gets downloaded.

## In the studio

The four Colab routes appear as **first-class engines** in Create, on any machine:

```
Hi3D 1.0 (Colab GPU)          ← recommended
Hunyuan3D-2.1 (Colab GPU)
TRELLIS.2 (Colab GPU)
Pixal3D (Colab GPU)
```

They are catalogue entries of their own rather than an option on a local engine, because
"is this ready" is a different question: a local route is ready when its venv and weights
are on *this* disk, while a Colab route is ready when the `colab` CLI can reach an account
and needs no local bytes at all. Their weights are marked `remote_only`, so Setup states
the size and never counts or removes anything on this disk.

Create's advanced panel gains a **GPU** and a **Detail** picker for these routes, and says
in words that the run costs compute units on someone else's hardware. Both choices travel
to the job as settings and end up on the supervisor's command line.

## Licensing

Renting a GPU changes where the code runs, not what you owe.

- The backend's own licence still applies to the weights and the output. Hi3D's
  `commercial-conditional` terms, Hunyuan's territorial restriction, TRELLIS's DINOv3
  encoder — all unchanged. See [hi3d.md](hi3d.md) and the Licensing section of the
  README. The prompt-to-motion chain adds: MiniMax H3's Community License (commercial use
  under $20M revenue with attribution; companies in the USA, EU, UK or South Korea must
  apply), and research-only terms for GVHMR, WiLoR and the SMPL-X body model — the
  viewer writes all four into every prompt-made animation's provenance record.
- The VM is someone else's disk. Nothing from this machine is uploaded except the picture,
  this package and the scripts; `output/`, `characters/` and `vendor/` stay here. The VM is
  discarded when the session stops, and any weights downloaded onto it go with it.
- Your picture does leave this machine. If that matters, do not use this route.

## Limits

- **Linux/NVIDIA backends only.** The Mac-only Hunyuan-MLX ports install into this
  machine's interpreter and have no remote story. (Pixal3D's runtime is C++/GGML, so its
  prebuilt CUDA build runs on the VM even though the local route is a Mac one.)
- **One job per session at a time.** The supervisor polls a single status file.
- **No cost cap.** The three-hour idle rule bounds a forgotten session; it does not bound
  a long one. `--timeout` (default six hours) is the hard stop.
- Setup is not resumable across sessions: a new VM starts from nothing. Within one
  session, a per-backend stamp makes it a no-op.
