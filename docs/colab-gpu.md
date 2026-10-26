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
| `--backend` | `hi3d` | `hi3d`, `hunyuan_cuda`, `trellis_cuda` or `pixal3d` |
| `--accelerator` | `G4` | `G4`, `A100`, `L4`, `T4`, `H100` |
| `--session` | a warm one, else new | Use this VM by name (implies `--keep`) |
| `--new-session` | off | Rent a fresh VM and pay the install again |
| `--keep` | off | Leave the VM running when the job finishes |
| `--skip-setup` | off | Trust an existing install on the VM |
| `--yes` | off | Skip the confirmation (the studio passes this after its own dialog) |

### Warm sessions

A fresh VM installs for 15–30 minutes before it can do anything. That is fine once and
unusable per job, so a run **reuses a session that is still inside its idle window** —
it already has the venv and the weights, and the job starts in seconds. Only sessions this
lab started for the *same backend* are candidates; a Hunyuan job will not be handed Hi3D's
venv. `--new-session` is for a clean-room run, and `--session NAME` pins one explicitly.

Idle sessions are reaped at the start of each job (that is the only long-lived process the
lab runs regularly), as well as by the 3-hour rule.

Anything after `--` is forwarded to the remote wrapper untouched:

```bash
python scripts/colab_generate.py photo.png out.glb --backend hi3d -- --resolution 1536
python scripts/colab_generate.py photo.png out.glb --backend pixal3d -- --seed 7
```

Pixal3D runs at its one resolution (1024), so it has no detail flag; its first job on a
fresh VM downloads 8.4 GB of weights (about ten minutes) instead of compiling anything.

### Which accelerator

`G4` is an **RTX PRO 6000 Blackwell** (sm_120, ~96 GB). It is the only one this lab has
verified Hi3D and Hunyuan on, so it is the default. `A100`, `L4`, `T4` and `H100` are
offered because the CLI accepts them, not because anything was measured here.

Availability is tier-gated. A `400` when allocating means no quota or entitlement for that
card on your account; the supervisor says so instead of retrying forever.

## The bill

A rented GPU costs Colab compute units from your account, by the second.

- **The session stops when the job finishes**, unless you pass `--keep`.
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

The supervisor itself is an ordinary local process printing ordinary stage lines, which is
why the studio can watch it exactly like a local run.

Detaching has one cost: nothing waits for the child, so a crash leaves a zombie and the
status file still says "running" — a hang with no error anywhere. `scripts/colab_supervisor.py`
is the waiter. It streams the output, writes a terminal status on a non-zero exit, and
records a clean exit too, because a job whose GLB exists but whose status was never updated
never gets downloaded.

## In the studio

The three Colab routes appear as **first-class engines** in Create, on any machine:

```
Hi3D 1.0 (Colab GPU)          ← recommended
Hunyuan3D-2.1 (Colab GPU)
TRELLIS.2 (Colab GPU)
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
  README.
- The VM is someone else's disk. Nothing from this machine is uploaded except the picture,
  this package and the scripts; `output/`, `characters/` and `vendor/` stay here. The VM is
  discarded when the session stops, and any weights downloaded onto it go with it.
- Your picture does leave this machine. If that matters, do not use this route.

## Limits

- **Linux/NVIDIA backends only.** The Mac-only routes (Pixal3D's Metal build, the
  Hunyuan-MLX ports) install into this machine's interpreter and have no remote story.
- **One job per session at a time.** The supervisor polls a single status file.
- **No cost cap.** The three-hour idle rule bounds a forgotten session; it does not bound
  a long one. `--timeout` (default six hours) is the hard stop.
- Setup is not resumable across sessions: a new VM starts from nothing. Within one
  session, a per-backend stamp makes it a no-op.
