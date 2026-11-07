# Deforum for ComfyUI

Deforum animation nodes for ComfyUI — 2D/3D camera motion, cadence, hybrid video,
schedules and prompt interpolation, driven frame by frame from inside a normal
ComfyUI graph.

![Deforum on SDXL](examples/deforum_base_preview.jpg)

Verified against **ComfyUI 0.44.2** and **deforum-studio 0.2.0**.

## Installation

### ComfyUI Manager

Look for **Deforum Nodes** by `XmYx`. The Manager installs `requirements.txt`,
which pulls in the deforum backend and everything else the nodes need.

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/XmYx/deforum-comfy-nodes.git
```

Then install the dependencies **into the same Python that runs ComfyUI**:

```bash
python -m pip install -r ComfyUI/custom_nodes/deforum-comfy-nodes/requirements.txt
```

On ComfyUI portable that is the bundled interpreter:

```bash
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\deforum-comfy-nodes\requirements.txt
```

Python 3.11 – 3.14 are supported. Nothing here pins or reinstalls torch, numpy or
moviepy, so it will not disturb an existing CUDA, ROCm, DirectML or Apple Silicon
install.

## Example workflows

Load these from `examples/` with **Workflow → Open**. Each one ships with a
checkpoint name you will need to point at your own model.

| Workflow | Model | Notes |
| --- | --- | --- |
| `deforum_base.json` | SDXL | 3D camera move with depth warping. The best starting point. |
| `deforum_cadence.json` | SDXL | Adds cadence frames between diffused frames for smoother motion. |
| `deforum_krea2_turbo.json` | Krea-2 Turbo | 8-step turbo model, `cfg 1`. |
| `deforum_flux3_klein.json` | FLUX.3 Image Klein (base) | 50 steps, `cfg 4.0`, `euler` / `simple`, with `ModelSamplingFlux` shift = ln(16). |
| `deforum_guided.json` | SDXL | Scheduled guide images — pin an image to a keyframe and keep pulling towards it. |
| `deforum_integrated.json` | SDXL | Whole animation inside one `Deforum Single Sample` node. |

![Deforum on Krea-2 Turbo](examples/deforum_krea2_turbo_preview.jpg)

![Deforum on FLUX.3 Image Klein](examples/deforum_flux3_klein_preview.jpg)

Older workflows live in `examples/legacy/`; see the README there for what still works.

### Running one

1. Load the workflow and set **Load Checkpoint** (or the UNET/CLIP/VAE loaders) to a model you have.
2. Queue the prompt. The first run renders frame 0.
3. The **Iterator** node ships with `enable_autoqueue` on, so the run drives
   itself until it reaches `max_frames`. You can also queue manually, or use
   ComfyUI's own auto-queue.
4. **Save Video** writes an `.mp4` to your output folder once `max_frames` frames
   are cached.

To restart an animation from frame 0, tick `reset_counter` and `reset_latent` on
the Iterator for one run, then untick them. Leaving them ticked pins the run to
frame 0 forever.

## Model support

The Iterator's `latent_type` defaults to `auto`. Wire your model into its
`model` input and it reads the latent shape straight from ComfyUI, so new
architectures work without waiting for this pack to be updated.

| Family | Latent | Works with `auto` |
| --- | --- | --- |
| SD 1.5 / SD 2.x / SDXL | 4 ch, /8 | yes |
| SD 3 / SD 3.5 | 16 ch, /8 | yes |
| FLUX.1 | 16 ch, /8 | yes |
| FLUX.2 / FLUX.3 | 128 ch, /16 | yes |
| Qwen-Image, Krea-2 | 16 ch, /8 | yes |
| Stable Cascade | 16 ch, /42 | yes |

Leave `model` unconnected only if you want to pick the shape by hand from the
dropdown.

**Distilled models** (Krea-2 Turbo, SDXL Turbo, FLUX schnell, the
`flux3-image-klein-*` distilled checkpoints, …) want `cfg_scale_schedule` set to
`0: (1)` in **Diffusion Parameters**. At a higher cfg they burn out.

**FLUX.3 Klein base** is the opposite: `flux3-image-klein-base-*` is the CFG
variant and wants real guidance. Follow Black Forest Labs' own settings —
`cfg_scale_schedule` `0: (4.0)`, `steps_schedule` `0: (50)`, `euler` / `simple`,
and a `ModelSamplingFlux` between the UNET and the samplers with both shifts set
to ln(alpha) = ln(16) = 2.77259. `deforum_flux3_klein.json` is wired that way.
Note that ComfyUI's own bundled FLUX.3 template uses cfg 1 for the *base* file;
that is the distilled model's recipe and gives mush on the base weights.

**Checkpoints without a text encoder.** SD 3.5, FLUX and Qwen-Image are usually
shipped as a bare diffusion model. `Load Checkpoint` then returns no CLIP and you
get `'NoneType' object has no attribute 'tokenize'`. Load the text encoders
separately with `CLIPLoader` / `DualCLIPLoader` and feed that CLIP into
**Blend Conditionings**, as `deforum_flux3_klein.json` does.

## Guide images

Two different things, depending on what you want.

**Seed the animation from an image.** Encode it and feed the Iterator's
`init_latent` input:

```
LoadImage ─► VAEEncode ─► (deforum) Iterator [init_latent]
```

That replaces the starting noise on frame 0 only. How closely frame 0 follows it
is `strength_schedule[0]` in **Diffusion Parameters**.

**Keep guiding later frames.** Use **Guide Image Schedule**, between
**Frame Warp** and the rest of the loop:

```
Frame Warp ─► Guide Image Schedule ─► Color Match ─► Add Noise ─► VAEEncode
                      ▲
   LoadImage ─► ImageBatch
```

It takes a batch of guide images and two schedules.

`image_schedule` picks which guide to use, by index into the batch.
`0:(0), 60:(1)` moves from the first guide to the second over 60 frames, and
with `crossfade_guides` on, fractional values blend between neighbouring guides
so the handover is smooth rather than a jump cut.

`strength_schedule` says how hard each frame is pulled towards it. `0` passes
the frame through untouched, `1` replaces it outright.

**Pulse it, don't hold it.** This node sits inside the feedback loop, so a
strength held across many frames compounds and the animation converges onto the
guide — at a steady `0.12` the guide has effectively taken over within a dozen
frames. What you usually want is a pulse at each keyframe, leaving the animation
free in between:

```
0:(1.0), 2:(0.0), 58:(0.0), 60:(0.6), 62:(0.0)
```

That pins the first guide at frame 0, lets the animation run, then pulls hard
towards the second guide at frame 60 and releases again. A sustained pull is
still useful for locking composition, but keep it around `0.02`–`0.05`.

How far the animation drifts back afterwards depends on `strength_schedule` in
**Diffusion Parameters** — that is the denoise. At the low denoise deforum
usually runs (0.2–0.5) the model has little room to redraw, so after a strong
pulse the animation carries on *from* the guide rather than returning to what it
was doing. That is what pinning a guide to a keyframe means; if you want the
guide to read as a suggestion instead, use a smaller pulse or raise the denoise.

Guides are resized to the frame automatically, and an optional `mask` input
restricts the guide to part of the frame. The node also outputs the resolved
guide for the current frame, which is worth hanging a Preview on while you dial
the schedules in.

`deforum_guided.json` is a working example; swap its two `LoadImage` nodes for
your own images.

For conditioning-level guidance instead (ControlNet, IPAdapter), use
**Schedule Value**: it reads any deforum schedule as a FLOAT for the current
frame, so you can drive a ControlNet strength from the deforum timeline.

## Schedules

Most parameter fields take a deforum schedule: `frame: (expression)`, comma or
newline separated, with `t` as the frame number.

```
0: (1.0), 30: (1.02), 60: (1.0)
0: (1.0025+0.002*sin(1.25*3.14*t/30))
```

Prompts are `frame: text`, one per line:

```
0: a lone astronaut walking across a vast red desert, cinematic
30: the desert dissolving into glowing crystal spires
```

Quotes around the prompt are optional and are stripped if present.

### Prompt timing and blending

A prompt scheduled at frame N is rendered exactly as written from frame N. To
crossfade into the next prompt, set `prompt_blend_frames` on the Iterator to the
number of frames the fade should take; the fade ends on the next prompt's
keyframe, so earlier prompts keep their timing. `prompt_blend_curve` picks
`linear` or `exponential`. `0` frames means no blending at all.

## Troubleshooting

**`IMPORT FAILED` / nodes missing from the menu.** The node pack now prints the
missing module and the exact pip command for your interpreter. The usual cause is
installing into a different Python than the one ComfyUI runs.

**`No module named 'deforum'`.**

```bash
python -m pip install git+https://github.com/XmYx/deforum-studio.git
```

**`No module named 'timm.layers'`.** `timm` is older than 1.0. Reinstall the
requirements, or `python -m pip install "timm>=1.0"`.

**`No module named 'moviepy.editor'`.** moviepy 2.x dropped that module. The
video nodes handle both 1.x and 2.x now, so update this pack.

**`'NoneType' object is not subscriptable` in KSampler.** The deforum cache was
empty. Branches that depend on a previous frame now stop quietly instead of
crashing, so this should no longer reach you.

**Frying / harsh edges after a few frames.** Lower `amount_schedule` and
`noise_schedule`, raise `strength_schedule`, and set `color_coherence` to `LAB`.
Running a model below its native resolution (SDXL under 1024) does this too.

**Colours drift over the clip.** `color_coherence` on **ColorMatch Parameters**
only does something if a **Color Match** node is actually in the graph, between
**Frame Warp** and **Add Noise**. `deforum_krea2_turbo.json` and
`deforum_flux3_klein.json` show the placement.

**The animation never advances past frame 0.** `reset_counter` / `reset_latent`
are ticked on the Iterator. If you are using cadence, check that
`diffusion_cadence` is smaller than `max_frames` — stepping by more than the
whole animation resets the counter on every queue.

**`Python integer -1 out of bounds for uint64`.** A `-1` in `seed_schedule`
(deforum's "pick one for me") reached the noise generator. `-1` is now resolved
to a random seed whatever `seed_behavior` is set to, so update this pack.

## Optional companion nodes

None of these are required, and this pack no longer installs anything on your
behalf:

- [rgthree-comfy](https://github.com/rgthree/rgthree-comfy)
- [ComfyUI-AudioScheduler](https://github.com/a1lazydog/ComfyUI-AudioScheduler)
- [ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus)
- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
- [ComfyUI-Advanced-ControlNet](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet)
- [ComfyUI_FizzNodes](https://github.com/FizzleDorf/ComfyUI_FizzNodes)
- [ComfyUI-Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation)

## Contributing

1. Fork the repository on GitHub.
2. Create a branch for your change.
3. Commit with clear messages.
4. Open a pull request.

## License

MIT. See [LICENSE](LICENSE).

## Community and support

[Join Discord](https://discord.gg/deforum) · [deforum.art](https://deforum.art)

![Deforum Website Logo](docs/logo.png)
