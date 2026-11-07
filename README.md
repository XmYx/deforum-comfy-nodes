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
| `deforum_flux3_klein.json` | FLUX.3 Image Klein | 15 steps, `cfg 1`, `er_sde` / `simple`. |
| `deforum_integrated.json` | SDXL | Whole animation inside one `Deforum Single Sample` node. |

![Deforum on Krea-2 Turbo](examples/deforum_krea2_turbo_preview.jpg)

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

**Distilled models** (FLUX, Krea Turbo, SDXL Turbo, …) want `cfg_scale_schedule`
set to `0: (1)` in **Diffusion Parameters**. At a higher cfg they burn out.

**Checkpoints without a text encoder.** SD 3.5, FLUX and Qwen-Image are usually
shipped as a bare diffusion model. `Load Checkpoint` then returns no CLIP and you
get `'NoneType' object has no attribute 'tokenize'`. Load the text encoders
separately with `CLIPLoader` / `DualCLIPLoader` and feed that CLIP into
**Blend Conditionings**, as `deforum_flux3_klein.json` does.

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

**The animation never advances past frame 0.** `reset_counter` / `reset_latent`
are ticked on the Iterator.

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
