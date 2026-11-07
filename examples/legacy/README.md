# Legacy examples

These workflows are kept for reference but are **not** part of the verified set.

| Workflow | Status |
| --- | --- |
| `deforum_ip_adapter.json` | Broken. Uses `IPAdapterApply`, which ComfyUI_IPAdapter_plus removed; rebuild it against `IPAdapterAdvanced`. |
| `deforum_simple.json` | Broken. Built around a `workflow/Deforum Sampler` group node from an old ComfyUI frontend that no longer loads. |
| `deforum_stablecascade.json` | Stable Cascade two-stage setup. The node types still exist, but the workflow has not been re-verified. |
| `deforum_stablecascade_legacy.json` | As above, using separate UNET/CLIP/VAE loaders. |

For a current starting point use one of the workflows in the parent folder.
