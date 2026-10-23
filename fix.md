# Multi-GPU Vulkan split for Flux: what needs fixing

**Status: abandoned, untested.** The work below was written in March 2026 and never committed. It can't be tested now because I no longer have any GPUs, and the feature needs at least two dedicated (non-integrated) GPUs. This file records what's left to fix, so it can be picked up again by me or anyone with the hardware.

## What the feature does

It lets a Flux or Flux 2 model that doesn't fit on one graphics card run across several, using the Vulkan backend. It works the same way llama.cpp splits a model across GPUs.

- `GGMLRunner` can take a list of GPU backends. It builds a `ggml_backend_sched` scheduler over those GPUs plus the CPU.
- `FluxRunner::alloc_params_buffer()` gives each GPU its own weight buffer. It divides the transformer blocks (`double_blocks.N`, `single_blocks.N`) evenly across the GPUs. All other weights stay on GPU 0.
- The scheduler runs each operation on the GPU that holds its weights, and copies intermediate results between GPUs automatically.
- `ggml_backend_vk_device_is_integrated()` (added in the `ggml` submodule) lets the split skip integrated GPUs.
- How it switches on:
  - **Manually:** set `SD_VK_SPLIT=1`.
  - **Automatically:** happens when there are two or more dedicated GPUs, `SD_VK_DEVICE` is not set, and the diffusion model is larger than 80% of GPU 0's free memory. To support that size check, `init_backend()` now runs after the model's tensor list is read.
- In split mode, model loading runs on one thread, because Vulkan device fences are not safe when several threads use them at once.

Changed files: `src/flux.hpp`, `src/ggml_extend.hpp`, `src/stable-diffusion.cpp`, `src/diffusion_model.hpp`, and in the `ggml` submodule `include/ggml-vulkan.h` and `src/ggml-vulkan/ggml-vulkan.cpp`. That's about 530 real lines.

The `ggml` submodule points at the upstream ggml project, so its change can't be committed here. It's saved as `patches/ggml-vulkan-device-is-integrated.patch`, made against ggml commit `a8db410a252c8c8f2d120c6f2e7133ebe032f35d`. Apply it with `git -C ggml apply ../patches/ggml-vulkan-device-is-integrated.patch` before building.

## Fixes needed, in order

Items 2 and 3 were done when this work was committed to the `multi-gpu-vulkan-split` branch. Items 1 and 4 are still open.

### 1. The scheduler is too small for Flux (would crash)

`src/ggml_extend.hpp`, around line 2014, in the multi-GPU `GGMLRunner` constructor:

```cpp
sched_ = ggml_backend_sched_new(sched_bes.data(), nullptr,
                                (int)sched_bes.size(),
                                GGML_DEFAULT_GRAPH_SIZE, false, true);
```

`GGML_DEFAULT_GRAPH_SIZE` is 2048. The scheduler sizes its internal tables from this number, and ggml checks it every time a graph is planned:

```cpp
GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
```

A Flux graph is far larger. This project builds it with `FLUX_GRAPH_SIZE` (10240), and the weights alone add several hundred leaf tensors. So the first generation would most likely stop at that assertion before producing anything.

**Fix:** pass the real graph size:

```cpp
sched_ = ggml_backend_sched_new(sched_bes.data(), nullptr,
                                (int)sched_bes.size(),
                                MAX_GRAPH_SIZE, false, true);
```

### 2. Line endings were converted (done)

The changed files had been re-saved with Windows (CRLF) line endings, while the repo uses LF, so the diff showed about 18,000 changed lines. They were converted back to LF before this branch was committed, so the commit holds only the real changes.

### 3. The surname in the author comments (done)

The surname was misspelled in nine author comments. All nine now read **Michael Burgus**.

### 4. The weight-splitting code exists twice

`GGMLRunner::alloc_params_buffer_split()` in `ggml_extend.hpp` (around line 2113) is a general version that matches block names in `params_ctx`. Flux doesn't use it: `FluxRunner::alloc_params_buffer()` in `flux.hpp` replaces it, using the full tensor names from `get_param_tensors()` and `ggml_tallocr`. Either remove the general version, or keep it deliberately for other model types and say so in a comment.

Also note that `GGMLRunner::alloc_params_buffer()` isn't `virtual`. The Flux version only runs because `FluxModel` calls it on a `FluxRunner` directly. Calling it through a `GGMLRunner` pointer or reference would silently use the general version instead.

## Not yet verified (once the fixes above are done)

- **Never compiled in its final form.** The last build (`build/bin/Release/sd-cli.exe`) is from March 27, but `flux.hpp` and `stable-diffusion.cpp` changed on March 28.
- **Never run on two GPUs**, as far as I know.
- **Flux and Flux 2 only.** Other model types (UNet, SD3/MMDiT, Wan, Qwen-Image, Z-Image) always use a single GPU.
- **Vulkan only.** No CUDA or other backends.
- **LoRA and other weight adapters** weren't tried in split mode.
- **Splitting by block count ignores memory size.** The split doesn't weigh each GPU's free memory, so with cards of different sizes the smaller one could run out of memory.

## How to test once there are 2+ GPUs

1. Build with Vulkan: `cmake -B build -DSD_VULKAN=ON` then `cmake --build build --config Release`.
2. Run a Flux model once on a single GPU (`SD_VK_DEVICE=0`), then again with `SD_VK_SPLIT=1`, using the same seed and settings.
3. In the log, check for `Multi-GPU split: N double + M single blocks across K GPUs` and one `GPUn params:` line per card.
4. The two images should match, or nearly so. Generation time and per-GPU memory use show whether the split pays off.
