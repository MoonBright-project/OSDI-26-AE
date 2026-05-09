# PageMold Artifact Evaluation Guide

**OSDI'26 Artifact Evaluation for _A GPU Memory Allocator with Device-Side Page Table Materialization and Deferred TLB Coherence_**

This README is the entry point for evaluating **PageMold**, a GPU memory allocator that accelerates virtual-memory operations by materializing page tables on the device and deferring TLB coherence. The artifact is organized as a set of repositories under the [MoonBright GitHub organization](https://github.com/MoonBright-project). This guide explains which repository to use, what must be installed first, and which scripts reproduce each result in the paper.

---

## 1. Artifact at a Glance

| Component | Repository | Purpose |
|---|---|---|
| PageMold runtime | <https://github.com/MoonBright-project/PageMold> | Core allocator and microbenchmarks |
| NVIDIA kernel module | <https://github.com/MoonBright-project/open-gpu-kernel-modules> | Modified NVIDIA driver path required by PageMold |
| AMD ROCm backend | <https://github.com/MoonBright-project/PageMold-ROCm-Backend> | Modified AMD driver/backend path required by PageMold |
| PyTorch integration | <https://github.com/MoonBright-project/PyTorch> | Training-memory defragmentation experiments |
| vLLM integration | <https://github.com/MoonBright-project/vLLM-fastmap> | Long-context and beam-search inference experiments |
| vAttention baseline | <https://github.com/Rash-598/vllm/tree/rash598/vattn> | Baseline for selected LLM inference comparisons |

We recommend starting with the **PageMold runtime** after the appropriate kernel module has been built and loaded. The application-level experiments depend on that runtime and are easier to evaluate once the core microbenchmarks are working.

---

## 2. Critical Setup Requirement: PageMold Needs a Modified GPU Driver

PageMold is **not** a pure user-space allocator and cannot be evaluated by only installing a shared library or docker file. Its design moves page-table materialization into the GPU execution path, so the corresponding GPU kernel module must be rebuilt, installed, and loaded before any PageMold experiment can run.

### NVIDIA path

Build and install our fork of `open-gpu-kernel-modules`:

<https://github.com/MoonBright-project/open-gpu-kernel-modules>

The user-space NVIDIA driver must match the kernel module version. Before loading our module, the stock NVIDIA kernel module must be unloaded.

### AMD path

Build the ROCm backend repository:

<https://github.com/MoonBright-project/PageMold-ROCm-Backend>

Because parts of the AMD GPU driver live in the Linux kernel tree, this path may require building a patched Linux kernel and rebooting into it.

### Practical implications for AE reviewers

Loading the modified driver requires **root privileges**, may require **disabling Secure Boot**, and temporarily replaces the vendor GPU driver for the evaluation run. We strongly recommend using a dedicated machine, a reserved test node, or an isolated AE environment instead of a shared workstation. Each driver repository includes step-by-step installation instructions and an `uninstall.sh` script to restore the stock driver.

---

## 3. Hardware and Software Requirements

### Validated hardware

The full artifact was evaluated on the following GPU configurations:

- NVIDIA A100
- NVIDIA H100
- AMD MI210

Other GPGPUs with virtual-address support may be adaptable, but they are outside the validated AE configuration.

### Required software

| Platform | Requirement |
|---|---|
| NVIDIA | CUDA 12.6 |
| AMD | ROCm 6.4.0 |
| Both | Linux environment capable of building and loading the modified GPU driver |

---

## 4. Recommended Evaluation Order

To minimize setup friction, we recommend evaluating the artifact in this order:

1. **Build and load the modified GPU driver** for the target platform.
2. **Build PageMold** from the core runtime repository.
3. **Run PageMold microbenchmarks** to validate the allocator and VMM path and CUDA/ROCm samples and HeCBench..
4. **Run application-level experiments** for training defragmentation and LLM inference, if the required hardware and time budget are available.

Reviewers with limited hardware access can evaluate a subset of the claims. NVIDIA-only environments can run the NVIDIA portions of C1, C2, C3, and C4. AMD-only environments can run the AMD portions of C1 and C2.

---

## 5. Core System: PageMold Runtime

- **Repository:** <https://github.com/MoonBright-project/PageMold>
- **AE branch/tag:** `main`
- **Build instructions:** see the repository `README.md`

PageMold must be linked against the modified GPU driver path described above. Building PageMold against the stock vendor module may produce binaries, but those binaries will not exercise the PageMold page-table materialization path and should not be used for AE measurements.

---

## 6. Evaluation Claims and Reproduction Scripts

Each claim below corresponds to a paper section and a set of AE scripts. Commands should be run from the relevant repository root unless the repository-specific README states otherwise.

### C1 — Primitive Efficiency of PageMold VMM APIs (§4.2)

**Goal.** Measure the latency and scalability of PageMold's virtual-memory primitives against CUDA VMM.

```bash
bash ae/scripts/driver_cost.sh     # map/unmap/reserve/create on A100, H100, and MI210
bash ae/scripts/multi_map.sh       # mapping cost on 1, 2, and 4 A100 GPUs
bash ae/scripts/pure_malloc.sh     # cudaMalloc vs. PageMold, 2 MB to 2 GB
```

---

### C2 — Generality Across GPGPU Workloads (§4.3)

**Goal.** Evaluate PageMold across a broad benchmark suite covering CUDA Samples, ROCm Samples, and HeCBench.

Benchmark sources:

- CUDA Samples: <https://github.com/NVIDIA/cuda-samples>
- ROCm Samples: <https://github.com/ROCm/rocm-examples>
- HeCBench: <https://github.com/zjin-lcf/HeCBench>

```bash
bash ae/scripts/malloc_overall.sh  # NVIDIA and AMD allocation-latency sweep
```

---

### C3 — Defragmentation for Deep-Learning Training (§4.4)

**Goal.** Evaluate whether PageMold improves GPU memory efficiency and enables larger trainable batch sizes under realistic training workloads.

- **Repository:** <https://github.com/MoonBright-project/PyTorch>
- **Baselines:** PyTorch default caching allocator and GMLake

```bash
bash ae/scripts/train_mem_eff.sh 
```

---

### C4 — Low-Latency LLM Inference (§4.5)

**Goal.** Evaluate PageMold's impact on long-context prefill, prefix caching, and beam-search inference.

- **PageMold integration:** <https://github.com/MoonBright-project/vLLM-fastmap>
- **vAttention baseline:** <https://github.com/Rash-598/vllm/tree/rash598/vattn>

```bash
bash ae/scripts/infer.sh 
```

---

## 7. Troubleshooting Notes

- If PageMold builds but the experiments fail immediately, first confirm that the modified kernel module is loaded rather than the stock vendor module.
- If the NVIDIA path is used, verify that the kernel module version matches the installed user-space NVIDIA driver.
- If the AMD path is used, verify that the machine has booted into the patched kernel when required.
- If root access, Secure Boot configuration, or driver replacement is not available, reviewers can inspect and build the repositories, but the full AE run cannot exercise the PageMold driver path.

---

## 8. Contact

For AE questions, please leave a comment on HotCRP. We appreciate your time reviewing the artifact and welcome feedback on both the artifact and the documentation.
