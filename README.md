# Kernels

Rust/CUDA kernel workspace for AI and machine-learning primitives using
[cuda-oxide](https://github.com/NVlabs/cuda-oxide).

This repo is a starting point. The intent is to keep kernel code, CPU reference
implementations, tests, and benchmarks together so each GPU kernel can be checked
for correctness and measured against a simple baseline.

## Goals

- Write CUDA kernels in Rust with cuda-oxide.
- Build small, reusable AI/ML primitives first: elementwise ops, reductions,
  normalization, matrix multiplication, and attention building blocks.
- Keep CPU reference implementations for every kernel.
- Add benchmarks before tuning kernel variants.
- Prefer clear, tested kernels over clever code until profiling shows a bottleneck.

## Toolchain

cuda-oxide is experimental and its API may change. Follow the official
[cuda-oxide book](https://nvlabs.github.io/cuda-oxide/) for current setup steps.
cuda-oxide currently targets Linux; use WSL, Linux, or a devcontainer for GPU
kernel builds and runs.

Expected local requirements:

- NVIDIA GPU with a CUDA-capable driver
- Rust nightly pinned in `rust-toolchain.toml`
- cuda-oxide tooling, including `cargo oxide`

Once the toolchain is installed, the normal workflow should be:

```powershell
cargo fmt
cargo test
cargo oxide doctor
cargo oxide build
```

Install `cargo oxide` with the same pinned nightly:

```powershell
cargo +nightly-2026-04-03 install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide
```

## Repository Layout

```text
Cargo.toml       # workspace definition
crates/
  kernels/
    src/        # device kernels, host launch wrappers, and CPU references
examples/       # runnable kernel demos
tests/          # integration and correctness tests
benches/        # benchmark harnesses and kernel comparisons
docs/           # kernel design notes and profiling results
```

## First Milestones

1. Create the minimal cuda-oxide workspace.
2. Add a vector-add smoke kernel.
3. Compare GPU output with a CPU reference implementation.
4. Add a benchmark harness.
5. Expand toward ML primitives: reductions, softmax, layer norm, matmul, and
   attention kernels.

## References

- [NVlabs cuda-oxide](https://github.com/NVlabs/cuda-oxide)
- [The cuda-oxide Book](https://nvlabs.github.io/cuda-oxide/)
