# How inference went from 4:57 to 3 seconds

## TL;DR
The accelerator was never the bottleneck. A debug **self-test** was running a
full **software convolution of every layer on the RISC-V CPU** to compare
against the hardware — and that CPU math was eating ~99% of the runtime.
Turning the self-test off dropped the run from **297 s to 3 s** (~100x), with
zero change to the actual result.

## The symptom
- Full inference on the NPU took **4 min 57 s**, every time.
- The runtime was *constant* — it did not change when the input changed, and it
  did not change when we cut 60% of the hardware weight-writes. That constant
  time was the clue: the cost was fixed work, not data movement.

## The false trail (and the real cause)
We first assumed the slowness was the host writing data to the accelerator one
32-bit word at a time over `/dev/mem`. We reduced those writes by ~60% — and
the runtime did **not budge**. That ruled out data movement as the bottleneck.

The real cause was the diagnostic block compiled in via `NPU_SELFTEST`:

```c
#if defined(USE_FPGA) && NPU_SELFTEST
    // ran a FULL software conv of this layer on the CPU, every layer,
    // purely to compare against the hardware output
    tensor_t sw; conv2d(x, w, ...); requant(&sw, ...);
    ... compare max|hw - sw| ...
#endif
```

`conv2d` is a plain quadruple-nested loop. On a small soft RISC-V core, doing
that for all five layers (layer 0 alone is 32x32x32 outputs x 27 MACs each,
layers 1-2 far more) is minutes of work. The hardware finished each layer in
microseconds and then sat idle while the CPU recomputed the same layer in
software for the comparison.

## The fix
One line:

```c
#define NPU_SELFTEST 0   // was 1
```

With it off, the per-layer software conv is gone. The CPU now only does the
cheap glue it was always meant to do (global average pool + the final 2-class
fully-connected layer), and the heavy convolutions run only on the accelerator.

We also reverted some experimental "signed activation" changes that had been
added while chasing a self-test mismatch — they were never needed once we
recognized the self-test itself was the issue, and they distorted bright input
pixels. The inference path is back to the version that produces correct
predictions.

## Smaller wins kept (harmless, low-risk)
- **Weight loader skips zero-padding.** The weight BRAM row is 128 channels
  wide, but the model has only 32-56 output channels per layer. We now write
  only the `ceil(out_ch/4)` sub-banks that hold real data instead of all 32 —
  ~60% fewer weight writes. (Minor on its own, but free.)
- **Activation read-back does one bus read per byte** instead of two (an
  earlier version did a redundant 32-bit read before the byte read).
- **Poll timeout tightened** so a genuine hardware stall fails fast instead of
  spinning for ~a minute.

## The lesson
Debug scaffolding can dominate runtime. A self-check that recomputes the whole
workload in software is invaluable for *verifying* correctness once, but it must
be compiled out for production runs — otherwise you are paying for the slow path
you were trying to avoid, plus the fast one, on every inference.

## Re-enabling verification later
If you ever want to re-verify the hardware against software, flip
`NPU_SELFTEST` back to `1`, accept one slow run, confirm the outputs, then set
it back to `0`. It is a diagnostic, not part of normal inference.
