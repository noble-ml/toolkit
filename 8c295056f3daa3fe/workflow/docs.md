# Workflow: complete documentation

*This file is the whole documentation in one page. It is generated from the individual pages; do not edit it by hand.*

## Contents

- [Install](#install)
- [Demo](#demo)
- [Getting started](#getting-started)
- [Layers](#layers)
- [Blocks](#blocks)
- [Routing](#routing)
- [Model](#model)
- [Architecture analysis](#architecture-analysis)
- [Detection](#detection)
- [Training](#training)
- [Benchmarking](#benchmarking)
- [CLI](#cli)
- [The .nbm format](#the-nbm-format)
- [Roadmap](#roadmap)

---

Noble Engineering's CNN framework: build custom architectures from named stages, with shapes, parameters,
FLOPs and routing checked for you; train a bounding-box detector on a Felix dataset; ship it as one `.nbm`
file. PyTorch is the compute backend; the API, architecture system and model format are Workflow's own.

```python
import felix as fx
import workflow as wf

model = wf.Model(
    wf.Stage("p1", wf.Conv2D(3, 16, 3, stride=2, padding=1), wf.BatchNorm2D(16), wf.SiLU()),
    wf.Stage("p2", wf.Conv2D(16, 32, 3, stride=2, padding=1), wf.BatchNorm2D(32), wf.SiLU(), wf.CSP(32)),
    wf.Stage("p3", wf.Conv2D(32, 64, 3, stride=2, padding=1), wf.BatchNorm2D(64), wf.SiLU(), wf.CSP(64)),
    wf.Stage("neck", wf.FPN(inputs=("p2", "p3"), channels=64, outputs=("n2", "n3"))),
    wf.Stage("head", wf.BBoxHead(inputs=("n2", "n3"), classes=3, outputs=("d2", "d3"))),
)

model.summary()                                   # shapes, parameters, FLOPs, memory
model.export_graph("architecture.png")            # the real computation graph
model.train(fx.load("dataset.felix"), epochs=100, batch_size=16, save_every=5)
model.predict("photo.jpg").save("prediction.png")
model.benchmark(dataset.test)                     # mAP, precision/recall/F1, per-class AP, confusion matrix
model.save("noble.nbm")                           # one file: architecture, weights, classes, training state
```

No channel arithmetic and no manual wiring: `input → p1 → p2 → …` is automatic, named routing is explicit, and a
wrong channel count, a bad residual, an unknown stage name or a class-count mismatch fails with a message that
names the stage and operation, before anything trains.

## What is in it

- **Architecture**: `Model`, `Stage`, named routing and multi-output operations; layers (`Conv2D`, `BatchNorm2D`,
  `ReLU`, `LeakyReLU`, `SiLU`, `Sigmoid`, `MaxPool2D`, `AvgPool2D`, `Upsample`) and blocks (`Residual`,
  `Bottleneck`, `CSP`, `SPPF`, `Concat`, `Add`, `FPN`, `PAN`, `BBoxHead`).
- **Analysis**: `summary()`, parameters, MACs/FLOPs, memory estimates, `plot()` / `export_graph()` (PNG/SVG).
- **Detection**: anchor-free `BBoxHead`, decoding, NMS, `predict()` with `Results`.
- **Training**: Felix datasets, validation and metrics every epoch, early stopping, mixed precision, EMA,
  periodic `.nbm` saves, safe Ctrl+C, and continuing from a saved model.
- **Benchmarking**: architecture speed/size, or accuracy on a dataset split.
- **Devices**: `model.device("auto" | "cuda" | "mps" | "cpu")`.
- **CLI**: `workflow info | devices | export (.pt, .onnx) | script (image, folder, video, webcam) | benchmark`.

---

## Install

Workflow needs Python 3.9+, PyTorch 2.1+, Pillow and rich. Datasets come from **Felix**, so install both.

```
pip install torch Pillow rich
pip install --no-deps --no-index --find-links BASE_URL/links.html felix workflow
```

`BASE_URL/links.html` lists every Noble wheel, so one command installs both packages. Check with
`workflow --version` and `felix --version`. If an older version is already installed, add `--upgrade`
(or `--force-reinstall`); pip otherwise keeps what it has.

Why the flags:

- **`--no-index`**: pip must not search PyPI. *Different, public* packages called `felix` and `workflow` live
  there and would be installed instead. Never use `--extra-index-url` for these.
- **`--no-deps`** and the separate `pip install torch Pillow rich`: with `--no-index`, pip can't fetch
  dependencies from this page, so install them first the normal way.

### Colab / notebooks

```
!pip install BASE_URL/files/felix-0.2.0-py3-none-any.whl BASE_URL/files/workflow-0.1.0-py3-none-any.whl
```

Direct wheel URLs need no extra flags: pip fetches the dependencies from PyPI itself, and there is no name clash.
Colab already has PyTorch with CUDA, so `model.device("auto")` picks the GPU.

### Optional extras

- Video and webcam inference scripts: `pip install opencv-python`.
- ONNX export works out of the box on current PyTorch; on a PyTorch without the legacy exporter, add
  `pip install onnx onnxscript`.

### GPU

`model.device("auto")` prefers CUDA, then Apple MPS, then CPU. Install the PyTorch build for your GPU from
pytorch.org; `workflow devices` shows what Workflow can see.

### Notes

Workflow and Felix are proprietary. This page is unlisted but not access-controlled: anyone with the link can
download the packages, and the packages are plain Python.

---

## Demo

`examples/demo.py` runs the whole stack in a few minutes: **Felix** draws a dataset of circles, squares and
triangles, **Workflow** builds a detector, trains it, benchmarks it, runs a prediction and saves everything.

```
pip install Pillow rich torch            # plus felix and workflow (see Install)
python examples/demo.py                  # ~25 epochs on 600 images
python examples/demo.py --epochs 40 --pairs 1200 --out my_demo
```

Outputs land in `demo_out/`: the dataset (`shapes_600.felix`), `architecture.png` / `.svg`, training runs
(`runs/last.nbm`, `best.nbm`, `epoch_N.nbm`), `evaluation.json`, `confusion_matrix.png`, `prediction.png`
and `demo.nbm`.

### What it does, step by step

**1. A Felix pipeline makes the data.** Three shape classes, 1-3 objects per image (the second and third use a
`threshold`), random scale, rotation, hue and brightness, a grey background, and a small grey distractor circle
painted *after* `fx.gen()` so it is in the image but not in the annotations.

**2. The architecture** is a backbone (`p1`…`p5` with CSP and SPPF blocks), `FPN`, `PAN` and a `BBoxHead`,
all wired by name:

```python
model = wf.Model(
    wf.Stage("p1", *cbs(3, 16)),
    wf.Stage("p2", *cbs(16, 32), wf.CSP(32)),
    wf.Stage("p3", *cbs(32, 64), wf.CSP(64, n=2)),
    wf.Stage("p4", *cbs(64, 128), wf.CSP(128, n=2)),
    wf.Stage("p5", *cbs(128, 256), wf.SPPF(256)),
    wf.Stage("neck", wf.FPN(inputs=("p3", "p4", "p5"), channels=64, outputs=("n3", "n4", "n5"))),
    wf.Stage("pan", wf.PAN(inputs=("n3", "n4", "n5"), channels=64, outputs=("m3", "m4", "m5"))),
    wf.Stage("head", wf.BBoxHead(inputs=("m3", "m4", "m5"), classes=3, outputs=("d3", "d4", "d5"))),
)
model.summary()                                  # shapes, parameters, FLOPs, memory
model.export_graph("architecture.png")           # the computation graph
```

**3. Training** is one call, and Ctrl+C is safe:

```python
model.train(dataset, epochs=25, batch_size=16, save_every=5)
```

**4. Evaluation and prediction** use the held-out test split and a single image:

```python
results = model.benchmark(dataset.test)          # mAP, precision/recall/F1, per-class AP, confusion matrix
pred = model.predict("test_image.png", conf=0.3)
pred.save("prediction.png")
```

**5. Deploy.** The model is one `.nbm` file:

```
workflow export demo_out/demo.nbm --format onnx
workflow script demo_out/demo.nbm --preset webcam      # also image, folder, video
```

### Typical result

The default run (600 generated images, 25 epochs, about 3 minutes on an Apple-silicon laptop using the GPU) reached
**mAP50 0.97 and mAP50-95 0.84** on the 60-image held-out test split (precision 0.96, recall 0.96), at about 150
images/s end to end. Your numbers vary with the random dataset and hardware.

![Predicted boxes](docs/img/prediction.png)

![The computation graph of the demo model (top part)](docs/img/architecture_top.png)

---

## Getting started

### The mental model

A **Model** is a list of named **Stages**. A stage is a sequence of **operations** (layers or blocks).

```python
import workflow as wf

model = wf.Model(
    wf.Stage("p1", wf.Conv2D(3, 32, 3, stride=2, padding=1), wf.SiLU()),
    wf.Stage("p2", wf.Conv2D(32, 64, 3, stride=2, padding=1), wf.SiLU()),
)
```

1. Every stage has a unique name (letters, digits, `_`; `input` is reserved for the model input).
2. The first stage receives the model input; every later stage receives the **previous stage's output**.
3. Operations inside a stage run in order.
4. A stage publishes its final output under its name. Later stages can read it by name
   ([Routing](#routing)).
5. Everything is validated when the `Model` is created, before any data or training.

### Shapes

Shapes are `(channels, height, width)` without the batch. Workflow propagates them for you:

```python
model.summary()                 # table of every operation's output shape and parameters
model.shapes()                  # {"p1": (32, 320, 320), "p2": (64, 160, 160)}
model.shapes(input=(3, 416, 416))
```

`wf.Model(..., input=(3, 640, 640))` sets the shape used for validation. Without it, the channels are
taken from the first convolution and the size defaults to 640×640. The model itself runs on any size.

### Running

```python
import torch
y = model(torch.randn(1, 3, 640, 640))       # output of the last stage
feats = model.features(x)                     # {"p1": ..., "p2": ...} every named tensor
```

If the last stage publishes several named outputs, `model(x)` returns a dict of them.

### Parameters you don't manage

- **Bias** is automatic: a `Conv2D` directly followed by `BatchNorm2D` has none; otherwise it has one.
- **Parameter counts** (trainable / non-trainable, per operation) come from `summary()`.
- **Memory**: intermediate tensors are freed as soon as nothing needs them.

### Device

```python
model.device("auto")   # CUDA, then Apple MPS, then CPU
model.device("cuda")   # or "cuda:1", "mps", "cpu"; a clear error if unavailable
model.device()         # the current torch.device
```

### Save and load

```python
model.save("noble.nbm")
model = wf.load("noble.nbm")            # rebuilds the architecture from the file
model.load("noble.nbm")                 # loads weights into an identical architecture
```

Runnable version of all of this: `examples/phase1.py`.

### Where next

- Build a detector and train it: [Demo](#demo), [Detection](#detection), [Training](#training).
- Inspect it: `model.summary()` and `model.export_graph(...)` ([Analysis](#architecture-analysis)).
- Wire necks and heads by name: [Routing](#routing) and [Blocks](#blocks).

---

## Layers

All layers live at `wf.<Name>`. Shapes are `(C, H, W)`; "in" is the tensor the layer receives.
Invalid arguments raise `WorkflowError` at construction; a bad input shape raises `ShapeError` naming
the stage and operation.

Integer arguments marked *pair* also accept `(h, w)` tuples.

### Conv2D

```python
wf.Conv2D(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1)
```

| Parameter | Type | Default | Requirement |
|---|---|---|---|
| `in_channels` | int | required | ≥ 1. Must equal the channels of the tensor it receives. |
| `out_channels` | int | required | ≥ 1 |
| `kernel_size` | int or pair | required | each ≥ 1 |
| `stride` | int or pair | 1 | each ≥ 1 |
| `padding` | int or pair | 0 | each ≥ 0 |
| `dilation` | int or pair | 1 | each ≥ 1 |

Output: `(out_channels, ⌊(H + 2·pad − dil·(k−1) − 1) / stride⌋ + 1, …)` (same for W).
Errors: wrong input channels; kernel does not fit the input.
`groups` and `bias` are not exposed: groups is always 1 and bias is automatic (none when the next
operation in the same sequence is `BatchNorm2D`).

### BatchNorm2D

```python
wf.BatchNorm2D(channels)
```

| Parameter | Type | Requirement |
|---|---|---|
| `channels` | int | ≥ 1; must equal the input's channels |

Output shape = input shape. Uses eps 1e-3 and momentum 0.03. Its running statistics count as
non-trainable parameters.

### Activations

`wf.ReLU()`, `wf.SiLU()`, `wf.Sigmoid()`: no arguments. `wf.LeakyReLU(slope=0.1)`: `slope` is the
gradient for negative inputs. All accept any shape and keep it.

### MaxPool2D / AvgPool2D

```python
wf.MaxPool2D(kernel_size, stride=None, padding=0)
wf.AvgPool2D(kernel_size, stride=None, padding=0)
```

| Parameter | Type | Default | Requirement |
|---|---|---|---|
| `kernel_size` | int or pair | required | each ≥ 1 |
| `stride` | int or pair | `kernel_size` | each ≥ 1 |
| `padding` | int or pair | 0 | each ≥ 0 and at most half of `kernel_size` |

Output: channels unchanged; H and W follow the same window formula as `Conv2D`. Error: window does not
fit the input.

### Upsample

```python
wf.Upsample(scale_factor=2)
```

`scale_factor`: integer ≥ 1. Nearest-neighbour. Output: `(C, H·scale, W·scale)`.

---

## Blocks

Blocks are operations built from layers. They go anywhere a layer goes.

### Implemented

#### Residual

```python
wf.Residual(*ops, shortcut=None)
```

`output = branch(x) + shortcut(x)`, where `x` is the block's **own input** (never an earlier stage).

| Parameter | Requirement |
|---|---|
| `*ops` | one or more layers/blocks forming the branch (no operations that use named inputs/outputs) |
| `shortcut` | optional op or list of ops (e.g. `wf.Conv2D(c_in, c_out, 1, stride=2)`) to project the input. Default: identity |

The branch output and the shortcut must have the **same shape**, or construction fails with the two
shapes and how to fix it. Output = that shape.

#### Bottleneck

```python
wf.Bottleneck(channels, expansion=0.5, shortcut=True)
```

1×1 Conv-BN-SiLU down to `channels × expansion`, then 3×3 Conv-BN-SiLU back up, added to the input.

| Parameter | Requirement |
|---|---|
| `channels` | int ≥ 1; must equal the input's channels |
| `expansion` | 0 < value ≤ 1 (hidden width = `channels × expansion`, at least 1) |
| `shortcut` | bool; `False` returns the branch only |

Returns one tensor with the input's shape.

#### CSP

```python
wf.CSP(channels, out_channels=None, n=1, expansion=0.5, shortcut=True)
```

Cross-stage-partial block: a 1×1 conv splits the input into two paths; one runs `n` Bottlenecks; the
paths are concatenated and fused by a 1×1 conv.

| Parameter | Requirement |
|---|---|
| `channels` | int ≥ 1; must equal the input's channels |
| `out_channels` | int ≥ 1; default `channels` |
| `n` | int ≥ 1, number of Bottlenecks |
| `expansion` | 0 < value ≤ 1; hidden width = `out_channels × expansion` |
| `shortcut` | bool, residual connection inside the Bottlenecks |

Output: `(out_channels, H, W)`.

#### SPPF

```python
wf.SPPF(channels, out_channels=None, kernel_size=5)
```

Fast spatial pyramid pooling: 1×1 conv to `channels // 2`, three chained max-pools (stride 1) concatenated
with the input, 1×1 conv to `out_channels`.

| Parameter | Requirement |
|---|---|
| `channels` | int ≥ 1; must equal the input's channels |
| `out_channels` | int ≥ 1; default `channels` |
| `kernel_size` | odd int ≥ 3 |

Output: `(out_channels, H, W)`.

### Routing blocks

These read **named** tensors (`inputs=`), so they sit directly in a `Stage` (not inside Residual/CSP).
See [Routing](#routing).

#### Add

```python
wf.Add(inputs=("a", "b", ...))
```

Elementwise sum. At least 2 distinct names; all inputs must have **identical** shapes. Output: that shape.

#### Concat

```python
wf.Concat(inputs=("a", "b", ...))
```

Concatenates along channels. At least 2 distinct names; heights and widths must match, channels may
differ. Output: `(sum of channels, H, W)`.

#### FPN

```python
wf.FPN(inputs=("p3", "p4", "p5"), channels=128, outputs=("n3", "n4", "n5"))
```

Top-down feature pyramid. Each input gets a 1×1 Conv-BN-SiLU to `channels`; starting at the coarsest
level, the merged map is upsampled (nearest) and added to the next finer level; each level then gets a
3×3 Conv-BN-SiLU.

| Parameter | Requirement |
|---|---|
| `inputs` | ≥ 2 distinct names, ordered **largest feature map first**; each smaller than the one before |
| `channels` | int ≥ 1 (all outputs have this many channels) |
| `outputs` | **required**: one distinct name per input, same order. There is no default naming |

Input channels may all differ; weights are created from the inferred channels when the `Model` is
built. Outputs keep their input's resolution. Must be the last operation of its stage.

#### PAN

```python
wf.PAN(inputs=("n3", "n4", "n5"), channels=128, outputs=("m3", "m4", "m5"))
```

Bottom-up path aggregation, meant for FPN outputs. Inputs with a different channel count are projected
with a 1×1 Conv-BN-SiLU. From the finest level, each level is downsampled with a stride-2 3×3 Conv-BN-SiLU,
added to the next level, and refined with a 3×3 Conv-BN-SiLU.

| Parameter | Requirement |
|---|---|
| `inputs` | ≥ 2 distinct names, largest first; **each exactly half the previous size, rounded up** |
| `channels` | int ≥ 1 |
| `outputs` | **required**, one name per input |

A mismatch (e.g. 80×80 followed by 20×20) fails when the model is built, naming both inputs.

### Detection head

#### BBoxHead

```python
wf.BBoxHead(inputs=("m3", "m4", "m5"), classes=3, outputs=("d3", "d4", "d5"), channels=None)
```

Anchor-free bounding-box head. It must be the **last operation of the last stage** for `train`,
`predict` and `benchmark(dataset)` to find it. Design and tensor format: [Detection](#detection).

| Parameter | Requirement |
|---|---|
| `inputs` | ≥ 1 distinct names of feature maps (usually PAN or FPN outputs), any order |
| `classes` | int ≥ 1, part of the architecture; must equal the dataset's class count when training |
| `outputs` | **required**, one distinct name per input |
| `channels` | optional int ≥ 1: width of the head's convs (default: each input level's own channels) |

Output: per level a raw map `(4 + 1 + classes, H, W)`.

---

## Routing

Ordinary stages need no wiring: each reads the previous stage's output. Routing is for everything else,
and it is always **by name**; Workflow never picks a feature map by position or channel count.

### Names

Every tensor that can be referenced has one name:

- the model input is `"input"` (reserved)
- a stage publishes its final output under the **stage name**
- an operation that produces several tensors publishes them under the names given by its `outputs=`

All published names live in one namespace and must be unique.

### Reading earlier outputs

**Whole stage from another point**: `input=` on the stage.

```python
wf.Stage("head", wf.Conv2D(64, 32, 1), input="p2")     # reads p2, not the previous stage
```

**Operations that combine tensors** ([Concat, Add, FPN, PAN](#routing-blocks), or your own) declare `inputs=("a", "b")`.
Such an operation reads exactly those names and ignores the running tensor.

### Several outputs

An operation declares `outputs=("n3", "n4", "n5")`, one name per tensor it returns. Rules:

1. It must be the **last** operation of its stage.
2. The stage then publishes those names, and **not** its own name.
3. The next stage can't silently take "the output": it must say which, with `input="n3"` or an
   operation with `inputs=(...)`.

```python
wf.Stage("neck", wf.FPN(inputs=("p3", "p4", "p5"), channels=128, outputs=("n3", "n4", "n5"))),
wf.Stage("pan",  wf.PAN(inputs=("n3", "n4", "n5"), channels=128, outputs=("m3", "m4", "m5"))),
wf.Stage("head3", wf.Conv2D(128, 64, 3, padding=1), input="m3"),
```

Here `neck` publishes `n3`, `n4`, `n5` (not `neck`), and `pan` reads exactly those. Full runnable
version: `examples/phase2.py`.

### Custom operations

Subclass `wf.Op`, implement `forward(*tensors)` and `infer(*shapes)`, set `self.inputs` /
`self.outputs` in `__init__`. Anything built this way is routed, validated and serialized like a
built-in. (See `tests/test_phase1.py` for tiny examples.)

### Validation and errors

Checked when the `Model` is created:

| Problem | Message (abridged) |
|---|---|
| unknown name | `Stage 'p2', op 1 FPN refers to unknown name 'p9' (did you mean 'p3'?). Names available here: input, p1, p2` |
| reference to a later stage | `… refers to 'x', which is produced by stage 'x' later in the model. Stages run in the order they are listed` |
| circular dependency | `circular dependency between stages (each one needs the next): a → b → a` |
| duplicate stage / published name | `duplicate stage name 'a'` / `the name 'n3' is published by both stage … and stage …` |
| ambiguous flow after multi-output | `Stage 't' follows stage 's', which publishes several outputs ('x', 'y'). Say which one it reads` |
| stage name used for a multi-output stage | `… stage 's' publishes several outputs ('x', 'y'); refer to one of them by name` |
| multi-output op not last / undeclared | `… must be the last operation of its stage` / `produces 2 outputs; name them with outputs=(...)` |
| FPN/PAN inputs not ordered / not halving | `inputs must be ordered from the largest feature map to the smallest, but 'p5' is 20x20 and 'p4' is 40x40` / `each input must be half the size of the previous one` |
| missing output names | `FPN needs outputs=(...): one name per input (3 names), e.g. outputs=('n0', 'n1', 'n2')` |
| shape mismatch | `Stage 'p2' → op 1 Conv2D (output of 'p1'): expects 64 input channels but received 32` |

Intermediate tensors that nothing reads later are freed automatically during the forward pass.

---

## Model

```python
wf.Model(*stages, input=None)
```

| Member | What it does |
|---|---|
| `model(x)` | forward pass on `(N, C, H, W)`; returns the last stage's tensor, or a dict if it publishes several |
| `model.features(x)` | dict of every named tensor (stages and named outputs) |
| `model.summary(input=None)` | prints per-operation output shape and parameters; returns the totals |
| `model.shapes(input=None)` | `{name: (C, H, W)}` for every named tensor |
| `model.output_shapes` | shapes of the model outputs |
| `model.analyze(input=None)` | parameters, MACs/FLOPs and memory as data ([Analysis](#architecture-analysis)) |
| `model.plot()` / `model.export_graph(path)` | draw the computation graph ([Analysis](#drawing-the-graph)) |
| `model.train(dataset, epochs, batch_size, save_every, ...)` | [Training](#training) |
| `model.predict(source, ...)` | [Detection](#modelpredict) |
| `model.benchmark(dataset=None)` | [Benchmarking](#benchmarking) |
| `model.history`, `model.class_names` | training history; class names (from the dataset, stored in `.nbm`) |
| `model.device(name=None)` | no argument: current device. `"auto"`/`"cuda"`/`"cuda:N"`/`"mps"`/`"cpu"`: move the model |
| `model.save(path)` / `model.load(path)` / `wf.load(path)` | [.nbm](#the-nbm-format) |

`Model` is a regular `torch.nn.Module`, so `.parameters()`, `.eval()`, `.to()`, `torch.compile(model)`
and friends work as usual.

### Summary

Parameters are split into **trainable** and **non-trainable** (BatchNorm running statistics). The
table lists top-level operations; a block's row includes everything inside it.

FLOPs/MACs, memory and graph drawing: see [Analysis](#architecture-analysis).

### Devices and speed

- `"auto"` prefers CUDA, then Apple MPS, then CPU. An unavailable request raises `DeviceError` with what
  *is* available. `workflow devices` shows the same from the terminal.
- On CUDA, Workflow enables cuDNN autotuning, TF32 and the channels-last memory layout.
- Run inference under `torch.inference_mode()`; intermediate tensors are freed as soon as they are
  unused, which keeps memory low.
- `torch.compile(model)` works for extra speed on supported setups.
- Training uses mixed precision on CUDA automatically; `model.predict` uses fp16 on CUDA.

Measured on an Apple M-series laptop, 640×640, batch 8, a small model: ~25 img/s on CPU, ~190 img/s on MPS.
`model.benchmark()` measures your own model on your own device.

---

## Architecture analysis

Everything here works on the architecture alone: no data, no training, no forward pass on real tensors.

### `model.summary()`

```python
totals = model.summary()                      # prints a table, returns the totals
model.summary(input=(3, 416, 416))            # for another input size
```

| Column | Meaning |
|---|---|
| stage / # / operation | where the operation sits (position inside its stage) |
| output | its output shape `(C, H, W)`; several for multi-output operations |
| params | trainable + non-trainable values of the operation (blocks include everything inside) |
| MACs | multiply-accumulates of its convolutions, in millions |

Totals: parameters (trainable / non-trainable), **GFLOPs**, and memory.

### `model.analyze(input=None)`

Returns an `Analysis` with the same numbers as data: `.params`, `.trainable`, `.non_trainable`,
`.macs`, `.flops`, `.param_bytes`, `.act_bytes`, `.peak_bytes`, and `.ops` (one entry per operation
with `.macs`, `.trainable`, `.out_shapes`, ...).

How the numbers are defined:

- **MACs** count convolutions (they dominate). BatchNorm folds into the convolution at inference;
  activations, pooling, upsampling and additions are not counted. **FLOPs = 2 × MACs.**
  They are measured by running the architecture on torch's `meta` device, so they follow the real
  computation of every block (including custom operations) and cost no memory or time.
- **Non-trainable parameters** are BatchNorm running statistics.
- **Memory** is an fp32, per-image estimate: *weights*; *activations* = the sum of every operation's output
  (what training keeps for backpropagation); *inference peak* = the most tensors alive at once when the
  forward pass frees tensors as soon as they are unused. Multiply the per-image numbers by the batch size.

### Drawing the graph

```python
model.plot()                                   # opens the image; returns a PIL image
model.export_graph("architecture.png")         # or .svg
model.export_graph("overview.svg", detail="stages")   # compact: one card per stage
```

It draws the actual computation graph, with no Graphviz or other tool to install:

- one **card per stage** with its output shapes, one row per **operation**, arrows for the sequential flow;
- **residual shortcuts** and **branches**: Residual, Bottleneck, CSP and SPPF are drawn with their real
  paths and the `+` (add) or `©` (concat) merge;
- **named routing**: red edges from the stage/port that publishes a tensor to the operation that reads it,
  labelled with the tensor name; ports on the right edge of multi-input and multi-output operations;
- the model input at the top and the model outputs at the bottom.

`detail="ops"` (default) shows every operation; `detail="stages"` is the compact overview. PNG files are
rendered at 2× for sharpness (`scale=`). In a notebook `model.plot()` displays the image inline.

---

## Detection

Workflow's first task is **bounding-box object detection**: boxes, classes and confidences.
(Segmentation, pose and classification are not part of this version.)

### The head: `BBoxHead`

```python
wf.Stage("head", wf.BBoxHead(inputs=("m3", "m4", "m5"), classes=3, outputs=("d3", "d4", "d5")))
```

The head is **anchor-free**. For every input level it applies two 3×3 Conv-BN-SiLU layers and a 1×1 conv.
Each level produces one raw map of shape `(N, 4 + 1 + classes, H, W)`:

| Channels | Meaning |
|---|---|
| 0-3 | distances **left, top, right, bottom** from the cell centre to the box edges, in units of the level's stride (softplus is applied when decoding, so they are positive) |
| 4 | confidence (objectness) **logit** |
| 5… | one **logit per class** |

`model(x)` returns these raw maps as a dict keyed by the names in `outputs`. The head's design decisions:

- Each level's stride is **derived from the image and map sizes** at decode time, so any input size that
  the architecture accepts works; nothing is hard-coded to 640.
- `classes` is explicit in the architecture; training checks it against the dataset and explains a mismatch.
- `outputs` is explicit, like every multi-output operation: one name per input level.
- Levels can be fed in any order; their order is the order of `inputs`.

### Decoding and NMS (inference only)

NMS is post-processing, not a network block. `model.predict` does, per image:

1. **Decode** every cell to a box: centre ± (softplus distances × stride).
2. **Score** = sigmoid(confidence) × sigmoid(class logit); each cell keeps its best class.
3. Drop scores ≤ `conf`; keep the best 1000.
4. **Class-aware NMS**: boxes of different classes never suppress each other; overlaps above `iou` are removed.
5. Keep at most `max_det` and map the boxes back to the original image (undoing the letterbox).

### `model.predict`

```python
results = model.predict("test.png")            # also: PIL image, RGB numpy array, torch tensor, or a list of them
results = model.predict(photo, conf=0.25, iou=0.45, imgsz=640, max_det=300)

results.show()                                  # open the annotated image
results.save("prediction.png")                  # .png / .jpg
print(results.xyxy)                             # (N, 4) pixel boxes x1, y1, x2, y2 in the original image
print(results.classes)                          # ['square', 'circle', ...]
print(results.confidences)                      # (N,) scores
```

| Argument | Default | Meaning |
|---|---|---|
| `conf` | 0.25 | minimum score |
| `iou` | 0.45 | NMS overlap threshold |
| `imgsz` | the model's input size | letterbox size (int or `(H, W)`); must fit the architecture (checked with a clear error) |
| `max_det` | 300 | maximum detections per image |
| `device` | auto | if you never called `model.device(...)`, predict uses the best available device |
| `half` | CUDA only | fp16 autocast on CUDA |

Images are **letterboxed** (resized keeping the aspect ratio, padded grey), so any image size works.
A list input is processed in batches of 16 and returns a list of results.
An RGB array is expected: OpenCV frames are BGR, so convert them with `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`.

#### `Results`

| Member | Type |
|---|---|
| `xyxy`, `xywh` | float tensors `(N, 4)` |
| `confidences` | float tensor `(N,)` |
| `class_ids` | long tensor `(N,)` |
| `classes` | list of class names (the ids as text if the model has no names) |
| `len(results)`, iteration | iterate dicts with `xyxy`, `confidence`, `class_id`, `class` |
| `to_dict()`, `plot()`, `show()`, `save(path)` | list of dicts, annotated PIL image, viewer, file |

Class names come from the dataset used for training and are stored in the `.nbm`.

### Training targets and loss (what `train` optimises)

- **Assignment** (FCOS-style, by object size): a pyramid level handles objects up to 8 × its stride (and above
  the previous level's range), so small objects train the fine maps and large objects the coarse ones.
  Cells inside the box and within 1.5 strides of its centre are positives; every object always gets at least
  its centre cell; where objects overlap the smaller one wins.
- **Loss** = 5 × (1 − GIoU) on positive cells + BCE confidence (positives 1, all other cells 0) + BCE class,
  normalised by the number of positives.

---

## Training

```python
import felix as fx
import workflow as wf

dataset = fx.load("./dataset.felix")           # or a folder; model.train also accepts the path directly
model.train(dataset, epochs=100, batch_size=16, save_every=5)
```

Training uses **Felix** datasets; Workflow has no loader of its own. A Felix dataset provides
`dataset.train`, `dataset.val` and `dataset.test`: datasets converted from COCO keep their `split`, generated
ones are split deterministically 80/10/10 (see the Felix docs, *Formats*). Training uses the train
split and validates on the val split every epoch.

### `model.train(...)`

| Argument | Default | Meaning |
|---|---|---|
| `dataset` | required | Felix dataset object or path |
| `epochs` | 100 | **total** target epoch count (see Continuing) |
| `batch_size` | 16 | images per step |
| `save_every` | 5 | also write `epoch_N.nbm` every N epochs (`None`: only last/best) |
| `lr` | 1e-3 | peak learning rate |
| `imgsz` | the model's input | training size (int or `(H, W)`) |
| `device` | auto | otherwise the device set with `model.device(...)` |
| `workers` | auto | data-loading processes: 0 on Windows/macOS, up to 4 on Linux |
| `patience` | 30 | early stopping: stop after this many epochs without improvement (`None` disables) |
| `save_dir` | `runs/train` | where `.nbm` files go |
| `amp` | auto | mixed precision (CUDA only) |
| `ema` | True | keep an exponential moving average of the weights, used for validation and saved as the model |
| `augment` | True | horizontal flip + brightness/contrast jitter |
| `val` | True | validate every epoch |
| `quiet` | False | no progress panel |

Returns the training **history**: one dict per epoch with `epoch`, `loss`, `box`, `obj`, `cls`, `lr`,
`seconds`, `map50`, `map50_95`, `precision`, `recall`, `f1` and `fitness`. It is also kept in `model.history`.

`model.train(False)` and `model.train()` keep their usual torch meaning (`eval()` / train mode).

### Defaults and what they do

- **Optimizer** AdamW, weight decay 5e-4 on weights only (not BatchNorm / biases).
- **Schedule** linear warm-up (about 3 epochs, at least 100 steps) then cosine decay to 1% of `lr`.
- **Gradient clipping** at norm 10; training stops with a clear error if the loss becomes NaN.
- **Best model** is chosen by validation fitness `0.1 × mAP50 + 0.9 × mAP50-95` (by training loss if there is no
  validation split).
- **Speed**: mixed precision on CUDA, cuDNN autotuning, pinned memory and worker processes where supported.

### Saved files

Written to `save_dir` as `.nbm`: `last.nbm` after every epoch, `best.nbm` on every new best, and
`epoch_N.nbm` every `save_every` epochs. Each holds the weights (EMA), architecture, class names, history,
optimizer state and the raw weights needed to continue exactly. See [.nbm format](#the-nbm-format).

### Safe interruption

Press **Ctrl+C** at any time: training stops, the current state is saved to `save_dir/last.nbm`, and
`model.train` returns normally with the history so far. The epoch counter in the file is the last
*completed* epoch.

### Continuing

```python
model = wf.load("runs/train/last.nbm")        # or model.load(...) into an identical architecture
model.train(dataset, epochs=150, batch_size=16)
```

`epochs` is the **total** target: a run that stopped at epoch 80 continues from epoch 81 to 150. The optimizer
state, raw weights, EMA and history are restored. If a file has no optimizer state, or it doesn't fit, a fresh
optimizer is created. The learning-rate schedule is a function of progress towards the new target, so it
stretches to the new `epochs` rather than restarting. If the model already reached the target, nothing runs.

### Errors you may meet

| Message | Cause |
|---|---|
| `the dataset has 3 classes … but the model's BBoxHead was built with classes=1` | rebuild the head with the dataset's class count |
| `this model has no BBoxHead` | end the last stage with `wf.BBoxHead(...)` |
| `the loss became NaN/inf` | lower `lr`, check the dataset's boxes |
| `the training split is empty` | the dataset has fewer than 10 pairs and no splits |

---

## Benchmarking

`model.benchmark(...)` has two modes. They answer different questions and are never mixed up.

### Architecture benchmark: how heavy and how fast is it?

```python
bench = model.benchmark()                 # no dataset: says nothing about accuracy
bench = model.benchmark(imgsz=640, batch_size=8, device="cuda", iters=50)
```

Measures, on the chosen device (default: auto): **parameters**, **FLOPs/MACs**, **memory** (weights,
activations, inference peak), **latency** (batch 1, forward pass only, no NMS) and **FPS** (1000 / latency), plus
**throughput** at `batch_size`. It warms up first and synchronises CUDA/MPS so the timings are real.
Returns an `ArchitectureBenchmark` (`.show()`, `.to_dict()`, `.save("bench.json")` or `.csv`).

### Dataset evaluation: how accurate is it?

```python
results = model.benchmark(dataset.test)   # a trained model and a dataset split
results.show()
results.save("evaluation.json")           # or .csv
results.plot_confusion_matrix("confusion.png")
```

| Field | Meaning |
|---|---|
| `map50`, `map50_95` | COCO-style AP (101-point interpolation) at IoU 0.5, and averaged over IoU 0.5-0.95 |
| `precision`, `recall`, `f1` | at the confidence threshold that maximises mean F1 (`conf_threshold`) |
| `per_class` | `{name: {"ap50", "ap50_95", "instances"}}` |
| `confusion_matrix` | `(classes + 1)²` counts at conf 0.25, IoU 0.45; rows predicted, columns true; the last row/column is background (false positives / missed objects) |
| `latency_ms`, `fps` | per image: model forward + decode + NMS (data loading excluded) |
| `images`, `device`, `class_names` | |

Notes:

- Pass `dataset.test` (a split). If you pass a whole Felix dataset, its test split is used (validation if there is none).
- Evaluation runs with `conf=0.001` and NMS `iou=0.6` so the AP curve is complete; override with `conf=`, `iou=`.
- Boxes are compared in the letterboxed image space; IoU is unaffected by that uniform scaling.
- An untrained model prints a note that the numbers describe random weights.

From the terminal: `workflow benchmark model.nbm` or `workflow benchmark model.nbm --dataset data.felix --split test`.

---

## CLI

`workflow` (or `python -m workflow`). Same minimal noir look as Felix; colour off with `NO_COLOR`.

```
workflow info FILE.nbm            summary of a saved model (classes, training progress, shapes, FLOPs)
workflow devices                  cpu / cuda / mps, and what "auto" picks
workflow export FILE.nbm [...]    convert to TorchScript (.pt) or ONNX (.onnx)
workflow script FILE.nbm [...]    generate an inference script (image, folder, video, webcam)
workflow benchmark FILE.nbm [...] architecture speed, or accuracy with --dataset
workflow --version
```

### export

```
workflow export noble.nbm                       # → noble.pt   (TorchScript)
workflow export noble.nbm --format onnx         # → noble.onnx
workflow export noble.nbm -o out/model.onnx --size 640 --dynamic
```

| Option | Meaning |
|---|---|
| `-o, --output` | output path; the extension (`.pt` / `.onnx`) picks the format |
| `--format pt\|onnx` | format when `-o` is not given (default `pt`) |
| `--size N` or `H W` | input size to trace with (default: the model's own) |
| `--device` | device to trace on (default `cpu`) |
| `--dynamic` | ONNX: variable height/width (batch is always dynamic) |
| `--opset` | ONNX opset (default 17) |
| `--overwrite` | replace an existing output |

`.pt` is a self-contained TorchScript file: load it with `torch.jit.load` without Workflow installed.
Inputs are named `images`; outputs use the model's output names.

### script

Writes a ready-to-run Python inference script for a model, based on a preset:

```
workflow script noble.nbm --preset image        # → infer_image.py
workflow script noble.nbm --preset webcam -o cam.py
```

| Preset | What the script does | Run it |
|---|---|---|
| `image` | one image → annotated `prediction.png`, prints every detection | `python infer_image.py photo.jpg` |
| `folder` | every image in a folder → annotated copies + `detections.json` | `python infer_folder.py photos/` |
| `video` | a video file → live preview, optional `--save out.mp4` | `python infer_video.py clip.mp4` |
| `webcam` | live detection from a camera (`--mirror` for selfie view) | `python infer_webcam.py` (or a camera index) |

Without `--preset`, an interactive terminal asks which one. All scripts take `--conf`, `--iou`, `--imgsz` and
`--device` (default `auto`) and refer to the model by its absolute path. `video` and `webcam` need OpenCV
(`pip install opencv-python`, or the `workflow[video]` extra); press `q` to quit the preview. The scripts are plain
Python meant to be edited.

### benchmark

```
workflow benchmark noble.nbm                                         # architecture: params, FLOPs, memory, latency, FPS
workflow benchmark noble.nbm --dataset data.felix --split test       # accuracy: mAP, P/R/F1, per-class AP
workflow benchmark noble.nbm --dataset data.felix --save eval.json --confusion confusion.png
```

Options: `--split train|val|test`, `--imgsz`, `--batch-size`, `--device`, `--save` (`.json`/`.csv`),
`--confusion` (PNG). Evaluation needs the `felix` package. See [Benchmarking](#benchmarking).

---

## The .nbm format

`.nbm` (Noble Model) is Workflow's only model file. There is no separate checkpoint format.

It is a zip archive:

| Entry | Content |
|---|---|
| `meta.json` | format version, Workflow version, creation time, **architecture spec** (stages, operations, input shape), `classes` (names), `detection` (head configuration) and `training` (epoch, target epochs, best epoch, history, configuration) |
| `weights.pt` | the inference weights (the EMA weights when training used EMA), read with `weights_only=True` so loading never runs code |
| `train.pt` *(only after training)* | optimizer state, the raw (non-EMA) weights and the EMA update count, so training continues exactly |

A `.nbm` can be loaded for **inference** (`wf.load`), **fine-tuned**, or used to **continue training** towards a new
epoch target. The learning-rate schedule is a function of progress, so its "state" is the epoch and the target
stored in `training`; there is nothing else to restore.

Files are written atomically (a `.part` file, then renamed), so an interrupted save never leaves a
corrupt model.

```python
model.save("noble.nbm")
m = wf.load("noble.nbm")        # architecture + weights, no source needed
model.load("noble.nbm")         # weights into an identical architecture (else SerializationError)
```

The architecture spec records every operation and its arguments, so any model, including ones using
custom `wf.Op` subclasses that are importable, can be rebuilt from the file alone.

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Model, Stage, named outputs, shape inference, layers, Residual, Bottleneck, device selection, `.nbm`, CLI basics | **done** |
| 2 | Concat, Add, FPN, PAN, CSP, SPPF, routing validation | **done** |
| 3 | FLOPs/MACs, memory estimates, `plot()`, `export_graph()` | **done** |
| 4 | BBoxHead, decoding, loss, NMS, `predict()` and `Results`, inference scripts | **done** |
| 5 | Felix integration, training loop, validation, metrics, mixed precision, early stopping, safe Ctrl+C | **done** |
| 6 | Optimizer/EMA state in `.nbm`, training continuation, `benchmark()` | **done** |

Possible next steps (not started): fusing Conv+BatchNorm for faster inference, exporting the decoder and NMS
inside ONNX/TorchScript, stronger augmentation (mosaic, scale/translate), `torch.compile` integration and
multi-GPU training.

Design notes:

- `model.train(dataset, ...)` coexists with `nn.Module.train(mode)`: a bool (or nothing) keeps the torch meaning.
- `model.head()` finds the BBoxHead; it is a method (not a property) because `torch.jit.trace` evaluates properties.
