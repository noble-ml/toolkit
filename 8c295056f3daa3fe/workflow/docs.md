# Workflow: complete documentation

*This file is the whole documentation in one page. It is generated from the individual pages; do not edit it by hand.*

## Contents

- [Install](#install)
- [Demo](#demo)
- [Cookbook](#cookbook)
- [Getting started](#getting-started)
- [Layers](#layers)
- [Blocks](#blocks)
- [Composition](#composition)
- [Model](#model)
- [Detection and inference](#detection-and-inference)
- [Training](#training)
- [Evaluation](#evaluation)
- [Quantisation and export](#quantisation-and-export)
- [Devices and backends](#devices-and-backends)
- [Errors](#errors)
- [CLI](#cli)
- [Changes in 0.2](#changes-in-02)

---

Noble Engineering's CNN framework for bounding-box detection, on its **own compute engine**: a native C core
for the CPU, its own CUDA kernels for NVIDIA GPUs, and an experimental Apple-GPU backend. No PyTorch.

```python
import felix as fx
import workflow as wf

data = fx.load("shapes.felix")

model = wf.stack(
    wf.Conv2D(32, kernel_size=3, stride=2, padding=1),
    wf.BatchNorm2D(),
    wf.SiLU(),

    wf.Conv2D(64, kernel_size=3, stride=2, padding=1),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.CSP(),
    wf.feature("p3"),

    wf.Conv2D(128, kernel_size=3, stride=2, padding=1),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.SPPF(),
    wf.feature("p4"),

    wf.FPN("p3", "p4", channels=64),
    wf.head(classes=3),
)

model.train(data.train, validation=data.val, epochs=100, device="cuda")
wf.eval(model, dataset=data.test)                 # mAP50, mAP50-95, precision, recall, FLOPs, FPS
model.train(data.train, validation=data.val, epochs=110, quantization=wf.QAT(precision="int8"))
model.export(format="nbm", output="detector-int8.nbm", precision="int8")
model.predict("photo.jpg")                        # annotated image + predictions.json in results/
```

Input channels, tensor shapes, biases, parameter counts and FLOPs are worked out for you. When something doesn't
fit, the error says **what** happened, **why**, and **how to fix it**, naming the layer.

## What is in it

- **Layers**: `Conv2D`, `DepthwiseConv2D`, `PointwiseConv2D`, `SeparableConv2D`, `TransposeConv2D`, `DilatedConv2D`,
  `BatchNorm2D`, `ReLU`, `LeakyReLU`, `SiLU`, `Sigmoid`, `MaxPool2D`, `AvgPool2D`, `Upsample`.
- **Blocks**: `Bottleneck`, `CSP`, `SPP`, `SPPF`, `SE`, `CBAM`, `Focus`.
- **Composition**: `wf.stack`, `wf.Residual`, `wf.parallel`, `wf.feature`, `wf.concat`, `wf.add`, `wf.FPN`, `wf.PAN`, `wf.head`.
- **Training**: one call with sensible defaults (AdamW, warm-up + cosine, GIoU + BCE loss, EMA, early stopping,
  `.nbm` checkpoints, safe Ctrl+C, continuing from a saved model).
- **Evaluation**: `wf.eval(model)` (size, FLOPs, latency, FPS, memory) and `wf.eval(model, dataset=val)` (accuracy).
- **Quantisation**: QAT during training, post-training quantisation, real INT8 / FP16 files.
- **Export**: `.nbm` at any precision, ONNX (FP32, FP16, INT8 as QDQ).
- **Inference**: `model.predict(...)` on images or batches, results saved to `results/`.
- **Devices**: `"cpu"`, `"cuda"`, `"gpu"`, `"mps"` (experimental), `"auto"`.
- **CLI**: `workflow init | info | predict | eval | quantize | export | script | devices`. `workflow init` writes a
  ready-to-run starter notebook.

---

## Install

Workflow needs Python 3.9+, `rich` and `Pillow`. There is no PyTorch: the compute engine ships inside the package.
Datasets come from **Felix**, so install both.

```
pip install rich Pillow numpy
pip install --no-deps --no-index --find-links BASE_URL/links.html felix workflow
```

Check with `workflow --version` and `workflow devices`. If an older version is installed, add `--upgrade`.

Why the flags: `--no-index` keeps pip from installing the unrelated public packages also called `felix` and
`workflow`; with it pip cannot fetch dependencies, so `--no-deps` plus the first line installs them.

### Colab / notebooks

```
!pip install BASE_URL/files/felix-0.4.0-py3-none-any.whl BASE_URL/files/workflow-0.2.0-py3-none-any.whl
```

Direct wheel URLs need no flags; pip fetches the dependencies from PyPI. Colab GPUs work with `device="cuda"`.

### Platforms

| Platform | CPU | GPU |
|---|---|---|
| Linux x86-64 | prebuilt (SSE2 / AVX / AVX2, picked automatically) | NVIDIA via `"cuda"` |
| Linux ARM64 (e.g. Raspberry Pi) | prebuilt | none |
| macOS Apple silicon | prebuilt | Apple GPU via `"mps"` (experimental, `pip install mlx`) |
| other Unix / macOS Intel | compiled on first import (needs a C compiler) | NVIDIA if present |
| Windows | not supported natively: use WSL2 | NVIDIA through WSL2 |

The prebuilt Linux libraries need glibc 2.34 or newer (Debian 12+, Ubuntu 22.04+). On anything else Workflow
compiles its C core once, into `~/.cache/workflow`, the first time it is imported.

### GPUs

- **NVIDIA**: only the NVIDIA driver is needed (`nvidia-smi` must work). Workflow ships its own CUDA kernels as PTX;
  no CUDA toolkit, cuDNN or PyTorch is required.
- **Apple**: `pip install mlx` enables `device="mps"`. It is marked experimental and prints a warning when used.

### Optional extras

- `pip install opencv-python` for the video and webcam inference scripts.
- Felix reads JPEG datasets through Pillow (installed above).

---

## Demo

`examples/demo.py` runs everything in one script: **Felix** draws a dataset of circles, squares and triangles,
**Workflow** builds a detector, evaluates it before training, trains it, fine-tunes it with INT8 QAT, exports it,
and runs predictions.

```
python examples/demo.py                              # 600 images, 20 epochs + 3 QAT epochs
python examples/demo.py --epochs 40 --device cuda
```

Outputs in `demo_out/`: the dataset, `architecture.png`, training checkpoints (`runs/`, `runs_qat/`),
`detector.nbm`, `detector-int8.nbm`, `detector.onnx`, and `results/` with annotated predictions and
`predictions.json`.

### What it does

1. **Data.** A Felix pipeline with three shape classes, 1-3 objects per image (thresholds), random scale, rotation,
   hue and brightness, and an unannotated grey decoy. Felix splits it into 480 train, 60 val and 60 test images.
2. **Architecture.** A backbone of strided Conv-BN-SiLU steps with CSP blocks and SPPF, three named features,
   `wf.FPN` + `wf.PAN` and `wf.head(classes=3)`: 1.29M parameters, 0.80 GFLOPs at 256×256.
3. **`wf.eval(model)`** before training: size, FLOPs, latency and FPS.
4. **Training**: `model.train(train, validation=val, epochs=20)`.
5. **QAT fine-tune**: `model.train(..., epochs=23, quantization=wf.QAT(precision="int8"))` continues from epoch
   21 with the trained weights and optimizer, simulating int8 for three epochs.
6. **Export**: an INT8 `.nbm` with real int8 weights, and an ONNX file.
7. **Prediction**: the INT8 model loaded from disk detects objects in three test images; results go to `results/`.

### Results

Measured on an Apple-silicon laptop CPU (Workflow's native C core, 8 threads):

| | |
|---|---|
| training | about 18 s per epoch (480 images, batch 16); QAT epochs about 24 s |
| test mAP50 / mAP50-95 (fp32) | **0.974 / 0.840** |
| test mAP50-95 after INT8 QAT | **0.840** (no loss) |
| model files | 19.7 MB `.nbm` with training state · **1.3 MB** INT8 `.nbm` · 4.9 MB ONNX |
| inference | about 57 FPS at batch 1, about 109 images/s at batch 8 (CPU) |

The same script trains on a GTX 1050 with `--device cuda` (about 11 s per epoch at this size). Numbers vary with the
random dataset and hardware.

![Predictions of the INT8 model: two squares and an overlapping triangle](docs/img/prediction.png)

![The computation graph of the demo model (top part)](docs/img/architecture_top.png)

---

## Cookbook

Every tool, explained like you're three, then *why* you'd reach for it and where it goes. Every example writes
out all the main arguments (with their defaults) so you can see every knob, even the ones you'd usually leave alone.

Felix makes the pictures. Workflow is the brain that learns to find things in them.

How to read an entry:

- **Toddler words**: what it is.
- **Why you'd use it**: the reasoning, and where it usually sits.
- **Example**: every main argument shown.
- **Watch out**: the one mistake people make (when there is one).

---

### Part 1 · Felix: making the pictures

#### `fx.class_set`

**Toddler words:** a toy box with a name. Every toy you make inside it is a *kind of thing* the brain must learn
to find.

**Why you'd use it:** the brain can only learn classes you define, and the boxes let you group them (animals,
vehicles) so you can later pick "any animal" or "any vehicle" at random. The class name comes from the variable
name, so `cat = ...` makes a class called `cat`.

```python
with fx.class_set("animals"):
    cat = fx.toolkit.circle(radius=30)              # class "cat"
    dog = fx.toolkit.rectangle(width=50, height=40) # class "dog"
```

**Watch out:** shapes made *outside* a class set are not classes: they get drawn but never labelled. That's how you
make decoys on purpose.

#### `fx.toolkit.circle`, `rectangle`, `polygon`

**Toddler words:** a ball, a box, and a "connect the dots" shape.

**Why you'd use it:** quick synthetic objects for testing a pipeline, or for detecting simple things (signs,
markers, parts). `position` is only a default spot; usually `fx.paint` decides where it goes.

```python
ball = fx.toolkit.circle(radius=20, position=None, color="red", name=None)
box  = fx.toolkit.rectangle(width=40, height=30, position=None, color="#1f77b4", name=None)
tent = fx.toolkit.polygon(vertices=[(0, 40), (20, 0), (40, 40)], position=None, color="green", name=None)
```

**Watch out:** without `color`, each class gets its own colour from a palette. Set it when the colour matters.

#### `fx.stack`

**Toddler words:** glue several shapes together into one toy. A big circle, two triangle ears and two dot eyes
make one cat, and the brain learns "cat", not "circle" and "triangle".

**Why you'd use it:** real objects are made of parts. A stack lets you build a recognisable object from simple
shapes and still get **one** class and **one** box around the whole thing. The class name is the variable name.
Each part's `position` says where its centre goes, measured from the middle of the stack.

```python
with fx.class_set("animals"):
    cat = fx.stack(
        fx.toolkit.polygon(vertices=[(0, 14), (7, 0), (14, 14)], position=(-13, -22), color=None),  # left ear
        fx.toolkit.polygon(vertices=[(0, 14), (7, 0), (14, 14)], position=(13, -22), color=None),   # right ear
        fx.toolkit.circle(radius=22, position=(0, 0), color=None),                                  # head
        fx.toolkit.circle(radius=3, position=(-8, -4), color="black"),                               # eyes
        fx.toolkit.circle(radius=3, position=(8, -4), color="black"),
        position=None, color="#e8a33d", name=None,
    )
```

**Watch out:** parts are drawn in order, so put the things that go *on top* (eyes, nose) last. Parts without a
colour use the stack's colour; parts with one keep it. Spinning or growing the stack moves all its parts together.

#### `fx.toolkit.image`

**Toddler words:** use a real sticker as a toy. The see-through parts of the sticker don't count as the toy.

**Why you'd use it:** cut-out photos of real objects (PNG with transparency) look far more like real life than
plain shapes. The box is fitted to the visible pixels only, so transparent padding never makes boxes too big.

```python
bike = fx.toolkit.image(source="assets/bicycle.png", width=80, height=None, position=None, name=None)
```

**Watch out:** give only `width` *or* `height` to keep the proportions; give both to stretch. JPEG stickers need
Pillow, and have no see-through parts.

#### `fx.step`

**Toddler words:** the recipe for one picture. Felix follows it again and again, rolling new dice each time.

**Why you'd use it:** everything you write inside it (background, painting, changes) happens once per image. Make
several steps (for example "busy street" and "empty street") and Felix shares the images between them.

```python
with fx.step(size=(640, 640), name=None):
    ...
```

#### `fx.canvas.background`

**Toddler words:** the paper you draw on: one colour, or a photo.

**Why you'd use it:** a plain colour is easy to learn; real photos (a folder of them, one picked per image) teach the
brain to ignore clutter, which is what you want for real-world use.

```python
fx.canvas.background(color="#eeeeee", image=None)
fx.canvas.background(color=None, image="backgrounds/")    # a different photo each time, cropped to fit
```

#### `fx.random.*`

**Toddler words:** dice. Every picture rolls them again.

**Why you'd use it:** variety. A brain that only ever sees a cat in the middle, upright and red will fail on
anything else. Randomness in position, size, angle and colour is what makes it robust.

```python
fx.random.uniform(0.5, 2.0, threshold=1.0)    # any number between 0.5 and 2
fx.random.integer(1, 6, threshold=1.0)        # a whole number, like a real die
fx.random.choice("red", "blue", threshold=1.0)
fx.random.position(margin=40, threshold=1.0)  # somewhere on the paper, at least 40 px from the edges
fx.random.color(threshold=1.0)
fx.random.class_set("animals", threshold=1.0) # pick a toy box (any box if no names)
fx.random.class_(None, threshold=1.0)         # pick a toy (from any box, or from the box you pass)
```

**Watch out:** `threshold=0.5` means "flip a coin first; if it says no, skip it". It's how you get a *varying number*
of objects. Each dice call is its own die: the same variable used twice gives the same roll within one picture.

#### `fx.paint`

**Toddler words:** put a toy on the paper.

**Why you'd use it:** this is what actually places objects. Pass a class, a random class, or a plain shape.

```python
obj = fx.paint(fx.random.class_(None, threshold=1.0), position=fx.random.position(margin=40))
```

#### `fx.toolkit.transformation`

**Toddler words:** change how the toy looks: spin it, grow it, recolour it, brighten it, fade it.

**Why you'd use it:** the same as dice: real objects come at all angles, sizes and lighting. Calls add up (two
rotations add, two scales multiply).

```python
fx.toolkit.transformation(obj, hue=fx.random.uniform(-20, 20), brightness=fx.random.uniform(0.8, 1.2),
                          rotation=fx.random.uniform(0, 360), scale=fx.random.uniform(0.7, 1.5), opacity=1.0)
```

**Watch out:** `hue` is in degrees around the colour wheel, `rotation` is in degrees counter-clockwise, and the
others are multipliers (1 = unchanged).

#### `fx.gen`

**Toddler words:** take a photo of where every toy is (the answer key). Toys painted *after* the photo are in
the picture but not in the answers.

**Why you'd use it:** it writes the boxes. Painting things after it adds **decoys** the brain must learn to
ignore, which cuts false alarms.

```python
fx.gen()
fx.paint(fx.toolkit.circle(radius=8, color="gray"), position=fx.random.position(margin=0))   # a decoy
```

#### `fx.make`

**Toddler words:** do the recipe lots of times and put everything in a folder (or one `.felix` file).

**Why you'd use it:** this is the moment the dataset is created. Same `seed` = the exact same dataset, which is
handy for comparing models fairly.

```python
fx.make(pairs=1000, output="shapes.felix", seed=7, overwrite=False, quiet=False)
```

#### `fx.load` and its splits

**Toddler words:** open the dataset. It already has three piles: `train` (learn from these), `val` (check while
learning), and `test` (the final exam nobody peeks at).

**Why you'd use it:** you pass these piles straight to Workflow. Generated datasets are split 80/10/10, the same
way every time. Converted datasets keep their own splits.

```python
data = fx.load("shapes.felix")
data.train, data.val, data.test          # the piles
data.classes                             # ['circle', 'square', 'triangle']
data.image(data.stems[0]).save("look.png")
```

#### `fx.partition` / `felix partition`

**Toddler words:** cut one big pile into smaller piles, each a complete dataset.

**Why you'd use it:** share work between machines, make a small dataset for quick tests, or split a converted
dataset your own way.

```python
fx.partition("shapes.felix", output=None, parts=4, ratio=None, size=None, by_split=False,
             names=None, shuffle=False, seed=0, pack=None, overwrite=False)
fx.partition("shapes.felix", ratio=(80, 10, 10), names=["train", "val", "test"], shuffle=True, seed=1)
```
```
felix partition shapes.felix --size 100
```

**Watch out:** choose exactly one of `parts`, `ratio`, `size` or `by_split`.

#### Other Felix commands

**Toddler words:** look inside a box, zip it up, unzip it, and turn someone else's box into yours.

**Why you'd use it:** checking what you made, sending one file instead of thousands, and training on real
Roboflow datasets.

```
felix info shapes.felix
felix pack ./dataset -o dataset.felix
felix unpack dataset.felix -o ./dataset
felix convert coco ./roboflow-export -o data.felix
```

---

### Part 2 · Workflow: the brain's building blocks

#### `wf.stack`

**Toddler words:** line up blocks. The picture goes through them one after another, like a slide with bumps.

**Why you'd use it:** it's the model itself (the outermost one) and also any group of layers you want to treat as
one. `input` is the picture size the model is built and checked for.

```python
model = wf.stack(
    wf.Conv2D(16, kernel_size=3, stride=1, padding=1),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.feature("p1"),
    input=(3, 640, 640),
)
```

**Watch out:** you never say how many channels go *in* to a layer. Workflow works that out.

#### `padding` (used by every layer with a window)

**Toddler words:** a cushion of empty pixels around the picture so the edges don't get squished away.

**Why you'd use it:** without padding, every 3×3 window shrinks the picture by 2 pixels. `padding = kernel_size // 2`
(1 for size 3, 2 for size 5) keeps it the same size at stride 1.

```python
padding=1                     # 1 pixel on every side
padding=(1, 2)                # 1 top and bottom, 2 left and right
padding=(0, 1, 0, 1)          # top, bottom, left, right
```

**Watch out:** only numbers. `"same"` and `"valid"` are refused (the error tells you the number to use).

#### `wf.Conv2D`

**Toddler words:** little magnifying glasses that slide over the picture, each looking for one kind of pattern (an
edge, a corner, a stripe). `filters=32` means 32 different glasses.

**Why you'd use it:** it's *the* building block. Early convs find edges; later ones find parts and whole objects.
`stride=2` takes big steps, halving the picture; that's how a backbone goes from big-and-shallow to small-and-deep.

```python
wf.Conv2D(32, kernel_size=3, stride=1, padding=1)    # same size, 32 channels
wf.Conv2D(64, kernel_size=3, stride=2, padding=1)    # half size, 64 channels: a "downsampling" step
```

#### `wf.DepthwiseConv2D`

**Toddler words:** one magnifying glass per colour channel, and each one only looks at its own channel.

**Why you'd use it:** it's very cheap. Mobile-style models use it to look around spatially for almost no cost,
then mix the channels with a pointwise conv.

```python
wf.DepthwiseConv2D(kernel_size=3, stride=1, padding=1)    # keeps the channel count
```

#### `wf.PointwiseConv2D`

**Toddler words:** a 1×1 glass. It doesn't look around; it just mixes the channels at each spot.

**Why you'd use it:** changing the number of channels cheaply: squeezing before expensive work, or matching
channels before an `add`.

```python
wf.PointwiseConv2D(64)       # kernel 1, stride 1, padding 0 are fixed: that's what "pointwise" means
```

#### `wf.SeparableConv2D`

**Toddler words:** depthwise, then pointwise: look around, then mix.

**Why you'd use it:** a drop-in for `Conv2D` that does nearly the same job for a fraction of the maths. Good for
small or fast models.

```python
wf.SeparableConv2D(64, kernel_size=3, stride=1, padding=1)
```

#### `wf.DilatedConv2D`

**Toddler words:** a magnifying glass with gaps between its eyes, so it sees farther without costing more.

**Why you'd use it:** a bigger view (more context) without shrinking the picture or adding work. Useful late in
a network or for big objects.

```python
wf.DilatedConv2D(32, kernel_size=3, dilation=2, stride=1, padding=2)   # padding = dilation keeps the size for kernel 3
```

#### `wf.TransposeConv2D`

**Toddler words:** a magnifying glass that makes the picture *bigger* (it un-squishes), and learns how.

**Why you'd use it:** growing a small, smart map back to a bigger size, for example in a neck. Unlike `Upsample`,
which just copies pixels, it *learns* how to fill in the new pixels.

```python
wf.TransposeConv2D(32, kernel_size=2, stride=2, padding=0)   # exactly twice the height and width
```

Output size: `(size − 1) × stride − 2 × padding + kernel_size`. With `kernel_size=2, stride=2, padding=0` that is
exactly double, which is why those are the defaults. `kernel_size=4, stride=2, padding=1` also doubles and
blends neighbours more smoothly.

**Watch out:** it stays float in int8 models (Workflow warns you). Use `Upsample` + `Conv2D` if everything must be int8.

#### `wf.BatchNorm2D`

**Toddler words:** keeps the numbers from getting too big or too small, so learning doesn't wobble.

**Why you'd use it:** almost always right after a convolution: `Conv2D → BatchNorm2D → activation`. Models train
faster and more reliably with it. The conv before it automatically drops its bias, which the BatchNorm would cancel
anyway.

```python
wf.BatchNorm2D()         # channels are worked out for you
```

#### `wf.ReLU`, `wf.LeakyReLU`, `wf.SiLU`, `wf.Sigmoid`

**Toddler words:** mood filters. ReLU says "no sad numbers, make them zero". LeakyReLU says "a tiny bit of sad is
okay". SiLU is a smooth ReLU. Sigmoid squishes everything between 0 and 1.

**Why you'd use it:** without them, stacking layers is no smarter than one layer. SiLU is the modern default for
detectors; ReLU is the cheapest; Sigmoid is for gates and probabilities, not between layers.

```python
wf.ReLU()
wf.LeakyReLU(slope=0.1)
wf.SiLU()
wf.Sigmoid()
```

#### `wf.MaxPool2D`, `wf.AvgPool2D`

**Toddler words:** shrink the picture. Max keeps the loudest pixel in each little square; Avg keeps the average.

**Why you'd use it:** a cheap way to halve the size with no weights to learn. Max is good at keeping "is the
pattern anywhere here?"; Avg smooths things out.

```python
wf.MaxPool2D(kernel_size=2, stride=None, padding=0)      # stride defaults to kernel_size: half the size
wf.AvgPool2D(kernel_size=3, stride=1, padding=1)         # same size, just smoothed
```

#### `wf.Upsample`

**Toddler words:** make the picture bigger by copying each pixel.

**Why you'd use it:** cheap resizing with nothing to learn, usually before joining a small map to a big one.

```python
wf.Upsample(scale=2)
```

---

### Part 3 · Workflow: bigger blocks

#### `wf.Bottleneck`

**Toddler words:** squeeze the picture thin (fewer channels), look around, puff it back up, then add the original
back so nothing gets lost.

**Why you'd use it:** to make a network deeper without making it slow. The expensive 3×3 look-around happens on
the thin version, and the added shortcut keeps learning stable in deep stacks.

```python
wf.Bottleneck(filters=None, kernel_size=3, stride=1, reduction=2, shortcut=True)   # same shape out as in
wf.Bottleneck(filters=128, kernel_size=3, stride=2, reduction=2, shortcut=True)    # wider and half size
```

**Watch out:** when `filters` or `stride` change the shape, the shortcut gets a 1×1 projection automatically,
so you never have to match the shapes yourself.

#### `wf.CSP`

**Toddler words:** split into two paths. One does hard work (bottlenecks), one takes it easy. Then glue them
together.

**Why you'd use it:** most of the accuracy of a deep stack of bottlenecks at a lot less cost. It's the standard
backbone block after each downsampling conv.

```python
wf.CSP(filters=None, n=1, shortcut=True)      # n = how many bottlenecks on the busy path
```

#### `wf.SPP`, `wf.SPPF`

**Toddler words:** look at the picture through small, medium and big windows at the same time, then glue what
you saw.

**Why you'd use it:** at the end of the backbone, so the brain sees both details and the big picture. SPPF gets
the same result faster with three chained small pools, so prefer it.

```python
wf.SPP(filters=None, kernels=(5, 9, 13))
wf.SPPF(filters=None, kernel_size=5)
```

#### `wf.SE`

**Toddler words:** each channel asks "am I important for *this* picture?" and turns its own volume up or down.

**Why you'd use it:** a small, cheap accuracy boost, usually after a backbone block. It learns to pay more
attention to the channels that matter for the image at hand.

```python
wf.SE(reduction=16)       # how much the "am I important?" helper squeezes; 16 is the classic value
```

#### `wf.CBAM`

**Toddler words:** SE, plus "which *spots* in the picture matter?": a spotlight for channels and places.

**Why you'd use it:** like SE, but it also helps with *where*, which suits detection. Costs a bit more.

```python
wf.CBAM(reduction=16, kernel_size=7)
```

#### `wf.Focus`

**Toddler words:** cut the picture into 2×2 tiles and stack the tiles as channels: half the size, nothing thrown
away.

**Why you'd use it:** as the very first layer, a fast way to shrink a big input without losing pixels.

```python
wf.Focus(filters=32, kernel_size=3)
```

**Watch out:** the input height and width must be even.

---

### Part 4 · Workflow: wiring

#### `wf.Residual`

**Toddler words:** do some stuff, then add what you started with. "Remember where you came from."

**Why you'd use it:** to build your own residual block. It lets very deep models train, because the brain can
learn "a small change to the input" instead of a whole new picture.

```python
wf.Residual(
    wf.Conv2D(64, kernel_size=3, stride=1, padding=1),
    wf.BatchNorm2D(),
    wf.SiLU(),
    shortcut=None,                                    # same shape in and out: add the input as it is
)

wf.Residual(
    wf.Conv2D(128, kernel_size=3, stride=2, padding=1),
    wf.BatchNorm2D(),
    wf.SiLU(),
    shortcut=wf.Conv2D(128, kernel_size=1, stride=2, padding=0),   # shape changes → project the input to match
)
```

**Watch out:** both paths must end the same shape. If they don't, the error tells you the exact `shortcut=` to add.

#### `wf.parallel`

**Toddler words:** give the same picture to several friends, then glue (or add up) their answers.

**Why you'd use it:** let the brain look at the same thing in different ways at once (small and big windows,
for example), Inception-style.

```python
wf.parallel(
    wf.Conv2D(16, kernel_size=1, stride=1, padding=0),
    wf.Conv2D(16, kernel_size=3, stride=1, padding=1),
    merge="concat",                                   # or merge="add" (then the outputs must match)
)
```

#### `wf.feature`

**Toddler words:** stick a note with a name on the current picture so you can grab it later.

**Why you'd use it:** detectors need maps from several depths (big and detailed, small and smart). You name them
in the backbone and use them in the neck.

```python
wf.feature("p3")
wf.feature(("n3", "n4", "n5"))        # after FPN/PAN: one name per map of the pyramid
```

#### `wf.concat`, `wf.add`

**Toddler words:** grab named notes and glue them side by side (concat) or add them up (add).

**Why you'd use it:** to hand-build skip connections or necks. Concat keeps both sets of channels (more
information, more channels); add mixes them (same channels, cheaper).

```python
wf.concat("p3", "up")      # needs the same height and width; channels add up
wf.add("a", "b")           # needs identical shapes
```

#### `wf.FPN`

**Toddler words:** big pictures see small details; small pictures know *what* things are. FPN passes the "what"
knowledge down into the big pictures.

**Why you'd use it:** so small objects (found on the big maps) are recognised as well as large ones. It takes your
named maps in any order, any channel counts, and handles the resizing and channel matching itself.

```python
wf.FPN("p3", "p4", "p5", channels=64)      # channels=None uses the smallest input's channel count
```

#### `wf.PAN`

**Toddler words:** the other way round: it passes the sharp details back up to the small pictures.

**Why you'd use it:** right after FPN, so the maps that find big objects also get precise edges. FPN + PAN is the
classic modern neck.

```python
wf.PAN(channels=64)                        # works on the pyramid FPN just produced
```

#### `wf.head`

**Toddler words:** the mouth. For every spot it says "there's a cat here, this big, and I'm this sure".

**Why you'd use it:** it's what turns features into boxes. It must be the last layer, and `classes` must match your
dataset (Workflow checks).

```python
wf.head(classes=3, channels=None)          # channels=None: as wide as each incoming map
```

---

### Part 5 · Workflow: using the brain

#### `model.summary()`

**Toddler words:** "tell me about yourself": every layer, how big, and how much work it does.

**Why you'd use it:** before training, to check the shapes come out as you expect and the model isn't accidentally
huge. It builds the model, so shape mistakes show up here, not halfway through training.

```python
model.summary(input=None, show=True)       # input=(3, 416, 416) checks another size
```

#### `model.export_graph`

**Toddler words:** draw a picture of the brain.

**Why you'd use it:** to see your wiring (shortcuts, branches, which feature goes where) or to share it. SVG for
documents, PNG for quick looks.

```python
model.export_graph("brain.png", detail="layers", scale=2)     # detail="compact" for one row per layer
```

#### `model.device`

**Toddler words:** pick which helper does the maths: the computer's own brain (`"cpu"`), a big NVIDIA graphics card
(`"cuda"`), any graphics card (`"gpu"`), or a Mac's graphics chip (`"mps"`, still learning to walk).

**Why you'd use it:** GPUs are much faster for training. Usually you just pass `device=` to `train` or `predict`
instead.

```python
model.device("cuda")      # or "cpu", "gpu", "mps", "auto"
model.device()            # which one am I on?
```

**Watch out:** if you ask for a GPU that isn't there, you get an error saying why, never a silent switch to CPU.

#### `model.train`

**Toddler words:** look at lots of pictures and their answers, again and again, until you get good. Press Ctrl+C
and it saves before stopping.

**Why you'd use it:** this is learning. The defaults (optimizer, learning-rate schedule, loss, augmentation,
checkpoints) are sensible, so most of the time you only set the data and `epochs`.

```python
model.train(data.train, validation=data.val, epochs=50, batch_size=16, lr=1e-3, device="auto",
            img_size=None, optimizer="adamw", patience=30, save_every=5, output="runs/train",
            quantization=None, augment=True, ema=True)
```

**Watch out:** `epochs` means "train *until* epoch N". To continue a model that reached 50, ask for `epochs=80` to get
30 more; `epochs=30` would do nothing.

#### `wf.eval`

**Toddler words:** a report card. Without a test: "how big and how fast am I?". With a test: "how many do I get
right?".

**Why you'd use it:** before training to compare architectures (size, speed); after training to measure accuracy on
the test pile. It never changes the model.

```python
wf.eval(model, dataset=None, device=None, img_size=None, batch_size=8, show=True)
wf.eval(model, dataset=data.test)              # adds mAP50, mAP50-95, precision, recall
```

#### `wf.QAT`

**Toddler words:** practise with smaller, rounder numbers while learning, so the brain still works after it's shrunk.

**Why you'd use it:** when you'll ship an int8 model (smaller, faster on small devices). Usually a short fine-tune
*after* normal training: it starts from your trained weights.

```python
model.train(data.train, validation=data.val, epochs=55, quantization=wf.QAT(precision="int8"))   # 5 epochs after 50
```

#### `model.quantize` / `workflow quantize`

**Toddler words:** shrink a finished brain so it's small and fast. Show it some pictures first (calibration) so it
learns how big the numbers get.

**Why you'd use it:** the quick way to int8 or fp16 without retraining. Try it first; use QAT if accuracy drops
too much.

```python
model.quantize("int8", calibration=data.val)     # "fp16" needs no calibration
```
```
workflow quantize detector.nbm int8 --calibration shapes.felix
```

#### `model.save`, `wf.load`, `model.load`

**Toddler words:** put the brain in a box (`.nbm`) and take it out again later.

**Why you'd use it:** keep a trained model, share it, or continue training it. The box holds the layers, weights,
class names and training progress.

```python
model.save("detector.nbm")
model = wf.load("detector.nbm")       # rebuilds everything from the box
model.load("detector.nbm")            # fills a model you already built the same way
```

#### `model.export`

**Toddler words:** pack the brain for other programs (ONNX), or make a smaller box.

**Why you'd use it:** ONNX runs in many runtimes (ONNX Runtime, mobile, servers); an int8 `.nbm` is the smallest
Workflow file.

```python
model.export(format="onnx", output="detector.onnx", precision="fp32", calibration=None, img_size=None)
model.export(format="nbm", output="detector-int8.nbm", precision="int8", calibration=data.val)
```

#### `model.predict`

**Toddler words:** "what's in this picture?". It draws boxes and puts the pictures and a list of answers in
`results/`.

**Why you'd use it:** to actually use the detector, on one image, many images, or video frames.

```python
r = model.predict("photo.jpg", device=None, confidence=0.25, iou=0.45, img_size=None,
                  output="results/", max_detections=300)
r.classes, r.confidences, r.xyxy
r.show()
```

**Watch out:** raise `confidence` for fewer, surer boxes; lower `iou` if one object gets several boxes.
`output=None` saves nothing, which is handy for video.

#### Errors

**Toddler words:** when something is wrong, Workflow tells you what broke, why, and how to fix it, like a kind
teacher.

**Why it matters:** you fix it from the message instead of digging through code. The first line also names the
layer.

```
layer 4 (Residual): Residual cannot add its two paths.
  why: the layers output (16, 15, 15) but the shortcut carries (8, 30, 30); addition needs identical shapes.
  fix: ... project the input with shortcut=wf.Conv2D(16, kernel_size=1, stride=2).
```

#### `workflow init`

**Toddler words:** "give me a notebook that already works." It writes a Jupyter notebook with a small brain, a
report card before and after learning, and saving.

**Why you'd use it:** the fastest way to start a new detector, or to show someone how Workflow fits together.
Every step is a cell you can read and change; the model is a flat list of layers you can grow. Without a dataset it
makes a practice one, so the first run always works.

```
workflow init detector.ipynb --dataset data.felix --img-size 256 --epochs 30 --device auto
```

**Watch out:** the image size must be a multiple of 32 (the starter model halves the picture four times).

#### Workflow commands

**Toddler words:** the same tools, typed in the terminal.

**Why you'd use it:** checking, converting and running models without writing Python.

```
workflow init my_detector.ipynb                     # a starter notebook
workflow info detector.nbm
workflow predict detector.nbm photo.jpg --confidence 0.25
workflow eval detector.nbm --dataset shapes.felix --split test
workflow quantize detector.nbm int8 --calibration shapes.felix
workflow export detector.nbm --format onnx --precision fp32
workflow script detector.nbm --preset webcam         # writes a ready-to-run camera program
workflow devices
```

---

### Part 6 · Recipes: how the pieces fit

**A downsampling step** (the most common pattern in a backbone): three lines that halve the picture and add
channels. You'll see it again and again in the detector below.

```python
wf.Conv2D(64, kernel_size=3, stride=2, padding=1),
wf.BatchNorm2D(),
wf.SiLU(),
```

**A small detector**: the backbone halves the picture five times, naming three depths with `wf.feature`; SPPF
widens the view; FPN + PAN mix the named maps; the head finds the boxes.

```python
model = wf.stack(
    # stem: 256 → 128
    wf.Conv2D(16, kernel_size=3, stride=2, padding=1),
    wf.BatchNorm2D(),
    wf.SiLU(),

    # 128 → 64
    wf.Conv2D(32, kernel_size=3, stride=2, padding=1),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.CSP(n=1),

    # 64 → 32: good for small objects
    wf.Conv2D(64, kernel_size=3, stride=2, padding=1),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.CSP(n=2),
    wf.feature("p3"),

    # 32 → 16: medium objects
    wf.Conv2D(128, kernel_size=3, stride=2, padding=1),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.CSP(n=2),
    wf.feature("p4"),

    # 16 → 8: large objects
    wf.Conv2D(256, kernel_size=3, stride=2, padding=1),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.SPPF(kernel_size=5),
    wf.feature("p5"),

    # neck and head
    wf.FPN("p3", "p4", "p5", channels=64),
    wf.PAN(channels=64),
    wf.head(classes=3),

    input=(3, 256, 256),
)
```

**Train, then shrink for deployment:**

```python
model.train(data.train, validation=data.val, epochs=50)
wf.eval(model, dataset=data.test)
model.train(data.train, validation=data.val, epochs=55, quantization=wf.QAT(precision="int8"))
model.export(format="nbm", output="detector-int8.nbm", precision="int8")
```

**A cheaper backbone block** (mobile-style): look around cheaply, mix, add back:

```python
wf.Residual(
    wf.DepthwiseConv2D(kernel_size=3, stride=1, padding=1),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.PointwiseConv2D(64),
    wf.BatchNorm2D(),
)
```

---

## Getting started

The quickest start is `workflow init`: it writes a notebook with a small model, benchmarks before and after
training, and saving, that runs as-is ([CLI](#init)). This page explains the pieces.

A model is a **stack** of layers. Each layer receives the previous layer's output, so the only numbers you write
are the ones you decide: filters, kernel sizes, strides, padding.

```python
import workflow as wf

model = wf.stack(
    wf.Conv2D(32, kernel_size=3, stride=2, padding=1),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.Bottleneck(),
    input=(3, 640, 640),
)
model.summary()
```

- **Input channels are inferred.** `wf.Conv2D(32, kernel_size=3)` works out how many channels it receives.
- **Shapes are checked before anything runs.** The first use of a model builds it for `input=` (default 3×640×640)
  and validates every layer.
- **Padding is explicit.** `padding=1` adds one pixel on each side; there is no `"same"`. For stride 1, `padding=kernel_size//2`
  keeps the size.
- **Bias is automatic.** A convolution right before `BatchNorm2D` has none; any other has one.

### A detector

One layer per line, top to bottom, exactly as the picture flows through them:

```python
model = wf.stack(
    # stem: 256 → 128
    wf.Conv2D(16, kernel_size=3, stride=2, padding=1),
    wf.BatchNorm2D(),
    wf.SiLU(),

    # 128 → 64
    wf.Conv2D(32, kernel_size=3, stride=2, padding=1),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.CSP(n=1),

    # 64 → 32: good for small objects
    wf.Conv2D(64, kernel_size=3, stride=2, padding=1),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.CSP(n=2),
    wf.feature("p3"),

    # 32 → 16: medium objects
    wf.Conv2D(128, kernel_size=3, stride=2, padding=1),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.CSP(n=2),
    wf.feature("p4"),

    # 16 → 8: large objects
    wf.Conv2D(256, kernel_size=3, stride=2, padding=1),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.SPPF(kernel_size=5),
    wf.feature("p5"),

    # neck and head
    wf.FPN("p3", "p4", "p5", channels=64),
    wf.PAN(channels=64),
    wf.head(classes=3),

    input=(3, 256, 256),
)
```

### Train, evaluate, use

```python
import felix as fx
data = fx.load("shapes.felix")                       # Felix gives train / val / test splits

model.train(data.train, validation=data.val, epochs=50)
wf.eval(model, dataset=data.test)
model.save("detector.nbm")

model = wf.load("detector.nbm")
results = model.predict("photo.jpg")                 # → results/photo.png and results/predictions.json
print(results.xyxy, results.classes, results.confidences)
```

Next: [Layers](#layers), [Blocks](#blocks) and [Composition](#composition) for building; [Training](#training),
[Evaluation](#evaluation) and [Quantisation](#quantisation-and-export) for the rest. A full runnable example is the [Demo](#demo).

---

## Layers

Every layer infers its input channels. Shapes below are `(channels, height, width)`; *pair* means an int or an
`(height, width)` tuple.

### Padding

Padding is always a number of pixels, never `"same"` / `"valid"` (those raise an error that suggests the number):

| `padding=` | Meaning |
|---|---|
| `0` | none (the old "valid") |
| `1`, `2`, ... | that many pixels on every side |
| `(ph, pw)` | `ph` top and bottom, `pw` left and right |
| `(top, bottom, left, right)` | asymmetric |

Output size along one axis: `⌊(size + pad_before + pad_after − dilation·(kernel − 1) − 1) / stride⌋ + 1`.
For stride 1, `padding = kernel_size // 2` keeps the size (odd kernels).

### Convolutions

| Layer | Signature | Output channels | Notes |
|---|---|---|---|
| `Conv2D` | `Conv2D(filters, kernel_size=3, stride=1, padding=0)` | `filters` | standard convolution |
| `DilatedConv2D` | `DilatedConv2D(filters, kernel_size=3, dilation=2, stride=1, padding=0)` | `filters` | taps spread `dilation` pixels apart; `padding=dilation` keeps the size for kernel 3 |
| `DepthwiseConv2D` | `DepthwiseConv2D(kernel_size=3, stride=1, padding=0)` | same as input | one filter per channel |
| `PointwiseConv2D` | `PointwiseConv2D(filters)` | `filters` | 1×1 convolution, keeps the resolution |
| `SeparableConv2D` | `SeparableConv2D(filters, kernel_size=3, stride=1, padding=0)` | `filters` | depthwise then pointwise |
| `TransposeConv2D` | `TransposeConv2D(filters, kernel_size=2, stride=2, padding=0)` | `filters` | upsampling convolution; output `(size − 1)·stride − 2·padding + kernel`. Symmetric padding only |

Requirements: `filters` ≥ 1; `kernel_size`, `stride`, `dilation` pairs ≥ 1; `padding` ≥ 0. If the kernel does not fit
the (padded) input the error says how big each one is. Weights are initialised uniformly in ±1/√fan_in.

### Normalisation and activations

| Layer | Signature | Notes |
|---|---|---|
| `BatchNorm2D` | `BatchNorm2D()` | per channel; eps 1e-3, momentum 0.03; channels inferred |
| `ReLU`, `SiLU`, `Sigmoid` | no arguments | keep the shape |
| `LeakyReLU` | `LeakyReLU(slope=0.1)` | `slope` is the gradient for negative inputs |

### Pooling and resizing

| Layer | Signature | Notes |
|---|---|---|
| `MaxPool2D` | `MaxPool2D(kernel_size=2, stride=None, padding=0)` | `stride` defaults to `kernel_size`; padding at most half the kernel |
| `AvgPool2D` | `AvgPool2D(kernel_size=2, stride=None, padding=0)` | same rules |
| `Upsample` | `Upsample(scale=2)` | nearest neighbour, integer factor |

---

## Blocks

Reusable architecture blocks. `filters` defaults to the input channel count wherever it is optional.

| Block | Signature | Output | What it does |
|---|---|---|---|
| `Bottleneck` | `Bottleneck(filters=None, kernel_size=3, stride=1, reduction=2, shortcut=True)` | `(filters, H/stride, W/stride)` | 1×1 reduce to `filters/reduction` → k×k spatial → 1×1 expand, each Conv-BN(-SiLU); the input is added back (directly, or through a 1×1 projection when channels or stride change), then SiLU |
| `CSP` | `CSP(filters=None, n=1, shortcut=True)` | `(filters, H, W)` | splits into two 1×1 paths, runs `n` Bottlenecks on one, concatenates, fuses with a 1×1 |
| `SPP` | `SPP(filters=None, kernels=(5, 9, 13))` | `(filters, H, W)` | 1×1 to half the channels, parallel max-pools of each size, concatenated with the input, 1×1 |
| `SPPF` | `SPPF(filters=None, kernel_size=5)` | `(filters, H, W)` | the fast SPP: three chained max-pools of one size (same receptive fields, cheaper) |
| `SE` | `SE(reduction=16)` | same | squeeze-and-excitation: global average → 1×1 (channels/reduction) → ReLU → 1×1 → sigmoid, scales each channel |
| `CBAM` | `CBAM(reduction=16, kernel_size=7)` | same | channel attention (average- and max-pooled through a shared MLP) then spatial attention (k×k conv over the channel mean and max) |
| `Focus` | `Focus(filters, kernel_size=3)` | `(filters, H/2, W/2)` | every 2×2 pixel block becomes 4 channels (nothing is lost), then Conv-BN-SiLU; needs an even height and width |

Requirements: kernels of `Bottleneck`, `CBAM`, `SPP` and `SPPF` must be odd (they keep the size by padding
`kernel_size//2`); `n`, `reduction` ≥ 1.

`Bottleneck` is a real bottleneck, not a container: the three convolutions are fixed by its arguments, and the
shortcut is matched automatically, so the two paths always have the same shape.

---

## Composition

| Function | Meaning |
|---|---|
| `wf.stack(*layers, input=None)` | layers in order. The outermost stack is the model; `input` is `(C, H, W)`, default `(3, 640, 640)` |
| `wf.Residual(*layers, shortcut=None)` | `layers(x) + x`. With `shortcut=` (a layer or list) the input is projected first. Both paths must end with the same shape |
| `wf.parallel(*branches, merge="concat")` | every branch receives the same input; results are concatenated along channels or summed (`merge="add"`) |
| `wf.feature("name")` | registers the current tensor under a name and passes it on unchanged. After a pyramid, give one name per map: `wf.feature(("n3", "n4", "n5"))` |
| `wf.concat("a", "b", ...)` | concatenates named features along channels (same height and width) |
| `wf.add("a", "b", ...)` | sums named features (identical shapes) |
| `wf.FPN("p3", "p4", "p5", channels=None)` | top-down pyramid over named features (any order, any channel counts); returns the pyramid, largest map first |
| `wf.PAN(*names, channels=None)` | bottom-up aggregation over the incoming pyramid (or named features) |
| `wf.head(classes=80, channels=None)` | detection head over the pyramid (or a single map); must be the last layer |

### Features

Features are how later layers reach back to earlier tensors. They are looked up by name only; Workflow never
picks a tensor by position or channel count. A missing name lists the names that exist (with a "did you mean"
when one is close), and a duplicate name is an error.

```python
model = wf.stack(
    wf.Conv2D(32, kernel_size=3, stride=2, padding=1),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.feature("p3"),                  # 1/2 size, 32 channels

    wf.Conv2D(64, kernel_size=3, stride=2, padding=1),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.feature("p4"),                  # 1/4 size, 64 channels

    wf.PointwiseConv2D(32),
    wf.Upsample(2),
    wf.feature("up"),                  # back to 1/2 size

    wf.concat("p3", "up"),             # 32 + 32 = 64 channels at 1/2 size
    input=(3, 256, 256),
)
```

### FPN and PAN

`wf.FPN` reads the named maps, orders them from the largest to the smallest, projects each to `channels` with a
1×1 Conv-BN-SiLU, adds every coarser map (resized, nearest) into the next finer one, and refines each level with a
3×3 Conv-BN-SiLU. Different sizes and channel counts are handled for you. `channels` defaults to the smallest
input channel count.

`wf.PAN` then goes back up: the finest map is reduced by a stride-2 3×3 Conv-BN-SiLU, resized if needed, added
to the next map, and refined.

Both return a **pyramid**: the next layer must accept several maps (`wf.PAN`, `wf.head`, or `wf.feature((...))`
to name them). A single-tensor layer after a pyramid raises an error that says so.

### Residual shapes

`wf.Residual` checks that its layers keep the input's shape. If they don't, the error shows both shapes and
the projection to add:

```
layer 2 (Residual): Residual cannot add its two paths.
  why: the layers output (16, 15, 15) but the shortcut carries (8, 30, 30); addition needs identical shapes.
  fix: keep the shape inside the block (same channels, stride 1, padding k//2), or project the input with
       shortcut=wf.Conv2D(16, kernel_size=1, stride=2).
```

---

## Model

The outermost `wf.stack(...)` is the model.

| Member | What it does |
|---|---|
| `model.summary(input=None)` | table of every layer's output shape, parameters and MACs; totals for parameters (trainable / non-trainable), size and GFLOPs. Returns the totals |
| `model.output_shape` | the output shape (a list of shapes for a pyramid / head) |
| `model.check(input)` | output shape for another input size, or a ShapeError |
| `model.count_parameters()` | `(trainable, non_trainable)` |
| `model.plot()` / `model.export_graph("arch.png")` | draw the computation graph (PNG or SVG; `detail="compact"` for one row per layer) |
| `model.device("cuda")` / `model.device()` | move the model / current device ([Devices](#devices-and-backends)) |
| `model.train(...)` | [Training](#training) |
| `model.predict(...)` | [Detection](#detection-and-inference) |
| `model.save(path)`, `model.load(path)`, `wf.load(path)` | `.nbm` files (below) |
| `model.export(format=..., output=..., precision=...)`, `model.quantize(...)` | [Quantisation and export](#quantisation-and-export) |
| `model(x, train=False)` | forward pass on a `(N, C, H, W)` Workflow tensor |
| `model.history`, `model.class_names`, `model.precision` | training history, class names (from the dataset), `fp32` / `fp16` / `int8` |

FLOPs count convolutions (2 per multiply-accumulate); BatchNorm folds into the convolution at inference and
activations, pooling and additions are negligible.

### The graph

`model.export_graph("architecture.png")` draws one row per layer with arrows, blocks with their real branches
(residual shortcuts, CSP and SPP splits, attention gates with a `×` merge) and named features as red edges from
`wf.feature` to the layers that read them. It needs Pillow for PNG; SVG needs nothing.

### `.nbm` files

`.nbm` (Noble Model) is the only model format. It is a zip archive:

| Entry | Content |
|---|---|
| `meta.json` | architecture, input shape, class names, detection settings, precision, quantisation ranges, training progress, and an index of every tensor |
| `weights.bin` | the tensors: float32, float16, or int8 with a scale per tensor |
| `train.bin` | after training: optimizer moments and the raw (non-EMA) weights, for exact continuation |

`wf.load(path)` rebuilds the architecture from the file; `model.load(path)` fills an identical architecture.
Saving is atomic (written to `.part`, then renamed). Files from Workflow 0.1 (the PyTorch version) cannot be
read: the error says so.

---

## Detection and inference

### The head

`wf.head(classes=80)` ends a detection model. It works on the pyramid coming from `wf.FPN` / `wf.PAN` (or on a
single map). Per map it applies two 3×3 Conv-BN-SiLU layers and a 1×1 convolution with `4 + 1 + classes`
output channels:

| Channels | Meaning |
|---|---|
| 0-3 | distances from the cell centre to the left, top, right and bottom box edges, in units of the map's stride (softplus keeps them positive) |
| 4 | confidence (objectness) logit |
| 5… | one logit per class |

It is **anchor-free**, and each map's stride is derived from the image and map sizes, so any input size the
architecture accepts works. `classes` is part of the architecture; training checks it against the dataset.

### Annotations

Workflow reads boxes exactly as Felix stores them (`bbox: [x, y, width, height]` in pixels, `class_id`), for
generated and converted datasets alike. Nothing is rewritten on disk.

### Default loss

You never configure it for ordinary training:

- Each object is assigned to the pyramid level whose stride suits its size (up to 8× the stride); cells inside the box
  within 1.5 strides of its centre are positives, every object keeps at least its centre cell, and the smaller object
  wins where boxes overlap.
- Loss = 5 × (1 − GIoU) on positive cells + binary cross-entropy for confidence (positives 1, everything else 0) +
  binary cross-entropy for the classes, normalised by the number of positives.
- `model.train(..., loss_weights=(box, confidence, classes))` changes the weights if you need to.

### Prediction

```python
model = wf.load("detector.nbm")
results = model.predict("image.jpg")
results = model.predict(["image1.jpg", "image2.jpg"])
results = model.predict("image.jpg", device="cuda", confidence=0.25, iou=0.45, img_size=640)
results = model.predict("image.jpg", output="predictions/")
```

| Argument | Default | Meaning |
|---|---|---|
| `source` | required | an image path, a PIL / Felix image, an RGB `(H, W, 3)` uint8 array, or a list of those |
| `device` | auto | `"cpu"`, `"cuda"`, `"gpu"`, `"mps"`; without it the model's device (or the best one) is used |
| `confidence` | 0.25 | minimum score (confidence × class probability) |
| `iou` | 0.45 | overlap above which a weaker box of the same class is removed (NMS) |
| `img_size` | the model's input | inference size (int or `(H, W)`); checked against the architecture |
| `output` | `"results/"` | folder for annotated images and `predictions.json` (created if needed); `None` writes nothing |
| `max_detections` | 300 | per image |

What happens: the image is letterboxed (resized keeping its aspect ratio, padded grey), run without gradients
(weights never change), decoded, filtered by `confidence`, cleaned with class-aware NMS, and mapped back to the
original pixels. Evaluation uses exactly the same preprocessing. FP32, FP16 and INT8 models all predict the same way.

Output on disk: `results/<image name>.png` with boxes and labels, and `results/predictions.json`:

```json
{"predictions": [{"image": "image.jpg", "width": 640, "height": 480, "annotated": "results/image.png",
  "detections": [{"class_id": 1, "class": "square", "confidence": 0.94, "xyxy": [138.0, 91.0, 206.0, 159.0]}]}]}
```

### Results

| Member | Type |
|---|---|
| `xyxy`, `xywh` | lists of `[x1, y1, x2, y2]` / `[x, y, w, h]` in original image pixels |
| `confidences` | list of floats |
| `class_ids`, `classes` | lists of ints and of names |
| `len(r)`, `for d in r` | count; dicts with `class_id`, `class`, `confidence`, `xyxy` |
| `r.show()`, `r.plot()`, `r.save(path)`, `r.to_dict()` | viewer, PIL image, file, plain data |

A list input returns `Predictions` (a list of `Results` with `.show()` and `.to_dict()`).

Unreadable inputs raise a `DataError` naming the file: missing files, unsupported extensions, damaged images.
OpenCV frames are BGR; convert with `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` (the generated scripts do).

---

## Training

```python
model.train(train, validation=val, epochs=100)
model.train(train, validation=val, epochs=100, device="cuda")
```

`train` and `validation` are Felix splits (`dataset.train`, `dataset.val`). A whole Felix dataset also works: its
own splits are used.

| Argument | Default | Meaning |
|---|---|---|
| `epochs` | 100 | **total** target epoch count (see Continuing) |
| `batch_size` | 16 | images per step |
| `lr` | 1e-3 | peak learning rate |
| `device` | auto | `"cpu"`, `"cuda"`, `"gpu"`, `"mps"`; unavailable devices raise an error instead of falling back silently |
| `img_size` | the model's input | training size |
| `optimizer` | `"adamw"` | or `"sgd"` (momentum 0.937) |
| `patience` | 30 | stop after this many epochs without improvement (`None` disables) |
| `save_every` | 5 | also save `epoch_N.nbm` every N epochs |
| `output` | `"runs/train"` | where checkpoints go: `last.nbm` every epoch, `best.nbm` on improvement |
| `quantization` | `None` | `wf.QAT(precision="int8")` for quantisation-aware training ([Quantisation](#quantisation-and-export)) |
| `augment` | True | horizontal flips and brightness / contrast jitter |
| `ema` | True | keep a moving average of the weights; it is validated and saved |
| `loss_weights` | (5, 1, 1) | box, confidence, class |
| `quiet` | False | hide the progress panel |

Returns the history: one dict per epoch (`epoch`, `loss`, `box`, `obj`, `cls`, `lr`, `seconds`, and with validation
`map50`, `map50_95`, `precision`, `recall`, `f1`, `fitness`). It is also kept in `model.history`.

### Defaults

- **Weights**: convolutions uniform in ±1/√fan_in; BatchNorm γ=1, β=0; the head starts with box distances of about
  1.3 strides and confidences of about 1%, which keeps the first steps stable.
- **Optimizer**: AdamW, weight decay 5e-4 on weights (not on biases and BatchNorm).
- **Schedule**: linear warm-up (about 3 epochs, at least 100 steps), then cosine decay to 1% of `lr`.
- **Stability**: gradients clipped at norm 10; a NaN/inf loss stops training with an explanation.
- **Best model**: highest `0.1 × mAP50 + 0.9 × mAP50-95` on the validation data (lowest loss without validation).

### Ctrl+C

Training stops at once, the current state is saved to `output/last.nbm`, and `model.train` returns the history
so far.

### Continuing

```python
model = wf.load("runs/train/last.nbm")
model.train(train, validation=val, epochs=150)
```

`epochs` is the total target: a run saved at epoch 80 continues from 81 to 150, with the optimizer state,
raw weights, EMA and history restored. A model already at the target does nothing (and says so). If the
saved optimizer state doesn't fit the model, a fresh optimizer is created with a note.

### Speed

Measured on the demo detector (1.3M parameters, 256×256, batch 16): about 32 images/s on an Apple M-series CPU,
about 47 images/s on a GTX 1050. Workflow's kernels are its own; they are slower than cuDNN-backed frameworks on
large models, which is the price of owning the stack.

---

## Evaluation

```python
wf.eval(model)                     # architecture: no data needed
wf.eval(model, dataset=val)        # plus accuracy
```

Without a dataset:

| Field | Meaning |
|---|---|
| `parameters`, `trainable` | values in the model |
| `size_mb` | file size of the weights at the model's precision (fp32 4 bytes, fp16 2, int8 1 per value) |
| `flops`, `macs` | per image at the evaluated size |
| `latency_ms`, `fps` | batch 1, forward pass only, measured on the device after warm-up |
| `throughput` | images/s at `batch_size` (default 8) |
| `memory_mb` | fp32 weights plus one image's activations |

With a dataset (a Felix split), additionally:

| Field | Meaning |
|---|---|
| `map50`, `map50_95` | COCO-style AP (101-point) at IoU 0.5, and averaged over IoU 0.5-0.95 |
| `precision_score`, `recall`, `f1` | at the confidence that maximises mean F1 (`conf_threshold`) |
| `per_class` | `{name: {"ap50", "ap50_95", "instances"}}` |
| `confusion_matrix` | (classes + 1)² counts at confidence 0.25 / IoU 0.45; the last row and column are background |

`wf.eval` prints a table and returns an `Evaluation` (`.show()`, `.to_dict()`, `.save("eval.json")` or `.csv`).
Options: `device`, `img_size`, `batch_size`, `confidence` (0.001 for a complete AP curve), `iou` (0.6), `show`.

It works before and after training, and never changes weights, BatchNorm statistics or the training state. Passing
a whole Felix dataset evaluates its validation split.

---

## Quantisation and export

### Precisions

| Precision | Weights on disk | Activations | Typical size |
|---|---|---|---|
| `fp32` | float32 | float32 | 1× |
| `fp16` | float16 | float32 | 0.5× |
| `int8` | int8, one scale per tensor (symmetric) | int8 at every convolution input (affine, per tensor) | 0.25× |

INT8 covers every convolution layer (`Conv2D`, `DepthwiseConv2D`, `PointwiseConv2D`, `SeparableConv2D`,
`DilatedConv2D`, and those inside blocks). `TransposeConv2D` stays float and is reported with a warning.

### QAT: quantisation-aware training

```python
model.train(train, validation=val, epochs=20, quantization=wf.QAT(precision="int8"))
```

Every training step rounds convolution weights and inputs to the int8 grid (gradients pass straight through), so
the model learns weights that survive quantisation, and the activation ranges are tracked while it trains.
Usually you fine-tune an fp32 model for a few epochs this way. After it, `model.precision` is `"int8"`.

### Post-training quantisation

```python
model.quantize("int8", calibration=val)       # in place; then model.save(...)
```

Or from the terminal:

```
workflow quantize detector.nbm int8 --calibration data.felix     # → detector-int8.nbm
workflow quantize detector.nbm fp16
```

`calibration` images fix the activation ranges. Without them only the weights are quantised, with a warning.
On the demo detector, INT8 post-training quantisation changed test mAP50-95 from 0.811 to 0.806 and the file from
19.7 MB (with training state) to 1.3 MB.

### Exporting

```python
model.save("detector.nbm")                                               # native, current precision
model.export(format="nbm", output="detector-int8.nbm", precision="int8") # real int8 weights
model.export(format="onnx", output="detector.onnx")                      # fp32 ONNX
model.export(format="onnx", output="detector-int8.onnx", precision="int8", calibration=val)
```

- **ONNX** (opset 13) has one input `images` `(batch, C, H, W)` and one raw output per head map (`output0`,
  `output1`, ...). The decoding described in [Detection](#detection-and-inference) is not inside the file. FP16 stores
  half-precision weights; INT8 uses QuantizeLinear / DequantizeLinear with int8 weights. Checked with ONNX Runtime:
  fp32 matches Workflow to 1e-5.
- **INT8 `.nbm`** files really store int8 weights and the activation ranges. Loading them gives a model that runs
  the int8 simulation, so its predictions match the quantised model.
- Before converting, Workflow validates the input size against the architecture and refuses unknown formats or
  precisions with an explanation. TorchScript `.pt` is gone along with PyTorch.

---

## Devices and backends

Workflow computes with its own engine. PyTorch is not used.

| Device | Backend | Status |
|---|---|---|
| `"cpu"` | **native C core**: im2col + cache-blocked GEMM, depthwise kernels, a thread pool, fused BatchNorm / activation kernels | stable |
| `"cuda"`, `"cuda:N"` | **Workflow's own CUDA kernels** (register-tiled GEMM convolution, BatchNorm, pooling, ...), shipped as PTX and run through the NVIDIA driver | stable |
| `"gpu"` | CUDA if present, else the Apple GPU | |
| `"mps"` | Apple GPU through **MLX**, Apple's open-source Metal array library (`pip install mlx`) | experimental: warns when used |
| `"auto"` | CUDA, then CPU (experimental MPS is never picked implicitly) | |

```python
model.device("cuda")                           # or model.train(..., device="cuda")
workflow devices                               # what this machine has, and why something is unavailable
```

A requested device that isn't available raises a `DeviceError` explaining why (no driver, no MLX, no GPU) and
what to do. Workflow never silently runs on another device than the one you asked for.

### Correctness

Every operation and gradient is tested against PyTorch on the CPU (`tests/test_core.py`, PyTorch as a test oracle
only), and every accelerator is tested against the CPU (`tests/test_parity.py`). On a GTX 1050 all 26 operations match
the CPU core to 4e-7 relative error, and on Apple MLX to 6e-7.

### Threads

The CPU core uses one thread per core, up to 16.

---

## Errors

Every Workflow error has three parts and names where it happened:

```
layer 3 (Concat): wf.concat refers to the feature 'b', which does not exist here.
  why: features must be registered with wf.feature(...) earlier in the model; registered so far: a.
  fix: add wf.feature("b") after the layer whose output you want.
```

| Class | Raised for |
|---|---|
| `wf.ShapeError` | a layer receives a shape it can't use (wrong channels, kernel larger than the input, residual paths that differ, a pyramid where one map is expected) |
| `wf.RoutingError` | missing, duplicate or mislabelled features |
| `wf.DeviceError` | a device that isn't available |
| `wf.BackendError` | the native core or a GPU backend can't load or run something (with the driver's reason) |
| `wf.DataError` | unreadable images, empty splits, class-count mismatches |
| `wf.SerializationError` | invalid model files, architecture mismatches, unsupported export formats |
| `wf.WorkflowError` | the base class of all of the above (also invalid arguments) |

Non-fatal issues are `wf.WorkflowWarning` warnings: the experimental Apple GPU, quantising without calibration data,
layers that stay float in an int8 model.

Errors carry `what`, `why`, `fix` and `where` attributes for programs that want them.

---

## CLI

`workflow` (or `python -m workflow`), in the same minimal style as Felix.

```
workflow init [NOTEBOOK.ipynb] [--dataset D]    starter notebook: small model, benchmarks before/after, saving
workflow info MODEL.nbm                        classes, precision, training progress, layer summary
workflow predict MODEL.nbm IMG [IMG ...]       detect; annotated images + predictions.json in results/
workflow eval MODEL.nbm [--dataset D --split test]   size/FLOPs/FPS, plus accuracy with a dataset
workflow quantize MODEL.nbm int8|fp16|fp32 [--calibration D] [-o OUT]   post-training quantisation
workflow export MODEL.nbm --format onnx|nbm [--precision ...] [-o OUT]
workflow script MODEL.nbm --preset image|folder|video|webcam    write an inference script
workflow devices                               cpu / cuda / mps, and why something is unavailable
```

`quantize` writes `MODEL-<type>.nbm` by default and prints the size before and after. Commands never overwrite
existing files without `--overwrite`. Errors print the what / why / fix explanation and exit with status 1.

### init

```
workflow init                                   # → workflow.ipynb
workflow init detector.ipynb --dataset data.felix --img-size 320 --epochs 50 --device cuda
```

Writes a Jupyter notebook that runs top to bottom:

1. settings (`DATASET`, `IMG_SIZE`, `EPOCHS`, `DEVICE`) in the first cell;
2. the data: your Felix dataset, or, if it doesn't exist yet, a small practice dataset of shapes that the notebook
   makes itself, so it works with no setup;
3. a small, readable detector written as one flat `wf.stack` (Conv-BN-SiLU steps, CSP, SPPF, two named features,
   FPN, head: about 0.4M parameters), its summary and its graph;
4. **benchmark before training**: `wf.eval(model, dataset=data.test)`;
5. training with a live progress bar (re-running the cell with a bigger `EPOCHS` continues);
6. **benchmark after training** and a before/after table;
7. saving: `model.nbm`, an int8 copy and an ONNX file, a reload check, and a prediction drawn inline.

| Option | Default | |
|---|---|---|
| `path` | `workflow.ipynb` | must end in `.ipynb` |
| `--dataset` | `practice_shapes.felix` | made by the notebook on first run if it doesn't exist |
| `--img-size` | 256 | a multiple of 32, at least 64 |
| `--epochs` | 30 | |
| `--device` | `auto` | `cpu`, `cuda`, `gpu`, `mps` |
| `--overwrite` | off | replace an existing notebook |

Open it with Jupyter, VS Code or Colab.

### Inference scripts

| Preset | Does | Run |
|---|---|---|
| `image` | one or more images → `results/` | `python infer_image.py photo.jpg` |
| `folder` | every image in a folder → `results/` | `python infer_folder.py photos/` |
| `video` | a video with live preview, optional `--save out.mp4` | `python infer_video.py clip.mp4` |
| `webcam` | live camera detection (`--mirror` for selfie view) | `python infer_webcam.py` |

All take `--confidence`, `--iou`, `--img-size` and `--device`. `video` and `webcam` need `opencv-python`.

---

## Changes in 0.2

Workflow 0.2 replaces PyTorch with its own engine and adopts a simpler API.

### Breaking changes

| 0.1 | 0.2 |
|---|---|
| `wf.Model(wf.Stage("p1", ...), ...)` | `wf.stack(...)`; named tensors come from `wf.feature("p1")` |
| `wf.Conv2D(in_channels, out_channels, k)` | `wf.Conv2D(filters, kernel_size=k)`; input channels are inferred |
| `wf.BatchNorm2D(channels)` | `wf.BatchNorm2D()` |
| `wf.FPN(inputs=(...), channels=..., outputs=(...))` | `wf.FPN("p3", "p4", channels=...)`; returns a pyramid |
| `wf.BBoxHead(inputs, classes, outputs)` | `wf.head(classes=...)` |
| `wf.Concat(inputs=...)`, `wf.Add(inputs=...)` | `wf.concat("a", "b")`, `wf.add("a", "b")` |
| `Stage(..., input="p3")` | `wf.feature` names and `wf.concat` / `wf.add` / `wf.FPN` |
| `model.train(dataset, epochs, batch_size, save_every)` | `model.train(train, validation=val, epochs=...)` |
| `model.predict(x, conf=, imgsz=)` | `model.predict(x, confidence=, img_size=)`; saves to `results/` by default |
| `model.benchmark()` / `model.benchmark(dataset)` | `wf.eval(model)` / `wf.eval(model, dataset=...)` |
| `workflow export --format pt` | removed (no PyTorch); ONNX and `.nbm` at any precision instead |
| `.nbm` format 1 | format 2 (no pickled tensors); 0.1 files cannot be loaded |

### New

- Own compute engine: native C CPU core, own CUDA kernels, experimental Apple GPU (MLX).
- Layers: `DepthwiseConv2D`, `PointwiseConv2D`, `SeparableConv2D`, `TransposeConv2D`, `DilatedConv2D`; explicit
  numeric and asymmetric padding.
- Blocks: `SPP`, `SE`, `CBAM`, `Focus`; `Bottleneck` is now a real reduce / spatial / expand bottleneck.
- Composition: `wf.parallel`, `wf.feature`, `wf.PAN` over pyramids.
- Quantisation: `wf.QAT`, post-training quantisation, `workflow quantize`, INT8 / FP16 `.nbm` and ONNX.
- Errors with what / why / fix everywhere.
