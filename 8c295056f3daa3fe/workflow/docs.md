# Workflow: complete documentation

*This file is the whole documentation in one page. It is generated from the individual pages; do not edit it by hand.*

## Contents

- [Install](#install)
- [Demo](#demo)
- [Cookbook](#cookbook)
- [Workflow library](#workflow-library)
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
- [Changes](#changes)

---

Noble Engineering's CNN framework for bounding-box detection, on its **own compute engine**: a native C core
for the CPU, its own CUDA kernels for NVIDIA GPUs, and an experimental Apple-GPU backend. No PyTorch.

```python
import felix as fx
import workflow as wf

data = fx.load("shapes.felix")

model = wf.stack(
    wf.input(channels=3, height=640, width=640),

    wf.Conv2D(32, kernel_size=3, stride=2, padding="same"),
    wf.BatchNorm2D(),
    wf.SiLU(),

    wf.Conv2D(64, kernel_size=3, stride=2, padding="same"),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.CSP(),
    wf.feature("p3"),

    wf.Conv2D(128, kernel_size=3, stride=2, padding="same"),
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
wf.inference("shapes.felix", model, "results/")   # a random test image → original/, annotated/, predictions.json
```

Input channels, tensor shapes, biases, parameter counts and FLOPs are worked out for you. When something doesn't
fit, the error says **what** happened, **why**, and **how to fix it**, naming the layer.

## What is in it

- **Layers**: `Conv2D`, `DepthwiseConv2D`, `PointwiseConv2D`, `SeparableConv2D`, `TransposeConv2D`, `DilatedConv2D`,
  `BatchNorm2D`, `ReLU`, `LeakyReLU`, `SiLU`, `Sigmoid`, `MaxPool2D`, `AvgPool2D`, `Upsample`.
- **Blocks**: `Bottleneck`, `CSP`, `SPP`, `SPPF`, `SE`, `CBAM`, `Focus`.
- **Composition**: `wf.input`, `wf.stack`, `wf.Residual`, `wf.parallel`, `wf.feature`, `wf.concat`, `wf.add`, `wf.FPN`, `wf.PAN`, `wf.head`.
- **Training**: one call with sensible defaults (AdamW, warm-up + cosine, GIoU + BCE loss, EMA, early stopping,
  `.nbm` checkpoints, crash protection: safe Ctrl+C / kill, autosave mid-epoch, exact resume; automatic low-memory
  mode on small machines).
- **Evaluation**: `wf.eval(model)` (size, FLOPs, latency, FPS, memory) and `wf.eval(model, dataset=val)` (accuracy).
- **Quantisation**: QAT during training, post-training quantisation, real INT8 / FP16 files.
- **Export**: `.nbm` at any precision, ONNX (FP32, FP16, INT8 as QDQ).
- **Inference**: `wf.inference(source, model, output)` on an image, video, folder or Felix dataset (one random test
  image by default, `all=True` for all; `"webcam"` for live detection), saving originals, annotated copies and `predictions.json`.
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
!pip install BASE_URL/files/felix-0.5.0-py3-none-any.whl BASE_URL/files/workflow-0.4.0-py3-none-any.whl
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

#### `wf.input`

**Toddler words:** the front door of the brain. It says what kind of pictures come in: how many colour channels
(3 for red, green and blue) and how big.

**Why you'd use it:** Workflow builds the whole model from it. It works out how many channels reach every layer,
makes the weights the right size, and checks that every layer fits *before* anything trains. It is also the size
pictures are resized to for training and prediction, unless you ask for another. It always goes on the very first
line, because that's where the picture comes in.

```python
model = wf.stack(
    wf.input(channels=3, height=256, width=256),
    wf.Conv2D(16, kernel_size=3, stride=2, padding="same"),
    ...
)
```

**Watch out:** only one, only at the top of the outermost `wf.stack` (blocks like `wf.Residual` take their input from
the layer before them). Leave it out and Workflow assumes 3 × 640 × 640. A trained model can still predict at other
sizes with `img_size=`.

#### `wf.stack`

**Toddler words:** line up blocks. The picture goes through them one after another, like a slide with bumps.

**Why you'd use it:** it's the model itself (the outermost one) and also any group of layers you want to treat as
one. Its first line is `wf.input(...)`, then one layer per line.

```python
model = wf.stack(
    wf.input(channels=3, height=640, width=640),

    wf.Conv2D(16, kernel_size=3, stride=1, padding="same"),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.feature("p1"),
)
```

**Watch out:** you never say how many channels go *in* to a layer. Workflow works that out.

#### `padding` (used by every layer with a window)

**Toddler words:** a cushion of empty pixels around the picture so the edges don't get squished away. You pick one
of two words: `"same"` (put a cushion on, keep the picture the same size) or `"valid"` (no cushion, the picture
shrinks a little).

**Why you'd use it:** without a cushion, every 3×3 window eats one pixel from each edge, so the picture shrinks a
bit at every layer. `"same"` stops that, which keeps sizes easy to reason about: a stride-1 layer keeps the size, and a
stride-2 layer exactly halves it (rounding up). Almost every layer in a detector uses `"same"`. `"valid"` (the
default) is for when you *want* the picture to shrink, or want the window to only ever see real pixels.

```python
wf.Conv2D(32, kernel_size=3, stride=1, padding="same")    # 256×256 in → 256×256 out
wf.Conv2D(32, kernel_size=3, stride=2, padding="same")    # 256×256 in → 128×128 out
wf.Conv2D(32, kernel_size=3, stride=1, padding="valid")   # 256×256 in → 254×254 out
```

**Watch out:** numbers like `padding=1` are not accepted. The error tells you which word to use (`1` meant
`"same"` for a 3×3, `0` meant `"valid"`).

#### `wf.Conv2D`

**Toddler words:** little magnifying glasses that slide over the picture, each looking for one kind of pattern (an
edge, a corner, a stripe). The first number is `out_channels`: how many different glasses, so how many channels come
*out*. `32` means 32 glasses, so 32 channels come out.

**Why you'd use it:** it's *the* building block. Early convs find edges; later ones find parts and whole objects.
`stride=2` takes big steps, halving the picture; that's how a backbone goes from big-and-shallow to small-and-deep.
You never write how many channels come *in*: Workflow reads that from the layer before.

```python
wf.Conv2D(32, kernel_size=3, stride=1, padding="same")    # 32 channels out, same size
wf.Conv2D(64, kernel_size=3, stride=2, padding="same")    # 64 channels out, half size: a "downsampling" step
wf.Conv2D(out_channels=64, kernel_size=3, stride=2, padding="same")   # the same, with the name written out
```

#### `wf.DepthwiseConv2D`

**Toddler words:** one magnifying glass per colour channel, and each one only looks at its own channel.

**Why you'd use it:** it's very cheap. Mobile-style models use it to look around spatially for almost no cost,
then mix the channels with a pointwise conv.

```python
wf.DepthwiseConv2D(kernel_size=3, stride=1, padding="same")    # keeps the channel count
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
wf.SeparableConv2D(64, kernel_size=3, stride=1, padding="same")
```

#### `wf.DilatedConv2D`

**Toddler words:** a magnifying glass with gaps between its eyes, so it sees farther without costing more.

**Why you'd use it:** a bigger view (more context) without shrinking the picture or adding work. Useful late in
a network or for big objects.

```python
wf.DilatedConv2D(32, kernel_size=3, dilation=2, stride=1, padding="same")   # sees a 5×5 area, keeps the size
```

#### `wf.TransposeConv2D`

**Toddler words:** a magnifying glass that makes the picture *bigger* (it un-squishes), and learns how.

**Why you'd use it:** growing a small, smart map back to a bigger size, for example in a neck. Unlike `Upsample`,
which just copies pixels, it *learns* how to fill in the new pixels.

```python
wf.TransposeConv2D(32, kernel_size=2, stride=2, padding="valid")   # exactly twice the height and width
wf.TransposeConv2D(32, kernel_size=4, stride=2, padding="same")    # also twice, blending neighbours more smoothly
```

Output size: with `"same"` it is exactly `size × stride`; with `"valid"` it is `(size − 1) × stride + kernel_size`.
With `kernel_size=2, stride=2` both give exactly double, which is why those are the defaults.

**Watch out:** `"same"` needs `kernel_size` at least as big as `stride` (the error says so). It stays float in int8
models (Workflow warns you); use `Upsample` + `Conv2D` if everything must be int8.

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

#### `wf.MaxPool2D`, `wf.MinPool2D`, `wf.AvgPool2D`

**Toddler words:** shrink the picture by looking at little squares and keeping one number from each. Max keeps the
loudest pixel, Min keeps the quietest, Avg keeps the average.

**Why you'd use it:** a cheap way to make the picture smaller (or smoother) with no weights to learn.
- **Max** answers "is the pattern anywhere in here?". It is the usual choice, and what `SPPF` uses.
- **Min** answers "is anything *missing* or dark here?", for dark objects on light backgrounds, gaps or holes.
- **Avg** smooths things out and keeps the overall level.

```python
wf.MaxPool2D(kernel_size=2, stride=None, padding="valid")   # stride defaults to kernel_size: half the size
wf.MaxPool2D(kernel_size=5, stride=1, padding="same")       # same size: every pixel becomes its 5×5 neighbourhood's max
wf.MinPool2D(kernel_size=3, stride=2, padding="same")       # half the size, keeping the smallest values
wf.AvgPool2D(kernel_size=3, stride=1, padding="same")       # same size, just smoothed
```

**Watch out:** the cushion pixels from `"same"` never win a max or a min, but they count as zero in an average,
so averages at the very edge come out a little darker.

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
wf.Bottleneck(out_channels=None, kernel_size=3, stride=1, reduction=2, shortcut=True)   # same shape out as in
wf.Bottleneck(out_channels=128, kernel_size=3, stride=2, reduction=2, shortcut=True)    # 128 channels out, half size
```

**Watch out:** when `out_channels` or `stride` change the shape, the shortcut gets a 1×1 projection automatically,
so you never have to match the shapes yourself.

#### `wf.CSP`

**Toddler words:** split into two paths. One does hard work (bottlenecks), one takes it easy. Then glue them
together.

**Why you'd use it:** most of the accuracy of a deep stack of bottlenecks at a lot less cost. It's the standard
backbone block after each downsampling conv.

```python
wf.CSP(out_channels=None, n=1, shortcut=True)      # n = how many bottlenecks on the busy path
```

#### `wf.SPP`, `wf.SPPF`

**Toddler words:** look at the picture through small, medium and big windows at the same time, then glue what
you saw.

**Why you'd use it:** at the end of the backbone, so the brain sees both details and the big picture. SPPF gets
the same result faster with three chained small pools, so prefer it.

```python
wf.SPP(out_channels=None, kernels=(5, 9, 13))
wf.SPPF(out_channels=None, kernel_size=5)
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
wf.Focus(out_channels=32, kernel_size=3)
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
    wf.Conv2D(64, kernel_size=3, stride=1, padding="same"),
    wf.BatchNorm2D(),
    wf.SiLU(),
    shortcut=None,                                    # same shape in and out: add the input as it is
)

wf.Residual(
    wf.Conv2D(128, kernel_size=3, stride=2, padding="same"),
    wf.BatchNorm2D(),
    wf.SiLU(),
    shortcut=wf.Conv2D(128, kernel_size=1, stride=2, padding="valid"),   # shape changes → project the input to match
)
```

**Watch out:** both paths must end the same shape. If they don't, the error tells you the exact `shortcut=` to add.

#### `wf.parallel`

**Toddler words:** give the same picture to several friends, then glue (or add up) their answers.

**Why you'd use it:** let the brain look at the same thing in different ways at once (small and big windows,
for example), Inception-style.

```python
wf.parallel(
    wf.Conv2D(16, kernel_size=1, stride=1, padding="valid"),
    wf.Conv2D(16, kernel_size=3, stride=1, padding="same"),
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
and it saves before stopping. It also saves by itself every 10 minutes, so if the computer crashes or the power goes
out, you lose minutes, not hours: load `last.nbm` and train again, and it carries on from the same step.

**Why you'd use it:** this is learning. The defaults (optimizer, learning-rate schedule, loss, augmentation,
checkpoints) are sensible, so most of the time you only set the data and `epochs`.

```python
model.train(data.train, validation=data.val, epochs=50, batch_size=16, lr=1e-3, device="auto",
            img_size=None, optimizer="adamw", patience=30, save_every=5, output="runs/train",
            quantization=None, augment=True, ema=True, low_memory=None, autosave=10)
```

**Not enough memory?** On CPU it checks first. If the batch won't fit it turns on `low_memory` (it forgets the
in-between results and works them out again when it needs them: same answers, a bit slower), and if even that won't
fit it tells you the biggest `batch_size` that does, before wasting any time.

While it learns you see a bar, plus a table that gains a line every epoch: the training loss, the validation loss
(how wrong it is on pictures it doesn't learn from) and the mAP scores. If the training loss keeps falling but the
validation loss rises, the brain is memorising instead of learning.

While it learns you see a bar, plus a table that gains a line every epoch: the training loss, the validation loss
(how wrong it is on pictures it doesn't learn from) and the mAP scores. If the training loss keeps falling but the
validation loss starts rising, the brain is memorising the training pictures instead of learning.

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

#### `wf.inference`

**Toddler words:** "go look at these and tell me what you see." Give it a picture, a video, a folder or a whole Felix
dataset, plus the brain, plus a folder for the answers. It figures out what you gave it by itself.

**Why you'd use it:** this is how you *use* a trained model. It keeps everything together in one folder: the
original pictures, the same pictures with boxes drawn on, and `predictions.json` with every box. Give it a Felix
dataset and it picks the test pile on its own (the pictures the model never trained on), unless you choose
another pile. From a big pile (a dataset or a folder) it looks at **one picture picked at random**, a quick spot
check. Say `all=True` when you want every picture done.

```python
model = wf.load("detector.nbm")
run = wf.inference("shapes.felix", model, "results/", all=False, split=None, confidence=0.25, iou=0.45,
                   img_size=None, device=None, batch_size=16)      # one random test picture
wf.inference("shapes.felix", model, "results/all/", all=True)       # every test picture
wf.inference("holiday.mp4", model, "results/video/")                # every frame, annotated video saved
wf.inference("photos/", model, "results/photos/", all=True)         # every image (and video) in the folder
wf.inference(data.val, model, "results/val/", all=True)             # you pick the pile
wf.inference("webcam", model)                                       # live, in a window (q to stop)
```

**Webcam:** `"webcam"` opens a window and detects live, with the FPS on the picture. Arrow keys or the number keys
switch camera (`1` is the second camera), `q` or `Esc` stops; the console lists the keys. Nothing is saved
unless `record=True`.

**Watch out:** videos and the webcam need `pip install opencv-python`. The folder can be reused: files are added, and names
never overwrite each other.

#### `model.predict`

**Toddler words:** "what's in this picture?", answered straight back to you in Python.

**Why you'd use it:** inside your own code, when you want the boxes as data and not a folder of files (for example
in a camera loop). `wf.inference` is built on it.

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
workflow predict detector.nbm photo.jpg --confidence 0.25     # also videos, folders, datasets (--all for every image)
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
wf.Conv2D(64, kernel_size=3, stride=2, padding="same"),
wf.BatchNorm2D(),
wf.SiLU(),
```

**A small detector**: the backbone halves the picture five times, naming three depths with `wf.feature`; SPPF
widens the view; FPN + PAN mix the named maps; the head finds the boxes.

```python
model = wf.stack(
    wf.input(channels=3, height=256, width=256),

    # stem: 256 → 128
    wf.Conv2D(16, kernel_size=3, stride=2, padding="same"),
    wf.BatchNorm2D(),
    wf.SiLU(),

    # 128 → 64
    wf.Conv2D(32, kernel_size=3, stride=2, padding="same"),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.CSP(n=1),

    # 64 → 32: good for small objects
    wf.Conv2D(64, kernel_size=3, stride=2, padding="same"),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.CSP(n=2),
    wf.feature("p3"),

    # 32 → 16: medium objects
    wf.Conv2D(128, kernel_size=3, stride=2, padding="same"),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.CSP(n=2),
    wf.feature("p4"),

    # 16 → 8: large objects
    wf.Conv2D(256, kernel_size=3, stride=2, padding="same"),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.SPPF(kernel_size=5),
    wf.feature("p5"),

    # neck and head
    wf.FPN("p3", "p4", "p5", channels=64),
    wf.PAN(channels=64),
    wf.head(classes=3),
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
    wf.DepthwiseConv2D(kernel_size=3, stride=1, padding="same"),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.PointwiseConv2D(64),
    wf.BatchNorm2D(),
)
```

---

## Workflow library

Everything you can put in a model, one line each. Write every layer/block on its own line inside `wf.stack(...)`.
Blocks build their own layers inside: you only pick their arguments, you don't put layers in between.
Only `wf.stack`, `wf.Residual` and `wf.parallel` take layers you write yourself.

Workflow 0.4.0. Full details: [Layers](#layers), [Blocks](#blocks), [Composition](#composition). `workflow init` writes this same table as `library.md` next to the notebook.

### Start

| Tool | What it does | You choose | Example |
|---|---|---|---|
| `wf.input` | says what picture comes in; must be the first line of the model | `channels=3`, `height=640`, `width=640` | `wf.input(channels=3, height=640, width=640)` |
| `wf.stack` | the model itself: runs its lines top to bottom, each feeding the next | the layers, one per line | see [wf.stack below](#a-whole-model) |

### Layers

Input channels are always worked out from the line above. `padding="same"` keeps the size (÷ stride),
`"valid"` adds none.

| Layer | What it does | You choose | Example |
|---|---|---|---|
| `wf.Conv2D` | a learned k×k filter bank: the basic building brick | `out_channels`, `kernel_size=3`, `stride=1`, `padding="valid"` | `wf.Conv2D(32, kernel_size=3, stride=2, padding="same")` |
| `wf.DilatedConv2D` | a k×k conv with gaps between its taps: sees wider for the same cost | `out_channels`, `kernel_size=3`, `dilation=2`, `stride=1`, `padding="valid"` | `wf.DilatedConv2D(64, kernel_size=3, dilation=2, padding="same")` |
| `wf.DepthwiseConv2D` | one k×k filter per channel (no mixing between channels); keeps the channel count | `kernel_size=3`, `stride=1`, `padding="valid"` | `wf.DepthwiseConv2D(kernel_size=3, stride=1, padding="same")` |
| `wf.PointwiseConv2D` | a 1×1 conv: mixes channels, never looks at neighbours | `out_channels` | `wf.PointwiseConv2D(64)` |
| `wf.SeparableConv2D` | builds a DepthwiseConv2D → PointwiseConv2D: a cheap stand-in for Conv2D | `out_channels`, `kernel_size=3`, `stride=1`, `padding="valid"` | `wf.SeparableConv2D(64, kernel_size=3, padding="same")` |
| `wf.TransposeConv2D` | a learned upsampling ("up" conv); the defaults double the size | `out_channels`, `kernel_size=2`, `stride=2`, `padding="valid"` | `wf.TransposeConv2D(32, kernel_size=2, stride=2)` |
| `wf.BatchNorm2D` | evens out each channel's numbers so training stays stable; channels inferred | nothing | `wf.BatchNorm2D()` |
| `wf.ReLU` | keeps positives, zeroes negatives | nothing | `wf.ReLU()` |
| `wf.LeakyReLU` | like ReLU but lets a little of the negatives through | `slope=0.1` | `wf.LeakyReLU(slope=0.1)` |
| `wf.SiLU` | smooth ReLU (x · sigmoid(x)); Workflow's default activation | nothing | `wf.SiLU()` |
| `wf.Sigmoid` | squashes to 0…1 | nothing | `wf.Sigmoid()` |
| `wf.MaxPool2D` | keeps the biggest value in each window | `kernel_size=2`, `stride=kernel_size`, `padding="valid"` | `wf.MaxPool2D(kernel_size=2, stride=2, padding="valid")` |
| `wf.MinPool2D` | keeps the smallest value in each window | `kernel_size=2`, `stride=kernel_size`, `padding="valid"` | `wf.MinPool2D(kernel_size=3, stride=1, padding="same")` |
| `wf.AvgPool2D` | averages each window | `kernel_size=2`, `stride=kernel_size`, `padding="valid"` | `wf.AvgPool2D(kernel_size=2, stride=2, padding="valid")` |
| `wf.Upsample` | makes the picture bigger by repeating pixels (no learning) | `scale=2` | `wf.Upsample(scale=2)` |

### Blocks

| Block | What it builds inside | You choose | Example |
|---|---|---|---|
| `wf.Bottleneck` | 1×1 conv → k×k conv → 1×1 conv (each with BatchNorm, SiLU on the first two), plus the shortcut add (projected by a 1×1 conv when channels or stride change), then SiLU | `out_channels=None` (same as input), `kernel_size=3`, `stride=1`, `reduction=2`, `shortcut=True` | `wf.Bottleneck(out_channels=64, kernel_size=3, stride=1)` |
| `wf.CSP` | splits into two 1×1 conv paths; one goes through `n` Bottlenecks; then concat → 1×1 conv | `out_channels=None`, `n=1`, `shortcut=True` | `wf.CSP(out_channels=64, n=2, shortcut=True)` |
| `wf.SPP` | 1×1 conv → max-pools of several sizes side by side (plus the unpooled map) → concat → 1×1 conv; keeps the size | `out_channels=None`, `kernels=(5, 9, 13)` | `wf.SPP(out_channels=128, kernels=(5, 9, 13))` |
| `wf.SPPF` | the fast SPP: 1×1 conv → one max-pool size chained 3× (plus the unpooled map) → concat → 1×1 conv; keeps the size | `out_channels=None`, `kernel_size=5` | `wf.SPPF(out_channels=128, kernel_size=5)` |
| `wf.SE` | channel attention: global average pool → 1×1 conv → ReLU → 1×1 conv → sigmoid → multiply each channel; keeps the shape | `reduction=16` | `wf.SE(reduction=16)` |
| `wf.CBAM` | SE-style channel attention (average and max pooled) then a "where to look" map: k×k conv over the channel mean and max → sigmoid → multiply; keeps the shape | `reduction=16`, `kernel_size=7` | `wf.CBAM(reduction=16, kernel_size=7)` |
| `wf.Focus` | spatial-to-channel: each 2×2 pixel block becomes 4 channels (halves height and width), then a conv with BatchNorm and SiLU | `out_channels`, `kernel_size=3` | `wf.Focus(32, kernel_size=3)` |

### Composition

| Tool | What it does | You choose | Example |
|---|---|---|---|
| `wf.Residual` | nothing of its own: wraps the layers you write inside it with a skip connection (output = layers(x) + x) | the layers, plus an optional `shortcut=` (a layer or list) when the shape changes | see [wf.Residual below](#residual) |
| `wf.parallel` | sends the same input down every branch and merges the results | the branches (2+), `merge="concat"` or `"add"` | see [wf.parallel below](#parallel) |
| `wf.feature` | names the current tensor so later lines can reach back to it; passes it on unchanged | a name, or a tuple of names after FPN/PAN | `wf.feature("p3")` |
| `wf.concat` | joins named features along channels (same height and width) | the feature names | `wf.concat("p3", "up")` |
| `wf.add` | sums named features (identical shapes) | the feature names | `wf.add("p3", "up")` |
| `wf.FPN` | top-down pyramid: 1×1 Conv-BN-SiLU per map, each coarser map resized and added into the next finer one, then a 3×3 Conv-BN-SiLU per level | the feature names, `channels=None` (smallest input) | `wf.FPN("p3", "p4", "p5", channels=64)` |
| `wf.PAN` | bottom-up pass over the pyramid: stride-2 3×3 Conv-BN-SiLU down, add, 3×3 Conv-BN-SiLU refine | `channels=None`, optional feature names | `wf.PAN(channels=64)` |
| `wf.head` | detection head; per map two 3×3 Conv-BN-SiLU and a 1×1 conv giving box, confidence and class scores; must be the last line | `classes=80`, `channels=None` | `wf.head(classes=20)` |

### Tools

| Tool | What it does | You choose | Example |
|---|---|---|---|
| `model.summary()` | prints every layer with its output shape, parameters and MACs (multiply-adds) | `input=` (another picture size), `show=True` | `model.summary()` |
| `model.train(...)` | trains on a Felix split; live bar and per-epoch table; saves `last.nbm` / `best.nbm` | `train`, `validation=`, `epochs=100`, `batch_size=16`, `lr=1e-3`, `device=`, `img_size=`, `optimizer="adamw"`, `patience=30`, `save_every=5`, `output="runs/train"`, `quantization=`, `augment=True`, `ema=True` | `model.train(data.train, validation=data.val, epochs=50, batch_size=16, device="mps")` |
| `wf.eval(model)` | size, FLOPs, latency, FPS, memory; with `dataset=` also mAP, precision, recall | `dataset=None`, `device=`, `img_size=`, `batch_size=8`, `confidence=0.001`, `iou=0.6` | `wf.eval(model, dataset=data.test)` |
| `model.save(path)` / `wf.load(path)` | saves / loads a `.nbm` (weights, training state, class names) | the path | `model.save("detector.nbm")` |
| `wf.inference(source, model, output)` | runs the model on an image, video, folder, Felix dataset or `"webcam"` (live window; ← → / 0–9 switch camera, q quits); saves `original/`, `annotated/`, `predictions.json` | `all=False` (one random image from a dataset/folder), `split=`, `confidence=0.25`, `iou=0.45`, `img_size=`, `device=`, `batch_size=16`, `record=False`, `mirror=False` (webcam) | `wf.inference("photo.jpg", model, "results/")` |
| `wf.QAT(...)` | quantisation-aware training: pass as `quantization=` to `model.train` | `precision="int8"` | `model.train(data.train, validation=data.val, epochs=60, quantization=wf.QAT("int8"))` |
| `model.quantize(...)` | post-training quantisation (also `workflow quantize <model> <type>`) | `precision="int8"`, `calibration=` | `model.quantize("int8", calibration=data.val)` |
| `model.export(...)` | writes ONNX or `.nbm` | `format="onnx"`, `output=`, `precision="fp32"`, `calibration=`, `img_size=` | `model.export("onnx", output="detector.onnx")` |
| `model.plot()` | draws the computation graph as a picture | `detail="layers"`, `show=True` | `model.plot(show=False)` |

### Examples

#### A whole model

`wf.stack` is the model. One line per layer or block, top to bottom, the way the picture flows:

```python
import workflow as wf

model = wf.stack(
    wf.input(channels=3, height=640, width=640),

    # 640 → 320
    wf.Conv2D(16, kernel_size=3, stride=2, padding="same"),
    wf.BatchNorm2D(),
    wf.SiLU(),

    # 320 → 160
    wf.Conv2D(32, kernel_size=3, stride=2, padding="same"),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.CSP(out_channels=32, n=1),

    # 160 → 80: small objects
    wf.Conv2D(64, kernel_size=3, stride=2, padding="same"),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.CSP(out_channels=64, n=2),
    wf.feature("p3"),

    # 80 → 40: medium objects
    wf.Conv2D(128, kernel_size=3, stride=2, padding="same"),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.Bottleneck(),
    wf.SE(reduction=16),
    wf.feature("p4"),

    # 40 → 20: large objects
    wf.Conv2D(256, kernel_size=3, stride=2, padding="same"),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.SPPF(out_channels=256, kernel_size=5),
    wf.feature("p5"),

    # neck and head
    wf.FPN("p3", "p4", "p5", channels=64),
    wf.PAN(channels=64),
    wf.head(classes=20),
)
model.summary()
```

#### Residual

Your own layers with a skip connection around them (they must keep the shape):

```python
wf.Residual(
    wf.Conv2D(64, kernel_size=3, padding="same"),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.Conv2D(64, kernel_size=3, padding="same"),
    wf.BatchNorm2D(),
),
```

If the layers change the shape, give the shortcut a projection:

```python
wf.Residual(
    wf.Conv2D(128, kernel_size=3, stride=2, padding="same"),
    wf.BatchNorm2D(),
    wf.SiLU(),
    shortcut=wf.Conv2D(128, kernel_size=1, stride=2),
),
```

#### Parallel

Each branch gets the same input; `wf.stack` groups the lines of one branch:

```python
wf.parallel(
    wf.stack(
        wf.Conv2D(32, kernel_size=1),
        wf.BatchNorm2D(),
        wf.SiLU(),
    ),
    wf.stack(
        wf.Conv2D(32, kernel_size=3, padding="same"),
        wf.BatchNorm2D(),
        wf.SiLU(),
    ),
    merge="concat",
),
```

#### Features, concat, add

Name a tensor with `wf.feature`, keep going, then reach back to it by name:

```python
    wf.Conv2D(32, kernel_size=3, stride=2, padding="same"),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.feature("p3"),                  # 32 channels

    wf.Conv2D(64, kernel_size=3, stride=2, padding="same"),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.PointwiseConv2D(32),
    wf.Upsample(scale=2),
    wf.feature("up"),                  # back to p3's size, 32 channels

    wf.concat("p3", "up"),             # 64 channels
```

#### Train, check, save, use

```python
import felix as fx

data = fx.load("shapes.felix")
model.train(data.train, validation=data.val, epochs=50, batch_size=16, device="mps")
wf.eval(model, dataset=data.test)
model.save("detector.nbm")

model = wf.load("detector.nbm")
wf.inference("photo.jpg", model, "results/")
wf.inference("webcam", model)                      # live; q to stop
```

---

## Getting started

The quickest start is `workflow init`: it writes a notebook with a small model, benchmarks before and after
training, and saving, that runs as-is ([CLI](#init)). This page explains the pieces.

A model is a **stack** of layers. Each layer receives the previous layer's output, so the only numbers you write
are the ones you decide: output channels, kernel sizes, strides, padding.

```python
import workflow as wf

model = wf.stack(
    wf.input(channels=3, height=640, width=640),

    wf.Conv2D(32, kernel_size=3, stride=2, padding="same"),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.Bottleneck(),
)
model.summary()
```

- **The first number is the output channels.** In `wf.Conv2D(32, kernel_size=3)`, 32 channels come out; the input
  channels are worked out from the previous layer.
- **The first line says what goes in.** `wf.input(channels=3, height=640, width=640)` builds the model for RGB
  pictures of that size and checks every layer before anything runs (3×640×640 if you leave it out).
- **Padding is a word.** `padding="same"` keeps the size (with `stride=2` it halves it); `padding="valid"` adds none,
  so the picture shrinks a little.
- **Bias is automatic.** A convolution right before `BatchNorm2D` has none; any other has one.

### A detector

One layer per line, top to bottom, exactly as the picture flows through them:

```python
model = wf.stack(
    wf.input(channels=3, height=256, width=256),

    # stem: 256 → 128
    wf.Conv2D(16, kernel_size=3, stride=2, padding="same"),
    wf.BatchNorm2D(),
    wf.SiLU(),

    # 128 → 64
    wf.Conv2D(32, kernel_size=3, stride=2, padding="same"),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.CSP(n=1),

    # 64 → 32: good for small objects
    wf.Conv2D(64, kernel_size=3, stride=2, padding="same"),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.CSP(n=2),
    wf.feature("p3"),

    # 32 → 16: medium objects
    wf.Conv2D(128, kernel_size=3, stride=2, padding="same"),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.CSP(n=2),
    wf.feature("p4"),

    # 16 → 8: large objects
    wf.Conv2D(256, kernel_size=3, stride=2, padding="same"),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.SPPF(kernel_size=5),
    wf.feature("p5"),

    # neck and head
    wf.FPN("p3", "p4", "p5", channels=64),
    wf.PAN(channels=64),
    wf.head(classes=3),
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
run = wf.inference("photo.jpg", model, "results/")   # → results/original/, results/annotated/, results/predictions.json
first = run.images[0]
print(first.xyxy, first.classes, first.confidences)
```

Next: [Layers](#layers), [Blocks](#blocks) and [Composition](#composition) for building; [Training](#training),
[Evaluation](#evaluation) and [Quantisation](#quantisation-and-export) for the rest. A full runnable example is the [Demo](#demo).

---

## Layers

Every layer infers its input channels. Shapes below are `(channels, height, width)`; *pair* means an int or an
`(height, width)` tuple.

### Padding

Every layer with a window (convolutions and pooling) takes `padding="same"` or `padding="valid"`:

| `padding=` | Meaning | Output size (per axis) |
|---|---|---|
| `"valid"` (default) | no padding: the window only visits real pixels, so the picture shrinks | `⌊(size − dilation·(kernel − 1) − 1) / stride⌋ + 1` |
| `"same"` | pads just enough to keep the size; with a stride, the size is divided by the stride and rounded up | `⌈size / stride⌉` |

How `"same"` pads: with an odd kernel (3, 5, 7, ...) it adds `(kernel − 1) / 2` pixels on every side, exactly like the
classic "padding 1 for a 3×3"; with an even kernel the extra pixel goes on the bottom and right. Padding is worked out
for each input size, so a model built at 640×640 also runs at 416×416. Numbers (`padding=1`) are not accepted: the
error tells you which name to use.

### Convolutions

The first argument is always **`out_channels`**: how many channels come *out* (how many different patterns the layer
looks for). Input channels are never written: Workflow works them out from the previous layer.

| Layer | Signature | Output channels | Notes |
|---|---|---|---|
| `Conv2D` | `Conv2D(out_channels, kernel_size=3, stride=1, padding="valid")` | `out_channels` | standard convolution |
| `DilatedConv2D` | `DilatedConv2D(out_channels, kernel_size=3, dilation=2, stride=1, padding="valid")` | `out_channels` | taps spread `dilation` pixels apart; `"same"` keeps the size |
| `DepthwiseConv2D` | `DepthwiseConv2D(kernel_size=3, stride=1, padding="valid")` | same as input | one filter per channel |
| `PointwiseConv2D` | `PointwiseConv2D(out_channels)` | `out_channels` | 1×1 convolution: kernel 1, stride 1, no padding (fixed) |
| `SeparableConv2D` | `SeparableConv2D(out_channels, kernel_size=3, stride=1, padding="valid")` | `out_channels` | depthwise then pointwise |
| `TransposeConv2D` | `TransposeConv2D(out_channels, kernel_size=2, stride=2, padding="valid")` | `out_channels` | upsampling convolution: `"valid"` gives `(size − 1)·stride + kernel`, `"same"` gives `size·stride` (needs `kernel_size ≥ stride`) |

Requirements: `out_channels` ≥ 1; `kernel_size`, `stride`, `dilation` ≥ 1 (an int, or `(height, width)`). If the kernel
does not fit the input (with `"valid"`), the error says how big each one is and suggests `"same"`. Weights are
initialised uniformly in ±1/√fan_in.

### Normalisation and activations

| Layer | Signature | Notes |
|---|---|---|
| `BatchNorm2D` | `BatchNorm2D()` | per channel; eps 1e-3, momentum 0.03; channels inferred |
| `ReLU`, `SiLU`, `Sigmoid` | no arguments | keep the shape |
| `LeakyReLU` | `LeakyReLU(slope=0.1)` | `slope` is the gradient for negative inputs |

### Pooling and resizing

Pooling slides a window over the picture and keeps one number per window. No weights are learned.

| Layer | Signature | Keeps |
|---|---|---|
| `MaxPool2D` | `MaxPool2D(kernel_size=2, stride=None, padding="valid")` | the largest value in each window |
| `MinPool2D` | `MinPool2D(kernel_size=2, stride=None, padding="valid")` | the smallest value in each window |
| `AvgPool2D` | `AvgPool2D(kernel_size=2, stride=None, padding="valid")` | the average of each window |
| `Upsample` | `Upsample(scale=2)` | (not a pool) makes the picture `scale` times bigger by copying pixels (nearest neighbour) |

- `stride` defaults to `kernel_size`, so `MaxPool2D(2)` halves the picture.
- `padding="same"` with `stride=1` keeps the size: `MaxPool2D(kernel_size=5, stride=1, padding="same")` is the
  pooling used inside `SPPF`.
- Padded pixels never win a max or a min. In an average they count as zero.

---

## Blocks

Reusable architecture blocks. `out_channels` defaults to the input channel count wherever it is optional.

| Block | Signature | Output | What it does |
|---|---|---|---|
| `Bottleneck` | `Bottleneck(out_channels=None, kernel_size=3, stride=1, reduction=2, shortcut=True)` | `(out_channels, ⌈H/stride⌉, ⌈W/stride⌉)` | 1×1 reduce to `out_channels/reduction` → k×k spatial → 1×1 expand, each Conv-BN(-SiLU); the input is added back (directly, or through a 1×1 projection when channels or stride change), then SiLU |
| `CSP` | `CSP(out_channels=None, n=1, shortcut=True)` | `(out_channels, H, W)` | splits into two 1×1 paths, runs `n` Bottlenecks on one, concatenates, fuses with a 1×1 |
| `SPP` | `SPP(out_channels=None, kernels=(5, 9, 13))` | `(out_channels, H, W)` | 1×1 to half the channels, parallel max-pools of each size, concatenated with the input, 1×1 |
| `SPPF` | `SPPF(out_channels=None, kernel_size=5)` | `(out_channels, H, W)` | the fast SPP: three chained max-pools of one size (same receptive fields, cheaper) |
| `SE` | `SE(reduction=16)` | same | squeeze-and-excitation: global average → 1×1 (channels/reduction) → ReLU → 1×1 → sigmoid, scales each channel |
| `CBAM` | `CBAM(reduction=16, kernel_size=7)` | same | channel attention (average- and max-pooled through a shared MLP) then spatial attention (k×k conv over the channel mean and max) |
| `Focus` | `Focus(out_channels, kernel_size=3)` | `(out_channels, H/2, W/2)` | every 2×2 pixel block becomes 4 channels (nothing is lost), then Conv-BN-SiLU; needs an even height and width |

Every block pads with `"same"` internally, so the picture keeps its size whatever the kernel (odd kernels such as 3
and 5 are the usual choice). `n` and `reduction` must be at least 1.

`Bottleneck` is a real bottleneck, not a container: the three convolutions are fixed by its arguments, and the
shortcut is matched automatically, so the two paths always have the same shape.

---

## Composition

| Function | Meaning |
|---|---|
| `wf.stack(wf.input(*None), *layers)` | layers in order. The outermost stack is the model; `input` is `(C, H, W)`, default `(3, 640, 640)` |
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
    wf.input(channels=3, height=256, width=256),

    wf.Conv2D(32, kernel_size=3, stride=2, padding="same"),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.feature("p3"),                  # 1/2 size, 32 channels

    wf.Conv2D(64, kernel_size=3, stride=2, padding="same"),
    wf.BatchNorm2D(),
    wf.SiLU(),
    wf.feature("p4"),                  # 1/4 size, 64 channels

    wf.PointwiseConv2D(32),
    wf.Upsample(scale=2),
    wf.feature("up"),                  # back to 1/2 size

    wf.concat("p3", "up"),             # 32 + 32 = 64 channels at 1/2 size
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
  fix: keep the shape inside the block (same channels, stride 1, padding='same'), or project the input with
       shortcut=wf.Conv2D(16, kernel_size=1, stride=2).
```

---

## Model

The outermost `wf.stack(...)` is the model.

| Member | What it does |
|---|---|
| `model.summary(input=None)` (`input=(3, 416, 416)` checks another size) | table of every layer's output shape, parameters and MACs; totals for parameters (trainable / non-trainable), size and GFLOPs. Returns the totals |
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

### Inference: `wf.inference`

The normal way to use a trained model:

```python
model = wf.load("detector.nbm")
run = wf.inference("photo.jpg", model, "results/")
```

`wf.inference(source, model, output)` works out what `source` is on its own:

| `source` | What runs |
|---|---|
| an image (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`) | that image |
| a video (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`, ...) | every frame; the annotated copy is an `.mp4` (needs `opencv-python`) |
| a folder | **one** image or video from it, picked at random (`all=True`: every one, not subfolders) |
| a Felix dataset (`.felix` file or dataset folder) | its **test** split (val if there is no test split, everything if there are no splits), and from it **one** image picked at random (`all=True`: every one) |
| a loaded dataset or split (`data`, `data.val`, ...) | the same: one random image of the test split, or of exactly the split you pass (`all=True`: every one) |
| `"webcam"` (or `"webcam:1"`, ...) | live detection from a camera in a window (needs `opencv-python`), see [Webcam](#webcam) |
| a list of any of these | exactly the items in the list |

Datasets and folders are usually many pictures, so by default `wf.inference` takes one random example to look at:
run it again for another, or pass `all=True` to process them all.

Choose a different split of a dataset with `split="val"` or `split="train"`, or by passing `data.val` yourself.
`model` can also be the path of a `.nbm`.

The output folder (created if needed) gets:

```
results/
├── original/            the inputs, untouched (dataset images are copied byte for byte)
├── annotated/           the same pictures with boxes and labels drawn (videos as .mp4)
└── predictions.json     every detection, plus which files belong together
```

```json
{"model": "detector.nbm", "classes": ["circle", "square", "triangle"], "source": "Felix dataset shapes.felix · test split (picked automatically) · 60 images · 1 picked at random (IMG13.png); all=True runs every item",
 "settings": {"confidence": 0.25, "iou": 0.45, "img_size": null},
 "images": [{"image": "IMG13.png", "original": "original/IMG13.png", "annotated": "annotated/IMG13.png", "width": 256, "height": 256,
             "detections": [{"class_id": 2, "class": "triangle", "confidence": 0.99, "xyxy": [154.1, 189.9, 198.4, 238.9]}]}],
 "videos": [{"video": "clip.mp4", "original": "original/clip.mp4", "annotated": "annotated/clip.mp4", "fps": 30.0, "frames": 25,
             "detections": [[...frame 1...], [...frame 2...]]}]}
```

| Argument | Default | Meaning |
|---|---|---|
| `output` | `"results/"` | the folder above |
| `all` | `False` | run every image of a dataset or folder (default: one, picked at random) |
| `split` | `None` (test) | which split of a Felix dataset |
| `confidence` | 0.25 | minimum score |
| `iou` | 0.45 | NMS overlap threshold |
| `img_size` | the model's input | inference size |
| `device` | auto | `"cpu"`, `"cuda"`, `"gpu"`, `"mps"` |
| `batch_size` | 16 | images (or video frames) per batch |
| `quiet` | False | hide the progress panel |
| `record` | False | webcam: save the session (both videos and `predictions.json`) |
| `mirror` | False | webcam: flip the picture horizontally (selfie view) |

It returns an `Inference`: `.images` (a `Results` for each image, see below), `.videos` (`{name: per-frame detections}`),
`.output`, `.json` and `.json_path`. Names never collide: two inputs called `photo.png` become `photo.png` and
`photo_2.png`. From the terminal: `workflow predict detector.nbm clip.mp4` (same rules, `--all`, `--split`, `-o`), or `workflow predict detector.nbm webcam`.

#### Webcam

```python
wf.inference("webcam", model)                                 # camera 0, live, nothing saved
wf.inference("webcam:1", model, "results/cam/", record=True)  # start on camera 1, save the session
```

A window called Workflow opens with the boxes (labels on solid tags, sized to the picture) and the FPS in the bottom-left. The keys work while the window has focus; the
console shows which camera is on and the controls:

| Key | Does |
|---|---|
| `→` / `↑` | next camera (after the last one, back to 0) |
| `←` / `↓` | previous camera |
| `0`–`9` | that camera (`0` is `"webcam"`, `1` is `"webcam:1"`, ...) |
| `q`, `Esc` (or closing the window, or Ctrl+C) | stop |

A camera that doesn't exist is reported in the console and the current one keeps running. Without `record=True`
nothing is written; with it, `original/webcam-<time>.mp4`, `annotated/webcam-<time>.mp4` and `predictions.json`
(every frame's detections, and `cameras`: which camera was on from which frame). The result's `.videos["webcam"]`
holds the per-frame detections either way. On a Mac, the app running Python (Terminal, VS Code, Jupyter) needs camera
access in System Settings → Privacy & Security → Camera. From the terminal: `workflow predict detector.nbm webcam`.

### Prediction in memory: `model.predict`

`wf.inference` is built on `model.predict`, which you can call directly when you want results in Python without
the folder of files:

```python
model = wf.load("detector.nbm")
results = model.predict("image.jpg")
results = model.predict(["image1.jpg", "image2.jpg"])
results = model.predict("image.jpg", device="cuda", confidence=0.25, iou=0.45, img_size=640)
results = model.predict("image.jpg", output="predictions/")
results = model.predict(frame_rgb_array, output=None)              # nothing written: handy inside your own loop
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
| `low_memory` | auto | `True`: recompute activations during backward instead of keeping them (identical results, ~30% slower, ~40% less memory); auto turns it on by itself on CPU when needed (see Memory) |
| `autosave` | 10 | minutes between mid-epoch saves of `last.nbm` (`None` turns it off; see Crash protection) |

Returns the history: one dict per epoch (`epoch`, `loss`, `box`, `obj`, `cls`, `lr`, `seconds`, and with validation
`val_loss`, `val_box`, `val_obj`, `val_cls`, `map50`, `map50_95`, `precision`, `recall`, `f1`, `fitness`). It is also kept in `model.history`.

### Progress

Training shows a live panel. In a terminal it redraws in place; in a notebook (Jupyter, VS Code, Colab) it redraws in
the cell's output, about twice a second. `quiet=True` hides it.

- **The bar**: epoch and step, and the running *training* loss of the current epoch with its parts (box, obj, cls) and the
  learning rate.
- **After each epoch**, a line is added to a table under the bar:

```
epoch   train loss   val loss   mAP50   mAP50-95
    1        7.963     14.018   0.000      0.000
    2        3.718     14.239   0.000      0.000
    3        3.099     13.865   0.000      0.000
    4        2.664      9.462   0.017      0.011
    5        2.497      4.697   0.585      0.191  ← best
```

The **validation loss** is the same loss as training (box + confidence + class), measured on the validation pictures
with the weights that get saved (the moving average, when `ema=True`). Early on it is often higher than the training
loss, because that average lags behind the newest weights; it catches up as training settles. The table shows the
latest 15 epochs (continued runs include the earlier ones); every epoch is in `model.history`, e.g.:

```python
for r in model.history:
    print(r["epoch"], r["loss"], r["val_loss"], r["map50_95"])
```

### Defaults

- **Weights**: convolutions uniform in ±1/√fan_in; BatchNorm γ=1, β=0; the head starts with box distances of about
  1.3 strides and confidences of about 1%, which keeps the first steps stable.
- **Optimizer**: AdamW, weight decay 5e-4 on weights (not on biases and BatchNorm).
- **Schedule**: linear warm-up (about 3 epochs, at least 100 steps), then cosine decay to 1% of `lr`.

### Learning rate

Two things adapt, one doesn't:

- **Per weight (adaptive)**: AdamW gives every weight its own effective step size, from the running size of its
  gradients. Weights with big, noisy gradients take smaller steps; quiet ones take bigger steps.
- **Over time (scheduled)**: the overall `lr` follows a fixed plan worked out from `epochs`: it ramps up from almost
  0 during warm-up, then glides down along a cosine curve to 1% of `lr` at the last epoch. The `lr` on the progress
  bar is this value.
- **Not adaptive**: it doesn't react to the validation loss (no "reduce on plateau"). Because the plan depends on
  `epochs`, a run with more epochs decays more slowly; a continued run keeps following the plan for the new total.
- **Stability**: gradients clipped at norm 10; a NaN/inf loss stops training with an explanation.
- **Best model**: highest `0.1 × mAP50 + 0.9 × mAP50-95` on the validation data (lowest loss without validation).

### Crash protection

- **Ctrl+C, `kill`, closing a service (SIGTERM)**: training finishes the step it is on (a few seconds), saves
  `output/last.nbm` with the exact position (epoch *and* step), and `model.train` returns the history so far. A second
  Ctrl+C stops at once.
- **Autosave**: inside an epoch, `last.nbm` is also saved every `autosave` minutes (default 10). A power cut, an
  out-of-memory kill or a crashed machine loses at most those minutes, not the epoch (an epoch at 640 px on a laptop
  CPU can take over an hour).
- **Errors**: if anything else goes wrong (a broken image, a full disk, ...), progress up to the last finished step is
  saved before the error is shown. A NaN / infinite loss is the exception: those weights are broken, so the last good
  save is kept.
- **Safe files**: checkpoints are written to a temporary file, flushed to disk and then renamed over the old one,
  so a crash during a save leaves the previous checkpoint, never half a file.

To continue, load `last.nbm` and train again with the same arguments:

```python
model = wf.load("runs/train/last.nbm")
model.train(train, validation=val, epochs=100)     # "continuing epoch 7 from step 412 of 996"
```

It resumes at the same step, with the same shuffled order for that epoch, the same optimizer state, EMA, and
BatchNorm statistics; without augmentation the result is the same as a run that never stopped.

### Memory

Each training step keeps every layer's output until the backward pass has used it, so memory grows with batch size,
picture size and model width. On CPU (Linux), training first measures what one step needs (a forward and backward pass
on one blank picture, taking a moment) and compares it with free RAM:

- **It fits**: normal training.
- **It fits only with low-memory mode**: it is switched on and a note says so. Each Conv → BatchNorm → activation unit
  and each block keeps only its input; its insides are recomputed during backward. Same numbers (gradients and
  BatchNorm statistics match normal training), about 30% slower, about 40% less memory.
- **It doesn't fit at all**: training stops before it starts, with the largest `batch_size` that fits, instead of the
  system killing it an hour in.

Pass `low_memory=True` to force it on any device, or `False` to turn the check off. Datasets are read from disk as
needed, so their size doesn't matter; only the batch does.

### Continuing

```python
model = wf.load("runs/train/last.nbm")
model.train(train, validation=val, epochs=150)
```

`epochs` is the total target: a run saved at epoch 80 continues from 81 to 150, with the optimizer state,
raw weights, EMA and history restored. A model already at the target does nothing (and says so). If the
saved optimizer state doesn't fit the model, a fresh optimizer is created with a note.

### Speed

Measured on the demo detector (1.3M parameters, 256×256, batch 16): about 47 images/s on a GTX 1050. The
`workflow init` model at 640×640, batch 16, on an Apple M2 GPU (`device="mps"`): about 45 images/s (0.3.7; 16 before).

On CPU (0.4), the `workflow init` model on a 2012 laptop (Intel i5-3210M, 2 cores, AVX, 4 GB RAM):

| | 0.3 | 0.4 |
|---|---|---|
| 320×320, batch 16 | 3.5 images/s | 11.7 images/s |
| one epoch, 642 pictures at 320, with validation | 182 s | 52 s |
| 640×640, batch 16 | 0.76 images/s (2.8 GB) | 3.0 images/s (1.8 GB) |

What changed: convolutions use a packed, register-blocked matrix multiply (the BLIS design) that reads the picture
directly instead of building a 9× larger copy first; BatchNorm and the activation run as one vectorised pass; and
backward frees each layer's saved values as soon as they are used. All of it is plain fp32 with the same results.

The cost of a step grows with the picture area: 640×640 is 2.4× the work of 416×416 and 4× of 320×320. If your
objects aren't tiny, a smaller `img_size` is the biggest speed-up there is. Workflow's kernels are its own; they are slower than cuDNN-backed frameworks on
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

### How the Apple GPU backend is fast

- Tensors live on the GPU channels-last (MLX's own layout), so convolutions never reshuffle memory.
- A training step is built as one graph and sent to the GPU at once, instead of waiting after every operation.
- Custom Metal kernels for BatchNorm statistics (one read of the data instead of several) and max pooling (the
  gradient goes straight to the winning pixel).
- BatchNorm followed by an activation (`BatchNorm2D()` then `SiLU()`, `ReLU()`, ...) runs as one fused operation.
- The next batch of pictures is decoded on the CPU while the GPU trains on the current one.

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
workflow predict <model_path> <image_path>      detect objects; original/, annotated/, predictions.json → results/
workflow eval MODEL.nbm [--dataset D --split test]   size/FLOPs/FPS, plus accuracy with a dataset
workflow quantize MODEL.nbm int8|fp16|fp32 [--calibration D] [-o OUT]   post-training quantisation
workflow export MODEL.nbm --format onnx|nbm [--precision ...] [-o OUT]
workflow script MODEL.nbm --preset image|folder|video|webcam    write an inference script
workflow devices                               cpu / cuda / mps, and why something is unavailable
```

`quantize` writes `MODEL-<type>.nbm` by default and prints the size before and after. Commands never overwrite
existing files without `--overwrite`. Errors print the what / why / fix explanation and exit with status 1.

### predict

```
workflow predict detector.nbm photo.jpg                    # one image → results/
workflow predict detector.nbm photo1.jpg photo2.jpg        # several images
workflow predict detector.nbm clip.mp4                     # a video, every frame
workflow predict detector.nbm photos/ --all                # every image in a folder (default: one at random)
workflow predict detector.nbm shapes.felix --all           # every test image of a Felix dataset
workflow predict detector.nbm webcam                       # live, in a window: ← → or 0–9 switch camera, q quits
workflow predict detector.nbm webcam:1 --record --mirror   # start on camera 1, save the session, selfie view
```

The model comes first, then the image (or images). Results go to `results/`: `original/`, `annotated/` and
`predictions.json`. Options: `-o` (another folder), `--all`, `--split train|val|test`, `--confidence`, `--iou`,
`--img-size`, `--device`, and for the webcam `--record` and `--mirror`. This is `wf.inference` from the terminal ([Detection](#detection-and-inference)).

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

It also writes **`library.md`** next to the notebook: one table of every layer, block, composition tool and model
tool, with what it builds inside and what you choose ([Library](#workflow-library)). An existing `library.md` is kept
unless you pass `--overwrite`.


### Inference scripts

| Preset | Does | Run |
|---|---|---|
| `image` | one or more images → `results/` | `python infer_image.py photo.jpg` |
| `folder` | every image in a folder → `results/` | `python infer_folder.py photos/` |
| `video` | a video with live preview, optional `--save out.mp4` | `python infer_video.py clip.mp4` |
| `webcam` | live camera detection (`--mirror` for selfie view) | `python infer_webcam.py` |

All take `--confidence`, `--iou`, `--img-size` and `--device`. `video` and `webcam` need `opencv-python`.

---

## Changes

### 0.4.0

- **CPU training 3.3–4× faster** (the `workflow init` model on a 2-core 2012 laptop: 3.5 → 11.7 images/s at
  320 px, 0.76 → 3.0 at 640 px), with the same results:
  - convolutions use a packed, register-blocked matrix multiply (6×16 micro-kernel; AVX, AVX2+FMA or NEON) that
    reads the picture directly (no im2col copy), forward, weight gradient and stride-1 input gradient;
  - BatchNorm + activation run as one vectorised pass and the BatchNorm output is no longer stored;
  - vectorised exp / sigmoid / SiLU; multi-threaded upsample and pooling backward; vectorised max pooling.
- **Less memory**: backward frees what each layer kept as soon as it is used (640 px batch 16: 2.8 GB in 0.3 → 1.8 GB).
- **`low_memory=`** for `model.train`: recompute activations instead of keeping them (identical gradients and
  BatchNorm statistics, ~30% slower, ~40% less memory). On CPU it is chosen automatically when needed, and a run
  that cannot fit stops before it starts, with the batch size that fits.
- **Crash protection**: autosave of `last.nbm` every 10 minutes inside an epoch (`autosave=`), exact resume at the
  saved step (reproducible shuffling), Ctrl+C and SIGTERM stop at the end of the step, other errors save first,
  checkpoints are flushed to disk before replacing the old file.

### 0.3.11

- Webcam window: thicker boxes and bigger labels on solid colour tags (both scale with the picture), FPS in the bottom-left, the window is just called "Workflow".

### 0.3.10

- `wf.inference("webcam", model)` and `workflow predict model.nbm webcam`: live detection in a window, FPS on the picture.
  `←` `→` (or `↑` `↓`) and `0`–`9` switch camera; the controls and the current camera are shown in the console.
  `record=True` / `--record` saves the session, `mirror=True` / `--mirror` flips it.

### 0.3.9

- `library.md` has an Example column (a ready-to-copy line per layer, block and tool) and full examples: a whole model, Residual, parallel, features, train → save → inference.

### 0.3.8

- `workflow init` also writes `library.md`: every layer, block and tool in one table (also in the docs as Library).

### 0.3.7

- Apple GPU (`mps`) training about 2.7× faster (the init model at 640: 16 → 45 images/s on an M2), same results.
- The next batch is prepared while the current one trains (all devices).
- Training shows validation loss, and a per-epoch table (0.3.6).

### 0.3

| 0.2 | 0.3 |
|---|---|
| `wf.Conv2D(filters, ...)` (and the other convolutions and blocks) | `wf.Conv2D(out_channels, ...)`: same position, clearer name |
| `padding=1`, `padding=0`, `padding=(t, b, l, r)` | `padding="same"` or `padding="valid"` |
| `MaxPool2D`, `AvgPool2D` | plus `MinPool2D`; all three take `kernel_size`, `stride`, `padding="same"`/`"valid"` |
| `model.predict(...)` as the main way to run a model | `wf.inference(source, model, output)`: images, videos, folders or Felix datasets (test split automatically), saving `original/`, `annotated/` and `predictions.json`. `model.predict` stays for in-memory use |
| `wf.stack(..., input=(3, 640, 640))` at the bottom | `wf.input(channels=3, height=640, width=640)` as the first line of the stack |

Models saved by 0.2 or an earlier 0.3 build still load and behave the same: their numeric padding, `filters=` and
`input=` are read as before.
Odd kernels with `"same"` pad exactly like the old `padding=kernel_size//2`.

### 0.2

Workflow 0.2 replaces PyTorch with its own engine and adopts a simpler API.

### Breaking changes

| 0.1 | 0.2 |
|---|---|
| `wf.Model(wf.Stage("p1", ...), ...)` | `wf.stack(...)`; named tensors come from `wf.feature("p1")` |
| `wf.Conv2D(in_channels, out_channels, k)` | `wf.Conv2D(out_channels, kernel_size=k)`; input channels are inferred |
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
