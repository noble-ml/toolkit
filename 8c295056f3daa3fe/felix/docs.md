# Felix: complete documentation

*This file is the whole documentation in one page. It is generated from the individual pages; do not edit it by hand.*

## Contents

- [Install](#install)
- [Getting started](#getting-started)
- [API reference](#api-reference)
- [CLI](#cli)
- [Dataset formats](#dataset-formats)

---

A lightweight dataset pipeline generator for object detection. Describe classes and a
generation step in a few lines of Python; Felix renders images, tight bounding boxes and
metadata. Felix renders with its own NumPy backend (shapes, transforms, compositing, PNG) and needs only NumPy and
rich; Pillow is optional, for JPEG files. Works on Linux, macOS and Windows (Python 3.8+).

```python
import felix as fx

with fx.class_set("animals"):
    cat = fx.toolkit.circle(radius=30)
    dog = fx.toolkit.rectangle(width=50, height=40)

with fx.step(size=(640, 640)):
    fx.canvas.background("#eeeeee")
    obj = fx.paint(fx.random.class_(), position=fx.random.position(margin=40))
    fx.toolkit.transformation(obj, rotation=fx.random.uniform(0, 360))
    fx.gen()

fx.make(pairs=10_000, output="./dataset")        # folder
# fx.make(pairs=10_000, output="./dataset.felix")  # one compiled file
```

Try the full example: [animals.py](examples/animals.py) (`python examples/animals.py` or `felix run examples/animals.py`).

## Install

See [Install](#install). Short version:

```
pip install numpy rich
pip install --no-deps --no-index --find-links BASE_URL/simple/felix/ felix
```

(`--no-index` stops pip from fetching the unrelated public `felix` from PyPI.)

---

## Install

### Colab / notebooks (direct link)

```
!pip install BASE_URL/files/felix-0.4.0-py3-none-any.whl
```

A direct wheel URL needs no extra flags: pip fetches NumPy and rich from PyPI itself, and
there is no name clash because it never searches an index for `felix`. Replace the version in
the file name for other releases (all wheels stay on the site).

### Anywhere else (index)

```
pip install numpy rich
pip install --no-deps --no-index --find-links BASE_URL/simple/felix/ felix
```

That's it. Works the same on Linux, macOS and Windows (Python 3.8+). Check with `felix --version`.

Why the extra flags:

- **`--no-index`**: pip must not search PyPI. A *different, public* package called `felix` lives
  there and would be installed instead of this one. Never use `--extra-index-url` for Felix.
- **`--no-deps`** and the separate `pip install numpy rich`: with `--no-index` pip can't fetch
  dependencies from this page, so install them first the normal way.

Upgrade to the newest version by adding `--upgrade`; pin one with `felix==0.4.0`.

Try the example pipeline: download [animals.py](examples/animals.py) and
[bicycle.png](examples/assets/bicycle.png) (keep it in an `assets/` folder next to the script), then
run `python animals.py`.

### JPEG

Felix reads and writes PNG on its own. Reading JPEG (for example Roboflow exports with `.jpg` images, or JPEG
sprites and backgrounds) needs Pillow: `pip install Pillow`. Without it Felix explains what is missing.

### Notes

Felix is proprietary software. This page is unlisted but not access-controlled: anyone with
the link can download the package, and the package is plain Python.

---

## Getting started

A Felix pipeline has five parts: **class sets → step → random → paint/transformation → gen/make**.

### 1. Class sets

```python
with fx.class_set("animals"):
    cat  = fx.toolkit.circle(radius=30, position=(32, 32))
    dog  = fx.toolkit.rectangle(width=50, height=40)
    bird = fx.toolkit.polygon(vertices=[(0, 30), (20, 0), (40, 30)])
with fx.class_set("vehicles"):
    bicycle = fx.toolkit.image(source="assets/bicycle.png")
```

Every shape created inside a `class_set` becomes a **class**. Its name is the variable
name (`cat`, `dog`, ...), or pass `name="..."`. Class ids are assigned in definition order.
Colours default to a built-in palette; override with `color="#ff8800"`. A shape's
`position` is its default location if `fx.paint` is not given one.

### 2. Step

```python
with fx.step(size=(640, 640)):
    ...
```

**The body runs once** to record a recipe. Calls to `fx.random.*` don't produce numbers
right away - they're *lazy* and are re-sampled for every generated image. This means:

- plain Python `if`/`for` on a random value will not work (it isn't a number yet);
- loops that *build* the scene do work: `for _ in range(5): fx.paint(...)` paints five
  objects, each with its own independent random draws;
- the same random node used twice in one image gives the same value both times (e.g.
  `selected_set` feeding `class_`).

### 3. Random + threshold

`fx.random.class_set()`, `fx.random.class_(set)`, `uniform`, `integer`, `choice`,
`position`, `color`. All accept `threshold=p` (0-1): the probability the value is
produced at all. On a failed roll the value is `None`: painting `None` does nothing and a
`None` transformation parameter is ignored. So `fx.paint(fx.random.class_(threshold=0.5))`
adds an object to about half the images.

### 4. Paint and transformation

`fx.paint(cls, position=...)` places a class and returns a handle.
`fx.toolkit.transformation(handle, hue=, brightness=, rotation=, scale=, opacity=)` modifies it.
Transformations are applied to the handle's object when the image is rendered, so you can call
them after `paint`.

### 5. gen and make

`fx.gen()` takes every object painted **before** it, measures each one's tight bounding box
(pixels that are fully transparent don't count - ideal for PNG cut-outs) and records the
annotations. Objects painted after `gen` are still drawn but are not annotated (handy for
distractors). Objects annotated by `gen` are frozen: later transformations on them are
ignored, so image and annotation always agree. The image itself is saved when the step ends.

`fx.make(pairs=10_000, output="./dataset")` repeats the step and writes the dataset.
Pass a path ending in `.felix` for a single compressed file. `seed=` makes runs
reproducible; the seed used is stored in `metadata.yml`.

Note: an object partly covered by a later object keeps its full box (occlusion is not
subtracted); an object fully off-canvas is skipped and reported.

### Rendering backend

Felix draws everything itself with NumPy: shapes are rasterised with 4×4 supersampling for smooth edges;
scaling uses an anti-aliased triangle filter; rotation is bilinear and grows the sprite so nothing is clipped;
hue shifts work in HSV; layers are composited with premultiplied alpha; images are written as PNG by Felix's own
encoder. Bounding boxes come from the sprite's alpha channel (pixels above 16/255), so transparent padding in
image sprites never inflates a box. About 75 images/s at 640×640 on a laptop CPU.

---

## API reference

`import felix as fx`

### Classes

#### `fx.class_set(name)`
Context manager. Shapes created inside become classes of the set `name`. Re-using a name adds
to the same set. Class names must be unique across the whole project.

#### `fx.toolkit` shapes
All return a shape; all accept `position=(x, y)` (default placement), `name=` (override the
inferred class name) and, except `image`, `color=` (hex, CSS name or RGB tuple).

| Function | Parameters |
|---|---|
| `circle` | `radius` |
| `rectangle` | `width`, `height` |
| `polygon` | `vertices=[(x, y), ...]` (≥ 3) |
| `image` | `source` path, optional `width` / `height` (keeps aspect if only one). Use PNGs with transparency for tight boxes. Relative paths resolve against the working directory, then the calling script's folder. |

#### `fx.stack(*parts, position=None, color=None, name=None)`
One object built from several shapes:

```python
with fx.class_set("animals"):
    cat = fx.stack(
        fx.toolkit.polygon(vertices=[(0, 14), (7, 0), (14, 14)], position=(-13, -22)),   # left ear
        fx.toolkit.polygon(vertices=[(0, 14), (7, 0), (14, 14)], position=(13, -22)),    # right ear
        fx.toolkit.circle(radius=22),                                                     # head
        fx.toolkit.circle(radius=3, position=(-8, -4), color="black"),                    # eyes
        fx.toolkit.circle(radius=3, position=(8, -4), color="black"),
        color="#e8a33d",
    )
```

- Inside `fx.class_set` the stack is **one class**, named after its variable (`cat`); its parts are not classes.
- Each part's `position` is where *its centre* sits relative to the stack's centre (default `(0, 0)`); later parts
  are drawn on top of earlier ones.
- Parts without a `color` take the stack's colour (or the class colour from the palette); parts with one keep it.
- The stack is painted, transformed (rotation, scale, hue, ...) and boxed as a whole: its box covers every visible
  part. `fx.paint(cat, position=...)` places the centre of the stack's overall extent.
- Stacks can contain stacks. Any shape works as a part: circles, rectangles, polygons, images.
- A shape that is already a registered class can't be reused as a part (create the part in the call instead).

### Steps

#### `fx.step(size=(640, 640), name=None)`
Context manager; see [Getting started](#getting-started). Several steps may be defined; `make`
splits `pairs` evenly between them. Steps cannot be nested.

#### `fx.canvas.background(color=None, image=None)`
`color` is a colour (or random node). `image` is a file, or a folder (one image picked at
random per sample); it is cover-fitted to the canvas. Default is white.

#### `fx.paint(target, position=None)` → handle
`target` can be a random class node, a class variable, a class name string, `None` (no-op), or
a plain shape that is *not* a class (drawn but never annotated). `position` is the object's
centre in pixels (value or random node). Default: the shape's `position`, else canvas centre.

#### `fx.toolkit.transformation(handle, hue=None, brightness=None, rotation=None, scale=None, opacity=None)`
- `hue`: degrees added to the hue (-180..180 covers everything)
- `brightness`: multiplier (1 = unchanged)
- `rotation`: degrees, counter-clockwise (the sprite grows so nothing is clipped)
- `scale`: multiplier, > 0
- `opacity`: multiplier 0-1

Repeated calls accumulate (hue/rotation add; the rest multiply). Any parameter may be a random
node. Ignored once `fx.gen()` has annotated the object.

#### `fx.gen()`
Annotate all not-yet-annotated objects painted so far. See [Getting started](#getting-started).

### Random (`fx.random`)

| Function | Produces |
|---|---|
| `uniform(lo, hi)` / `value` | float; `lo`/`hi` may be tuples (per component) |
| `integer(lo, hi)` | int, inclusive |
| `choice(*options)` | one option (options may be random nodes) |
| `position(margin=0)` | random `(x, y)` on the canvas |
| `color()` | random `#rrggbb` |
| `class_set(*names)` | a class set (optionally only among `names`) |
| `class_(from_=None)` / `cls` | a class from a set node, a set name, or any class if `None` |

All take `threshold=` (0-1, default 1). `class` is a Python keyword, hence `class_`.

### Output

#### `fx.make(pairs, output, seed=None, overwrite=False, quiet=False)`
Writes `pairs` images + annotations. `output` ending in `.felix` writes a compiled file,
otherwise a folder. Refuses to overwrite an existing dataset unless `overwrite=True`. Prints a
live progress panel (disable with `quiet=True`). Returns `{"pairs", "seed", "boxes", "output"}`.

#### `fx.load(path)` → `Dataset`
Read a folder or `.felix`: `len(ds)`, `ds.metadata`, `ds.stems`, `ds.classes` (names ordered by class id),
`ds.annotation(stem)`, `ds.image(stem)` (a Felix `Image`, see below), `ds.image_bytes(stem)`, iteration yields
`(stem, annotation)`. Use as a context manager or call `ds.close()`. Datasets are picklable (each
DataLoader worker re-opens its own file handle).

**Splits.** `ds.train`, `ds.val` (alias `ds.valid`) and `ds.test` are lightweight `Subset` views with the same
read API (`len`, `stems`, `annotation`, `image`, iteration, `classes`, `metadata`); `ds.splits` gives the counts.

- If the annotations carry a `split` field (converted datasets), it is used (`valid`/`validation` count as `val`).
- Otherwise the dataset is split **deterministically 80/10/10**: shuffled with the dataset's `seed`, so the
  same file always gives the same splits. Datasets with fewer than 10 pairs are not split (all `train`).

#### `fx.partition(source, output=None, parts=None, ratio=None, size=None, by_split=False, names=None, shuffle=False, seed=0, pack=None, overwrite=False, quiet=False)` → list of paths
Split a dataset into several complete datasets; the Python form of [`felix partition`](#cli). Give exactly one of
`parts=N`, `ratio=(80, 10, 10)`, `size=K` or `by_split=True`. `source` is a path or a loaded `Dataset`.

#### `felix.raster.Image`
What `ds.image(stem)` returns: an RGB or RGBA image backed by a NumPy array. `.size`, `.width`, `.height`,
`.mode`, `.array` (uint8, height × width × channels), `.convert("RGB" | "RGBA")`, `.getpixel((x, y))`, `.tobytes()`,
`.save("x.png")` (other formats via Pillow), `.to_pil()`, `.show()`. `Image.open(path)` reads PNG natively and
other formats through Pillow.

#### `fx.reset()`
Clear all classes and steps (tests, notebooks).

### Errors
All user-facing problems raise `fx.FelixError`.

---

## CLI

`felix` (or `python -m felix`). Output is a minimal monochrome ("noir") panel built on rich, redrawn in place on a terminal (colour unless
`NO_COLOR` is set); piped or on CI it prints just the final frame.

```
felix run SCRIPT [args...]            run a pipeline script
felix pack FOLDER [-o OUT.felix]      folder -> single compressed .felix
felix unpack FILE.felix [-o FOLDER]   .felix -> folder
felix info PATH                       summary + per-class box counts (folder or .felix)
felix partition SRC (--parts N | ...)  split a dataset into several smaller datasets
felix convert coco SRC -o OUT         Roboflow/COCO -> Felix
felix --version
```

All writing commands refuse to overwrite existing output unless `--overwrite` is given.

### Partitioning a dataset

`felix partition` splits one dataset into several smaller ones. Every piece is a complete dataset (images,
annotations, `metadata.yml`), so it can be loaded, packed, inspected or partitioned again on its own.

```
felix partition data.felix --parts 4                     # 4 equal pieces: part1 … part4
felix partition data.felix --ratio 80,10,10 --names train,val,test
felix partition ./data --size 1000                       # pieces of at most 1000 pairs
felix partition data.felix --by-split                    # one piece per train / val / test split
```

Choose exactly one way to split:

| Option | Pieces |
|---|---|
| `--parts N` | N equal pieces (sizes differ by at most one pair) |
| `--ratio A,B,...` | proportional pieces; any positive numbers, they need not sum to 100 |
| `--size K` | as many pieces of K pairs as needed; the last one may be smaller |
| `--by-split` | one piece per non-empty split (`train`, `val`, `test`), using the dataset's own splits |

Other options:

| Option | Meaning |
|---|---|
| `-o, --output DIR` | folder that receives the pieces (default `<source>_parts`) |
| `--names A,B,...` | piece names, one per piece (default `part1`, `part2`, ...) |
| `--shuffle`, `--seed S` | shuffle before splitting, reproducibly (default: keep the dataset's order) |
| `--pack` / `--folders` | write `.felix` files or folders (default: the same kind as the source) |
| `--overwrite` | replace existing pieces |

Every pair lands in exactly one piece. Files keep their names (`IMG42.png` stays `IMG42.png`), and each piece's
`metadata.yml` records where it came from (`partition: {source, name, index, of}`). With `--ratio`, leftover pairs
go to the pieces with the largest remainders.

### Converting a Roboflow COCO export

Export from Roboflow in **COCO** format, then:

```
felix convert coco ./my-export       -o ./my-felix-dataset
felix convert coco ./my-export.zip   -o ./my-dataset.felix
```

- `SRC` is the export folder or its `.zip`. Every `_annotations.coco.json` found
  (usually `train/`, `valid/`, `test/`) is converted; the split is stored per annotation
  (`split: train`).
- Images are copied untouched as `IMG<n>.<original ext>` (fast, no size growth). Add `--png`
  to re-encode everything to PNG instead - note JPEG→PNG makes files roughly 8-10x larger.
- Roboflow's placeholder parent category is dropped; each category's `supercategory`
  becomes its class set (`default` if none). Class ids are renumbered from 0.
- Boxes stay `xywh` pixels. Missing image files are skipped and reported.

---

## Dataset formats

### Folder

```
dataset/
├── images/IMG1.png ...        (converted datasets may keep .jpg)
├── annotations/IMG1.yml ...
└── metadata.yml
```

### `.felix`

A single file for the same tree - a standard ZIP archive (PNG/JPEG stored, YAML deflated),
so `unzip` and Python's `zipfile` can open it. `felix pack` / `felix unpack` convert between
the two; `fx.make(output="x.felix")` writes it directly (via a `.part` file, so a crash never
leaves a half-written dataset).

### `annotations/IMG1.yml`

```yaml
image: IMG1.png
width: 640
height: 640
step: step-1          # generated datasets
split: train          # converted datasets only
objects:
  - class: cat
    class_id: 0
    class_set: animals
    bbox: [316, 163, 62, 62]   # x, y, width, height in pixels, top-left origin
```

### `metadata.yml`

```yaml
felix_version: "0.1.0"
created: "2026-11-10T08:17:58Z"
pairs: 200
seed: 1                 # generated datasets: re-run with the same seed to reproduce
image_format: png
bbox_format: xywh
steps:
  - name: step-1
    size: [640, 640]
classes:
  - id: 0
    name: cat
    set: animals
```

Splits are not stored for generated datasets: `ds.train` / `ds.val` / `ds.test` derive them deterministically from
the seed (see the API page), and converted datasets keep their `split` field.

The YAML is plain and valid; any YAML parser reads it. Felix itself uses a tiny built-in
reader so it needs no dependencies.
