# Felix: complete documentation

*This file is the whole documentation in one page. It is generated from the individual pages; do not edit it by hand.*

## Contents

- [Install](#install)
- [Getting started](#getting-started)
- [Sources](#sources)
- [API reference](#api-reference)
- [Licences and provenance](#licences-and-provenance)
- [CLI](#cli)
- [Dataset formats](#dataset-formats)

---

A Python toolkit for building computer-vision datasets from real images: collect them from your files, videos,
cameras and online collections, transform them, label them with a model, and export one clean dataset, with every
image's source and licence recorded and the licence and citation files written for you.

```python
import felix as fx
import workflow as wf

fx.load_class("./classes.yml")

fx.acquire([
    fx.sources.local("./raw/leaves"),
    fx.sources.urls("./leaf_urls.txt"),
    fx.sources.video("./greenhouse.mp4", sampling=fx.sampling.auto()),
    fx.sources.wikimedia(query="plant leaves", limit=1000),
])

fx.transform.center_crop(size=640)

model = wf.load("./models/leaf-detector.nbm")
fx.annotate(model=model)

fx.export("./datasets/leaf-center.felix")
```

No pipeline objects, no `run()`: each line adds a step, and `fx.export` runs them in order. Images stream through,
so `fx.export(..., pairs=5000)` stops after 5,000 pairs; without `pairs` it keeps going until the sources run out or
you stop it (Ctrl+C), and the dataset written so far is closed properly.

## What is in it

- **Sources** (`fx.sources`), 20 of them: your files (folders, single images, archives, existing COCO / YOLO /
  Felix datasets), video files, cameras and streams, web URLs, Wikimedia Commons, Openverse, Flickr, Unsplash, Pexels,
  iNaturalist, GBIF, NASA, the Art Institute of Chicago, the Smithsonian, Roboflow, Open Images, COCO 2017,
  Hugging Face, Kaggle and Mapillary. Each uses its provider's real API and keeps what the provider says about
  licence and attribution.
- **Video sampling** (`fx.sampling`): `auto()` (only when the scene changes), `interval(seconds=)`, `random(count=)`.
- **Transforms** (`fx.transform`): `center_crop`, `letterbox`, `consequential` (object-focused crops plus negative
  samples), `dedupe`, `map_classes`, `filter_classes`, `filter_images`, `blur_faces`, and `custom` for your own.
  Boxes and masks move with the pixels.
- **Annotation** (`fx.annotate`): with your own Workflow model, or the built-in `fx.models.yolo26n`, `fx.models.yoloe`
  (open vocabulary, prompted from classes.yml; both bundled, so they work offline) and `fx.models.sam` (masks for boxes). Images the model fails on are
  never mistaken for images without objects.
- **Export** (`fx.export`): a `.felix` file, a folder, or a folder of `.sh_felix` shards, with LICENSE.txt,
  ATTRIBUTIONS.md and CITATION.cff, for an open (`license="CC-BY-4.0"` or `"auto"`) or closed dataset.
- **Dataset files**: `fx.load` (what Workflow trains from), `fx.split`, `fx.partition`, `fx.shard`, and the `felix`
  command (`info`, `pack`, `convert`, `library`, …).

---

## Install

### Anywhere (index)

```
pip install Pillow rich
pip install --no-deps --no-index --find-links BASE_URL/simple/felix/ felix
```

Works on Linux, macOS and Windows (Python 3.8+). Check with `felix --version`.

Why the extra flags:

- **`--no-index`**: pip must not search PyPI. A *different, public* package called `felix` lives there and would be
  installed instead of this one. Never use `--extra-index-url` for Felix.
- **`--no-deps`** and the separate `pip install Pillow rich`: with `--no-index` pip can't fetch dependencies from this
  page, so install them first the normal way.

Upgrade by adding `--upgrade --no-cache-dir`; pin a version with `felix==1.1.0`.

### Colab / notebooks (direct link)

```
!pip install BASE_URL/files/felix-1.1.0-py3-none-any.whl
```

### Optional parts

| For | Install |
|---|---|
| video files, cameras, streams, face blurring | `pip install opencv-python` |
| built-in annotation models (`fx.models.yolo26n`, `yoloe`, `sam`) | `pip install ultralytics` (pulls in PyTorch; AGPL-3.0 licensed). YOLO26n and YOLOE-26s weights ship inside Felix; others download once |
| your own trained models | Workflow (`wf.load("model.nbm")`), same site |

Without them Felix says what is missing when you use the feature.

### Credentials and contact

Online sources that need keys read them from the call, environment variables or `~/.config/felix/credentials.yml`
(see [Sources](#credentials)). Set `FELIX_CONTACT` to an email or URL so providers can reach you, as
Wikimedia's policy asks.

### Caches

Large downloads (Open Images CSVs, COCO, Roboflow exports, model weights) are cached in `~/.cache/felix`
(`FELIX_CACHE` to move it). Working copies made during an export live in a temporary folder that is removed
afterwards.

---

## Getting started

A Felix script is five kinds of lines, run top to bottom: **classes → acquire → transform → annotate → export**.
Nothing happens until `fx.export`: the earlier lines describe the dataset, export builds it.

### 1. Classes

```yaml
# classes.yml
classes:
  - name: leaf
    prompt: a green plant leaf           # what an open-vocabulary model (fx.models.yoloe) looks for
    aliases: [Leaf, leaves]              # names that mean "leaf" in datasets you import or in a model's labels
  - name: flower
confidence: 0.3                          # minimum model confidence (optional; per class too)
```

```python
import felix as fx

fx.load_class("./classes.yml")
```

Imported boxes and model detections are mapped onto these classes; anything else is left out. Without a class file,
every class is kept and models detect everything they know.

### 2. Acquire

```python
fx.acquire([
    fx.sources.local("./raw/leaves"),                       # a folder of photos, as it is
    fx.sources.video("./greenhouse.mp4", sampling=fx.sampling.auto()),
    fx.sources.wikimedia(query="plant leaves", limit=500),
])
```

Any mix of [sources](#sources). Local folders need no preparation; existing datasets (COCO, YOLO, Felix) bring their
labels and splits along. Online sources download politely (one request at a time per provider, as their rules ask)
and keep each image's licence, author and page.

### 3. Transform

```python
fx.transform.center_crop(size=640)       # or letterbox(size=640), dedupe(), blur_faces(), …
```

Steps run in the order written. Boxes and masks move with the pixels.

### 4. Annotate

```python
import workflow as wf

fx.annotate(model=wf.load("./models/leaf-detector.nbm"))   # your own model
# fx.annotate(model=fx.models.yolo26n)                     # or a built-in one
```

Only images without labels are annotated (`replace=True` to redo all). An image the model fails on is marked failed
and left out of the dataset, not saved as "no objects".

### 5. Export

```python
fx.export("./datasets/leaves.felix")                       # until the sources run out
fx.export("./datasets/leaves.felix", pairs=2000)           # exactly 2,000 image/annotation pairs
fx.export("./datasets/leaves", license="CC-BY-4.0")        # a folder, released openly
fx.export("./datasets/leaves.sh_felix")                    # a folder of shards, for very large datasets
```

The dataset comes with LICENSE.txt, ATTRIBUTIONS.md (every image's licence and credit) and CITATION.cff. Workflow
trains from it directly: `model.train("./datasets/leaves.felix")`.

To start a new dataset in the same program, call `fx.reset()`.

---

## Sources

Pass any mix to `fx.acquire([...])`. Every image keeps a `source` record in its annotation: where it came from,
its licence, who to credit and when it was collected. `limit=None` (the default for online sources) means "as many as
the provider gives": with `fx.export(..., pairs=N)` the export simply stops asking once it has N pairs.

Online sources send a User-Agent naming Felix. Set `FELIX_CONTACT` (an email or URL) or `contact:` in
`~/.config/felix/credentials.yml` to include a way to reach you, which Wikimedia's policy asks for. Requests to one
provider go one at a time, spaced as its rules ask; different providers run in parallel.

### Credentials

Keys go in the call (`api_key=...`), in an environment variable, or in `~/.config/felix/credentials.yml`:

```yaml
contact: you@example.com
roboflow_api_key: ...
unsplash_access_key: ...
pexels_api_key: ...
flickr_api_key: ...
api_data_gov_key: ...        # Smithsonian
mapillary_token: ...
huggingface_token: hf_...    # only for gated / private datasets
```

| Source | Environment variable | Where to get it |
|---|---|---|
| roboflow | `ROBOFLOW_API_KEY` | Roboflow → Settings → API Keys |
| unsplash | `UNSPLASH_ACCESS_KEY` | unsplash.com/oauth/applications (Access Key) |
| pexels | `PEXELS_API_KEY` | pexels.com/api |
| flickr | `FLICKR_API_KEY` | flickr.com/services/api/keys |
| smithsonian | `API_DATA_GOV_KEY` | api.data.gov/signup (works without, on a shared demo key) |
| mapillary | `MAPILLARY_TOKEN` | mapillary.com/dashboard/developers (client token) |
| huggingface | `HF_TOKEN` | huggingface.co/settings/tokens |
| kaggle | `KAGGLE_USERNAME` + `KAGGLE_KEY` | Kaggle → Settings → API (or `~/.kaggle/kaggle.json`) |

### On this computer

#### `fx.sources.local(path, recursive=True, labels=None, license=None)`
- An image file, a folder of images (any nesting; `recursive=False` for the top level only) or an archive (`.zip`,
  `.tar`, `.tar.gz`, …, unpacked safely into a working folder).
- Existing datasets are recognised and bring their labels and splits: **Felix** (`.felix`, folders, `.sh_felix`
  shards), **COCO** (any `*.json` with `images` / `annotations` / `categories`, e.g. Roboflow's
  `_annotations.coco.json` in `train/ valid/ test/`), **YOLO** (`data.yaml` with `names` and `images/` + `labels/`).
- `labels="folders"`: a folder of class folders (`cat/`, `dog/`, …) becomes image labels (classification).
- Photos stored sideways by a camera (EXIF rotation) are turned upright.
- `license`: the images' licence if you know it; otherwise they are recorded as your own files.

#### `fx.sources.video(path, sampling=fx.sampling.auto())`
Frames of a video file (OpenCV: `pip install opencv-python`). Each frame records the video, frame number and time.
See [Sampling](#sampling).

#### `fx.sources.camera(device=0, sampling=fx.sampling.interval(seconds=1), count=None, duration=None)`
A webcam / USB camera (`device=0, 1, …`) or a stream URL (`"rtsp://…"`, `"http://…/video.mjpg"`). Runs until `count`
frames, `duration` seconds, the export's `pairs`, or Ctrl+C. `random` sampling is not available for live cameras.

#### Sampling
- `fx.sampling.auto(threshold=0.04, min_gap=0.5, max_gap=None, check_fps=4, limit=None)`: keep a frame when it differs
  from the last kept one by at least `threshold` (mean colour difference of small thumbnails), at least `min_gap`
  seconds apart. A still scene gives one frame, not hundreds. `max_gap` forces a frame every so often.
- `fx.sampling.interval(seconds=2, start=0, limit=None)`: one frame every `seconds`.
- `fx.sampling.random(count=1000, seed=42)`: `count` frames spread at random over the whole video, reproducibly.

### On the web

#### `fx.sources.urls(source, license=None, attribution=None)`
One URL, a list, or a text file with one URL per line (`#` comments allowed). Their licences are unknown unless you
say (`license=`, `attribution=`); export then reminds you to check your rights.

#### `fx.sources.wikimedia(query, limit=None, licenses=None, max_side=2048, min_side=256)`
Wikimedia Commons (MediaWiki API, File namespace, bitmap images). Records licence, author, credit and the file page,
and builds the attribution line. `licenses=["CC0", "Public domain", "CC BY 4.0"]` keeps only those (by the start of
Commons' short licence name). Downloads a scaled copy up to `max_side`, not 50-megapixel originals.

#### `fx.sources.openverse(query, limit=None, licenses=None, commercial=True, source=None)`
Openverse: openly licensed images from Flickr, Wikimedia and many museums, in one search (no key). `commercial=True`
keeps licences that allow commercial use; `licenses=["cc0", "by", "by-sa"]` to choose; `source="flickr"` for one
provider. Openverse supplies the attribution text.

#### `fx.sources.flickr(query, limit=None, licenses=(4, 5, 7, 8, 9, 10, 11, 12), api_key=None)`
Flickr search filtered by licence id (default: CC BY, CC BY-SA, CC0, public domain and "no known copyright", i.e.
commercial use and changes allowed). Flickr's API is for non-commercial use unless you arrange otherwise with
Flickr, and returns at most 4,000 results per search.

#### `fx.sources.unsplash(query, limit=None, size="regular", orientation=None, api_key=None)`
Unsplash (Unsplash License). As Unsplash's API guidelines require, each download is reported to Unsplash. Demo
apps are limited to 50 requests an hour.

#### `fx.sources.pexels(query, limit=None, size="large2x", orientation=None, api_key=None)`
Pexels (Pexels License; 200 requests an hour, so Felix waits ~18 s between result pages). The "Photo by … on Pexels"
credit is recorded.

Unsplash and Pexels photos are free to use, but both licences forbid compiling their photos into a similar or
competing service; export flags them when you release a dataset openly.


### Open collections

#### `fx.sources.inaturalist(taxon, limit=None, licenses=("cc0", "cc-by", "cc-by-sa"), size="large", quality="research", label=False, place_id=None)`
Research-grade iNaturalist observation photos for a taxon (scientific or common name), with each photo's own licence
and attribution. `label=True` labels each image with its species (classification). Pages stop at 10,000 results.

#### `fx.sources.gbif(scientific_name=None, query=None, limit=None, licenses=("CC0_1_0", "CC_BY_4_0"), country=None)`
Occurrence images from GBIF, with the publishing institution's licence and rights holder.

#### `fx.sources.nasa(query, limit=None, size="large")`
The NASA Image and Video Library. NASA imagery is generally not copyrighted in the US, but some items are third-party
or show people and logos: check NASA's media guidelines.

#### `fx.sources.artic(query, limit=None)` · `fx.sources.smithsonian(query, limit=None, api_key=None)`
Public-domain (CC0) images from the Art Institute of Chicago (843 px IIIF images, as the museum asks) and the
Smithsonian's Open Access.

Not included, on purpose: **Pixabay** (its terms forbid automated collection and machine-learning use without
permission), **The Met** and the **Library of Congress** (their servers only serve browsers: automated requests get a
bot check, and Felix does not work around those).

### Dataset hubs

#### `fx.sources.roboflow(project, version, workspace=None, api_key=None)`
A Roboflow dataset version (your workspace or Universe: `project="workspace/project"`), downloaded as a COCO export
and imported with boxes and splits. Cached in `~/.cache/felix/roboflow`. A Universe project without a licence means
all rights reserved.

#### `fx.sources.open_images(classes, split="validation", limit=None, seed=0, group_of=True)`
Open Images V7 images containing any of `classes` (display names: "Car", "Person", …), with all their boxes of those
classes. The annotation CSVs are cached in `~/.cache/felix/open_images` (validation ~40 MB, test ~120 MB, train
several GB). Each image keeps its own licence, author and Flickr page; annotations are CC BY 4.0 (Google). Images that
Open Images lists as rotated are skipped, as their boxes may not line up.

#### `fx.sources.coco(split="val2017", limit=None)`
COCO 2017 from the official files (val2017: 5k images, ~1 GB with annotations; train2017: 118k images, ~18 GB),
cached in `~/.cache/felix/coco`. With a class file only images containing those classes are kept. Each image keeps
its Flickr licence from COCO's metadata.

#### `fx.sources.huggingface(dataset, split="train", config=None, limit=None, label_column=None, bbox_format="xywh", token=None)`
A Hugging Face image dataset through the Dataset Viewer API (only the rows you take are downloaded). A class-label
column becomes the image label; an `objects` column with `bbox` and `category` gives boxes. The dataset card's licence
is recorded.

#### `fx.sources.kaggle(dataset, labels=None)`
A Kaggle dataset (`"owner/name"`), downloaded and imported like `local` (format detected automatically).

#### `fx.sources.mapillary(bbox, limit=None, access_token=None)`
Street-level images inside a map area smaller than 0.01° on each side (Mapillary's limit), CC BY-SA 4.0.

### Your own source

Subclass `fx.sources.Source` with `collect(self, collector, progress)` and call
`collector.add(path=... or data=..., name=..., source={...}, objects=None, label=None)` for each image.

---

## API reference

`import felix as fx`. The calls build a dataset step by step; `fx.export` runs them in the order they were made.

### `fx.load_class(path)` → list of class names
Load the target classes from YAML:

```yaml
classes:
  - leaf                                # just a name
  - name: fruit
    prompt: a ripe tomato on the vine   # for open-vocabulary models (default: the name)
    aliases: [tomato, Tomato]           # names that mean this class in imported data / model labels
    confidence: 0.4                     # optional minimum model confidence for this class
confidence: 0.25                        # optional default
```

A bare list (`classes: [leaf, fruit]` or a top-level list) works too. Class ids follow the file's order. Imported boxes
and detections are mapped onto these names (case-insensitive, through aliases); others are left out. Without a class
file, every class is kept and models detect all they know.

### `fx.acquire(sources, blur_faces=False)`
Add sources (a list of [`fx.sources`](#sources); call again to add more). Nothing downloads yet. `blur_faces=True`
puts `fx.transform.blur_faces()` first, so every image is blurred before any other step.

### `fx.sampling`
`auto(threshold=0.04, min_gap=0.5, max_gap=None, check_fps=4, limit=None)`, `interval(seconds, start=0, limit=None)`,
`random(count, seed=0)`: which video frames to keep (see [Sources](#sampling)).

### `fx.transform`
Each call adds a step. Boxes and segmentation polygons move with the pixels; boxes keeping less than
`min_visibility` of their area after a crop are dropped.

| Step | What it does |
|---|---|
| `center_crop(size=640, min_visibility=0.3)` | the central region at the target shape (`640` or `(w, h)`), resized |
| `letterbox(size=640, fill=(114, 114, 114))` | the whole picture, resized to fit and padded; nothing is cut |
| `consequential(size=640, negative_rate=0.15, margin=0.25, min_visibility=0.3, overlap=0.6, seed=0)` | square crops around the annotated objects plus negative crops (see below). Must follow annotation; final for geometry |
| `dedupe(threshold=4)` | drops near-duplicates (64-bit difference hash; `threshold` bits may differ) |
| `map_classes({"Car": "vehicle"})` | renames classes; mapping to `None` removes them |
| `filter_classes(keep=[...])` / `(drop=[...])` | keeps only / removes boxes of some classes |
| `filter_images(min_width=0, min_height=0, max_aspect=None, require_objects=False, where=None)` | drops images |
| `blur_faces(strength=1.0, min_size=20)` | blurs faces found by OpenCV's face detectors (best effort) |
| `custom(fn, geometry=False, name=None)` | your own: `fn(image, objects) -> (image, objects)` or `None` to drop |

**consequential.** For each object not already well covered by a crop, a square around it with `margin` of its size on
each side, at least `size` pixels where the image allows (small objects are not blown up), kept inside the image
(grey padding only for objects larger than the short side). Every object inside a crop keeps its box. A crop
overlapping an earlier crop of the same image by more than `overlap` (IoU) is skipped. Negatives (squares touching no
object, from the same images or from annotated images without objects) are added so they make up `negative_rate` of
the output. Images whose annotation failed are left out, never used as negatives. Calling it before `fx.annotate`, or
a geometry step after it, raises an error at once.

### `fx.annotate(model, confidence=None, replace=False, batch_size=8)`
Boxes for every image without labels. `model`:
- a Workflow model: `wf.load("custom.nbm")` (detection gives boxes; a classification model gives image labels);
- a built-in model: `fx.models.yolo26n` (also `yolo26s/m/l/x`, `yolo26n_seg`), `fx.models.yoloe` (open vocabulary:
  prompted with classes.yml's prompts; without a class file, its prompt-free version), `fx.models.sam` (adds masks to
  existing boxes); call one to choose other weights, e.g. `fx.models.yolo26n("yolo26s.pt")`;
- a function `fn(image) -> [{"class", "bbox": [x, y, w, h] or "xyxy": [x1, y1, x2, y2], "confidence"}]`.

`confidence`: the minimum score; by default 0.25, and 0.1 for `fx.models.yoloe` when it is prompted by a class file
(open-vocabulary scores run lower: a "shoe" prompt scores ~0.16 where a closed-set "person" scores ~0.9). classes.yml
can set it per class.

**Built-in models** run through Ultralytics (`pip install ultralytics`). Bundled inside Felix, so they work offline:
`yolo26n.pt`, `yoloe-26s-seg.pt` (text prompts) and `yoloe-26s-seg-pf.pt` (prompt-free). Other weights
(`fx.models.yolo26s`, `fx.models.sam`, or `fx.models.yoloe("yoloe-26l-seg.pt")`) are downloaded once into
`~/.cache/felix/models`. YOLOE's prompts come from classes.yml (`prompt:`, else the class name) and its detections are
named after your classes; the first prompt downloads Ultralytics' text encoder (~254 MB) into the same cache. With
several prompts, one class's boxes never suppress another's. Prompts work best for whole objects ("shoe", "jacket",
"bus"); parts ("tyre", "bus window") are often missed even by the larger YOLOE. `device=` picks where a built-in
model runs: `fx.models.yoloe(device="cpu")`. Ultralytics models are AGPL-3.0 (or an Ultralytics Enterprise licence).

`replace=True` also re-annotates images that came with labels (otherwise model boxes are added only to unlabelled
images). Each box records its confidence and origin; each image records the model. An image the model fails on is
marked `failed`, reported, and left out of the export.

### `fx.export(path, pairs=None, *, license=None, title=None, authors=None, description=None, version=None, strict=False, shard_size=1000, overwrite=False, quiet=False)` → summary dict
Runs everything and writes the dataset:
- `path`: `name.felix` (one file), a folder, or `name.sh_felix` (the folder `name/` of shards, `shard_size` pairs each);
- `pairs`: stop after this many image/annotation pairs; `None` = until the sources run out, or Ctrl+C / `kill`,
  after which the dataset written so far is closed properly;
- `license`: `None` (closed: all rights reserved for the compilation), an open licence id (`"CC-BY-4.0"`,
  `"CC-BY-SA-4.0"`, `"CC0-1.0"`, `"CC-BY-NC-4.0"`, `"ODbL-1.0"`, …) or `"auto"` (the most permissive one every image
  allows); `strict=True` leaves out images whose licence does not fit; see [Licences](#licences-and-provenance);
- `title`, `authors`, `description`, `version`: for LICENSE.txt and CITATION.cff.

Images without a split get a stable 80 / 10 / 10 train / val / test split from their name; imported splits are kept.
Returns `{"pairs", "path", "license", "splits", "licence_issues", "failed", "stopped", "sources"}`.

### `fx.reset()`
Forget the class file, sources and steps (start another dataset in the same program).

### Reading and managing datasets
- `fx.load(path)` → `Dataset`: a `.felix` file, a folder, a folder of shards, a single `.sh_felix`, or class folders.
  `len(ds)`, `ds.stems`, `ds.classes`, `ds.metadata`, `ds.task`, `ds.splits`, `ds.train` / `ds.val` / `ds.test`,
  `ds.annotation(stem)`, `ds.image(stem)` (a Pillow image), `ds.image_bytes(stem)`; shards also `ds.shards`,
  `ds.shard_of`, `ds.load_shard(i)` / `ds.release_shard(i)`.
- `fx.split(source, fraction=None, output=None, *, size=None, seed=0)`: a smaller copy (same share of every split and
  class).
- `fx.partition(source, output=None, parts=None, ratio=None, size=None, by_split=False, …)`: several smaller datasets.
- `fx.shard(source, output=None, *, size=None, shards=None)`: a folder of `.sh_felix` shards.

### Errors
Every problem raises `fx.FelixError` with a message saying what to change.

---

## Licences and provenance

Publicly visible images are not automatically free to reuse. Felix records what each provider says about an image,
checks it against how you release the dataset, and writes the files that go with it. This is guidance from the
licence names providers report, **not legal advice**: check the terms before publishing or using a dataset
commercially.

### What every image records

Its annotation file holds a `source` block, for example:

```yaml
source:
  type: wikimedia
  title: "File:Sawtooth Oak Leaves.jpg"
  url: https://upload.wikimedia.org/…/1024px-….jpg
  page: https://commons.wikimedia.org/wiki/File:…
  license: CC BY-SA 3.0
  license_url: https://creativecommons.org/licenses/by-sa/3.0
  author: Derek Ramsey
  attribution: "Sawtooth Oak Leaves.jpg by Derek Ramsey, CC BY-SA 3.0, via Wikimedia Commons (https://…)"
  query: plant leaves
  acquired: "2026-11-21T09:15:02Z"
```

Depending on the source: original URL and file name, dataset id, licence and its URL, author and their page,
attribution text, the provider's usage terms, acquisition time, and for video frames the video file, frame number and
time. Transforms and annotation add `history`; face blurring adds `privacy`.

### Open or closed

```python
fx.export("leaves.felix")                          # closed (default)
fx.export("leaves.felix", license="CC-BY-4.0")     # open, under this licence
fx.export("leaves.felix", license="auto")          # open, under the most permissive licence every image allows
fx.export("leaves.felix", license="CC-BY-4.0", strict=True)   # leave out images that do not fit
```

- **Closed**: the compilation and annotations are all rights reserved (LICENSE.txt says so, naming `authors` as the
  owner). Third-party images keep their licences and are still credited. Felix flags images with unknown licences,
  all-rights-reserved images, non-commercial images (in case you use the data commercially) and no-derivatives
  images that were cropped.
- **Open**: everything above, plus conflicts with the chosen licence: non-commercial images under a licence that allows
  commercial use, share-alike images under a licence without share-alike, attribution-required images under CC0, your
  own files (confirm you hold the rights and people's consent), and photos whose licence does not clearly allow
  redistribution in a dataset (Unsplash, Pexels). `"auto"` picks CC0 if every image is free of conditions, CC BY 4.0
  if some need credit, CC BY-SA 4.0 with share-alike images, and the NC variants if any image is non-commercial.

Problems never stop the export (unless `strict=True` leaves those images out). They are listed in ATTRIBUTIONS.md
(marked ⚠) and counted in `metadata.yml` under `license`.

### The files

| File | Contents |
|---|---|
| `LICENSE.txt` | the dataset's licence (or the closed notice), that images keep their own licences, and a count per image licence |
| `ATTRIBUTIONS.md` | every image: file, source, licence (with any ⚠ issue), author / credit, link |
| `CITATION.cff` | how to cite the dataset (title, authors, date, licence), plus the datasets and collections it draws on (Open Images, COCO, Wikimedia Commons, iNaturalist, …) |

They sit next to the data: inside a `.felix` file, at the top of a dataset folder, or beside the shards.

### Faces

`fx.transform.blur_faces()` (or `fx.acquire(..., blur_faces=True)`) blurs faces found by OpenCV's frontal and
profile face detectors, and records how many in each image. It is best effort: small, turned, partly hidden or unusual
faces can be missed. Do not rely on it alone for sensitive data.

---

## CLI

`felix` (or `python -m felix`). Output is a minimal monochrome ("noir") panel built on rich, redrawn in place on a terminal (colour unless
`NO_COLOR` is set); piped or on CI it prints just the final frame.

```
felix run SCRIPT [args...]            run a Felix script (load_class, acquire, transform, annotate, export)
felix pack FOLDER [-o OUT.felix]      folder -> single compressed .felix
felix unpack FILE.felix [-o FOLDER]   .felix -> folder
felix info PATH                       summary + per-class box counts (or images per class and split for class folders)
felix partition SRC (--parts N | ...)  split a dataset into several smaller datasets
felix shard SRC [--size N|--shards N]  a folder of .sh_felix shards, for very large datasets
felix split SRC 0.5 [-o OUT]          a smaller copy (here half) for quick experiments; SRC is unchanged
                                      (classification: every class keeps its share in every split)
felix convert SRC [-o OUT]            any dataset or image folder (COCO, YOLO, class folders, images) -> Felix
felix library [SEARCH]                browse every function and command; type to search
felix --version
```

All writing commands refuse to overwrite existing output unless `--overwrite` is given.


### Browsing the library: `felix library`

```
felix library                 # every function and command, grouped; ↑↓ to move
felix library split           # start with a search
```

A full-screen browser in the same look as the rest of the CLI. On the left, everything you can use, grouped,
with its arguments under each name; type anywhere to search (every word must match; name matches come first). On
the right, the whole entry: what it does, the toddler-words explanation, why you'd use it, examples, the catch,
and the exact signature from the code.

| Key | Does |
|---|---|
| ↑ ↓ | move through the list (or scroll the entry, after Tab) |
| Tab / → / Enter | read the entry: ↑ ↓ then scroll it; Shift+Tab / ← goes back |
| PgUp / PgDn | scroll the entry |
| Home / End | first / last entry |
| typing, Backspace | search |
| Esc | back to the list, then clear the search, then quit (Ctrl+C quits at once) |

It needs a terminal (not a notebook or a pipe). The content is the same as the docs, shipped inside the package,
so it works offline and always matches the installed version.

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

### A smaller copy for experiments: `felix split`

```
felix split voc.felix 0.5                 # → voc_split_50.felix, half the pairs; voc.felix is unchanged
felix split voc.felix 0.1 --seed 2        # → voc_split_10.felix, another random tenth
felix split voc.felix --size 1000         # → voc_split_1000.felix, exactly 1000 pairs
felix split ./data 0.25 -o small          # a folder dataset → folder copy "small"
```

The copy is a complete dataset like any other. Every train / val / test split keeps the same share (half of train,
half of val, half of test), and every pair stays in the split it was in, so test pictures never become training
pictures. Which pairs are picked is random but repeatable: the same `--seed` (default 0) gives the same subset.
A split that had pairs keeps at least one. `metadata.yml` records where it came from
(`split_from: {source, pairs, seed, fraction}`).

| Option | Meaning |
|---|---|
| `fraction` | share to keep, above 0 and at most 1 (`0.5` = half; `50` is refused with a hint) |
| `--size N` | keep N pairs instead of a fraction |
| `-o OUT` | output path (default `<name>_split_<percent>.felix`, or a folder for a folder source) |
| `--seed S` | which random subset |
| `--overwrite` | replace an existing output (never the source) |

### Shards for very large datasets: `felix shard`

```
felix shard voc.felix                    # → voc_shards/ with 1000 pairs per shard
felix shard voc.felix --size 5000        # 5000 pairs per shard
felix shard ./imagenet --shards 256 -o imagenet_shards
felix pack voc_shards -o voc.felix       # back to one file
```

Every shard is a complete `.sh_felix` file; the folder loads as one dataset (`fx.load("voc_shards")`), with the same
pairs and splits as the source, which stays unchanged. Workflow trains from it one shard at a time. `felix info`,
`split` and `partition` work on shard folders too.

### Converting a dataset: `felix convert`

```
felix convert ./roboflow-export                    # COCO export (folder or .zip) → roboflow-export.felix
felix convert ./yolo-dataset -o data.felix         # YOLO (data.yaml + images/ + labels/)
felix convert ./animals --labels folders           # class folders → a classification dataset
felix convert ./photos -o photos.sh_felix          # plain images (no labels yet), as shards
felix convert ./export.zip --license CC-BY-4.0     # released openly, with licence files
```

The same as `fx.acquire([fx.sources.local(SRC)])` followed by `fx.export(OUT)`: labels and splits come along, the
files are copied as they are, and LICENSE.txt, ATTRIBUTIONS.md and CITATION.cff are written.

---

## Dataset formats

### Folder

```
dataset/
├── images/IMG1.png ...        (JPEG, PNG or WebP, kept as they were)
├── annotations/IMG1.yml ...
└── metadata.yml
```

### Shards: a folder of `.sh_felix` files

For datasets too big for one file, a folder holds many shards:

```
voc_shards/
├── voc_shards-00000.sh_felix
├── voc_shards-00001.sh_felix
└── …
```

Each shard is a complete dataset of its own (a zip like `.felix`: `images/`, `annotations/`, its own `metadata.yml`
with `shard: {index, of, dataset}`), plus `index.tsv` (stem, split and class id per line, so a folder of thousands of
shards loads without reading every annotation). `fx.load("voc_shards/")` reads the folder as one dataset; a single
shard also loads on its own. Shards open on demand, at most 64 at a time, and reads are thread-safe.

Make them with `felix shard` / `fx.shard(...)` from any dataset, or straight from generation with
`fx.export("name.sh_felix")`. Splits are the same as the unsharded dataset's. `felix pack voc_shards/ -o voc.felix`
puts them back into one file. All shards in a folder must come from the same dataset (same classes, no file in two
shards); Felix says which shard breaks that.

### Class folders (classification)

An ImageNet-style folder is a dataset as it is: one folder per class with its images, optionally inside `train/`,
`val/` and `test/` (also `valid`, `validation`, `training`, `testing`).

```
animals/                      animals/
├── cat/001.jpg …             ├── train/cat/… dog/…
├── dog/…                     ├── val/cat/… dog/…
└── bird/…                    └── test/cat/… dog/…
```

`fx.load("animals/")` reads it without writing anything next to the images: the class names are the folder names (in
natural order, ids follow), each image is one pair, and its annotation is made up on the fly:

```yaml
image: cat/001.jpg
split: train
class: cat
class_id: 1
```

with `metadata.yml` holding `task: classification`, the classes and the split counts. Without split folders every class
is split 80 / 10 / 10 on its own (stratified, seed 0, the same every time). PNG, JPEG, BMP, WebP, GIF and TIFF are
read; hidden files and folders (`.` or `_` first) are skipped. `felix pack` turns it into a regular `.felix` file with
those annotations, and `felix info`, `split` and `partition` work on both.

### `.felix`

A single file for the same tree - a standard ZIP archive (PNG/JPEG stored, YAML deflated),
so `unzip` and Python's `zipfile` can open it. `felix pack` / `felix unpack` convert between
the two; `fx.export("x.felix")` writes it directly (via a `.part` file, so a crash never
leaves a half-written dataset).

### `annotations/<name>.yml`

```yaml
image: leaf_0001.jpg
width: 640
height: 640
split: train                     # train / val / test
objects:
  - class: leaf
    class_id: 0
    bbox: [316, 163, 62, 62]     # x, y, width, height in pixels, top-left origin
    confidence: 0.91             # boxes from a model
    segmentation: [[316, 163, 378, 170, …]]   # optional polygons (x, y, x, y, …)
    origin: model                # dataset (came with the data) or model
annotation:
  status: model                  # dataset, model, negative (deliberately empty) or none (never annotated)
  model: leaf-detector
source:                          # where the image came from: see Licences and provenance
  type: wikimedia
  license: CC BY-SA 4.0
  author: …
  page: https://commons.wikimedia.org/wiki/File:…
history: [center_crop 640x640, annotated by leaf-detector]
```

Classification datasets have `class` and `class_id` at the top instead of `objects`.

### `metadata.yml`

```yaml
felix_version: "1.0.0"
created: "2026-11-21T08:17:58Z"
title: leaves
pairs: 4210
task: detection                  # or classification
bbox_format: xywh
classes:
  - {id: 0, name: leaf, prompt: a green plant leaf}
splits: {train: 3368, val: 421, test: 421}
sources: [{type: local, label: …, images: 812, skipped: 3}, {type: wikimedia, query: plant leaves, …}]
steps: [{step: center_crop, size: [640, 640]}, {step: annotate, model: {…}, confidence: 0.25}]
license: {dataset: CC-BY-4.0, access: open, images: {CC BY-SA 4.0: 230, …}, issues: 12, files: [LICENSE.txt, ATTRIBUTIONS.md, CITATION.cff]}
excluded: {annotation_failed: 2, licence: 0}
```

Exported datasets also hold `LICENSE.txt`, `ATTRIBUTIONS.md` and `CITATION.cff` next to `metadata.yml`.
