# Felix: complete documentation

*This file is the whole documentation in one page. It is generated from the individual pages; do not edit it by hand.*

## Contents

- [Install](#install)
- [Getting started](#getting-started)
- [Felix cookbook](#felix-cookbook)
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
  licence and attribution. Give a list of queries to run several searches in one source, or leave the query out and
  a search source takes whatever it has (then a ~4,600-subject vocabulary, from the RAM++ tag list). Downloads run in parallel
  over kept-alive connections (10–20× faster than 1.1), and downloaded images are cached for fast re-runs.
- **Video sampling** (`fx.sampling`): `auto()` (only when the scene changes), `interval(seconds=)`, `random(count=)`.
- **Transforms** (`fx.transform`): `center_crop`, `letterbox`, `consequential` (object-focused crops plus negative
  samples), `augment` (flips, rotation, scale, shifts, colour, blur, noise, cutout, JPEG damage), `mosaic`, `dedupe`,
  `map_classes`, `filter_classes`, `filter_images`, `blur_faces`, and `custom` for your own. Boxes and masks move with
  the pixels; augmented copies, mosaics and crops stay in their original's train / val / test split.
- **Annotation** (`fx.annotate`): with your own Workflow model, or the built-in `fx.models.yolo26n`, `fx.models.yoloe`
  (open vocabulary, prompted from classes.yml, several prompts per class allowed; both bundled, so they work offline) and `fx.models.sam` (masks for boxes). Images the model fails on are
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

Upgrade by adding `--upgrade --no-cache-dir`; pin a version with `felix==1.4.0`.

### Colab / notebooks (direct link)

```
!pip install BASE_URL/files/felix-1.4.0-py3-none-any.whl
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

**Toddler words:** write the shopping list: the things you want boxes for.

```yaml
# classes.yml
classes:
  - name: leaf
    prompt: a green plant leaf           # what an open-vocabulary model (fx.models.yoloe) looks for
    aliases: [Leaf, leaves]              # names that mean "leaf" in datasets you import or in a model's labels
  - name: flower
    prompt: [flower, blossom]            # several prompts for one class: any of them finds it
confidence: 0.3                          # minimum model confidence (optional; per class too)
```

```python
import felix as fx

fx.load_class("./classes.yml")
```

Imported boxes and model detections are mapped onto these classes; anything else is left out. Without a class file,
every class is kept and models detect everything they know.

### 2. Acquire

**Toddler words:** say where the pictures come from: your folder, a video, free picture libraries.

```python
fx.acquire([
    fx.sources.local("./raw/leaves"),                       # a folder of photos, as it is
    fx.sources.video("./greenhouse.mp4", sampling=fx.sampling.auto()),
    fx.sources.wikimedia(query="plant leaves", limit=500),
])
```

Any mix of [sources](#sources). Local folders need no preparation; existing datasets (COCO, YOLO, Felix) bring their
labels and splits along. Online sources download many pictures at once, ask each provider's API as politely as its
rules say, and keep each image's licence, author and page. Leave out the query to take whatever a source has.

### 3. Transform

**Toddler words:** tidy the pictures up: same size, no copies, faces blurred.

```python
fx.transform.center_crop(size=640)       # or letterbox(size=640), dedupe(), blur_faces(), …
```

Steps run in the order written. Boxes and masks move with the pixels.

### 4. Annotate

**Toddler words:** a helper draws the boxes on every picture that has none.

```python
import workflow as wf

fx.annotate(model=wf.load("./models/leaf-detector.nbm"))   # your own model
# fx.annotate(model=fx.models.yolo26n)                     # or a built-in one
```

Only images without labels are annotated (`replace=True` to redo all). An image the model fails on is marked failed
and left out of the dataset, not saved as "no objects".

### 5. Export

**Toddler words:** run everything and pack the finished dataset into a box, with a label saying what's inside and who to credit.

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

## Felix cookbook

Every Felix tool, explained like you're three, then *why* you'd reach for it and where it goes. After the tools come
**recipes**: whole datasets, start to finish, that you can copy and change.

How to read an entry:

- **Toddler words**: what it is.
- **Why you'd use it**: the reasoning, and where it usually sits.
- **Example**: the main arguments shown, with their defaults.
- **Watch out**: the one mistake people make (when there is one).

A Felix program is a list of steps, in order. Nothing downloads or changes until `fx.export` runs them:

```python
import felix as fx

fx.load_class("./classes.yml")          # 1. what we are looking for       (optional)
fx.acquire([...])                       # 2. where the pictures come from
fx.transform.center_crop(size=640)      # 3. change the pictures            (any number, in order)
fx.annotate(model=fx.models.yoloe)      # 4. draw the boxes                 (if the pictures have none)
fx.transform.augment(copies=2)          # 5. more changes, after the boxes  (optional)
fx.export("./datasets/leaves.felix")    # 6. run it all and write the dataset
```

---

### Part 1 · What to look for

#### `fx.load_class`

**Toddler words:** the shopping list. Before going out, write down what you're looking for: "leaf", "tomato", "white
tiger". Felix only keeps boxes for things on the list.

**Why you'd use it:** it decides the dataset's classes (and their ids, in file order). An open-vocabulary model such
as `fx.models.yoloe` is asked to find exactly these, by name or by `prompt`. `aliases` say which names in other
datasets or in a model's labels mean the same thing.

```python
fx.load_class("./classes.yml")
```
```yaml
classes:
  - leaf                                  # the name is also the prompt
  - tomato
  - name: white tiger
    prompt: a white bengal tiger          # optional: a more precise description for the model
  - name: dog
    prompt: [dog, puppy, a dog lying down]   # optional: several ways to say it; any of them finds a dog
    aliases: [White tiger, tiger_white]   # optional: other names that mean this class
    confidence: 0.3                       # optional: this class's own minimum score
confidence: 0.25                          # optional: the default minimum score
```

**Several prompts:** when one description misses things, give a few. The model looks for each, every box is named after
the class, and two prompts boxing the same dog become one box.

**Watch out:** without a class file, everything is kept: a COCO import brings all 80 classes, and a model labels
everything it knows. Good for "collect anything", noisy for a focused detector. A prompt can belong to one class only.

---

### Part 2 · Where the pictures come from

#### `fx.acquire`

**Toddler words:** the shopping trip. Point at the places pictures come from; Felix goes there when the dataset is
built.

**Why you'd use it:** real datasets mix sources: your own photos, a video, a museum, a public dataset. Every picture
remembers where it came from, its licence and who to credit. Call `fx.acquire` again to add more sources.

```python
fx.acquire([
    fx.sources.local("./raw/leaves"),
    fx.sources.wikimedia(query="plant leaves", limit=500),
], blur_faces=False)
```

`blur_faces=True` blurs every face before anything else happens (see `blur_faces` below).

#### `fx.sources.local`

**Toddler words:** the pictures already in your drawer.

**Why you'd use it:** your own photos, a zip someone sent you, or a dataset you downloaded. COCO, YOLO and Felix
datasets are recognised by their content and bring their boxes and splits. A folder of class folders (`cat/`,
`dog/`) becomes a classification dataset.

```python
fx.sources.local("./raw/leaves", recursive=True, labels=None, license=None)
fx.sources.local("./downloads/roboflow-export.zip")          # COCO inside: boxes come along
fx.sources.local("./pets", labels="folders")                  # cat/, dog/ → image labels
fx.sources.local("./bought-photos", license="CC BY 4.0")      # say the licence when you know it
```

**Watch out:** with no `license`, Felix records the files as your own. Say so if they are not.

#### `fx.sources.video`

**Toddler words:** a flip book. A video is thousands of pictures in a row; Felix keeps some pages, not all.

**Why you'd use it:** your own footage (a greenhouse walk, a dashcam) is the closest thing to what the model will see
later. Each frame remembers its video, frame number and time.

```python
fx.sources.video("./greenhouse.mp4", sampling=fx.sampling.auto())
fx.sources.video("./drive.mp4", sampling=fx.sampling.interval(seconds=2))
```

**Watch out:** video needs OpenCV: `pip install "felix[video]"`.

#### `fx.sources.camera`

**Toddler words:** a camera that keeps taking pictures while you watch, and Felix keeps the good ones.

**Why you'd use it:** collect data live from a webcam, a USB camera or a network stream, right where the model will
be used.

```python
fx.sources.camera(device=0, sampling=fx.sampling.interval(seconds=1), count=500, duration=None)
fx.sources.camera("rtsp://192.168.1.20/stream", sampling=fx.sampling.auto(), duration=600)
```

**Watch out:** it runs until `count`, `duration`, the export's `pairs`, or Ctrl+C. `fx.sampling.random` can't be used
(there is no "whole video" to pick from).

#### `fx.sampling.auto`

**Toddler words:** only turn the page when something changed. A still scene gives one picture, not a hundred copies.

**Why you'd use it:** the best default. A minute of video is 1,800 frames, nearly all the same; `auto` keeps more
frames when things move and fewer when nothing happens.

```python
fx.sampling.auto(threshold=0.04, min_gap=0.5, max_gap=None, check_fps=4, limit=None)
```

**Watch out:** if it keeps too few frames, lower `threshold` (0.02); too many, raise it. `max_gap=10` forces a frame
at least every 10 seconds.

#### `fx.sampling.interval`

**Toddler words:** take a picture every few seconds, like a clock.

**Why you'd use it:** predictable spacing, e.g. one frame a second from a camera, or every 2 seconds of a drive.

```python
fx.sampling.interval(seconds=2, start=0, limit=None)
```

#### `fx.sampling.random`

**Toddler words:** close your eyes and point at a thousand pages of the flip book.

**Why you'd use it:** frames spread over the whole video, the same ones every time (`seed`). Good for long videos
where you want a fixed number of frames.

```python
fx.sampling.random(count=1000, seed=42)
```

**Watch out:** not available for live cameras (there is no "whole video" to pick from).

#### `fx.sources.urls`

**Toddler words:** a list of addresses; Felix goes to each one and brings back the picture.

**Why you'd use it:** you already have links to the pictures you want (from a spreadsheet, a scrape you're allowed to
do, a partner's list).

```python
fx.sources.urls("./leaf_urls.txt", license=None, attribution=None)
fx.sources.urls(["https://example.org/a.jpg", "https://example.org/b.jpg"], license="CC0-1.0")
```

**Watch out:** "public on the internet" is not "free to use". These licences are unknown unless you say; export
reminds you.

#### Choosing an online source

**Toddler words:** libraries of free pictures. Ask for "oak trees" and they hand you oak trees, with a note saying
who took each one and how you may use it.

**Why you'd use it:** lots of real, varied pictures with clear licences, for free. Pick by what you need:

| Need | Source |
|---|---|
| anything, any topic | `wikimedia`, `openverse`, `flickr` |
| plants and animals | `inaturalist` (by taxon), `gbif` |
| space, science | `nasa` |
| art, objects, history | `artic`, `smithsonian` |
| high-quality stock photos | `unsplash`, `pexels` (keys needed) |
| street scenes | `mapillary` |
| already labelled | `open_images`, `coco`, `roboflow`, `huggingface`, `kaggle` |

**Several searches in one source:** give a list. The searches run side by side, share the `limit` equally, and a
picture found by two of them is taken once. Each picture records which search found it.

```python
fx.sources.wikimedia(query=["oak leaf", "maple leaf", "beech leaf"], limit=900)    # 300 of each
fx.sources.inaturalist(taxon=["Quercus", "Acer", "Fagus"], limit=1500, label=True)
fx.sources.gbif(scientific_name=["Pica pica", "Corvus corax"], limit=400)
```

**No target: take whatever comes.** Leave the query out and a source gives you whatever it has: its own "anything"
feed (random Commons files, the newest Unsplash photos, any species on iNaturalist). A source without such a feed, or
once its feed runs dry, works through a built-in vocabulary of about 4,600 subjects: the tag list of Recognize
Anything (RAM++), from "abacus" to "wind turbine", plus everyday subjects. It asks for them in shuffled order, at most
25 pictures per subject, so no subject takes over.

```python
fx.acquire([fx.sources.wikimedia(), fx.sources.openverse(), fx.sources.artic()])
fx.sources.openverse(limit=500).using(vocabulary="general")                          # ~150 everyday subjects only
fx.sources.openverse(limit=500).using(vocabulary=["oak", "maple", "pine"], per_term=50)   # your own subjects
```

**Watch out:** a slow or paused provider never holds up the others: it waits in the background while the rest keep
going, and gives up quietly when the export is done.

#### `fx.sources.wikimedia`

**Toddler words:** the picture library behind Wikipedia: millions of photos anyone may reuse, each with a note saying
who made it.

**Why you'd use it:** almost every subject, mostly real photos, clear licences. Without a query: random files.

```python
fx.sources.wikimedia(query="plant leaves", limit=500, licenses=["CC0", "CC BY"], max_side=2048, min_side=256)
fx.sources.wikimedia(limit=1000)                         # random Commons pictures
```

**Watch out:** set `FELIX_CONTACT` (your email or a URL) first; Wikimedia refuses programs that don't say how to reach
their operator.

#### `fx.sources.openverse`

**Toddler words:** one search box over many free picture libraries at once (Flickr, Wikimedia, museums).

**Why you'd use it:** the widest single search for openly licensed photos, no key needed, attribution text included.

```python
fx.sources.openverse(query="tomato plant", limit=500, licenses=None, commercial=True, source=None)
fx.sources.openverse(query="bicycle", source="flickr")    # only one of its providers
```

**Watch out:** `commercial=True` (the default) keeps only licences that allow commercial use.

#### `fx.sources.flickr`

**Toddler words:** the giant photo-sharing site, but only the photos their owners said everyone may reuse.

**Why you'd use it:** huge variety of everyday photos. Without a query: recent uploads with reusable licences.

```python
fx.sources.flickr(query="street market", limit=1000, licenses=(4, 5, 7, 8, 9, 10, 11, 12), api_key=None)
```

**Watch out:** needs a free API key (`FLICKR_API_KEY`); a search returns at most 4,000 results; the API is for
non-commercial use unless you arrange otherwise with Flickr.

#### `fx.sources.unsplash`

**Toddler words:** a shop of beautiful, free professional photos.

**Why you'd use it:** sharp, well-lit pictures. Without a query: the newest photos.

```python
fx.sources.unsplash(query="coffee cup", limit=200, size="regular", orientation=None, api_key=None)
```

**Watch out:** needs a key (`UNSPLASH_ACCESS_KEY`); demo keys allow 50 requests an hour. Its licence forbids building
a competing photo service, so export flags Unsplash photos in openly released datasets.

#### `fx.sources.pexels`

**Toddler words:** another shop of free professional photos.

**Why you'd use it:** like Unsplash, more stock-style subjects. Without a query: Pexels' curated photos.

```python
fx.sources.pexels(query="kitchen", limit=200, size="large2x", orientation=None, api_key=None)
```

**Watch out:** needs a key (`PEXELS_API_KEY`); 200 requests an hour, so pages come about 18 seconds apart. Same
licence caveat as Unsplash.

#### `fx.sources.inaturalist`

**Toddler words:** nature lovers' photos of plants and animals, each checked by experts so the species name is right.

**Why you'd use it:** the best source for species: real photos in the wild, with the species name, which
`label=True` turns into the image's class.

```python
fx.sources.inaturalist(taxon="Quercus", limit=500, licenses=("cc0", "cc-by", "cc-by-sa"), size="large",
                       quality="research", label=False, place_id=None)
fx.sources.inaturalist(limit=1000)                       # any species
```

**Watch out:** each search stops at 10,000 results; use narrower taxa for more. A name means exactly that taxon
and everything under it ("Pinus": every pine; "red oak": *Quercus rubra*); a name iNaturalist doesn't know stops with
the closest matches.

#### `fx.sources.gbif`

**Toddler words:** the world's big list of "where was this plant or animal seen", from museums and scientists, with
photos.

**Why you'd use it:** museum specimens and field photos, including species iNaturalist has few of.

```python
fx.sources.gbif(scientific_name="Pica pica", query=None, limit=500, licenses=("CC0_1_0", "CC_BY_4_0"), country=None)
```

**Watch out:** many pictures are herbarium sheets or drawers of pinned insects: great for some tasks, odd for others.

#### `fx.sources.nasa`

**Toddler words:** NASA's photo album: rockets, planets, astronauts, Earth from space.

```python
fx.sources.nasa(query="mars rover", limit=200, size="large")
fx.sources.nasa(limit=500)                               # anything
```

**Watch out:** generally not copyrighted in the US, but some items are third-party or show people and logos.

#### `fx.sources.artic`

**Toddler words:** the Art Institute of Chicago's paintings and objects that belong to everyone (public domain).

```python
fx.sources.artic(query="still life", limit=300)
```

**Watch out:** its API sits behind a bot check that sometimes refuses programs for a while; Felix then says so.
Images are 843 pixels wide, as the museum asks.

#### `fx.sources.smithsonian`

**Toddler words:** the Smithsonian museums' free pictures: fossils, planes, stamps, art, animals.

```python
fx.sources.smithsonian(query="aircraft", limit=300, api_key=None)
```

**Watch out:** works without a key on a shared demo key; get your own (`API_DATA_GOV_KEY`) for anything sizeable.

#### `fx.sources.open_images`

**Toddler words:** Google's huge box of photos that already have boxes drawn on them, 600 kinds of things.

**Why you'd use it:** ready-labelled pictures. Without `classes`: your classes.yml's classes it has, or all 600.

```python
fx.sources.open_images(classes=["Car", "Person"], split="validation", limit=1000, seed=0, group_of=True)
```

**Watch out:** the `train` split's annotation tables are several GB (downloaded once); `validation` is ~40 MB.

#### `fx.sources.coco`

**Toddler words:** the most famous box of labelled photos: everyday scenes, 80 kinds of things.

```python
fx.sources.coco(split="val2017", limit=None)             # 5,000 images, ~1 GB
```

**Watch out:** `train2017` is 118,000 images (~18 GB). With a class file only images containing those classes come.

#### `fx.sources.roboflow`

**Toddler words:** a dataset you (or someone on Roboflow Universe) already made, fetched with its boxes.

```python
fx.sources.roboflow("my-workspace/leaves", version=3, workspace=None, api_key=None)
```

**Watch out:** needs `ROBOFLOW_API_KEY`. A Universe project without a licence means all rights reserved.

#### `fx.sources.huggingface`

**Toddler words:** a dataset from the Hugging Face hub; only the rows you take are downloaded.

```python
fx.sources.huggingface("cats_vs_dogs", split="train", config=None, limit=2000, label_column=None, bbox_format="xywh")
```

**Watch out:** gated or private datasets need `HF_TOKEN`.

#### `fx.sources.kaggle`

**Toddler words:** a dataset from Kaggle, downloaded and opened like a folder on your computer.

```python
fx.sources.kaggle("owner/dataset-name", labels=None)     # labels="folders" for class folders
```

**Watch out:** needs your Kaggle API key (`~/.kaggle/kaggle.json`). Check the dataset's licence on its Kaggle page.

#### `fx.sources.mapillary`

**Toddler words:** street pictures taken by people driving and walking around, pinned on a map.

```python
fx.sources.mapillary(bbox=(13.40, 52.51, 13.41, 52.52), limit=500, access_token=None)   # lon/lat box
```

**Watch out:** the map area must be smaller than 0.01° on each side; needs `MAPILLARY_TOKEN`. Images are CC BY-SA.

#### Credentials

**Toddler words:** some libraries want a membership card (a key) before they hand out pictures.

**Why you'd use it:** Unsplash, Pexels, Flickr, Roboflow, Mapillary and Kaggle need one; Smithsonian and Hugging
Face work better with one. Put keys in the call, an environment variable, or one file:

```yaml
# ~/.config/felix/credentials.yml
contact: you@example.com
unsplash_access_key: ...
pexels_api_key: ...
flickr_api_key: ...
```

**Watch out:** never put keys in a script you share; use the file or environment variables.

#### How fast does it go?

**Toddler words:** Felix sends many helpers at once, each fetching a picture, instead of one helper doing every trip.

Each source downloads up to 16 pictures at a time over reused connections, different sources run side by side, and
search APIs are still asked as politely as their rules say. A server that says "slow down" pauses only itself.
Downloaded pictures are kept in `~/.cache/felix/images`, so running the same export again (after changing a
transform or the model) is almost instant. See [Sources → Speed](#speed).

```python
fx.acquire([fx.sources.open_images(classes=["Car"], limit=2000)])
fx.export("./datasets/cars.felix")        # first run: downloads, 16 at a time
# change a transform, then run again: pictures come from ~/.cache/felix/images
```
```bash
FELIX_IMAGE_CACHE_GB=50 python make_cars.py   # a bigger download cache
FELIX_IMAGE_CACHE=0 python make_cars.py       # no cache at all
```

---

### Part 3 · Changing the pictures

#### `fx.transform`

**Toddler words:** the workbench. Each line puts one tool on the bench (crop, resize, copy, blur…), and every picture
goes past the tools in the order you wrote them.

**Why you'd use it:** raw pictures are all sizes and shapes; models want them tidy. The boxes always move with the
pixels, and a box that loses most of its area (`min_visibility`) is dropped.

```python
fx.transform.dedupe()                     # 1. no copies
fx.transform.center_crop(size=640)        # 2. same size
fx.annotate(model=fx.models.yoloe)        #    (boxes)
fx.transform.augment(copies=2)            # 3. more variety
```

**Watch out:** order matters. Crop *before* annotating (faster, and the model sees what the dataset keeps); augment
*after* (copies inherit the boxes); nothing that resizes or turns pixels after `consequential`.

#### `fx.transform.center_crop`

**Toddler words:** cut the edges off so every picture is the same size, keeping the middle.

**Why you'd use it:** shrink big photos before annotating (faster, and the subject is usually in the middle). Boxes
cut off at the edge are trimmed, or dropped if most of them is gone.

```python
fx.transform.center_crop(size=640, min_visibility=0.3)
fx.transform.center_crop(size=(800, 600))
```

**Watch out:** things near the edge are lost. If they matter, use `letterbox`.

#### `fx.transform.letterbox`

**Toddler words:** shrink the whole picture to fit the frame and fill the empty strips with grey, like a widescreen
film on an old TV.

**Why you'd use it:** nothing is cut, so every object survives. Suits general detectors and wide street scenes.

```python
fx.transform.letterbox(size=640, fill=(114, 114, 114))
```

**Watch out:** tall or wide pictures get large grey bars, so their objects end up smaller.

#### `fx.transform.consequential`

**Toddler words:** zoom in on each toy and take a photo of it, plus a few photos of empty floor so the brain learns
what "nothing" looks like.

**Why you'd use it:** in a 4,000-pixel photo a leaf may be 40 pixels: resized to 640 for training it nearly vanishes.
Crops around each object keep them big enough to learn; `negative_rate` of the crops show no objects.

```python
fx.annotate(model=model)                                   # first: it needs the boxes
fx.transform.consequential(size=640, negative_rate=0.15, margin=0.25, min_visibility=0.3, overlap=0.6, seed=0)
```

**Watch out:** it must come after `fx.annotate`, and nothing that resizes, turns or shifts pixels may follow it
(`augment` with only flips and colour may). Felix stops you if the order is wrong.

#### `fx.transform.augment`

**Toddler words:** take each photo and make a few funny copies: flipped like in a mirror, a little turned, darker,
brighter, blurrier. It's still the same leaf, and the box moves with it.

**Why you'd use it:** a model that only ever saw leaves facing left, in sunshine, gets confused by a leaf facing
right on a cloudy day. Copies that vary the things that *shouldn't* matter teach it to ignore them. Most useful when
you have few pictures.

```python
fx.annotate(model=model)                     # first: label the real pictures, the copies inherit the labels
fx.transform.augment(
    copies=2,                # new versions per picture; the original stays (keep_original=True)
    flip=0.5,                # chance of a mirror image
    vflip=0.0,               # chance of upside down (aerial photos: yes; street scenes: no)
    rotate=10,               # up to ±10 degrees
    scale=0.2,               # 80 % to 120 % size
    translate=0.1,           # shifted up to 10 % of the size
    brightness=0.2, contrast=0.2, saturation=0.2, hue=5,
    blur=0.05, noise=0.0, grayscale=0.02, jpeg=0.0,
    cutout=0.0,              # chance of a grey patch (teaches "a partly hidden leaf is still a leaf")
    seed=0,
)
```

**Watch out:** only vary what really varies where the model will be used. Upside-down street scenes or purple
tomatoes teach it things that never happen. Copies always stay in their original's train / val / test pile, so the
final exam is never seen during learning.

#### `fx.transform.mosaic`

**Toddler words:** four photos glued into one, like a window with four panes.

**Why you'd use it:** each picture shows more objects, at smaller sizes, against more different backgrounds. That
helps detectors see small things and stop relying on the background.

```python
fx.transform.mosaic(size=640, rate=0.25, seed=0)    # one mosaic for every four pictures, originals kept
```

**Watch out:** only pictures from the same train / val / test pile are glued together. Each mosaic credits all four
of its parts.

#### `fx.transform.dedupe`

**Toddler words:** throw away the second copy of the same photo.

**Why you'd use it:** web searches and videos are full of near-copies. Copies make the model over-learn some
pictures, and copies across train and test make test scores lie.

```python
fx.transform.dedupe(threshold=4)        # higher = also drop pictures that are only nearly the same
```

#### `fx.transform.map_classes`

**Toddler words:** rename things: "Car" and "Truck" both become "vehicle".

**Why you'd use it:** datasets and models name things differently; this merges them into your names. `None` deletes
a class's boxes.

```python
fx.transform.map_classes({"Car": "vehicle", "Truck": "vehicle", "Person": None})
```

#### `fx.transform.filter_classes`

**Toddler words:** keep only the boxes you care about, or throw away the ones you don't.

```python
fx.transform.filter_classes(keep=["vehicle", "bicycle"])
fx.transform.filter_classes(drop=["person"])
```

**Watch out:** the picture stays even if all its boxes go; add `filter_images(require_objects=True)` to drop those too.

#### `fx.transform.filter_images`

**Toddler words:** throw away pictures that are too small, oddly shaped, or don't pass your own test.

```python
fx.transform.filter_images(min_width=320, min_height=320, max_aspect=3.0, require_objects=False)
fx.transform.filter_images(where=lambda image, objects: len(objects) <= 50)   # your own rule
```

#### `fx.transform.blur_faces`

**Toddler words:** put a fuzzy sticker over everyone's face.

**Why you'd use it:** street, shop or crowd pictures show people who never agreed to be in your dataset. Each image
records that it was blurred.

```python
fx.transform.blur_faces(strength=1.0, min_size=20)
fx.acquire([...], blur_faces=True)        # the same, as the very first step
```

**Watch out:** it's best effort. Small or turned-away faces can be missed: check before publishing.

#### `fx.transform.custom`

**Toddler words:** your own magic trick, done to every picture.

```python
def sepia(image, objects):
    from PIL import ImageOps
    return ImageOps.colorize(image.convert("L"), "#402010", "#f0e0c0"), objects   # or None to drop the image

fx.transform.custom(sepia, geometry=False, name="sepia")
```

**Watch out:** if your trick moves pixels, set `geometry=True` and move the boxes yourself.

---

### Part 4 · Drawing the boxes

#### `fx.annotate`

**Toddler words:** a helper who looks at every picture and draws the boxes for you.

**Why you'd use it:** drawing boxes by hand takes forever. Pick a helper:

| Helper | Knows | Use it when |
|---|---|---|
| `fx.models.yolo26n` (also `s/m/l/x`) | 80 everyday things (people, cars, dogs, cups…) | your classes are among them |
| `fx.models.yoloe` | whatever classes.yml names; without it, ~4,600 things | your classes are anything else |
| `fx.models.sam` | outlines for boxes that already exist | you need masks, not only boxes |
| `wf.load("model.nbm")` | what you trained it on | you have your own Workflow model |
| your function | whatever you write | another tool or a rule of your own |

```python
fx.annotate(model=fx.models.yoloe, confidence=None, replace=False, batch_size=8)
fx.annotate(model=wf.load("./models/leaf-detector.nbm"))
fx.annotate(model=lambda image: [{"class": "leaf", "xyxy": [10, 10, 80, 90], "confidence": 0.9}])
```

**Watch out:** a picture the model chokes on is marked failed and left out, never saved as "nothing here". Pictures
that arrived with boxes (COCO, Open Images, …) keep them unless `replace=True`.

#### `fx.models.yolo26n`

**Toddler words:** a fast helper who already knows 80 everyday things: people, cars, dogs, chairs, cups, phones.

**Why you'd use it:** your classes are among those 80. It's small, quick, works offline (it comes inside Felix) and is
very sure of itself on what it knows.

```python
fx.load_class("./classes.yml")            # e.g. classes: [person, car] (aliases map other names)
fx.annotate(model=fx.models.yolo26n)
fx.annotate(model=fx.models.yolo26n(device="cuda", imgsz=1280))   # bigger input: small objects found more often
```

**Watch out:** it only finds those 80 classes; Felix tells you if none of your classes is one of them.

#### `fx.models.yolo26s`

**Toddler words:** the same helper's bigger, more careful siblings: s (small), m, l, x (extra large).

**Why you'd use it:** more accurate boxes when speed matters less. Downloaded once into `~/.cache/felix/models`.

```python
fx.annotate(model=fx.models.yolo26s)
fx.annotate(model=fx.models.yolo26x(device="cuda"))
```

#### `fx.models.yoloe`

**Toddler words:** a helper you can *tell* what to look for. Write "white tiger" in your list and it looks for white
tigers. Give it no list and it names whatever it recognises, out of about 4,600 things.

**Why you'd use it:** your classes are not everyday COCO things: leaves, tomatoes, a particular tool. The prompts are
your class names (or `prompt:` from classes.yml), and its boxes come back with your class names.

```python
fx.load_class("./classes.yml")            # classes: [leaf, tomato, white tiger]
fx.annotate(model=fx.models.yoloe)        # prompted with those three
fx.annotate(model=fx.models.yoloe("yoloe-26l-seg.pt"))   # larger: better on hard prompts
```

**Watch out:** prompted scores run low (0.1–0.4 is normal), so its default minimum is 0.1. Whole objects work best;
parts ("bus window") are often missed. The first prompt downloads a ~254 MB text encoder once.

#### `fx.models.sam`

**Toddler words:** a helper who colours in exactly the shape of each thing inside a box someone already drew.

**Why you'd use it:** segmentation datasets (outlines, not only boxes). Use it after a box model, or on a dataset that
came with boxes.

```python
fx.annotate(model=fx.models.yoloe)        # boxes first
fx.annotate(model=fx.models.sam)          # then outlines for them
```

**Watch out:** it adds masks to existing boxes and never finds new objects; pictures without boxes are left as they
are.

#### `fx.use`

**Toddler words:** tell Felix once which helper does the looking (the computer's brain, or a graphics card), so
you don't have to say it again for every model.

**Why you'd use it:** annotating thousands of pictures is much faster on a GPU. Without it, Ultralytics guesses.
With `fx.use("cuda")` (or `"mps"` on a Mac) every model in the session uses it.

```python
fx.use("cuda")                                    # every fx.annotate from now on
fx.annotate(model=fx.models.yoloe)                # → on the GPU
fx.annotate(model=fx.models.sam(device="cpu"))    # a model's own device= still wins

with fx.use("cpu"):                               # only inside the block
    fx.annotate(model=fx.models.yolo26n)

fx.use()                                          # what is it now, and where does that come from?
```

Setting `FELIX_DEVICE=cuda` in the environment does the same without touching the code, so one notebook runs on a
laptop and on a GPU box unchanged.

**Watch out:** asking for a device that isn't there fails straight away instead of quietly using the CPU.
`"modal"` belongs to Workflow (cloud training), not Felix.

### Part 5 · Writing the dataset

#### `fx.export`

**Toddler words:** pack the box and write the label on it: what's inside, where each picture came from, and who may
use it.

**Why you'd use it:** it runs every step above and writes the dataset Workflow trains from, plus LICENSE.txt,
ATTRIBUTIONS.md and CITATION.cff. `pairs=` stops at a number; without it, it keeps going until the sources run out
or you press Ctrl+C (the dataset is still closed properly).

```python
fx.export("./datasets/leaves.felix", pairs=None, license=None, title=None, authors=None, strict=False)
fx.export("./datasets/leaves", license="CC-BY-4.0", title="Leaves", authors=["Ann Lee"])   # a folder, released openly
fx.export("./datasets/leaves.sh_felix", shard_size=1000)   # a folder of shards
```

**Watch out:** `license=None` means closed (all rights reserved). For an open dataset, read the ⚠ lines in
ATTRIBUTIONS.md, or use `license="auto"` (the most open licence every picture allows) and `strict=True` (leave out
pictures that don't fit). See [Licences](#licences-and-provenance).

#### `fx.load` and its splits

**Toddler words:** open the box. It already has three piles: `train` (learn from these), `val` (check while
learning), `test` (the final exam).

```python
data = fx.load("./datasets/leaves.felix")
data.train, data.val, data.test, data.classes, data.splits
```

Imported datasets keep their piles; everything else is split 80 / 10 / 10, the same way every time. Copies, crops and
mosaics follow their original.

#### `fx.split`

**Toddler words:** photocopy the pile, then keep only part of the copy. The big pile stays as it was.

**Why you'd use it:** quick experiments on a tenth of the data; every split and class keeps its share.

```python
fx.split("leaves.felix", 0.1, output="leaves-small.felix", size=None, seed=0)
fx.split("leaves.felix", size=500)                                 # exactly 500 pairs
```

#### `fx.partition`

**Toddler words:** cut the pile into several smaller piles, each a complete dataset.

**Why you'd use it:** sharing work between people or machines, or separate datasets per split.

```python
fx.partition("leaves.felix", parts=4)
fx.partition("leaves.felix", ratio=[70, 30], names=["big", "small"], shuffle=True)
fx.partition("leaves.felix", by_split=True)                       # one per train / val / test
```

#### `fx.shard`

**Toddler words:** spread a huge pile over many small boxes, so you only carry one box at a time.

**Why you'd use it:** datasets too big for memory or one file. Workflow trains from the folder, one shard at a time.

```python
fx.shard("leaves.felix", output=None, size=1000)
fx.export("./datasets/big.sh_felix", shard_size=1000)             # or export straight into shards
```

#### `fx.reset`

**Toddler words:** wipe the whiteboard: forget the shopping list, the places and the tools, and start a new dataset.

```python
fx.export("./datasets/leaves.felix")
fx.reset()                                # the next lines describe a different dataset
fx.acquire([fx.sources.nasa(query="rocket", limit=200)])
fx.export("./datasets/rockets.felix")
```

#### `fx.FelixError`

**Toddler words:** when something's wrong, Felix stops and tells you, in plain words, what to change.

```python
try:
    fx.export("./datasets/leaves.felix")
except fx.FelixError as e:
    print("fix this:", e)
```

**Watch out:** problems with single pictures (a dead link, an unreadable file) never stop the run: they're counted
and reported at the end.

#### `felix run`

**Toddler words:** "do the recipe in this file".

```
felix run make_leaves.py
felix run make_leaves.py --pairs 500          # extra words go to the script (sys.argv)
```

#### `felix info`

**Toddler words:** read the label on the box: how many pictures, which classes, which piles.

```
felix info leaves.felix
felix info ./animals                          # class folders: images per class and split
```

#### `felix pack`

**Toddler words:** put a folder of pictures into one tidy box (one `.felix` file) that's easy to copy.

```
felix pack ./leaves -o leaves.felix
felix pack ./leaves_shards -o leaves.felix    # shards back into one file
```

#### `felix unpack`

**Toddler words:** open the box and spread everything out in a folder.

```
felix unpack leaves.felix -o ./leaves
```

#### `felix convert`

**Toddler words:** take someone else's box (COCO, YOLO, class folders, a zip of pictures) and repack it the Felix way.

```
felix convert ./roboflow-export -o data.felix
felix convert ./pets --labels folders -o pets.felix --license CC0-1.0
```

#### `felix split`

**Toddler words:** photocopy the pile, then keep only part of the copy. The big pile stays as it was.

**Why you'd use it:** quick experiments on a tenth of the data; every split and class keeps its share.

```
felix split leaves.felix 0.1 -o leaves-small.felix
```

#### `felix partition`

**Toddler words:** cut the pile into several smaller piles, each a complete dataset.

```
felix partition leaves.felix --parts 4
felix partition leaves.felix --ratio 70,30 --names big,small --shuffle
felix partition leaves.felix --by-split          # one dataset per train / val / test
```

#### `felix shard`

**Toddler words:** spread a huge pile over many small boxes, so you only carry one box at a time.

**Why you'd use it:** datasets too big for memory or one file. Workflow trains from the folder, one shard at a time.

```
felix shard leaves.felix --size 1000
felix shard leaves.felix --shards 16
```

#### `felix library`

**Toddler words:** this book, in your terminal: every tool, with what it is, why, and an example. Type to search.

```
felix library
felix library augment
```

### Part 6 · Recipes

#### 1. A detector for your own thing, from your own photos

**Toddler words:** Your own photos, a helper draws the boxes, you zoom in on each fruit, and make a few funny copies.

You have 300 photos of tomato plants and want a fruit detector. YOLOE draws the first boxes, crops keep small fruit
big, and augmentation stretches 300 photos further.

```python
import felix as fx

fx.load_class("./classes.yml")              # classes: [tomato]
fx.acquire([fx.sources.local("./raw/tomatoes")])
fx.annotate(model=fx.models.yoloe)
fx.transform.consequential(size=640, negative_rate=0.15)
fx.transform.augment(copies=2, flip=0.5, scale=0, translate=0, brightness=0.25, hue=4)   # after crops: flips + colour
fx.export("./datasets/tomatoes.felix")
```

#### 2. Topping up with the web

**Toddler words:** Your photos are a good start; borrow more from free libraries so the model sees more kinds of tomato.

Same classes, more variety: your photos plus openly licensed ones, without copies.

```python
fx.load_class("./classes.yml")
fx.acquire([
    fx.sources.local("./raw/tomatoes"),
    fx.sources.openverse(query=["tomato plant", "cherry tomatoes", "tomato harvest"], limit=1500),
    fx.sources.inaturalist(taxon="Solanum lycopersicum", limit=1000),
    fx.sources.open_images(classes=["Tomato"], limit=1000),     # already labelled: kept as they are
])
fx.transform.dedupe()
fx.transform.center_crop(size=1024)
fx.annotate(model=fx.models.yoloe)
fx.export("./datasets/tomatoes-plus.felix")
```

#### 3. A general-purpose dataset of anything

**Toddler words:** Grab whatever pictures the libraries have and let the helper name everything it sees.

No class file and no queries: random pictures from several collections, labelled with whatever the prompt-free
model recognises. Runs until 10,000 pairs (or Ctrl+C).

```python
fx.acquire([fx.sources.wikimedia(), fx.sources.openverse(), fx.sources.nasa(), fx.sources.open_images()])
fx.transform.dedupe()
fx.transform.letterbox(size=640)
fx.annotate(model=fx.models.yoloe)
fx.export("./datasets/anything.sh_felix", pairs=10_000, shard_size=1000)
```

#### 4. A species classifier

**Toddler words:** One word per picture ("oak", "maple", "beech") instead of boxes, straight from nature photographers.

One label per picture instead of boxes: iNaturalist names each photo's species.

```python
fx.acquire([
    fx.sources.inaturalist(taxon=["Quercus robur", "Acer pseudoplatanus", "Fagus sylvatica"], limit=2400, label=True),
])
fx.transform.center_crop(size=384)
fx.transform.augment(copies=1, flip=0.5, rotate=15, brightness=0.2)
fx.export("./datasets/trees.felix")
```

#### 5. From a video to a dataset

**Toddler words:** Film it once; Felix keeps the frames where something changed and blurs faces.

```python
fx.load_class("./classes.yml")
fx.acquire([fx.sources.video("./greenhouse.mp4", sampling=fx.sampling.auto())], blur_faces=True)
fx.annotate(model=wf.load("./models/leaf-detector.nbm"))
fx.export("./datasets/greenhouse.felix")
```

#### 6. Releasing a dataset openly

**Toddler words:** Share it with the world, with only the pictures everyone may reuse, and a note crediting every photographer.

Let Felix pick the most open licence every picture allows, and leave out any picture that doesn't fit.

```python
fx.acquire([fx.sources.wikimedia(query="bicycle", limit=2000), fx.sources.openverse(query="bicycle", limit=2000)])
fx.annotate(model=fx.models.yolo26n)
fx.export("./release/bicycles", license="auto", strict=True, title="Bicycles", authors=["Ann Lee"], version="1.0")
```

#### 7. Training on it, then fine-tuning

**Toddler words:** Teach the brain with the new box of pictures, starting from what it already knows.

The dataset goes straight into Workflow (its train / val piles are used as they are). To adapt a model you
already trained to the new dataset, fine-tune it: the backbone stays frozen and only the later layers learn.

```python
import workflow as wf
model = wf.fine_tune("./models/leaf-detector.nbm", "./datasets/tomatoes.felix", epochs=30, freeze="backbone")
```

#### 8. Trying different settings quickly

**Toddler words:** The pictures are already in the cupboard (the cache), so trying another setting takes seconds.

The first export downloads; every later run of the same sources comes from the download cache. Change a transform
or the model and run again: it takes seconds, not minutes.

```python
fx.reset()                                  # forget the previous steps (same program)
fx.acquire([fx.sources.open_images(classes=["Tomato"], limit=1000, seed=0)])   # same seed: same pictures
fx.transform.letterbox(size=512)
fx.export("./datasets/try-512.felix", overwrite=True)
```

---

## Sources

Pass any mix to `fx.acquire([...])`. Every image keeps a `source` record in its annotation: where it came from,
its licence, who to credit and when it was collected. `limit=None` (the default for online sources) means "as many as
the provider gives": with `fx.export(..., pairs=N)` the export simply stops asking once it has N pairs.

**Several searches.** Every search argument also takes a list: `query=[...]` (Wikimedia, Openverse, Flickr, Unsplash,
Pexels, NASA, the Art Institute, the Smithsonian), `taxon=[...]` (iNaturalist), `scientific_name=[...]` (GBIF). The
searches run side by side, share `limit` equally (`limit=900` with three queries: 300 each), and a picture found by
two is taken once. Each picture's `source` records the search that found it.

**No target, no problem.** The query, taxon or class of the search sources is optional: `fx.sources.wikimedia()`
gives random Commons files, `openverse()` / `nasa()` / `smithsonian()` / `gbif()` / `inaturalist()` whatever they
list, `unsplash()` the newest photos, `pexels()` the curated ones, `flickr()` recent uploads (licence-filtered), and
`open_images()` the classes of your classes.yml (all 600 box classes without one). Pair it with `fx.annotate` to
label what comes in.

A source with no "anything" feed of its own (the Art Institute), or whose feed runs dry (Openverse stops after 240),
goes on with a **vocabulary**: by default about 4,600 subjects, the Recognize Anything (RAM++) tag list (Apache-2.0,
bundled) plus everyday subjects, asked in shuffled order, at most 25 pictures each. Choose another with
`.using(vocabulary="general")` (~150 everyday subjects), `"ram++"`, or your own list, and `per_term=` for the number
of pictures per subject. Pictures found by two subjects are taken once.

Online sources send a User-Agent naming Felix. Set `FELIX_CONTACT` (an email or URL) or `contact:` in
`~/.config/felix/credentials.yml` to include a way to reach you. **Wikimedia requires it**: without it, its image
server refuses Felix, and Felix stops with a message saying so.

#### Speed
- **Parallel downloads.** Each source downloads up to 16 images at once, over kept-alive connections (no new
  handshake per image). Image hosts built for it (S3 buckets, CDNs) take up to 16 at a time; others take 2–4.
  Different sources run side by side.
- **Polite APIs.** Search APIs are still asked one page at a time, spaced as their rules ask (iNaturalist about one
  request a second, Pexels 200 an hour, …).
- **Slow-downs.** A server that answers "too many requests" (429 / 503) pauses only that server, for as long as it
  asks; everything else carries on.
- **Download cache.** Downloaded images are kept in `~/.cache/felix/images` (most recently used first, up to
  `FELIX_IMAGE_CACHE_GB`, default 10). Re-running an export after changing a transform or the model needs no network.
  `FELIX_IMAGE_CACHE=0` turns it off.
- **Limits are met.** A failed download frees its place, so `limit=` and `pairs=` still get their full count.

Measured on a home connection, 48 images, cache off: Open Images 0.5 → 9.4 img/s (Felix 1.1 → 1.2); iNaturalist
0.3 → 6.2 img/s; a cached re-run 40–60 img/s.

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

#### `fx.sources.wikimedia(query=None, limit=None, licenses=None, max_side=2048, min_side=256)`
Wikimedia Commons (MediaWiki API, File namespace, bitmap images). Records licence, author, credit and the file page,
and builds the attribution line. `licenses=["CC0", "Public domain", "CC BY 4.0"]` keeps only those (by the start of
Commons' short licence name). Downloads a scaled copy up to `max_side`, not 50-megapixel originals. No
query: random files.

#### `fx.sources.openverse(query=None, limit=None, licenses=None, commercial=True, source=None)`
Openverse: openly licensed images from Flickr, Wikimedia and many museums, in one search (no key). `commercial=True`
keeps licences that allow commercial use; `licenses=["cc0", "by", "by-sa"]` to choose; `source="flickr"` for one
provider. Openverse supplies the attribution text.

#### `fx.sources.flickr(query=None, limit=None, licenses=(4, 5, 7, 8, 9, 10, 11, 12), api_key=None)`
Flickr search filtered by licence id (default: CC BY, CC BY-SA, CC0, public domain and "no known copyright", i.e.
commercial use and changes allowed). Flickr's API is for non-commercial use unless you arrange otherwise with
Flickr, and returns at most 4,000 results per search. No query: recent uploads with those licences.

#### `fx.sources.unsplash(query=None, limit=None, size="regular", orientation=None, api_key=None)`
Unsplash (Unsplash License). As Unsplash's API guidelines require, each download is reported to Unsplash. Demo
apps are limited to 50 requests an hour. No query: the newest photos.

#### `fx.sources.pexels(query=None, limit=None, size="large2x", orientation=None, api_key=None)`
Pexels (Pexels License; 200 requests an hour, so Felix waits ~18 s between result pages). The "Photo by … on Pexels"
credit is recorded. No query: Pexels' curated photos.

Unsplash and Pexels photos are free to use, but both licences forbid compiling their photos into a similar or
competing service; export flags them when you release a dataset openly.


### Open collections

#### `fx.sources.inaturalist(taxon=None, limit=None, licenses=("cc0", "cc-by", "cc-by-sa"), size="large", quality="research", label=False, place_id=None)`
Research-grade iNaturalist observation photos for a taxon (scientific or common name), with each photo's own licence
and attribution. `label=True` labels each image with its species (classification). Pages stop at 10,000 results. No taxon: any species. The
name is looked up first and means exactly that taxon and everything under it ("Pinus": all pines, not the pine
siskin *Spinus pinus*; common names work: "red oak" is *Quercus rubra*).

#### `fx.sources.gbif(scientific_name=None, query=None, limit=None, licenses=("CC0_1_0", "CC_BY_4_0"), country=None)`
Occurrence images from GBIF, with the publishing institution's licence and rights holder. Neither name nor
query: any occurrence with an image.

#### `fx.sources.nasa(query=None, limit=None, size="large")`
The NASA Image and Video Library. NASA imagery is generally not copyrighted in the US, but some items are third-party
or show people and logos: check NASA's media guidelines.

#### `fx.sources.artic(query=None, limit=None)` · `fx.sources.smithsonian(query=None, limit=None, api_key=None)`
Public-domain (CC0) images from the Art Institute of Chicago (843 px IIIF images, as the museum asks) and the
Smithsonian's Open Access. The Art Institute's API sits behind a bot check that sometimes
refuses programs for a while; Felix then says so instead of failing silently.

Not included, on purpose: **Pixabay** (its terms forbid automated collection and machine-learning use without
permission), **The Met** and the **Library of Congress** (their servers only serve browsers: automated requests get a
bot check, and Felix does not work around those).

### Dataset hubs

#### `fx.sources.roboflow(project, version, workspace=None, api_key=None)`
A Roboflow dataset version (your workspace or Universe: `project="workspace/project"`), downloaded as a COCO export
and imported with boxes and splits. Cached in `~/.cache/felix/roboflow`. A Universe project without a licence means
all rights reserved.

#### `fx.sources.open_images(classes=None, split="validation", limit=None, seed=0, group_of=True)`
Open Images V7 images containing any of `classes` (display names: "Car", "Person", …), with all their boxes of those
classes. Without `classes`: your classes.yml's classes that Open Images has (names or aliases), or all 600
box classes. The annotation CSVs are cached in `~/.cache/felix/open_images` (validation ~40 MB, test ~120 MB, train
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
  - name: dog
    prompt: [dog, puppy, a dog lying down]   # several prompts: any of them finds the class
    aliases: [tomato, Tomato]           # names that mean this class in imported data / model labels
    confidence: 0.4                     # optional minimum model confidence for this class
confidence: 0.25                        # optional default
```

Several prompts (`prompt: [...]`, or `prompts:`) all look for the same class: YOLOE is asked for each, the boxes
are named after the class, and boxes of one class overlapping by more than 60 % (two prompts finding the same
object) are merged, keeping the most confident. A prompt may belong to one class only. Use it for synonyms
("car", "automobile"), kinds ("sedan", "hatchback") and looks ("a dog lying down").

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
| `augment(copies=1, keep_original=True, flip=0.5, vflip=0, rotate=0, scale=0.1, translate=0.1, brightness=0.2, contrast=0.2, saturation=0.2, hue=5, blur=0.05, noise=0, grayscale=0.02, jpeg=0, cutout=0, min_visibility=0.3, seed=0)` | extra, randomly changed copies of every image; labels follow (see below) |
| `mosaic(size=640, rate=0.25, seed=0)` | 2 × 2 mosaics of four images with all their boxes, added alongside the originals |

**consequential.** For each object not already well covered by a crop, a square around it with `margin` of its size on
each side, at least `size` pixels where the image allows (small objects are not blown up), kept inside the image
(grey padding only for objects larger than the short side). Every object inside a crop keeps its box. A crop
overlapping an earlier crop of the same image by more than `overlap` (IoU) is skipped. Negatives (squares touching no
object, from the same images or from annotated images without objects) are added so they make up `negative_rate` of
the output. Images whose annotation failed are left out, never used as negatives. Calling it before `fx.annotate`, or
a geometry step after it, raises an error at once.

**augment.** Each image gets `copies` new versions; the original stays unless `keep_original=False`.
- Geometry (boxes and masks move with the pixels): `flip` / `vflip` are probabilities, `rotate` is ± degrees, `scale`
  and `translate` are ± fractions (`scale=0.2`: 80–120 %). Uncovered corners are grey. Boxes keeping less than
  `min_visibility` of their area are dropped.
- Colour: `brightness`, `contrast`, `saturation` (± fractions), `hue` (± degrees).
- Damage: `blur`, `grayscale`, `jpeg` (probabilities of a slight blur, no colour, heavy compression), `noise` (largest
  noise as a fraction of the brightness range), `cutout` (probability of a grey patch; boxes more than 60 % hidden
  are dropped).
- Set an option to `0` to turn it off. With `rotate`, `scale` and `translate` at `0` (flips and colour are fine)
  it may come after `consequential`.
- Augment *after* `fx.annotate`, so the model labels the real pictures and the copies inherit those labels.

**mosaic.** `rate` mosaics per input image (`0.25`: one per four). Each mosaic puts four whole images, each fitted into
a quarter, in one `size` picture. It lists its four parts as its source, so attribution and licence checks cover all
of them.

**Splits stay clean.** Augmented copies, mosaics and consequential crops always land in the same train / val / test
split as the image they came from, so no version of a test image is ever trained on.

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
`yolo26n.pt`, `yoloe-26s-seg.pt` (text prompts) and `yoloe-11s-seg-pf.pt` (prompt-free: YOLOE-26's own prompt-free weights give nonsense under current Ultralytics). Other weights
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

### `fx.use(device)`
The device Felix's models run on when they don't name one: `"cpu"`, `"cuda"`, `"cuda:N"` (or `"N"`), `"mps"`, or
`"auto"` / `None` (back to automatic). Takes effect at once; `with fx.use("cuda"):` sets it only inside the block, and
`fx.use()` shows the current choice. First match wins: the model's own `device=` (`fx.models.yoloe(device="cuda")`),
then `fx.use`, then the `FELIX_DEVICE` environment variable, then automatic (Ultralytics' choice; a Workflow model's
own). A device that isn't on this machine raises an error at once; `"modal"` is a Workflow training device, not a
Felix one. It applies to `fx.annotate` with the built-in models and with Workflow models, and isn't cleared by
`fx.reset()`.

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

**Toddler words:** every picture carries a little note: where it came from, who made it, and the rules for using it.

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

**Toddler words:** decide whether everyone may use your dataset (open) or only you (closed); Felix checks every picture's rules against that.

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

**Toddler words:** three notes that travel in the box: the rules for the whole dataset, a thank-you list for every photographer, and how to cite it.

```bash
unzip -p leaves.felix ATTRIBUTIONS.md | head -20      # read the credits inside a .felix file
unzip -p leaves.felix LICENSE.txt
```

| File | Contents |
|---|---|
| `LICENSE.txt` | the dataset's licence (or the closed notice), that images keep their own licences, and a count per image licence |
| `ATTRIBUTIONS.md` | every image: file, source, licence (with any ⚠ issue), author / credit, link |
| `CITATION.cff` | how to cite the dataset (title, authors, date, licence), plus the datasets and collections it draws on (Open Images, COCO, Wikimedia Commons, iNaturalist, …) |

They sit next to the data: inside a `.felix` file, at the top of a dataset folder, or beside the shards.

### Faces

**Toddler words:** put a fuzzy sticker over people's faces, so nobody can be recognised.

```python
fx.acquire([fx.sources.local("./street-photos")], blur_faces=True)   # blur before anything else
fx.transform.blur_faces(strength=1.0, min_size=20)                    # or as a step of its own
```

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

**Toddler words:** the dataset spread out in a folder: pictures in one drawer, their notes in another.

```
dataset/
├── images/IMG1.png ...        (JPEG, PNG or WebP, kept as they were)
├── annotations/IMG1.yml ...
└── metadata.yml
```

### Shards: a folder of `.sh_felix` files

**Toddler words:** a huge dataset split into many small boxes, so you only carry one at a time.

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

**Toddler words:** one folder per kind of thing: every picture in `cat/` is a cat.

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

**Toddler words:** the whole folder zipped into one file, easy to copy and share.

```bash
felix pack ./leaves -o leaves.felix     # folder → one file
felix unpack leaves.felix -o ./leaves   # and back
unzip -l leaves.felix | head            # it's a normal zip
```

A single file for the same tree - a standard ZIP archive (PNG/JPEG stored, YAML deflated),
so `unzip` and Python's `zipfile` can open it. `felix pack` / `felix unpack` convert between
the two; `fx.export("x.felix")` writes it directly (via a `.part` file, so a crash never
leaves a half-written dataset).

### `annotations/<name>.yml`

**Toddler words:** the note next to each picture: what's in it, where the boxes are, and where the picture came from.

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

**Toddler words:** the label on the outside of the box: what's in the dataset, how many pictures, which classes.

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
