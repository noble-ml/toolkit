"""Example A: a leaf detector dataset, center-cropped before annotation.

Collect leaf photos from a folder, a URL list, a greenhouse video (only frames where the scene changes) and Wikimedia
Commons; shrink them to 640 x 640 by center cropping; annotate with a custom Workflow model; export."""
import felix as fx
import workflow as wf

fx.load_class("./classes.yml")

fx.acquire([
    fx.sources.local("./raw/leaves"),

    fx.sources.urls("./leaf_urls.txt"),

    fx.sources.video(
        "./greenhouse.mp4",
        sampling=fx.sampling.auto()
    ),

    fx.sources.wikimedia(
        query="plant leaves",
        limit=1000
    )
])

fx.transform.center_crop(size=640)

model = wf.load("./models/leaf-detector.nbm")

fx.annotate(model=model)

fx.export("./datasets/leaf-center.felix")
