"""Example C: a general object detector dataset, letterboxed.

Combine local photos, Open Images, a Roboflow project, frames from a street video (one every 2 seconds) and
Wikimedia Commons; letterbox everything to 640 x 640 so no part of a scene is cut off; annotate with the built-in
YOLO26n detector (pip install ultralytics). The Roboflow source needs ROBOFLOW_API_KEY (see docs/sources.md)."""
import felix as fx

fx.load_class("./classes.yml")       # e.g. classes: [car, person, bicycle] with aliases [Car, Person, Bicycle]

fx.acquire([
    fx.sources.local("./raw/general"),

    fx.sources.open_images(
        classes=["Car", "Person", "Bicycle"]
    ),

    fx.sources.roboflow(
        project="general-objects",
        version=1
    ),

    fx.sources.video(
        "./recordings/street.mp4",
        sampling=fx.sampling.interval(seconds=2)
    ),

    fx.sources.wikimedia(
        query="cars and bicycles",
        limit=1000
    )
])

fx.transform.letterbox(size=640)

fx.annotate(
    model=fx.models.yolo26n
)

fx.export("./datasets/general.felix")
