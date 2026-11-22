"""Example B: a leaf detector dataset of object-focused crops.

Annotate the original high-resolution images first, then cut square crops around the leaves (plus 15% negative
crops without leaves), so small leaves in big photos become well-sized training examples."""
import felix as fx
import workflow as wf

fx.load_class("./classes.yml")

fx.acquire([
    fx.sources.local("./raw/leaves"),

    fx.sources.video(
        "./greenhouse.mp4",
        sampling=fx.sampling.auto()
    ),

    fx.sources.wikimedia(
        query="plant leaves",
        limit=2000
    )
])

model = wf.load("./models/leaf-detector.nbm")

fx.annotate(model=model)

fx.transform.consequential(
    size=640,
    negative_rate=0.15
)

fx.export("./datasets/leaf-consequential.felix")
