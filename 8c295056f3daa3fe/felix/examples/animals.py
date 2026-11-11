"""Example Felix pipeline: python examples/animals.py   (or: felix run examples/animals.py)"""
import felix as fx

# 1. classes ------------------------------------------------------------
with fx.class_set("animals"):
    cat = fx.toolkit.circle(radius=30, position=(32, 32))
    dog = fx.toolkit.rectangle(width=50, height=40)
    bird = fx.toolkit.polygon(vertices=[(0, 30), (20, 0), (40, 30)])

with fx.class_set("vehicles"):
    car = fx.toolkit.rectangle(width=80, height=40)
    bicycle = fx.toolkit.image(source="assets/bicycle.png")  # transparent PNG

# 2. one generation step (the body is recorded once, re-sampled per image) ----
with fx.step(size=(640, 640)):
    fx.canvas.background("#eeeeee")

    # first object: always present
    selected_set = fx.random.class_set()
    selected_class = fx.random.class_(selected_set)
    obj = fx.paint(selected_class, position=fx.random.position(margin=60))
    fx.toolkit.transformation(
        obj,
        hue=fx.random.uniform(-180, 180),
        brightness=fx.random.uniform(0.7, 1.3),
        rotation=fx.random.uniform(0, 360),
        scale=fx.random.uniform(0.6, 1.6),
    )

    # second object: only appears in ~50% of images (threshold)
    other = fx.paint(
        fx.random.class_(threshold=0.5),
        position=fx.random.position(margin=60),
    )
    fx.toolkit.transformation(other, rotation=fx.random.uniform(0, 360))

    fx.gen()  # annotate everything above; what follows is drawn but NOT annotated

    # a distractor: visible in the image, absent from the annotations
    fx.paint(fx.toolkit.circle(radius=12, color="#888888"), position=fx.random.position())

# 3. generate -------------------------------------------------------------
if __name__ == "__main__":
    fx.make(pairs=200, output="./dataset", seed=1, overwrite=True)
    # fx.make(pairs=200, output="./dataset.felix", overwrite=True)   # single compiled file
