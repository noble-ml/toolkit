"""Workflow demo: Felix generates a dataset, Workflow builds, trains, benchmarks and runs a detector.

    python examples/demo.py                      # ~25 epochs on a small synthetic dataset
    python examples/demo.py --epochs 40 --pairs 1200

Needs both packages (`felix` and `workflow`). Outputs go to ./demo_out/.
"""
import argparse
import os

import felix as fx
import workflow as wf

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", type=int, default=600, help="images to generate")
ap.add_argument("--epochs", type=int, default=25)
ap.add_argument("--imgsz", type=int, default=256)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--out", default="demo_out")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
SIZE = args.imgsz


# 1. data: a Felix pipeline draws circles, squares and triangles ------------------------------
def make_dataset(path):
    fx.reset()
    s = SIZE / 256
    with fx.class_set("shapes"):
        circle = fx.toolkit.circle(radius=20 * s, color="#d62728")
        square = fx.toolkit.rectangle(width=38 * s, height=38 * s, color="#1f77b4")
        triangle = fx.toolkit.polygon(vertices=[(0, 40 * s), (22 * s, 0), (44 * s, 40 * s)], color="#2ca02c")
    with fx.step(size=(SIZE, SIZE)):
        fx.canvas.background("#e8e8e8")
        for threshold in (1.0, 0.7, 0.4):                         # 1 to 3 objects per image
            obj = fx.paint(fx.random.class_(threshold=threshold), position=fx.random.position(margin=40 * s))
            fx.toolkit.transformation(obj, scale=fx.random.uniform(0.7, 1.5), rotation=fx.random.uniform(0, 360),
                                      hue=fx.random.uniform(-12, 12), brightness=fx.random.uniform(0.85, 1.15))
        fx.gen()                                                   # annotate the objects above...
        fx.paint(fx.toolkit.circle(radius=8 * s, color="#999999"), position=fx.random.position())  # ...not this
    fx.make(pairs=args.pairs, output=path, seed=7, overwrite=True)


data_path = os.path.join(args.out, f"shapes_{args.pairs}.felix")
if not os.path.exists(data_path):
    make_dataset(data_path)
dataset = fx.load(data_path)
print(f"\ndataset {data_path}: {len(dataset)} pairs, splits {dataset.splits}, classes {dataset.classes}\n")


# 2. architecture: backbone → FPN → PAN → BBoxHead, all wired by name ----------------------------
def cbs(c_in, c_out):
    return [wf.Conv2D(c_in, c_out, 3, stride=2, padding=1), wf.BatchNorm2D(c_out), wf.SiLU()]


model = wf.Model(
    wf.Stage("p1", *cbs(3, 16)),
    wf.Stage("p2", *cbs(16, 32), wf.CSP(32)),
    wf.Stage("p3", *cbs(32, 64), wf.CSP(64, n=2)),
    wf.Stage("p4", *cbs(64, 128), wf.CSP(128, n=2)),
    wf.Stage("p5", *cbs(128, 256), wf.SPPF(256)),
    wf.Stage("neck", wf.FPN(inputs=("p3", "p4", "p5"), channels=64, outputs=("n3", "n4", "n5"))),
    wf.Stage("pan", wf.PAN(inputs=("n3", "n4", "n5"), channels=64, outputs=("m3", "m4", "m5"))),
    wf.Stage("head", wf.BBoxHead(inputs=("m3", "m4", "m5"), classes=len(dataset.classes), outputs=("d3", "d4", "d5"))),
    input=(3, SIZE, SIZE),
)
model.summary()
model.export_graph(os.path.join(args.out, "architecture.png"))
model.export_graph(os.path.join(args.out, "architecture.svg"), detail="stages")
model.benchmark()                                   # architecture-level: params, FLOPs, memory, latency, FPS

# 3. train (Ctrl+C is safe: the state is saved to runs/demo/last.nbm and can be continued) ---------
model.device("auto")
model.train(dataset, epochs=args.epochs, batch_size=args.batch, save_every=5, save_dir=os.path.join(args.out, "runs"))

# 4. evaluate on the held-out test split ---------------------------------------------------------
results = model.benchmark(dataset.test)
results.plot_confusion_matrix(os.path.join(args.out, "confusion_matrix.png"))
results.save(os.path.join(args.out, "evaluation.json"))

# 5. predict -----------------------------------------------------------------------------------
stem = dataset.test.stems[0]
dataset.test.image(stem).convert("RGB").save(os.path.join(args.out, "test_image.png"))
pred = model.predict(os.path.join(args.out, "test_image.png"), conf=0.3)
pred.save(os.path.join(args.out, "prediction.png"))
print(f"\n{pred}\nboxes (xyxy): {pred.xyxy.round().tolist()}\nclasses: {pred.classes}\nconfidences: {[round(c, 2) for c in pred.confidences.tolist()]}")

# 6. one file with everything; convert and script it from the terminal ---------------------------
model.save(os.path.join(args.out, "demo.nbm"))
print(f"""
saved {args.out}/demo.nbm. Next:
  workflow export {args.out}/demo.nbm --format onnx        # or --format pt
  workflow script {args.out}/demo.nbm --preset webcam      # also: image, folder, video
""")
