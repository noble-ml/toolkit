"""Workflow demo: Felix generates a dataset; Workflow builds, trains, evaluates, quantises, exports and predicts.

    python examples/demo.py                          # ~20 epochs on 600 synthetic images
    python examples/demo.py --epochs 40 --device cuda

Needs `felix` and `workflow`. Everything is written to ./demo_out/.
"""
import argparse
import os

import felix as fx
import workflow as wf

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", type=int, default=600)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--qat-epochs", type=int, default=3)
ap.add_argument("--device", default="auto", help="auto, cpu, cuda, gpu or mps")
ap.add_argument("--out", default="demo_out")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
SIZE = 256


# 1. data: circles, squares and triangles with their boxes (drawn with Pillow, stored through Felix) ------------
def make_dataset(path):
    wf.practice_dataset(path, pairs=args.pairs, img_size=SIZE, seed=7, max_objects=3, overwrite=True)


data_path = os.path.join(args.out, f"shapes_{args.pairs}.felix")
if not os.path.exists(data_path):
    make_dataset(data_path)
dataset = fx.load(data_path)
train, val, test = dataset.train, dataset.val, dataset.test
print(f"\n{data_path}: {dataset.splits}, classes {dataset.classes}\n")

# 2. architecture: a CSP backbone, FPN + PAN neck and a detection head, all inferred -----------------
model = wf.stack(
    wf.input(channels=3, height=SIZE, width=SIZE),

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
    wf.head(classes=len(dataset.classes)),
)
model.summary()
model.export_graph(os.path.join(args.out, "architecture.png"))
wf.eval(model, device=args.device)                    # before training: size, FLOPs, latency, FPS, memory

# 3. train (Ctrl+C is safe: the state is saved and model.train(...) can continue it later) ------------
model.train(train, validation=val, epochs=args.epochs, device=args.device, output=os.path.join(args.out, "runs"))
fp32 = wf.eval(model, dataset=test)
model.save(os.path.join(args.out, "detector.nbm"))

# 4. quantisation-aware fine-tuning, then a real int8 model and an ONNX file --------------------------
if args.qat_epochs:
    model.train(train, validation=val, epochs=args.epochs + args.qat_epochs, quantization=wf.QAT(precision="int8"),
                output=os.path.join(args.out, "runs_qat"))
    int8 = wf.eval(model, dataset=test)
    model.export(format="nbm", output=os.path.join(args.out, "detector-int8.nbm"), precision="int8")
    print(f"\nfp32 mAP50-95 {fp32.map50_95:.3f} → int8 (QAT) {int8.map50_95:.3f}")
model.export(format="onnx", output=os.path.join(args.out, "detector.onnx"))

# 5. inference: original/, annotated/ and predictions.json go to results/ -----------------------------
detector = wf.load(os.path.join(args.out, "detector-int8.nbm" if args.qat_epochs else "detector.nbm"))
run = wf.inference(data_path, detector, os.path.join(args.out, "results"), all=True)   # every test image
print(run)
for r in run.images[:3]:
    print(r.path, r, [round(c, 2) for c in r.confidences])
print(f"""
Next, from the terminal:
  workflow quantize {args.out}/detector.nbm int8 --calibration {data_path}
  workflow script {args.out}/detector.nbm --preset webcam
""")
