"""Phase 2 example: backbone → FPN → PAN with named routing. Run: python examples/phase2.py"""
import torch
import workflow as wf


def down(c_in, c_out):
    return [wf.Conv2D(c_in, c_out, 3, stride=2, padding=1), wf.BatchNorm2D(c_out), wf.SiLU()]


model = wf.Model(
    wf.Stage("p1", *down(3, 16)),
    wf.Stage("p2", *down(16, 32), wf.CSP(32)),
    wf.Stage("p3", *down(32, 64), wf.CSP(64)),
    wf.Stage("p4", *down(64, 128), wf.CSP(128)),
    wf.Stage("p5", *down(128, 256), wf.SPPF(256)),
    # the neck names exactly which stages it reads, and what it publishes
    wf.Stage("neck", wf.FPN(inputs=("p3", "p4", "p5"), channels=64, outputs=("n3", "n4", "n5"))),
    wf.Stage("pan", wf.PAN(inputs=("n3", "n4", "n5"), channels=64, outputs=("m3", "m4", "m5"))),
)

model.summary()
out = model(torch.randn(1, 3, 640, 640))   # last stage publishes several names → dict
print({k: tuple(v.shape) for k, v in out.items()})
