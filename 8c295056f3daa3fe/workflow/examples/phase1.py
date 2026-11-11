"""Phase 1 example: the architecture from the spec. Run: python examples/phase1.py"""
import torch
import workflow as wf

model = wf.Model(
    wf.Stage(
        "p1",
        wf.Conv2D(3, 32, 3, stride=2, padding=1),
        wf.SiLU(),
    ),
    wf.Stage(
        "p2",
        wf.Conv2D(32, 64, 3, stride=2, padding=1),
        wf.SiLU(),
        wf.Residual(
            wf.Conv2D(64, 64, 3, padding=1),
            wf.SiLU(),
            wf.Conv2D(64, 64, 3, padding=1),
        ),
    ),
    wf.Stage(
        "p3",
        wf.Conv2D(64, 128, 3, stride=2, padding=1),
        wf.Bottleneck(128),
        wf.Bottleneck(128),
    ),
)

model.summary()                       # shapes + parameters, no forward pass needed
dev = model.device("auto")            # CUDA → MPS → CPU
x = torch.randn(2, 3, 640, 640, device=dev.device())
print({name: tuple(t.shape) for name, t in model.features(x).items()})

model.save("example.nbm")             # one file: architecture + weights
print(type(wf.load("example.nbm")).__name__, "reloaded from example.nbm")
