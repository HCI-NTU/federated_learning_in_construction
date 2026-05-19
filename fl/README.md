# Federated Learning Layer (Stage 3)

Bridges Flower's federated learning framework with Ultralytics YOLO11.
Implements the three methods compared in this study: FedAvg, FedProx, FedPer.

## Components

- `parameter_utils.py` — Round-trip conversion between Ultralytics YOLO
  state_dict and Flower's NumPy parameter list. Also identifies which
  parameters belong to the detection head (for FedPer's head-local
  federation).
- `yolo_client.py` — `YOLOClient(fl.client.NumPyClient)`: wraps Ultralytics
  YOLO11 so each FL client runs real Ultralytics training/validation per
  round. Supports both full-parameter exchange (FedAvg/FedProx) and shared-only
  exchange (FedPer).
- `strategies.py` — `build_strategy(method, ...)`: returns a Flower Strategy
  configured for the requested method. All three methods use plain FedAvg as
  the server-side aggregator; differences are entirely client-side:
  - FedAvg: full state_dict, no proximal term
  - FedProx: full state_dict, proximal_mu sent via fit_config
  - FedPer: shared parameters only (clients keep their heads local)
- `server.py` — `run_fl_server(method, num_rounds, ...)`: top-level entry
  for starting a Flower server. Returns serialized per-round history.
- `run_client.py` — CLI launcher for a single Flower client process.

## Key design decisions

### Single-strategy server, role-aware clients

All three FL methods are implemented using **Flower's plain FedAvg strategy
on the server side**. What differs is what each client sends. This keeps the
server logic uniform and makes the comparison fair: any performance
difference between methods is attributable to the federation rule, not to
strategy implementation differences.

- FedAvg: client sends full state_dict
- FedProx: client sends full state_dict; adds (mu/2)||w - w_global||² to its
  local loss via a training callback
- FedPer: client sends backbone+neck only; keeps its own detection head
  across rounds. When the server sends back aggregated shared parameters,
  the client merges them with its local head before training.

### num_classes (nc) must match across server and clients

Ultralytics defaults YOLO11 to 80 classes (COCO). If the server constructs
the initial model with the default nc=80 but the clients' data.yaml
declares 17 classes, Ultralytics' trainer silently resizes the head — and
the resized weights no longer match the server's initial shape in round 2.

Both `YOLOClient.__init__` and `server.build_initial_parameters` accept an
explicit `nc` argument. The client infers it from the data.yaml if not
provided. Build the model with `DetectionModel(cfg=..., nc=nc)` rather than
`YOLO(yaml)` to ensure the head shape is correct from the start.

### FedProx implementation

FedProx is FedAvg + a proximal regularizer added to each client's local
loss. Ultralytics doesn't expose its loss object cleanly, so the proximal
gradient is injected directly via the `on_train_batch_end` callback —
after backward(), the client adds `mu * (w_local - w_global)` to each
parameter's `.grad`. Mathematically equivalent to integrating the term in
the loss; pragmatically simpler given Ultralytics' API.

### Parameter exchange uses independent NumPy copies

A subtle bug: `tensor.numpy()` returns a view sharing memory with the
underlying tensor. If Flower exchanges these views, then any subsequent
mutation of the local model would silently corrupt the parameters in
Flower's internal state. `state_dict_to_numpy` therefore calls `.copy()`
on every array to guarantee independence.

## Usage

### Programmatic (within a script)

```python
from fl.server import run_fl_server

history = run_fl_server(
    method="fedper",
    num_rounds=10,
    num_clients=3,
    model_variant="yolo11s.yaml",
    nc=17,                          # MUST match data.yaml class count
    server_address="0.0.0.0:8080",
    epochs_per_round=1,
    results_path="results/exp1_fedper_n4000.json",
)
```

### CLI (multi-process)

Server:
```bash
python -m fl.server \
    --method fedper \
    --num-rounds 10 \
    --num-clients 3 \
    --model-variant yolo11s.yaml \
    --nc 17 \
    --server-address 0.0.0.0:8080 \
    --epochs-per-round 1 \
    --results-path results/exp1_fedper.json
```

Each client (one process per client):
```bash
python -m fl.run_client \
    --client-name MOCS \
    --data-yaml work/configs/MOCS_n4000_tier_full_s42.yaml \
    --model-variant yolo11s.yaml \
    --method fedper \
    --server-address 127.0.0.1:8080 \
    --epochs-per-round 1 \
    --imgsz 640 \
    --batch 16 \
    --device cuda:0
```

For the experiment driver (Stage 5), this is wrapped in a single command
that launches server + clients in coordinated subprocesses.

### Simulation (for testing without networking)

`flwr.simulation.run_simulation` requires `pip install "flwr[simulation]"`
(installs Ray). This is used only by `tests/test_fl_integration.py` for
in-process verification. The real experiment runs use multi-process via
the CLI.

## Configuration knobs

| Parameter | Default | Purpose |
|---|---|---|
| `model_variant` | `yolo11s.yaml` | YOLO architecture (use `.yaml` for from-scratch, `.pt` for pretrained) |
| `nc` | 17 | Detection class count (must match data.yaml) |
| `epochs_per_round` | 1 | Local epochs per FL round; raising trades comm for compute |
| `imgsz` | 640 | Training/inference image size |
| `batch` | 16 | Local batch size |
| `proximal_mu` | 0.01 | FedProx coefficient (ignored for fedavg/fedper) |
| `seed` | 42 | Reproducibility seed |
