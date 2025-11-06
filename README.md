# Synaptron

High-performance multi-modal inference engine with dynamic model graph and auto-optimization.

## Features

### 🔧 Core Features

1. **Model Management**
   - Single model loading and execution (example: bert-base-uncased)
   - Model cache system: downloaded models stored in `./models_cache`
   - Auto-download: Downloads from Hugging Face if model not present
   - Model metadata: Reads type, layer count, etc. from config.json

2. **Inference Engine**
   - Runs inference with single input (text)
   - CPU-based inference (Candle, Rust-Bert)
   - Simple async API: Accepts requests via POST /predict endpoint
   - Returns model result + latency measurement (in PredictResponse struct)

3. **REST API (Axum)**
   - `/predict` — Single text prediction
   - `/models` — Lists loaded models
   - `/models/activate` — Switches model
   - `/health` — System health check

4. **Preprocessing**
   - Cleaning: Unicode normalization + removes unnecessary whitespace/special characters
   - Tokenization: Hugging Face compatible tokenization with tokenizers crate
   - Max input length control (e.g., 512 token limit)

5. **Monitoring**
   - Tracelog: Structured logging with tracing
   - Metrics: Total request count, average latency
   - Benchmark: Throughput measurement with simple BenchmarkResult struct

### ⚡ Advanced Features

1. **Multi-Modal Pipeline**
   - Text, image, and audio input support
   - Automatically detects input type and routes to appropriate model
   - Examples: Text → BERT, Image → CLIP, Audio → Whisper

2. **Dynamic Model Graph**
   - Creates model chains (pipeline graph)
   - "Output → another model's input" chaining
   - Config-based: "which model in which order" determined in graph.yaml

3. **Auto-Optimization Layer**
   - Acceleration with OpenVINO, TensorRT, or ONNX Runtime
   - Model quantization (FP16, INT8)
   - Selects best backend in benchmark mode (auto-select backend)

4. **Memory & Performance**
   - LRU cache (stores recently used embeddings)
   - Async batching (processes concurrent requests as single batch)
   - Dynamic thread scaling (increases CPU threads as load increases)

5. **Advanced API & Dashboard**
   - `/metrics` → Prometheus-compatible metrics endpoint
   - `/dashboard` → (optional web UI) performance graphs
   - WebSocket streaming (real-time inference results)

## Installation

```bash
cargo build --release
```

## Usage

```bash
cargo run --release
```

## Configuration

The application can be configured using the `config.yaml` file or environment variables with the `SYNAPTRON_` prefix.

## API Endpoints

- `POST /predict` - Run inference on text input
- `GET /models` - List loaded models
- `POST /models/activate` - Activate a model
- `GET /health` - Health check
- `GET /metrics` - Performance metrics

## License

MIT License - For educational and research purposes

DEMO: https://synaptron.netlify.app/#api

Not: The Ai is simulation version for that can be not working good
