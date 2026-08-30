"""
Compile ONNX model to TensorRT engine.

The .engine file is loaded by the C++ runtime for inference.
"""

import tensorrt as trt

ONNX_PATH = "lstm.onnx"
ENGINE_PATH = "lstm.engine"

# TensorRT uses a "logger" for build-time warnings/errors
logger = trt.Logger(trt.Logger.INFO)

# Builder: creates the engine
builder = trt.Builder(logger)

# Network: defines the computation graph (parsed from ONNX)
network = builder.create_network()

# Parser: reads ONNX file into the network
parser = trt.OnnxParser(network, logger)

with open(ONNX_PATH, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(f"ONNX Parse Error: {parser.get_error(i)}")
        raise RuntimeError("Failed to parse ONNX")

print(f"Network inputs:  {network.get_input(0).name} {network.get_input(0).shape}")
print(f"Network outputs: {network.get_output(0).name} {network.get_output(0).shape}")

# Build configuration
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB workspace

# TF32 is enabled by default on Ada (tensor core acceleration)
# TensorRT 11 auto-selects FP16/TF32 where beneficial

# Set optimization profile for dynamic batch size
profile = builder.create_optimization_profile()
# min/opt/max batch sizes
profile.set_shape("input", (1, 50, 40), (1, 50, 40), (32, 50, 40))
config.add_optimization_profile(profile)

# Build the engine (this takes 10-60 seconds — profiling your GPU)
print("Building TensorRT engine (this may take a minute)...")
engine_bytes = builder.build_serialized_network(network, config)

if engine_bytes is None:
    raise RuntimeError("Engine build failed")

# Save to disk
with open(ENGINE_PATH, "wb") as f:
    f.write(engine_bytes)

print(f"Engine saved to {ENGINE_PATH} ({engine_bytes.nbytes / 1024:.1f} KB)")
