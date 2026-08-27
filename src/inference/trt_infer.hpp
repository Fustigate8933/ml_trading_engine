#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>

// TensorRT requires a logger. Minimal implementation.
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TRT] " << msg << std::endl;
        }
    }
};

// Custom deleters for TensorRT RAII
struct TRTDestroyer {
    template <typename T>
    void operator()(T* obj) const {
        if (obj) delete obj;
    }
};

/**
 * TRTInfer — TensorRT inference engine wrapper.
 *
 * Loads a serialized .engine file and runs inference on the GPU.
 * Manages GPU memory (input/output buffers) and CUDA stream.
 *
 * Usage:
 *   TRTInfer engine("models/baseline.engine");
 *   float input[50 * 40] = { ... };  // one LOB window
 *   float output[3];                  // 3-class probabilities
 *   engine.infer(input, output);
 */
class TRTInfer {
private:
    TRTLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime, TRTDestroyer> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TRTDestroyer> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TRTDestroyer> context_;

    // GPU memory buffers
    void* d_input_ = nullptr;   // device (GPU) input buffer
    void* d_output_ = nullptr;  // device (GPU) output buffer

    // Pinned host memory (for CUDA Graph path — DMA-friendly)
    float* h_input_ = nullptr;
    float* h_output_ = nullptr;

    // CUDA stream for async operations
    cudaStream_t stream_ = nullptr;

    // CUDA Graph (captured inference pipeline)
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    bool graph_captured_ = false;

    // Dimensions
    size_t input_size_ = 0;   // bytes
    size_t output_size_ = 0;  // bytes

public:
    TRTInfer(const std::string& engine_path) {
        // 1. Load serialized engine from file
        std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open engine file: " + engine_path);
        }

        size_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> engine_data(file_size);
        file.read(engine_data.data(), file_size);

        // 2. Create runtime (needed to deserialize)
        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) throw std::runtime_error("Failed to create TRT runtime");

        // 3. Deserialize engine from binary blob
        engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), file_size));
        if (!engine_) throw std::runtime_error("Failed to deserialize engine");

        // 4. Create execution context (holds intermediate activation memory)
        context_.reset(engine_->createExecutionContext());
        if (!context_) throw std::runtime_error("Failed to create execution context");

        // 5. Set input shape (batch=1, window=50, features=40)
        context_->setInputShape("input", nvinfer1::Dims3{1, 50, 40});

        // 6. Allocate GPU buffers
        // Input: 1 * 50 * 40 * sizeof(float) = 8000 bytes
        // Output: 1 * 3 * sizeof(float) = 12 bytes
        input_size_ = 1 * 50 * 40 * sizeof(float);
        output_size_ = 1 * 3 * sizeof(float);

        cudaMalloc(&d_input_, input_size_);
        cudaMalloc(&d_output_, output_size_);

        // 7. Create CUDA stream
        cudaStreamCreate(&stream_);

        // 8. Allocate pinned host memory (for CUDA Graph path)
        cudaMallocHost(&h_input_, input_size_);
        cudaMallocHost(&h_output_, output_size_);

        // 9. Bind buffers to context
        context_->setTensorAddress("input", d_input_);
        context_->setTensorAddress("output", d_output_);
    }

    ~TRTInfer() {
        if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
        if (graph_) cudaGraphDestroy(graph_);
        if (h_input_) cudaFreeHost(h_input_);
        if (h_output_) cudaFreeHost(h_output_);
        if (d_input_) cudaFree(d_input_);
        if (d_output_) cudaFree(d_output_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    // No copy
    TRTInfer(const TRTInfer&) = delete;
    TRTInfer& operator=(const TRTInfer&) = delete;

    /**
     * Run inference on one sample.
     *
     * @param input  Host pointer to 50*40 floats (one LOB window)
     * @param output Host pointer to 3 floats (logits: down/stable/up)
     */
    void infer(const float* input, float* output) {
        // Copy input from CPU → GPU
        cudaMemcpyAsync(d_input_, input, input_size_, cudaMemcpyHostToDevice, stream_);

        // Execute inference
        context_->enqueueV3(stream_);

        // Copy output from GPU → CPU
        cudaMemcpyAsync(output, d_output_, output_size_, cudaMemcpyDeviceToHost, stream_);

        // Wait for completion
        cudaStreamSynchronize(stream_);
    }

    /**
     * Run inference WITHOUT synchronization (for pipelining).
     * Caller must sync the stream manually.
     */
    void infer_async(const float* input, float* output) {
        cudaMemcpyAsync(d_input_, input, input_size_, cudaMemcpyHostToDevice, stream_);
        context_->enqueueV3(stream_);
        cudaMemcpyAsync(output, d_output_, output_size_, cudaMemcpyDeviceToHost, stream_);
    }

    void sync() {
        cudaStreamSynchronize(stream_);
    }

    cudaStream_t get_stream() const { return stream_; }

    /**
     * Capture the inference pipeline as a CUDA Graph.
     * After capture, call infer_graph() instead of infer() — eliminates
     * per-launch kernel overhead by replaying a pre-recorded command sequence.
     */
    void capture_graph() {
        // Warm up first (TRT may JIT-compile on first run)
        cudaMemcpyAsync(d_input_, d_input_, input_size_, cudaMemcpyDeviceToDevice, stream_);
        context_->enqueueV3(stream_);
        cudaStreamSynchronize(stream_);

        // Begin capture
        cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);

        cudaMemcpyAsync(d_input_, h_input_, input_size_, cudaMemcpyHostToDevice, stream_);
        context_->enqueueV3(stream_);
        cudaMemcpyAsync(h_output_, d_output_, output_size_, cudaMemcpyDeviceToHost, stream_);

        cudaStreamEndCapture(stream_, &graph_);
        cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0);
        graph_captured_ = true;
    }

    /**
     * Run inference via captured CUDA Graph (much lower launch overhead).
     * Must call capture_graph() first.
     * Input: write to pinned_input() before calling.
     * Output: read from pinned_output() after calling.
     */
    void infer_graph() {
        cudaGraphLaunch(graph_exec_, stream_);
        cudaStreamSynchronize(stream_);
    }

    float* pinned_input() { return h_input_; }
    float* pinned_output() { return h_output_; }
    bool has_graph() const { return graph_captured_; }
};
