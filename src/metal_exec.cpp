#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>

#include "../include/metal_exec.h"

#include <unordered_map>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <functional>
#include <iostream>

// ── MSL shader source ────────────────────────────────────────────────────────
static const char* SHADER_SRC = R"msl(
#include <metal_stdlib>
using namespace metal;

// ─── Forward ─────────────────────────────────────────────────────────────────

kernel void matmul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float*       C [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= N || gid.y >= M) return;
    float sum = 0.0f;
    for (uint k = 0; k < K; k++)
        sum += A[gid.y * K + k] * B[k * N + gid.x];
    C[gid.y * N + gid.x] = sum;
}

kernel void matmul_relu(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float*       C [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= N || gid.y >= M) return;
    float sum = 0.0f;
    for (uint k = 0; k < K; k++)
        sum += A[gid.y * K + k] * B[k * N + gid.x];
    C[gid.y * N + gid.x] = sum > 0.0f ? sum : 0.0f;
}

kernel void relu(
    device const float* inp [[buffer(0)]],
    device float*       out [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    out[gid] = inp[gid] > 0.0f ? inp[gid] : 0.0f;
}

// One thread per batch row — numerically stable
kernel void softmax(
    device const float* inp    [[buffer(0)]],
    device float*       out    [[buffer(1)]],
    constant uint& batch       [[buffer(2)]],
    constant uint& classes     [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= batch) return;
    uint b = gid;
    float maxv = inp[b * classes];
    for (uint c = 1; c < classes; c++)
        maxv = max(maxv, inp[b * classes + c]);
    float sum = 0.0f;
    for (uint c = 0; c < classes; c++)
        sum += exp(inp[b * classes + c] - maxv);
    for (uint c = 0; c < classes; c++)
        out[b * classes + c] = exp(inp[b * classes + c] - maxv) / sum;
}

// Single-thread reduction (small batch/classes in practice)
kernel void cross_entropy(
    device float*       output       [[buffer(0)]],
    device const float* input        [[buffer(1)]],
    device const float* ground_truth [[buffer(2)]],
    constant uint& batch             [[buffer(3)]],
    constant uint& classes           [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid > 0) return;
    float total = 0.0f;
    for (uint b = 0; b < batch; b++) {
        float sample = 0.0f;
        for (uint c = 0; c < classes; c++) {
            float pred = input[b * classes + c] + 1e-8f;
            sample += ground_truth[b * classes + c] * log(pred);
        }
        total -= sample;
    }
    output[0] = total / float(batch);
}

kernel void mse(
    device float*       output       [[buffer(0)]],
    device const float* input        [[buffer(1)]],
    device const float* ground_truth [[buffer(2)]],
    constant uint& size              [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid > 0) return;
    float sum = 0.0f;
    for (uint i = 0; i < size; i++) {
        float diff = input[i] - ground_truth[i];
        sum += diff * diff;
    }
    output[0] = sum / float(size);
}

// ─── Backward ────────────────────────────────────────────────────────────────

kernel void relu_backward(
    device float*       input_grad  [[buffer(0)]],
    device const float* output_grad [[buffer(1)]],
    device const float* output      [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    input_grad[gid] += output_grad[gid] * (output[gid] > 0.0f ? 1.0f : 0.0f);
}

// grad_lhs[i,k] += sum_j  output_grad[i,j] * rhs[k,j]
// dispatch (K, M): gid.x = k, gid.y = i
kernel void matmul_backward_lhs(
    device float*       lhs_grad    [[buffer(0)]],
    device const float* output_grad [[buffer(1)]],
    device const float* rhs         [[buffer(2)]],
    constant uint& M                [[buffer(3)]],
    constant uint& K                [[buffer(4)]],
    constant uint& N                [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= K || gid.y >= M) return;
    float sum = 0.0f;
    for (uint j = 0; j < N; j++)
        sum += output_grad[gid.y * N + j] * rhs[gid.x * N + j];
    lhs_grad[gid.y * K + gid.x] += sum;
}

// grad_rhs[k,j] += sum_i  lhs[i,k] * output_grad[i,j]
// dispatch (N, K): gid.x = j, gid.y = k
kernel void matmul_backward_rhs(
    device float*       rhs_grad    [[buffer(0)]],
    device const float* lhs         [[buffer(1)]],
    device const float* output_grad [[buffer(2)]],
    constant uint& M                [[buffer(3)]],
    constant uint& K                [[buffer(4)]],
    constant uint& N                [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= N || gid.y >= K) return;
    float sum = 0.0f;
    for (uint i = 0; i < M; i++)
        sum += lhs[i * K + gid.y] * output_grad[i * N + gid.x];
    rhs_grad[gid.y * N + gid.x] += sum;
}

// Same as matmul_backward_lhs but weighted by relu mask from fused output
kernel void matmul_relu_backward_lhs(
    device float*       lhs_grad    [[buffer(0)]],
    device const float* output_grad [[buffer(1)]],
    device const float* rhs         [[buffer(2)]],
    device const float* output      [[buffer(3)]],
    constant uint& M                [[buffer(4)]],
    constant uint& K                [[buffer(5)]],
    constant uint& N                [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= K || gid.y >= M) return;
    float sum = 0.0f;
    for (uint j = 0; j < N; j++) {
        float mask = output[gid.y * N + j] > 0.0f ? 1.0f : 0.0f;
        sum += output_grad[gid.y * N + j] * mask * rhs[gid.x * N + j];
    }
    lhs_grad[gid.y * K + gid.x] += sum;
}

kernel void matmul_relu_backward_rhs(
    device float*       rhs_grad    [[buffer(0)]],
    device const float* lhs         [[buffer(1)]],
    device const float* output_grad [[buffer(2)]],
    device const float* output      [[buffer(3)]],
    constant uint& M                [[buffer(4)]],
    constant uint& K                [[buffer(5)]],
    constant uint& N                [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= N || gid.y >= K) return;
    float sum = 0.0f;
    for (uint i = 0; i < M; i++) {
        float mask = output[i * N + gid.x] > 0.0f ? 1.0f : 0.0f;
        sum += lhs[i * K + gid.y] * output_grad[i * N + gid.x] * mask;
    }
    rhs_grad[gid.y * N + gid.x] += sum;
}

// One thread per batch row — Jacobian-vector product
kernel void softmax_backward(
    device float*       input_grad  [[buffer(0)]],
    device const float* output_grad [[buffer(1)]],
    device const float* output      [[buffer(2)]],
    constant uint& batch            [[buffer(3)]],
    constant uint& classes          [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= batch) return;
    uint b = gid;
    float dot = 0.0f;
    for (uint c = 0; c < classes; c++)
        dot += output_grad[b * classes + c] * output[b * classes + c];
    for (uint c = 0; c < classes; c++) {
        float s = output[b * classes + c];
        float g = output_grad[b * classes + c];
        input_grad[b * classes + c] += s * (g - dot);
    }
}

// One thread per element
kernel void cross_entropy_backward(
    device float*       input_grad   [[buffer(0)]],
    device const float* output_grad  [[buffer(1)]],
    device const float* input        [[buffer(2)]],
    device const float* ground_truth [[buffer(3)]],
    constant uint& batch             [[buffer(4)]],
    constant uint& classes           [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= batch * classes) return;
    float pred = input[gid] + 1e-8f;
    float grad = (-ground_truth[gid] / pred) * output_grad[0] / float(batch);
    input_grad[gid] += grad;
}

kernel void mse_backward(
    device float*       input_grad   [[buffer(0)]],
    device const float* output_grad  [[buffer(1)]],
    device const float* input        [[buffer(2)]],
    device const float* ground_truth [[buffer(3)]],
    constant uint& size              [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    input_grad[gid] += (2.0f * (input[gid] - ground_truth[gid]) / float(size)) * output_grad[0];
}

// ─── Utility ──────────────────────────────────────────────────────────────────

kernel void sgd_update(
    device float*       storage [[buffer(0)]],
    device const float* grad    [[buffer(1)]],
    constant float& lr          [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    storage[gid] -= lr * grad[gid];
}

kernel void zero_buffer(
    device float* buf [[buffer(0)]],
    uint gid [[thread_position_in_grid]])
{
    buf[gid] = 0.0f;
}
)msl";

// ── Context ───────────────────────────────────────────────────────────────────

struct MetalContext {
    MTL::Device*    device = nullptr;
    MTL::CommandQueue* queue = nullptr;
    MTL::Library*   lib    = nullptr;

    MTL::ComputePipelineState* matmulPSO              = nullptr;
    MTL::ComputePipelineState* matmulReluPSO           = nullptr;
    MTL::ComputePipelineState* reluPSO                 = nullptr;
    MTL::ComputePipelineState* softmaxPSO              = nullptr;
    MTL::ComputePipelineState* crossEntropyPSO         = nullptr;
    MTL::ComputePipelineState* msePSO                  = nullptr;
    MTL::ComputePipelineState* reluBackwardPSO         = nullptr;
    MTL::ComputePipelineState* matmulBackwardLhsPSO    = nullptr;
    MTL::ComputePipelineState* matmulBackwardRhsPSO    = nullptr;
    MTL::ComputePipelineState* matmulReluBackwardLhsPSO = nullptr;
    MTL::ComputePipelineState* matmulReluBackwardRhsPSO = nullptr;
    MTL::ComputePipelineState* softmaxBackwardPSO      = nullptr;
    MTL::ComputePipelineState* crossEntropyBackwardPSO = nullptr;
    MTL::ComputePipelineState* mseBackwardPSO          = nullptr;
    MTL::ComputePipelineState* sgdUpdatePSO            = nullptr;
    MTL::ComputePipelineState* zeroBufferPSO           = nullptr;
};

static MetalContext* g_ctx = nullptr;

// float* (contents ptr) → MTL::Buffer*
static std::unordered_map<float*, MTL::Buffer*> g_bufferMap;

static MTL::ComputePipelineState* makePSO(MTL::Device* dev, MTL::Library* lib, const char* name) {
    NS::String* fname = NS::String::string(name, NS::UTF8StringEncoding);
    MTL::Function* fn = lib->newFunction(fname);
    if (!fn) {
        throw std::runtime_error(std::string("Metal: function not found: ") + name);
    }
    NS::Error* err = nullptr;
    MTL::ComputePipelineState* pso = dev->newComputePipelineState(fn, &err);
    if (!pso) {
        throw std::runtime_error(std::string("Metal: PSO creation failed for: ") + name);
    }
    fn->release();
    return pso;
}

static MetalContext& getCtx() {
    if (g_ctx) return *g_ctx;

    g_ctx = new MetalContext();
    g_ctx->device = MTLCreateSystemDefaultDevice();
    if (!g_ctx->device) throw std::runtime_error("Metal: no device found");

    std::cout << "[Metal] GPU: " << g_ctx->device->name()->utf8String() << "\n";

    g_ctx->queue = g_ctx->device->newCommandQueue();

    NS::String* src = NS::String::string(SHADER_SRC, NS::UTF8StringEncoding);
    NS::Error* err  = nullptr;
    g_ctx->lib = g_ctx->device->newLibrary(src, nullptr, &err);
    if (!g_ctx->lib) {
        throw std::runtime_error(std::string("Metal: shader compile failed: ") +
                                 err->localizedDescription()->utf8String());
    }

    g_ctx->matmulPSO               = makePSO(g_ctx->device, g_ctx->lib, "matmul");
    g_ctx->matmulReluPSO           = makePSO(g_ctx->device, g_ctx->lib, "matmul_relu");
    g_ctx->reluPSO                 = makePSO(g_ctx->device, g_ctx->lib, "relu");
    g_ctx->softmaxPSO              = makePSO(g_ctx->device, g_ctx->lib, "softmax");
    g_ctx->crossEntropyPSO         = makePSO(g_ctx->device, g_ctx->lib, "cross_entropy");
    g_ctx->msePSO                  = makePSO(g_ctx->device, g_ctx->lib, "mse");
    g_ctx->reluBackwardPSO         = makePSO(g_ctx->device, g_ctx->lib, "relu_backward");
    g_ctx->matmulBackwardLhsPSO    = makePSO(g_ctx->device, g_ctx->lib, "matmul_backward_lhs");
    g_ctx->matmulBackwardRhsPSO    = makePSO(g_ctx->device, g_ctx->lib, "matmul_backward_rhs");
    g_ctx->matmulReluBackwardLhsPSO = makePSO(g_ctx->device, g_ctx->lib, "matmul_relu_backward_lhs");
    g_ctx->matmulReluBackwardRhsPSO = makePSO(g_ctx->device, g_ctx->lib, "matmul_relu_backward_rhs");
    g_ctx->softmaxBackwardPSO      = makePSO(g_ctx->device, g_ctx->lib, "softmax_backward");
    g_ctx->crossEntropyBackwardPSO = makePSO(g_ctx->device, g_ctx->lib, "cross_entropy_backward");
    g_ctx->mseBackwardPSO          = makePSO(g_ctx->device, g_ctx->lib, "mse_backward");
    g_ctx->sgdUpdatePSO            = makePSO(g_ctx->device, g_ctx->lib, "sgd_update");
    g_ctx->zeroBufferPSO           = makePSO(g_ctx->device, g_ctx->lib, "zero_buffer");

    return *g_ctx;
}

// ── Internal helpers ──────────────────────────────────────────────────────────

static MTL::Buffer* getBuf(float* ptr) {
    auto it = g_bufferMap.find(ptr);
    if (it == g_bufferMap.end())
        throw std::runtime_error("metalExec: unregistered pointer — was metalMalloc called?");
    return it->second;
}

static void dispatch(MTL::ComputePipelineState* pso,
                     std::function<void(MTL::ComputeCommandEncoder*)> setup,
                     MTL::Size threads, MTL::Size tg) {
    auto& ctx = getCtx();
    MTL::CommandBuffer*         cmd = ctx.queue->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    setup(enc);
    enc->dispatchThreads(threads, tg);
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
}

static void dispatch1D(MTL::ComputePipelineState* pso,
                       std::function<void(MTL::ComputeCommandEncoder*)> setup,
                       int n) {
    uint tn = (uint)std::min(n, 256);
    dispatch(pso, setup, MTL::Size(n, 1, 1), MTL::Size(tn, 1, 1));
}

static void dispatch2D(MTL::ComputePipelineState* pso,
                       std::function<void(MTL::ComputeCommandEncoder*)> setup,
                       int w, int h) {
    uint tw = (uint)std::min(w, 16);
    uint th = (uint)std::min(h, 16);
    dispatch(pso, setup, MTL::Size(w, h, 1), MTL::Size(tw, th, 1));
}

// ── Memory ────────────────────────────────────────────────────────────────────

void metalMalloc(float** ptr, size_t bytes) {
    auto& ctx = getCtx();
    MTL::Buffer* buf = ctx.device->newBuffer(bytes, MTL::ResourceStorageModeShared);
    if (!buf) throw std::runtime_error("metalMalloc: allocation failed");
    *ptr = static_cast<float*>(buf->contents());
    g_bufferMap[*ptr] = buf;
}

void metalFree(float* ptr) {
    auto it = g_bufferMap.find(ptr);
    if (it != g_bufferMap.end()) {
        it->second->release();
        g_bufferMap.erase(it);
    }
}

// Unified memory — CPU and GPU share the same physical memory on M1
void metalCopyToDevice(float* d_dst, const float* h_src, size_t bytes) {
    std::memcpy(d_dst, h_src, bytes);
}

void metalCopyToHost(float* h_dst, const float* d_src, size_t bytes) {
    std::memcpy(h_dst, d_src, bytes);
}

void metalZeroDevice(float* ptr, int size) {
    auto& ctx = getCtx();
    dispatch1D(ctx.zeroBufferPSO, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(getBuf(ptr), 0, 0);
    }, size);
}

// ── Forward ───────────────────────────────────────────────────────────────────

void metalMatmulDevice(float* C, float* A, float* B, int M, int N, int K) {
    auto& ctx = getCtx();
    uint uM = M, uN = N, uK = K;
    // dispatch (N, M): gid.x = col, gid.y = row
    dispatch2D(ctx.matmulPSO, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(getBuf(A), 0, 0);
        enc->setBuffer(getBuf(B), 0, 1);
        enc->setBuffer(getBuf(C), 0, 2);
        enc->setBytes(&uM, sizeof(uint), 3);
        enc->setBytes(&uN, sizeof(uint), 4);
        enc->setBytes(&uK, sizeof(uint), 5);
    }, N, M);
}

void metalReluDevice(float* output, float* input, int size) {
    auto& ctx = getCtx();
    dispatch1D(ctx.reluPSO, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(getBuf(input),  0, 0);
        enc->setBuffer(getBuf(output), 0, 1);
    }, size);
}

void metalMatmulReluDevice(float* C, float* A, float* B, int M, int N, int K) {
    auto& ctx = getCtx();
    uint uM = M, uN = N, uK = K;
    dispatch2D(ctx.matmulReluPSO, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(getBuf(A), 0, 0);
        enc->setBuffer(getBuf(B), 0, 1);
        enc->setBuffer(getBuf(C), 0, 2);
        enc->setBytes(&uM, sizeof(uint), 3);
        enc->setBytes(&uN, sizeof(uint), 4);
        enc->setBytes(&uK, sizeof(uint), 5);
    }, N, M);
}

void metalSoftmaxDevice(float* output, float* input, int batch, int classes) {
    auto& ctx = getCtx();
    uint ub = batch, uc = classes;
    // one thread per batch row
    dispatch1D(ctx.softmaxPSO, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(getBuf(input),  0, 0);
        enc->setBuffer(getBuf(output), 0, 1);
        enc->setBytes(&ub, sizeof(uint), 2);
        enc->setBytes(&uc, sizeof(uint), 3);
    }, batch);
}

void metalCrossEntropyDevice(float* output, float* input, float* ground_truth,
                              int batch, int classes) {
    auto& ctx = getCtx();
    uint ub = batch, uc = classes;
    dispatch1D(ctx.crossEntropyPSO, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(getBuf(output),       0, 0);
        enc->setBuffer(getBuf(input),        0, 1);
        enc->setBuffer(getBuf(ground_truth), 0, 2);
        enc->setBytes(&ub, sizeof(uint), 3);
        enc->setBytes(&uc, sizeof(uint), 4);
    }, 1);
}

void metalMseDevice(float* output, float* input, float* ground_truth, int size) {
    auto& ctx = getCtx();
    uint us = size;
    dispatch1D(ctx.msePSO, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(getBuf(output),       0, 0);
        enc->setBuffer(getBuf(input),        0, 1);
        enc->setBuffer(getBuf(ground_truth), 0, 2);
        enc->setBytes(&us, sizeof(uint), 3);
    }, 1);
}

// ── Backward ──────────────────────────────────────────────────────────────────

void metalReluBackwardDevice(float* input_grad, float* output_grad, float* output, int size) {
    auto& ctx = getCtx();
    dispatch1D(ctx.reluBackwardPSO, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(getBuf(input_grad),  0, 0);
        enc->setBuffer(getBuf(output_grad), 0, 1);
        enc->setBuffer(getBuf(output),      0, 2);
    }, size);
}

void metalMatmulBackwardDevice(float* lhs_grad, float* rhs_grad, float* output_grad,
                               float* lhs, float* rhs, int M, int K, int N) {
    auto& ctx = getCtx();
    uint uM = M, uK = K, uN = N;

    // grad_lhs[M,K]: dispatch (K, M) — gid.x=k, gid.y=i
    dispatch2D(ctx.matmulBackwardLhsPSO, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(getBuf(lhs_grad),    0, 0);
        enc->setBuffer(getBuf(output_grad), 0, 1);
        enc->setBuffer(getBuf(rhs),         0, 2);
        enc->setBytes(&uM, sizeof(uint), 3);
        enc->setBytes(&uK, sizeof(uint), 4);
        enc->setBytes(&uN, sizeof(uint), 5);
    }, K, M);

    // grad_rhs[K,N]: dispatch (N, K) — gid.x=j, gid.y=k
    dispatch2D(ctx.matmulBackwardRhsPSO, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(getBuf(rhs_grad),    0, 0);
        enc->setBuffer(getBuf(lhs),         0, 1);
        enc->setBuffer(getBuf(output_grad), 0, 2);
        enc->setBytes(&uM, sizeof(uint), 3);
        enc->setBytes(&uK, sizeof(uint), 4);
        enc->setBytes(&uN, sizeof(uint), 5);
    }, N, K);
}

void metalMatmulReluBackwardDevice(float* lhs_grad, float* rhs_grad, float* output_grad,
                                   float* lhs, float* rhs, float* output, int M, int K, int N) {
    auto& ctx = getCtx();
    uint uM = M, uK = K, uN = N;

    dispatch2D(ctx.matmulReluBackwardLhsPSO, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(getBuf(lhs_grad),    0, 0);
        enc->setBuffer(getBuf(output_grad), 0, 1);
        enc->setBuffer(getBuf(rhs),         0, 2);
        enc->setBuffer(getBuf(output),      0, 3);
        enc->setBytes(&uM, sizeof(uint), 4);
        enc->setBytes(&uK, sizeof(uint), 5);
        enc->setBytes(&uN, sizeof(uint), 6);
    }, K, M);

    dispatch2D(ctx.matmulReluBackwardRhsPSO, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(getBuf(rhs_grad),    0, 0);
        enc->setBuffer(getBuf(lhs),         0, 1);
        enc->setBuffer(getBuf(output_grad), 0, 2);
        enc->setBuffer(getBuf(output),      0, 3);
        enc->setBytes(&uM, sizeof(uint), 4);
        enc->setBytes(&uK, sizeof(uint), 5);
        enc->setBytes(&uN, sizeof(uint), 6);
    }, N, K);
}

void metalSoftmaxBackwardDevice(float* input_grad, float* output_grad, float* output,
                                int batch, int classes) {
    auto& ctx = getCtx();
    uint ub = batch, uc = classes;
    dispatch1D(ctx.softmaxBackwardPSO, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(getBuf(input_grad),  0, 0);
        enc->setBuffer(getBuf(output_grad), 0, 1);
        enc->setBuffer(getBuf(output),      0, 2);
        enc->setBytes(&ub, sizeof(uint), 3);
        enc->setBytes(&uc, sizeof(uint), 4);
    }, batch);
}

void metalCrossEntropyBackwardDevice(float* input_grad, float* output_grad,
                                     float* input, float* ground_truth,
                                     int batch, int classes) {
    auto& ctx = getCtx();
    uint ub = batch, uc = classes;
    dispatch1D(ctx.crossEntropyBackwardPSO, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(getBuf(input_grad),   0, 0);
        enc->setBuffer(getBuf(output_grad),  0, 1);
        enc->setBuffer(getBuf(input),        0, 2);
        enc->setBuffer(getBuf(ground_truth), 0, 3);
        enc->setBytes(&ub, sizeof(uint), 4);
        enc->setBytes(&uc, sizeof(uint), 5);
    }, batch * classes);
}

void metalMseBackwardDevice(float* input_grad, float* output_grad,
                            float* input, float* ground_truth, int size) {
    auto& ctx = getCtx();
    uint us = size;
    dispatch1D(ctx.mseBackwardPSO, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(getBuf(input_grad),   0, 0);
        enc->setBuffer(getBuf(output_grad),  0, 1);
        enc->setBuffer(getBuf(input),        0, 2);
        enc->setBuffer(getBuf(ground_truth), 0, 3);
        enc->setBytes(&us, sizeof(uint), 4);
    }, size);
}

// ── Training ──────────────────────────────────────────────────────────────────

void metalSgdUpdateDevice(float* storage, float* grad, float lr, int size) {
    auto& ctx = getCtx();
    dispatch1D(ctx.sgdUpdatePSO, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(getBuf(storage), 0, 0);
        enc->setBuffer(getBuf(grad),    0, 1);
        enc->setBytes(&lr, sizeof(float), 2);
    }, size);
}
