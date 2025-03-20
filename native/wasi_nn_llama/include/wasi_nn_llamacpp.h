#ifndef WASI_NN_LLAMACPP_H
#define WASI_NN_LLAMACPP_H

#include "wasi_nn_types.h"
#include <stdint.h>
#include <stdbool.h>

// Forward declarations for llama types
struct llama_context;
struct llama_model;
typedef int llama_token;
typedef int llama_pos;
typedef int llama_seq_id;

// Configuration structure for llama model
struct wasi_nn_llama_config {
    // Backend parameters
    bool enable_log;
    bool enable_debug_log;
    bool stream_stdout;
    bool embedding;
    int32_t n_predict;
    char *reverse_prompt;

    // LLaVA parameters
    char *mmproj;
    char *image;

    // Model parameters
    int32_t n_gpu_layers;
    int32_t main_gpu;
    float *tensor_split;
    bool use_mmap;

    // Context parameters
    uint32_t ctx_size;
    uint32_t batch_size;
    uint32_t ubatch_size;
    uint32_t threads;

    // Sampling parameters
    float temp;
    float topP;
    float repeat_penalty;
    float presence_penalty;
    float frequency_penalty;
};

// Context structure for llama model
struct LlamaContext {
    struct llama_context *ctx;
    struct llama_model *model;
    llama_token *prompt;
    size_t prompt_len;
    llama_token *generation;
    size_t generation_len;
    struct wasi_nn_llama_config config;
};

// Public API functions
__attribute__((visibility("default"))) wasi_nn_error init_backend(void **ctx);
__attribute__((visibility("default"))) wasi_nn_error deinit_backend(void *ctx);
__attribute__((visibility("default"))) wasi_nn_error load(void *ctx, graph_builder_array *builder, 
                                                         graph_encoding encoding, execution_target target, 
                                                         graph *g);
__attribute__((visibility("default"))) wasi_nn_error load_by_name(void *ctx, const char *filename, 
                                                                 uint32_t filename_len, graph *g);
__attribute__((visibility("default"))) wasi_nn_error load_by_name_with_config(void *ctx, const char *filename, 
                                                                             uint32_t filename_len, const char *config, 
                                                                             uint32_t config_len, graph *g);
__attribute__((visibility("default"))) wasi_nn_error init_execution_context(void *ctx, graph g, 
                                                                          graph_execution_context *exec_ctx);
__attribute__((visibility("default"))) wasi_nn_error set_input(void *ctx, graph_execution_context exec_ctx, 
                                                              uint32_t index, tensor *input_tensor);
__attribute__((visibility("default"))) wasi_nn_error compute(void *ctx, graph_execution_context exec_ctx);
__attribute__((visibility("default"))) wasi_nn_error get_output(void *ctx, graph_execution_context exec_ctx, 
                                                               uint32_t index, tensor_data output_tensor, 
                                                               uint32_t *output_tensor_size);

#endif // WASI_NN_LLAMACPP_H