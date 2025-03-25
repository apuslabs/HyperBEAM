#ifndef WASI_NN_LLAMACPP_H
#define WASI_NN_LLAMACPP_H

#include "wasi_nn_types.h"
#include <erl_nif.h>
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>

// Function pointer types
typedef wasi_nn_error (*init_backend_fn)(void **ctx);
typedef wasi_nn_error (*deinit_backend_fn)(void *ctx);
typedef wasi_nn_error (*load_by_name_fn)(void *ctx, const char *filename, uint32_t filename_len, graph *g);
typedef wasi_nn_error (*init_execution_context_fn)(void *ctx, graph g, graph_execution_context *exec_ctx);
typedef wasi_nn_error (*set_input_fn)(void *ctx, graph_execution_context exec_ctx, uint32_t index, tensor *tensor);
typedef wasi_nn_error (*compute_fn)(void *ctx, graph_execution_context exec_ctx);
typedef wasi_nn_error (*get_output_fn)(void *ctx, graph_execution_context exec_ctx, uint32_t index, tensor_data output, uint32_t *output_size);
typedef wasi_nn_error (*load_by_name_with_config_fn)(void *ctx, const char *filename, uint32_t filename_len,
	const char *config, uint32_t config_len, graph *g);
// Structure to hold all function pointers
typedef struct {
    void* handle;
    init_backend_fn init_backend;
    deinit_backend_fn deinit_backend;
    load_by_name_fn load_by_name;
    init_execution_context_fn init_execution_context;
    set_input_fn set_input;
    compute_fn compute;
    get_output_fn get_output;
	load_by_name_with_config_fn load_by_name_with_config;
} WasiNnFunctions;

#endif // WASI_NN_LLAMACPP_H