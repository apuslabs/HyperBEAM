#ifndef WASI_NN_LLAMACPP_H
#define WASI_NN_LLAMACPP_H
#include "llama_runtime.h"
#include <erl_nif.h>
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>




// Function pointer types
typedef LlamaHandle (*initialize_llama_runtime_fn)(const char* model_path, const char* config, char* error_buffer , size_t ERROR_BUFFER_SIZE);
typedef bool (*run_inference_fn)(LlamaHandle, const char*, char*, size_t, char*, size_t);
typedef void (*cleanup_llama_runtime_fn)(LlamaHandle);

// Structure to hold all function pointers
typedef struct {
    void* handle;
    initialize_llama_runtime_fn initialize_llama_runtime;
    run_inference_fn run_inference;
    cleanup_llama_runtime_fn cleanup_llama_runtime;
} llama_backend_apis;

#endif // WASI_NN_LLAMACPP_H