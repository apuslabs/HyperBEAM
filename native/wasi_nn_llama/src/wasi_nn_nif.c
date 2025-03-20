#include <erl_nif.h>
#include <stdio.h>
#include <string.h>
#include <wasi_nn.h>
//#include "../include/wasi_nn_llamacpp.h"

// Define the NIF context structure
typedef struct {
    void* ctx;
    graph g;
    graph_execution_context exec_ctx;
    // Define a simple config structure if the original is not available
    struct {
        bool enable_log;
        bool enable_debug_log;
        bool stream_stdout;
        bool embedding;
        int32_t n_predict;
        char *reverse_prompt;
        char *mmproj;
        char *image;
        int32_t n_gpu_layers;
        int32_t main_gpu;
        float *tensor_split;
        bool use_mmap;
        uint32_t ctx_size;
        uint32_t batch_size;
        uint32_t ubatch_size;
        uint32_t threads;
        float temp;
        float topP;
        float repeat_penalty;
        float presence_penalty;
        float frequency_penalty;
    } config;
} LlamaNifContext;

// Resource type for the context
static ErlNifResourceType* llama_context_resource;


// Resource destructor
static void llama_context_destructor(ErlNifEnv* env, void* obj)
{
    LlamaNifContext* nif_ctx = (LlamaNifContext*)obj;
    if (nif_ctx->ctx) {
        // deinit_backend(nif_ctx->ctx);
    }
}
static int Testload(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info)
{
    printf("Load nif\n");
    llama_context_resource = enif_open_resource_type(env, NULL, "llama_context",
        llama_context_destructor, ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER, NULL);
    return 0;
}
static void unload(ErlNifEnv* env, void* priv_data)
{
    // Cleanup code if needed
}

// Function implementations


static ERL_NIF_TERM nif_load_model(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    printf("Loading model...\n");
    char model_path[256];
    LlamaNifContext* nif_ctx;
    
    if (!enif_get_string(env, argv[0], model_path, sizeof(model_path), ERL_NIF_LATIN1)) {
        return enif_make_tuple2(env, 
            enif_make_atom(env, "error"),
            enif_make_atom(env, "invalid_path"));
    }

    nif_ctx = enif_alloc_resource(llama_context_resource, sizeof(LlamaNifContext));
    if (!nif_ctx) {
        return enif_make_tuple2(env,
            enif_make_atom(env, "error"),
            enif_make_atom(env, "allocation_failed"));
    }

    // Initialize backend first
    if (load_by_name(model_path, &nif_ctx->g) != success) {
        enif_release_resource(nif_ctx);
        return enif_make_tuple2(env,
            enif_make_atom(env, "error"),
            enif_make_atom(env, "init_failed"));
    }

    // Initialize the config with default values
    nif_ctx->config.enable_log = true;
    nif_ctx->config.enable_debug_log = false;
    nif_ctx->config.n_predict = 512;
    nif_ctx->config.ctx_size = 2048;
    nif_ctx->config.threads = 4;

    // if (load_by_name(nif_ctx->ctx, model_path, strlen(model_path), &nif_ctx->g) != success) {
    //     enif_release_resource(nif_ctx);
    //     return enif_make_tuple2(env,
    //         enif_make_atom(env, "error"),
    //         enif_make_atom(env, "load_failed"));
    // }

    ERL_NIF_TERM result = enif_make_resource(env, nif_ctx);
    enif_release_resource(nif_ctx);
    return enif_make_tuple2(env,
        enif_make_atom(env, "ok"),
        result);
}

static ErlNifFunc nif_funcs[] = {
    {"load_model", 1, nif_load_model},
};

ERL_NIF_INIT(dev_wasi_nn_nif, nif_funcs, Testload, unload, NULL, NULL)