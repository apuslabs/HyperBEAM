// Replace wasi_nn.h with wasi_nn_llamacpp.h
#include <erl_nif.h>
#include <string.h>
#include <stdio.h>
#include "../include/wasi_nn_llamacpp.h"

// ... rest of the code ...
typedef struct {
    void* ctx;
    graph g;
    graph_execution_context exec_ctx;
    struct wasi_nn_llama_config config;
} LlamaNifContext;

static ErlNifResourceType* llama_context_resource;
// Add destructor implementation
static void llama_context_destructor(ErlNifEnv* env, void* obj)
{
    LlamaNifContext* nif_ctx = (LlamaNifContext*)obj;
    if (nif_ctx->ctx) {
        deinit_backend(nif_ctx->ctx);
    }
}
static int nif_load(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info)
{
    printf("Load nif\n");
    llama_context_resource = enif_open_resource_type(env, NULL, "llama_context",
        llama_context_destructor, ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER, NULL);
    return 0;
}
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
    if (init_backend(&nif_ctx->ctx) != success) {
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

    // Use the wrapper's load_by_name which takes ctx as first parameter
    if (load_by_name(nif_ctx->ctx, model_path, strlen(model_path), &nif_ctx->g) != success) {
        deinit_backend(nif_ctx->ctx);
        enif_release_resource(nif_ctx);
        return enif_make_tuple2(env,
            enif_make_atom(env, "error"),
            enif_make_atom(env, "load_failed"));
    }

    // Use the wrapper's init_execution_context
    if (init_execution_context(nif_ctx->ctx, nif_ctx->g, &nif_ctx->exec_ctx) != success) {
        deinit_backend(nif_ctx->ctx);
        enif_release_resource(nif_ctx);
        return enif_make_tuple2(env,
            enif_make_atom(env, "error"),
            enif_make_atom(env, "init_context_failed"));
    }

    ERL_NIF_TERM result = enif_make_resource(env, nif_ctx);
    enif_release_resource(nif_ctx);
    return enif_make_tuple2(env,
        enif_make_atom(env, "ok"),
        result);
}



static ErlNifFunc nif_funcs[] = {
    {"load_model", 1, nif_load_model},
};

ERL_NIF_INIT(dev_wasi_nn_nif, nif_funcs, nif_load, NULL, NULL, NULL)