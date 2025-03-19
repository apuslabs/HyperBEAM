#include <erl_nif.h>
#include <string.h>
#include "../include/wasi_nn_nif.h"

static ErlNifResourceType* llama_context_resource;

typedef struct {
    void* ctx;
    graph g;
    graph_execution_context exec_ctx;
} LlamaNifContext;

static void
llama_context_destructor(ErlNifEnv* env, void* obj)
{
    LlamaNifContext* nif_ctx = (LlamaNifContext*)obj;
    if (nif_ctx->ctx) {
        deinit_backend(nif_ctx->ctx);
    }
}

static int
load(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info)
{
    llama_context_resource = enif_open_resource_type(env, NULL, "llama_context",
        llama_context_destructor, ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER, NULL);
    return 0;
}

static ERL_NIF_TERM
nif_load_model(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    char model_path[256];
    LlamaNifContext* nif_ctx;
    
    if (!enif_get_string(env, argv[0], model_path, sizeof(model_path), ERL_NIF_LATIN1))
        return enif_make_badarg(env);

    nif_ctx = enif_alloc_resource(llama_context_resource, sizeof(LlamaNifContext));
    if (init_backend(&nif_ctx->ctx) != success)
        return enif_make_atom(env, "error");

    if (load_by_name(nif_ctx->ctx, model_path, strlen(model_path), &nif_ctx->g) != success)
        return enif_make_atom(env, "error");

    ERL_NIF_TERM result = enif_make_resource(env, nif_ctx);
    enif_release_resource(nif_ctx);
    return result;
}

// Add other NIF functions here...

static ErlNifFunc nif_funcs[] = {
    {"load_model", 1, nif_load_model},
    // Add other functions...
};

ERL_NIF_INIT(wasi_nn_llama, nif_funcs, load, NULL, NULL, NULL)