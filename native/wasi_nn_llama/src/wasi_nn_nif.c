#include "../include/wasi_nn_llamacpp.h"
#include "../include/wasi_nn_logging.h"
#define LIB_PATH "./native/wasi_nn_llama/libllama_runtime.so"
#define MAX_MODEL_PATH 256
#define MAX_INPUT_SIZE 4096
#define MAX_CONFIG_SIZE 1024
#define MAX_OUTPUT_SIZE 4096
#define ERROR_BUFFER_SIZE 256

static llama_backend_apis llama_backend_functions = {0};
static ErlNifResourceType* llama_context_resource;

typedef struct {
    LlamaHandle llamahandle;
} LlamaContext;

static void llama_context_destructor(ErlNifEnv* env, void* obj)
{
    LlamaContext* ctx = (LlamaContext*)obj;
	if (ctx) {
        // Cleanup  context
        if (ctx->llamahandle && llama_backend_functions.cleanup_llama_runtime) {
            llama_backend_functions.cleanup_llama_runtime(ctx->llamahandle);
            ctx->llamahandle = NULL;
        }
        memset(ctx, 0, sizeof(LlamaContext));
    }
}



static int load(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info)
{
    DRV_PRINT("Load nif start");
	
	llama_backend_functions.handle = dlopen(LIB_PATH, RTLD_LAZY);
    if (!llama_backend_functions.handle) {
        DRV_PRINT("Failed to load wasi library: %s", dlerror());
        return 1;
    }
	// Load all required functions once
	llama_backend_functions.initialize_llama_runtime = (initialize_llama_runtime_fn)dlsym(llama_backend_functions.handle, "initialize_llama_runtime");
    llama_backend_functions.run_inference = (run_inference_fn)dlsym(llama_backend_functions.handle, "run_inference");
    llama_backend_functions.cleanup_llama_runtime = (cleanup_llama_runtime_fn)dlsym(llama_backend_functions.handle, "cleanup_llama_runtime");
    if (!llama_backend_functions.initialize_llama_runtime ||
		!llama_backend_functions.run_inference ||
		!llama_backend_functions.cleanup_llama_runtime 
		)
	{
        dlclose(llama_backend_functions.handle);
        return 1;
    }
	
	llama_context_resource = enif_open_resource_type(env, NULL, "llama_context",
        llama_context_destructor, ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER, NULL);

    return llama_context_resource ? 0 : 1;
}




static ERL_NIF_TERM nif_init_llama(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{

	LlamaContext* ctx = enif_alloc_resource(llama_context_resource, sizeof(LlamaContext));
    if (!ctx) {
        DRV_PRINT("Failed to allocate LlamaHandle resource");
        return enif_make_tuple2(env, enif_make_atom(env, "error"), 
                              enif_make_atom(env, "allocation_failed"));
    }
	DRV_PRINT("Initializing llama...");

	char *model_path = (char *)malloc(MAX_MODEL_PATH * sizeof(char));
	char *config = (char *)malloc(MAX_CONFIG_SIZE * sizeof(char));
	char *error_buffer = (char *)malloc(ERROR_BUFFER_SIZE * sizeof(char));
	// parse model path
    if (!enif_get_string(env, argv[0], model_path, MAX_MODEL_PATH, ERL_NIF_LATIN1)) {
		free(model_path);
        free(config);
        free(error_buffer);
		return enif_make_tuple2(env, enif_make_atom(env, "error"), 
                              enif_make_atom(env, "invalid_model_path"));
    }
	DRV_PRINT("Loading model: %s\n ", model_path);
	// parse config path
	if (!enif_get_string(env, argv[1], config, MAX_CONFIG_SIZE, ERL_NIF_LATIN1)) {
		free(model_path);
        free(config);
        free(error_buffer);
		return enif_make_tuple2(env, enif_make_atom(env, "error"),
                              enif_make_atom(env, "invalid_config"));
    }

	DRV_PRINT("Loading model: %s\n Configs %s\n", model_path, config);
	ctx->llamahandle = llama_backend_functions.initialize_llama_runtime(model_path, config, error_buffer, ERROR_BUFFER_SIZE);
	
    if (!ctx) {
        DRV_PRINT("Backend initialization failed with error: %s", error_buffer);
        enif_release_resource(ctx);
        return enif_make_tuple2(env, enif_make_atom(env, "error"), 
                              enif_make_atom(env, "init_failed"));
    }
	DRV_PRINT("nif_init_backend finished");

    ERL_NIF_TERM ctx_term = enif_make_resource(env, ctx);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), ctx_term);
}

static ERL_NIF_TERM nif_run_inference(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
	//check context
    LlamaContext *ctx;
	if(!enif_get_resource(env, argv[0], llama_context_resource, (void**)&ctx))
	{
		DRV_PRINT("Invalid context\n");
		return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "invalid_context"));
	}

	char *prompt = (char *)malloc(MAX_INPUT_SIZE * sizeof(char));
	char *result_buffer = (char *)malloc(MAX_INPUT_SIZE * sizeof(char));
	char *error_buffer = (char *)malloc(ERROR_BUFFER_SIZE * sizeof(char));
	// get prompt
    if (!enif_get_string(env, argv[1], prompt, MAX_INPUT_SIZE, ERL_NIF_LATIN1)) {
        
		return enif_make_tuple2(env, enif_make_atom(env, "error"), 
                              enif_make_atom(env, "invalid_prompt_input"));
    }
	// get result buffer

	DRV_PRINT("Prompt : %s \n", prompt);

	if(!llama_backend_functions.run_inference(ctx->llamahandle, prompt, result_buffer, MAX_OUTPUT_SIZE, error_buffer, ERROR_BUFFER_SIZE))
	{
		DRV_PRINT("run_inference failed with error: %s", error_buffer);
		return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "run_inference_failed"));
	}
	// Create a new binary term in Erlang
    ERL_NIF_TERM result_bin;
    unsigned char* bin_data = enif_make_new_binary(env, MAX_OUTPUT_SIZE, &result_bin);
    if (!bin_data) {
        free(result_buffer);
        return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "binary_creation_failed"));
    }
	// print output_buffer
	DRV_PRINT("Output: %s\n", result_buffer);
    // Copy the output_buffer into the Erlang binary
    memcpy(bin_data, result_buffer, MAX_OUTPUT_SIZE);

    // Free the output_buffer as it's no longer needed
    free(result_buffer);

    return enif_make_tuple2(env,
        enif_make_atom(env, "ok"),  
		result_bin);

}
static ERL_NIF_TERM nif_cleanup_llama(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
	DRV_PRINT("in clean up");
    LlamaContext *ctx;
    if (!enif_get_resource(env, argv[0], llama_context_resource, (void**)&ctx)) {
        return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "invalid_args"));
    }
	llama_backend_functions.cleanup_llama_runtime(ctx->llamahandle);
    return enif_make_atom(env, "ok");
}
static ErlNifFunc nif_funcs[] = {
    {"initialize_llama_runtime", 2, nif_init_llama},
    {"run_inference", 2, nif_run_inference},
    {"cleanup_llama_runtime", 1, nif_cleanup_llama}
};


static void unload(ErlNifEnv* env, void* priv_data)
{
	// The resource destructor will be called automatically for any remaining resources

}
ERL_NIF_INIT(dev_wasi_nn_nif, nif_funcs, load, NULL, NULL, unload)