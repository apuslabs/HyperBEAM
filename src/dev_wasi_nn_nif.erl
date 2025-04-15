-module(dev_wasi_nn_nif).
-include("include/hb.hrl").
-include_lib("eunit/include/eunit.hrl").
-on_load(init/0).
-export([run_inference/2,initialize_llama_runtime/2,cleanup_llama_runtime/1]).
-hb_debug(print).
init() ->
    PrivDir = code:priv_dir(hb),
    Path = filename:join(PrivDir, "wasi_nn"),
    io:format("Loading NIF from: ~p~n", [Path]),
    case erlang:load_nif(Path, 0) of
        ok ->
            io:format("NIF loaded successfully~n"),
            ok;
        {error, {load_failed, Reason}} ->
            io:format("Failed to load NIF: ~p~n", [Reason]),
            exit({load_failed, {load_failed, Reason}});
        {error, Reason} ->
            io:format("Failed to load NIF with error: ~p~n", [Reason]),
            exit({load_failed, Reason})
    end.



initialize_llama_runtime(_Path, _Config) ->
    erlang:nif_error("NIF library not loaded").

cleanup_llama_runtime(Context) ->
    erlang:nif_error("NIF library not loaded").

run_inference(Context,Prompt) ->
	erlang:nif_error("NIF library not loaded").

run_inference_test() ->
	ModelPath = "test/Qwen2.5-1.5B-Instruct.Q2_K.gguf",
    Config = "{\"n_gpu_layers\":98,\"n_ctx\":2048,\"stream-stdout\":true,\"enable_debug_log\":true}",
	Prompt = "Hello, who are you ",
	{ok, Context} = dev_wasi_nn_nif:initialize_llama_runtime(ModelPath, Config),
	{ok, Output} = dev_wasi_nn_nif:run_inference(Context,Prompt),
	ok = dev_wasi_nn_nif:cleanup_llama_runtime(Context),
	?assertNotEqual("",Output).