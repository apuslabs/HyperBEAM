-module(dev_wasi_nn_nif).
-include("include/hb.hrl").
-include_lib("eunit/include/eunit.hrl").
-on_load(init/0).
-export([init_backend/0,load_by_name_with_config/3,init_execution_context/1,set_input/2,get_output/1,compute/1,deinit_backend/1,run_inference/2]).

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

init_backend() ->
    erlang:nif_error("NIF library not loaded").

load_by_name_with_config(_Context, _Path, _Config) ->
    erlang:nif_error("NIF library not loaded").

init_execution_context(_Context) ->
    erlang:nif_error("NIF library not loaded").

set_input(_Context, _Prompt) ->
    erlang:nif_error("NIF library not loaded").

compute(_Context) ->
    erlang:nif_error("NIF library not loaded").
get_output(_Context) ->
    erlang:nif_error("NIF library not loaded").
deinit_backend(_Context) ->
    erlang:nif_error("NIF library not loaded").
run_inference(_Context,_Prompt) ->
	erlang:nif_error("NIF library not loaded").

run_inference_test() ->
	Path = "test/Qwen2.5-1.5B-Instruct.Q2_K.gguf",
	Config = "{\"n_gpu_layers\":98,\"ctx_size\":2048,\"stream-stdout\":true,\"enable_debug_log\":true}",
	Prompt = "What is the meaning of life",
	{ok, Context} = init_backend(),
	?event({load_model, Context, Path, Config}),
	load_by_name_with_config(Context, Path, Config),
	init_execution_context(Context),
	{ok, Output} = run_inference(Context,Prompt),
	?assertNotEqual(Output, "").

