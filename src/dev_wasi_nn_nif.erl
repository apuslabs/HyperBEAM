-module(dev_wasi_nn_nif).
-include("include/hb.hrl").
-include_lib("eunit/include/eunit.hrl").
-on_load(init/0).
-export([run_inference/1]).

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
run_inference(Prompt) ->
    ModelPath = "test/Qwen2.5-1.5B-Instruct.Q2_K.gguf",
    Config = "{\"n_gpu_layers\":32}",
    case filelib:is_regular(ModelPath) of
        true ->
            {ok, Context} = init_backend(),
            try
                ?assertNotEqual(undefined, Context),
				% TODO: check rets
                load_by_name_with_config(Context, ModelPath, Config),
                init_execution_context(Context),
                set_input(Context, binary_to_list(Prompt)),
                compute(Context),
                get_output(Context)
            catch
                Error:Reason ->
                    io:format("Test failed: ~p:~p~n", [Error, Reason]),
                    erlang:error(Reason)
            after
                deinit_backend(Context)
            end;
        false ->
            ?event("Skipping test - model file not found")
    end.
