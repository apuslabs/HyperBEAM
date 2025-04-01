-module(dev_wasi_nn_nif).
-include("include/hb.hrl").
-include_lib("eunit/include/eunit.hrl").
-hb_debug(print).
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

load_by_name_with_config(_Context,_Path, _Config) ->
    erlang:nif_error("NIF library not loaded").

init_execution_context(_Context) ->
    erlang:nif_error("NIF library not loaded").

set_input(_Context,_Prompt) ->
    erlang:nif_error("NIF library not loaded").

compute(_Context) ->
    erlang:nif_error("NIF library not loaded").
get_output(_Context) ->
    erlang:nif_error("NIF library not loaded").
deinit_backend(_Context) ->
	erlang:nif_error("NIF library not loaded").
run_inference(Prompt)->
	ModelPath = "test/qwen1_5-0_5b-chat-q2_k.gguf",
	Config = "{\"n_gpu_layers\":20}",
	case filelib:is_regular(ModelPath) of
		true ->
			?event(ModelPath),
			% Test init_backend
			{ok, Context} = init_backend(),
			try
				% Print the context type for debugging
				?assertNotEqual(undefined, Context),
				ok = load_by_name_with_config(Context,ModelPath,Config),
				% Test init_execution_context
				ok = init_execution_context(Context),
				% Test set_input
				ok = set_input(Context,Prompt),
				% Test compute
				ok = compute(Context),
				% Test get_output
				{ok, Output} = get_output(Context),
				?event(Output)
			catch
				Error:Reason ->
					io:format("Test failed: ~p:~p~n", [Error, Reason]),
					erlang:error(Reason)
			after
			% Cleanup
			ok = deinit_backend(Context)
			end;
		false ->
			?event("Skipping test - model file not found")
	end.
load_model_test() ->
% Skip test if model doesn't exist
ModelPath = "test/qwen1_5-0_5b-chat-q2_k.gguf",
case filelib:is_regular(ModelPath) of
	true ->
		?event(ModelPath),
		% Test init_backend
		{ok, Context} = init_backend(),
		Prompt = "Hello",
		try
			% Print the context type for debugging
			
			?assertNotEqual(undefined, Context),
			
			% Test load_model_with_config
			Config = "{\"n_gpu_layers\":20}",
			
			ok = load_by_name_with_config(Context,ModelPath, Config),
			
			% Test init_execution_context
			ok = init_execution_context(Context),
			
			% Test set_input
			ok = set_input(Context,Prompt),
			
			% Test compute
			ok = compute(Context),
			
			% Test get_output
			{ok, Output} = get_output(Context),
			?assertNotEqual("", Output)
		catch
			Error:Reason ->
				io:format("Test failed: ~p:~p~n", [Error, Reason]),
				erlang:error(Reason)
		after
		% Cleanup
		ok = deinit_backend(Context)
		end;
	false ->
		?debugMsg("Skipping test - model file not found")
end.