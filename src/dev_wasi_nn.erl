%%% @doc WASI-NN device implementation for HyperBEAM
%%% Implements wasi_nn API functions as imported functions by WASM modules
-module(dev_wasi_nn).
-export([init/3]).
-export([load/3, load_by_name/3, load_by_name_with_config/3]).
-export([init_execution_context/3, set_input/3, get_output/3]).
-export([run_inference/3, run_inference_http/3]).
-include("include/hb.hrl").
-include_lib("eunit/include/eunit.hrl").

%% @doc Initialize device state
init(M1, _M2, Opts) ->
    ?event(initializing_wasi_nn),
    MsgWithLib =
        hb_ao:set(
            M1,
            #{
                <<"wasm/stdlib/wasi_ephemeral_nn">> =>
                    #{<<"device">> => <<"WASI-NN@1.0">>}
            },
            Opts
        ),
    {ok, MsgWithLib}.

%% Load model from builders
load(M1, M2, Opts) ->
    % Not Implemented
    S = hb_ao:get(<<"state">>, M1, Opts),
    {ok, #{<<"state">> => S, <<"results">> => [0]}}.

load_by_name(M1, M2, Opts) ->
    [FilenamePtr, FilenameLen, GraphPtr | _] = hb_ao:get(<<"args">>, M2, Opts),
    ?event({path_open, FilenamePtr, FilenameLen, GraphPtr}),
    % TODO: read Filename from WASM memory
    Signature = hb_ao:get(<<"func-sig">>, M2, Opts),
    ?event({signature, Signature}),

    % TODO: call dev_wasi_nn_nif

    {ok, #{
        <<"state">> =>
            hb_ao:set(
                M1,
                <<"wasi-nn/graph">>,
                GraphPtr
            ),
        <<"results">> => [0]
    }}.

load_by_name_with_config(Msg1, Msg2, Opts) ->
	State = hb_ao:get(<<"state">>, Msg1, Opts),
    Instance = hb_private:get(<<"wasm/instance">>, State, Opts),
	[FilenamePtr, FilenameLen, ConfigPtr, ConfigLen, GraphPtr | _] = hb_ao:get(<<"args">>, Msg2, Opts),
    {ok, Filename} = hb_beamr_io:read(Instance, FilenamePtr, FilenameLen),
	{ok, Config} = hb_beamr_io:read(Instance, ConfigPtr, ConfigLen),
	?event({load_by_name_with_config, Filename, Config}),
	hb_beamr_io:write(Instance, GraphPtr, <<0>>),
    {ok, #{<<"state">> => State, <<"results">> => [0]}}.

init_execution_context(Msg1, Msg2, Opts) ->
	State = hb_ao:get(<<"state">>, Msg1, Opts),
    Instance = hb_private:get(<<"wasm/instance">>, State, Opts),
    [Graph, CtxPtr | _] = hb_ao:get(<<"args">>, Msg2, Opts),
    ?event({init_execution_context, Graph}),
	hb_beamr_io:write(Instance, CtxPtr, <<0>>),
    {ok, #{<<"state">> => State, <<"results">> => [0]}}.

set_input(M1, M2, Opts) ->
    Graph = hb_ao:get(<<"wasi-nn/graph">>, M2, Opts),
    [Ctx, Idx, TensorPtr | _] = hb_ao:get(<<"args">>, M2, Opts),
    ?event({set_input, Graph, Ctx, Idx, TensorPtr}),
    Signature = hb_ao:get(<<"func-sig">>, M2, Opts),
    ?event({signature, Signature}),

    % TODO: call dev_wasi_nn_nif

    {ok, #{
        <<"state">> =>
            hb_ao:set(
                M1,
                <<"wasi-nn/prompt">>,
                % TODO: read prompt and set to state
                TensorPtr
            ),
        <<"results">> => [0]
    }}.

% compute(M1, M2, Opts) ->
%     Graph = hb_ao:get(<<"wasi-nn/graph">>, M2, Opts),
%     [Ctx | _] = hb_ao:get(<<"args">>, M2, Opts),
%     ?event({set_input, Graph, Ctx}),
%     {ok, #{<<"results">> => [0]}}.

get_output(M1, M2, Opts) ->
    Graph = hb_ao:get(<<"wasi-nn/graph">>, M2, Opts),
    [Ctx, Idx, OutputPtr, OutputMaxSize, OutputSizePtr | _] = hb_ao:get(<<"args">>, M2, Opts),
    ?event({get_output, Graph, Ctx, Idx, OutputPtr, OutputMaxSize, OutputSizePtr}),
    Signature = hb_ao:get(<<"func-sig">>, M2, Opts),
    ?event({signature, Signature}),

    % TODO: call dev_wasi_nn_nif

    {ok, #{<<"results">> => [0]}}.

run_inference(Msg1, Msg2, Opts) ->
    State = hb_ao:get(<<"state">>, Msg1, Opts),
    Instance = hb_private:get(<<"wasm/instance">>, State, Opts),
    [VecsPtr, Len] = hb_ao:get(<<"args">>, Msg2, Opts),
    {ok, Prompt} = hb_beamr_io:read(Instance, VecsPtr, Len),
    ?event({inference_prompt, Prompt, is_binary(Prompt)}),
    {ok, Output} = dev_wasi_nn_nif:run_inference(Prompt),
    ?event({inference_output, Output}),
    {ok, Ptr} = hb_beamr_io:write_string(Instance, Output),
    {ok, #{<<"state">> => State, <<"results">> => [Ptr]}}.

%%% Tests

init() ->
    application:ensure_all_started(hb).

generate_wasi_nn_stack(File, Func, Params) ->
    init(),
    Msg0 = dev_wasm:cache_wasm_image(File),
    Msg1 = Msg0#{
        <<"device">> => <<"stack@1.0">>,
        <<"device-stack">> => [<<"WASI-NN@1.0">>, <<"WASI@1.0">>, <<"WASM-64@1.0">>],
        <<"output-prefixes">> => [<<"wasm">>, <<"wasm">>, <<"wasm">>],
        <<"stack-keys">> => [<<"init">>, <<"init">>, <<"compute">>],
        <<"function">> => Func,
        <<"params">> => Params
    },
    {ok, Msg2} = hb_ao:resolve(Msg1, <<"init">>, #{}),
    Msg2.

run_inference_http(_, Request, NodeMsg) ->
	case hb_ao:get(<<"method">>, Request, NodeMsg) of
        <<"GET">> ->
            % ?event({get_config_req, Request, NodeMsg}),
			Prompt = hb_ao:get(<<"prompt">>, Request, NodeMsg),
			?assertNotEqual(not_found, Prompt),
			?event({inference_prompt, Prompt}),
			Config = hb_ao:get(<<"config">>, Request, NodeMsg),
			?event({inference_config, Config}),
			% if Config is not configured, use default
			ConfigBin = case Config of
				X when X =:= <<"">>; X =:= not_found -> <<"{\"enable_debug\": false}">>;
				_ -> Config
			end,
			Init = generate_wasi_nn_stack("test/wasi-nn.wasm", <<"handle">>, []),
			Instance = hb_private:get(<<"wasm/instance">>, Init, #{}),
			{ok, PromptPtr} = hb_beamr_io:write_string(Instance, Prompt),
			?assertNotEqual(0, PromptPtr),
			{ok, ConfigPtr} = hb_beamr_io:write_string(Instance, ConfigBin),
			?assertNotEqual(0, ConfigPtr),
			Ready = Init#{ <<"parameters">> => [PromptPtr, ConfigPtr] },
			{ok, StateRes} = hb_ao:resolve(Ready, <<"compute">>, #{}),
			[Ptr] = hb_ao:get(<<"results/wasm/output">>, StateRes),
			{ok, Output} = hb_beamr_io:read_string(Instance, Ptr),
			?event({wasm_output, Output}),
            {ok, Output};
            % {ok, <<"Hello World">>};
        _ -> {error, <<"Unsupported method">>}
    end.
    
wasi_nn_exec_test() ->
    Init = generate_wasi_nn_stack("test/wasi-nn.wasm", <<"handle">>, []),
    Instance = hb_private:get(<<"wasm/instance">>, Init, #{}),
    {ok, StateRes} = hb_ao:resolve(Init, <<"compute">>, #{}),
    [Ptr] = hb_ao:get(<<"results/wasm/output">>, StateRes),
    {ok, Output} = hb_beamr_io:read_string(Instance, Ptr),
    ?event({wasm_output, Output}),
    ?assertNotEqual(<<"">>, Output).