%%% @doc WASI-NN device implementation for HyperBEAM
%%% Implements wasi_nn API functions as imported functions by WASM modules
-module(dev_wasi_nn).
-export([init/3]).
-export([load/3, load_by_name/3, set_input/3, get_output/3]).
-export([
    load_by_name_with_config/3, init_execution_context/3, run_inference/3, run_inference_http/3
]).
-include("include/hb.hrl").
-include_lib("eunit/include/eunit.hrl").

%% @doc Initialize device state
init(Msg1, _Msg2, Opts) ->
    ?event(initializing_wasi_nn),
    MsgWithLib =
        hb_ao:set(
            Msg1,
            #{
                <<"wasm/stdlib/wasi_ephemeral_nn">> =>
                    #{<<"device">> => <<"WASI-NN@1.0">>}
            },
            Opts
        ),
    {ok, MsgWithLib}.

get_instance(Msg1, _Msg2, Opts) ->
    NNInstance = hb_ao:get(<<"state/wasi-nn/instance">>, Msg1, Opts),
    NNInstance2 =
        case NNInstance of
            not_found ->
                ?event({wasi_nn_instance_not_found}),
                % First time init the instance
                {ok, Context} = dev_wasi_nn_nif:init_backend(),
                Context;
            _ ->
                NNInstance
        end,
    ?event({instance, NNInstance2}),
    NNInstance2.

%% Load model from builders
load(_Msg1, _Msg2, _Opts) ->
    % Not Implemented
    erlang:error("Not Implemented").

load_by_name(_Msg1, _Msg2, _Opts) ->
    % Not Implemented
    erlang:error("Not Implemented").

load_by_name_with_config(Msg1, Msg2, Opts) ->
    State = hb_ao:get(<<"state">>, Msg1, Opts),
    Instance = hb_private:get(<<"wasm/instance">>, State, Opts),
    [FilenamePtr, FilenameLen, ConfigPtr, ConfigLen, GraphPtr | _] = hb_ao:get(
        <<"args">>, Msg2, Opts
    ),
    {ok, Filename} = hb_beamr_io:read(Instance, FilenamePtr, FilenameLen),
    {ok, Config} = hb_beamr_io:read(Instance, ConfigPtr, ConfigLen),
    ?event({load_by_name_with_config, Filename, Config}),
    % Use the once version of the function to ensure we only load the model once
    % regardless of how many WASM instances are created
    {ok, NNInstance} = dev_wasi_nn_nif:load_by_name_with_config_once(
        dummy_context, "test/qwen2.5-14b-instruct-q2_k.gguf", binary_to_list(Config)
    ),
    % Write Graph to the model
    hb_beamr_io:write(Instance, GraphPtr, <<0>>),
    {ok, #{
        <<"state">> => hb_ao:set(
            State,
            #{ <<"wasi-nn/instance">> => NNInstance },
			Opts
		),
        <<"results">> => [0]
    }}.

init_execution_context(Msg1, Msg2, Opts) ->
    ?event({init_Msgs, Msg1, Msg2, Opts}),
    State = hb_ao:get(<<"state">>, Msg1, Opts),
    Instance = hb_private:get(<<"wasm/instance">>, State, Opts),
    [Graph, CtxPtr | _] = hb_ao:get(<<"args">>, Msg2, Opts),
    ?event({init_execution_context, Graph}),
    NNInstance = get_instance(Msg1, Msg2, Opts),
    dev_wasi_nn_nif:init_execution_context_once(NNInstance),
    hb_beamr_io:write(Instance, CtxPtr, <<0>>),
    {ok, #{<<"state">> => State, <<"results">> => [0]}}.

set_input(_Msg1, _Msg2, _Opts) ->
    % Not Implemented
    erlang:error("Not Implemented").

% compute(M1, M2, Opts) ->
%     Graph = hb_ao:get(<<"wasi-nn/graph">>, M2, Opts),
%     [Ctx | _] = hb_ao:get(<<"args">>, M2, Opts),
%     ?event({set_input, Graph, Ctx}),
%     {ok, #{<<"results">> => [0]}}.

get_output(_Msg1, _Msg2, _Opts) ->
    % Not Implemented
    erlang:error("Not Implemented").

run_inference(Msg1, Msg2, Opts) ->
    State = hb_ao:get(<<"state">>, Msg1, Opts),
    Instance = hb_private:get(<<"wasm/instance">>, State, Opts),
    [VecsPtr, Len] = hb_ao:get(<<"args">>, Msg2, Opts),
    {ok, Prompt} = hb_beamr_io:read(Instance, VecsPtr, Len),
    ?event({inference_prompt, Prompt}),
    NNInstance = get_instance(Msg1, Msg2, Opts),
    {ok, Output} = dev_wasi_nn_nif:run_inference(NNInstance, binary_to_list(Prompt)),
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
            ConfigBin =
                case Config of
                    X when X =:= <<"">>; X =:= not_found -> <<"{\"enable_debug\": false}">>;
                    _ -> Config
                end,
            Init = generate_wasi_nn_stack("test/wasi-nn.wasm", <<"handle">>, []),
            Instance = hb_private:get(<<"wasm/instance">>, Init, #{}),
            {ok, PromptPtr} = hb_beamr_io:write_string(Instance, Prompt),
            ?assertNotEqual(0, PromptPtr),
            {ok, ConfigPtr} = hb_beamr_io:write_string(Instance, ConfigBin),
            ?assertNotEqual(0, ConfigPtr),
            Ready = Init#{<<"parameters">> => [PromptPtr, ConfigPtr]},
            {ok, StateRes} = hb_ao:resolve(Ready, <<"compute">>, #{}),
            [Ptr] = hb_ao:get(<<"results/wasm/output">>, StateRes),
            {ok, Output} = hb_beamr_io:read_string(Instance, Ptr),
            ?event({wasm_output, Output}),
            {ok, Output};
        % {ok, <<"Hello World">>};
        _ ->
            {error, <<"Unsupported method">>}
    end.

wasi_nn_exec_test() ->
    Init = generate_wasi_nn_stack("test/wasi-nn.wasm", <<"handle">>, []),
    Instance = hb_private:get(<<"wasm/instance">>, Init, #{}),
    {ok, StateRes} = hb_ao:resolve(Init, <<"compute">>, #{}),
    [Ptr] = hb_ao:get(<<"results/wasm/output">>, StateRes),
    {ok, Output} = hb_beamr_io:read_string(Instance, Ptr),
    ?event({wasm_output, Output}),
    ?assertNotEqual(<<"">>, Output).
