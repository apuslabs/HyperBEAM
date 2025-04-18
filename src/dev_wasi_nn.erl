%%% @doc WASI-NN device implementation for HyperBEAM
%%% Implements wasi_nn API functions as imported functions by WASM modules
-module(dev_wasi_nn).
-export([init/3]).
-export([load/3, load_by_name/3, load_by_name_with_config/3]).
-export([init_execution_context/3, set_input/3, get_output/3]).
-export([run_inference/3,run_inference_http/3]).
-include("include/hb.hrl").
-include_lib("eunit/include/eunit.hrl").


%% @doc Initialize device state
init(M1, _M2, Opts) ->
    ?event(initializing_wasi_nn),
    MsgWithLib =
        hb_converge:set(
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
    S = hb_converge:get(<<"state">>, M1, Opts),
    {ok, #{<<"state">> => S, <<"results">> => [0]}}.

load_by_name(M1, M2, Opts) ->
    [FilenamePtr, FilenameLen, GraphPtr | _] = hb_converge:get(<<"args">>, M2, Opts),
    ?event({path_open, FilenamePtr, FilenameLen, GraphPtr}),
    % TODO: read Filename from WASM memory
    Signature = hb_converge:get(<<"func-sig">>, M2, Opts),
    ?event({signature, Signature}),

    % TODO: call dev_wasi_nn_nif

    {ok, #{
        <<"state">> =>
            hb_converge:set(
                M1,
                <<"wasi-nn/graph">>,
                GraphPtr
            ),
        <<"results">> => [0]
    }}.

load_by_name_with_config(M1, M2, Opts) ->
    [FilenamePtr, FilenameLen, ConfigPtr, ConfigLen, GraphPtr | _] = hb_converge:get(
        <<"args">>, M2, Opts
    ),
    ?event({path_open, FilenamePtr, FilenameLen, ConfigPtr, ConfigLen, GraphPtr}),
    % TODO: read Filename and Config from WASM memory


    % TODO: call dev_wasi_nn_nif

    {ok, #{<<"results">> => [0]}}.

init_execution_context(M1, M2, Opts) ->
    [Graph, CtxPtr | _] = hb_converge:get(<<"args">>, M2, Opts),
    ?event({init_execution_context, Graph, CtxPtr}),
    Signature = hb_converge:get(<<"func-sig">>, M2, Opts),
    ?event({signature, Signature}),

    % TODO: call dev_wasi_nn_nif

    {ok, #{
        <<"state">> =>
            hb_converge:set(
                M1,
                <<"wasi-nn/graph">>,
                Graph
            ),
        <<"results">> => [0]
    }}.

set_input(M1, M2, Opts) ->
    Graph = hb_converge:get(<<"wasi-nn/graph">>, M2, Opts),
    [Ctx, Idx, TensorPtr | _] = hb_converge:get(<<"args">>, M2, Opts),
    ?event({set_input, Graph, Ctx, Idx, TensorPtr}),
    Signature = hb_converge:get(<<"func-sig">>, M2, Opts),
    ?event({signature, Signature}),

    % TODO: call dev_wasi_nn_nif

    {ok, #{
        <<"state">> =>
            hb_converge:set(
                M1,
                <<"wasi-nn/prompt">>,
                % TODO: read prompt and set to state
                TensorPtr
            ),
        <<"results">> => [0]
    }}.

% compute(M1, M2, Opts) ->
%     Graph = hb_converge:get(<<"wasi-nn/graph">>, M2, Opts),
%     [Ctx | _] = hb_converge:get(<<"args">>, M2, Opts),
%     ?event({set_input, Graph, Ctx}),
%     {ok, #{<<"results">> => [0]}}.

get_output(M1, M2, Opts) ->
    Graph = hb_converge:get(<<"wasi-nn/graph">>, M2, Opts),
    [Ctx, Idx, OutputPtr, OutputMaxSize, OutputSizePtr | _] = hb_converge:get(<<"args">>, M2, Opts),
    ?event({get_output, Graph, Ctx, Idx, OutputPtr, OutputMaxSize, OutputSizePtr}),
    Signature = hb_converge:get(<<"func-sig">>, M2, Opts),
    ?event({signature, Signature}),

    % TODO: call dev_wasi_nn_nif

    {ok, #{<<"results">> => [0]}}.
%%% Tests

init() ->
    application:ensure_all_started(hb).

generate_wasi_nn_stack(File, Func, Params) ->
    init(),
    Msg0 = dev_wasm:cache_wasm_image(File),
    Msg1 = Msg0#{
        <<"device">> => <<"Stack@1.0">>,
        <<"device-stack">> => [<<"WASI-NN@1.0">>, <<"WASI@1.0">>, <<"WASM-64@1.0">>],
        <<"output-prefixes">> => [<<"wasm">>, <<"wasm">>, <<"wasm">>],
        <<"stack-keys">> => [<<"init">>, <<"init">>, <<"compute">>],
        <<"wasm-function">> => Func,
        <<"wasm-params">> => Params
    },
    {ok, Msg2} = hb_converge:resolve(Msg1, <<"init">>, #{}),
    Msg2.

run_inference(M1,M2,Opts)->
	State = hb_converge:get(<<"state">>, M1, Opts),
    Instance = hb_private:get(<<"wasm/instance">>, State, Opts),
    [VecsPtr, Len] = hb_converge:get(<<"args">>, M2, Opts),
	{ok,Prompt} = hb_beamr_io:read(Instance, VecsPtr, Len),
	?event({inference_prompt, Prompt, is_binary(Prompt)}),
	{ok, Output} = dev_wasi_nn_nif:run_inference(Prompt),
	?event({inference_output, Output}),
	{ok, Ptr} = hb_beamr_io:write_string(Instance, Output),
	{ok, #{ <<"state">> => State, <<"results">> => [Ptr] }}.

run_inference_http(_, Request, NodeMsg) ->
	Prompt = hb_converge:get(<<"prompt">>, Request, NodeMsg),
	?event({inference_prompt, Prompt}),
	Init = generate_wasi_nn_stack("test/wasi-nn.wasm", <<"lib_main">>, []),
	Instance = hb_private:get(<<"wasm/instance">>, Init, #{}),
	{ok, StateRes} = hb_converge:resolve(Init, <<"compute">>, #{}),
	[Ptr] = hb_converge:get(<<"results/wasm/output">>, StateRes),
	{ok, Output} = hb_beamr_io:read_string(Instance, Ptr),
	?event({wasm_output, Output}),
	embed_status({ok, Output}).

embed_status({ErlStatus, Res}) when is_map(Res) ->
    case lists:member(<<"status">>, hb_message:committed(Res)) of
        false ->
            HTTPCode = status_code({ErlStatus, Res}),
            {ok, Res#{ <<"status">> => HTTPCode }};
        true ->
            {ok, Res}
    end;
embed_status({ErlStatus, Res}) ->
    HTTPCode = status_code({ErlStatus, Res}),
    {ok, #{ <<"status">> => HTTPCode, <<"body">> => Res }}.

status_code({ErlStatus, Msg}) ->
    case message_to_status(Msg) of
        default -> status_code(ErlStatus);
        RawStatus -> RawStatus
    end;
status_code(ok) -> 200;
status_code(error) -> 400;
status_code(created) -> 201;
status_code(not_found) -> 404;
status_code(unavailable) -> 503.

message_to_status(#{ <<"body">> := Status }) when is_atom(Status) ->
    status_code(Status);
message_to_status(Item) when is_map(Item) ->
    % Note: We use `dev_message' directly here, such that we do not cause 
    % additional AO-Core calls for every request. This is particularly important
    % if a remote server is being used for all AO-Core requests by a node.
    case dev_message:get(<<"status">>, Item) of
        {ok, RawStatus} when is_integer(RawStatus) -> RawStatus;
        {ok, RawStatus} when is_atom(RawStatus) -> status_code(RawStatus);
        {ok, RawStatus} -> binary_to_integer(RawStatus);
        _ -> default
    end;
message_to_status(Item) when is_atom(Item) ->
    status_code(Item);
message_to_status(_Item) ->
    default.

filter_node_msg(Msg) when is_map(Msg) ->
    maps:map(fun(_, Value) -> filter_node_msg(Value) end, hb_private:reset(Msg));
filter_node_msg(Msg) when is_list(Msg) ->
    lists:map(fun filter_node_msg/1, Msg);
filter_node_msg(Tuple) when is_tuple(Tuple) ->
    <<"Unencodable value.">>;
filter_node_msg(Other) ->
    Other.

wasi_nn_exec_test() ->
	Init = generate_wasi_nn_stack("test/wasi-nn.wasm", <<"lib_main">>, []),
	Instance = hb_private:get(<<"wasm/instance">>, Init, #{}),
	{ok, StateRes} = hb_converge:resolve(Init, <<"compute">>, #{}),
	[Ptr] = hb_converge:get(<<"results/wasm/output">>, StateRes),
	{ok, Output} = hb_beamr_io:read_string(Instance, Ptr),
	?event({wasm_output, Output}),
	?assertNotEqual(<<"">>, Output).

aos_wasi_nn_exec_test() ->
	Init = generate_wasi_nn_stack("test/aos-wasi-nn.wasm", <<"handle">>, []),
	Msg = gen_test_aos_msg("return 1 + 1"),
	Env = gen_test_env(),
	Instance = hb_private:get(<<"wasm/instance">>, Init, #{}),
	{ok, Ptr1} = hb_beamr_io:malloc(Instance, byte_size(Msg)),
	?assertNotEqual(0, Ptr1),
	hb_beamr_io:write(Instance, Ptr1, Msg),
	{ok, Ptr2} = hb_beamr_io:malloc(Instance, byte_size(Env)),
	?assertNotEqual(0, Ptr2),
	hb_beamr_io:write(Instance, Ptr2, Env),
	% Read the strings to validate they are correctly passed
	{ok, MsgBin} = hb_beamr_io:read(Instance, Ptr1, byte_size(Msg)),
	{ok, EnvBin} = hb_beamr_io:read(Instance, Ptr2, byte_size(Env)),
	?assertEqual(Env, EnvBin),
	?assertEqual(Msg, MsgBin),
	Ready = Init#{ <<"wasm-params">> => [Ptr1, Ptr2] },
	{ok, StateRes} = hb_converge:resolve(Ready, <<"compute">>, #{}),
	[Ptr] = hb_converge:get(<<"results/wasm/output">>, StateRes),
	{ok, Output} = hb_beamr_io:read_string(Instance, Ptr),
	?event({got_output, Output}),
	#{ <<"response">> := #{ <<"Output">> := #{ <<"data">> := Data }} }
		= jiffy:decode(Output, [return_maps]),
	?assertEqual(<<"2">>, Data).

%%% Test Helpers
gen_test_env() ->
    <<"{\"Process\":{\"Id\":\"AOS\",\"Owner\":\"FOOBAR\",\"Tags\":[{\"name\":\"Name\",\"value\":\"Thomas\"}, {\"name\":\"Authority\",\"value\":\"FOOBAR\"}]}}\0">>.

gen_test_aos_msg(Command) ->
    <<"{\"From\":\"FOOBAR\",\"Block-Height\":\"1\",\"Target\":\"AOS\",\"Owner\":\"FOOBAR\",\"Id\":\"1\",\"Module\":\"W\",\"Tags\":[{\"name\":\"Action\",\"value\":\"Eval\"}],\"Data\":\"", (list_to_binary(Command))/binary, "\"}\0">>.