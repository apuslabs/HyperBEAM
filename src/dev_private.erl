%%% @doc Private conversation device for end-to-end encrypted messaging.
%%%
%%% This device provides session-based encrypted communication where:
%%% 1. Sessions are created with unique IDs and AES keys
%%% 2. Messages are encrypted before being stored on-chain
%%% 3. Users can decrypt messages on their side using session keys
%%% 4. Each session has its own encryption key for isolation
-module(dev_private).
-export([info/1, info/3, create_session/3, get_key/3, encrypt/3, decrypt/3]).
-include("include/hb.hrl").
-include_lib("eunit/include/eunit.hrl").

%% @doc Controls which functions are exposed via the device API.
%%
%% This function defines the security boundary for the private conversation device
%% by explicitly listing which functions are available through the API.
%%
%% @param _ Ignored parameter
%% @returns A map with the `exports' key containing a list of allowed functions
info(_) -> 
    #{ exports => [info, create_session, get_key, encrypt, decrypt] }.

%% @doc Provides information about the private conversation device and its API.
%%
%% This function returns detailed documentation about the device, including:
%% 1. A high-level description of the device's purpose
%% 2. Version information
%% 3. Available API endpoints with their parameters and descriptions
%%
%% @param _Msg1 Ignored parameter
%% @param _Msg2 Ignored parameter
%% @param _Opts A map of configuration options
%% @returns {ok, Map} containing the device information and documentation
info(_Msg1, _Msg2, _Opts) ->
    InfoBody = #{
        <<"description">> => 
            <<"Private conversation device for end-to-end encrypted messaging">>,
        <<"version">> => <<"1.0">>,
        <<"api">> => #{
            <<"info">> => #{
                <<"description">> => <<"Get device info">>
            },
            <<"create_session">> => #{
                <<"description">> => <<"Create a new private conversation session">>,
                <<"parameters">> => #{
                    <<"session_name">> => <<"Optional human-readable name for the session">>
                },
                <<"returns">> => #{
                    <<"session_id">> => <<"Unique identifier for the session">>,
                    <<"created_at">> => <<"Timestamp when session was created">>
                }
            },
            <<"get_key">> => #{
                <<"description">> => <<"Retrieve the encryption key for a session">>,
                <<"parameters">> => #{
                    <<"session_id">> => <<"Session identifier from create_session">>
                },
                <<"returns">> => #{
                    <<"key">> => <<"Base64-encoded session encryption key">>,
                    <<"iv">> => <<"Initialization vector for this session">>
                }
            },
            <<"encrypt">> => #{
                <<"description">> => <<"Encrypt a message for a session">>,
                <<"parameters">> => #{
                    <<"session_id">> => <<"Session identifier">>,
                    <<"message">> => <<"Message to encrypt (string or binary)">>
                },
                <<"returns">> => #{
                    <<"encrypted_message">> => <<"Base64-encoded encrypted message">>,
                    <<"iv">> => <<"Initialization vector used for encryption">>
                }
            },
            <<"decrypt">> => #{
                <<"description">> => <<"Decrypt a message for a session">>,
                <<"parameters">> => #{
                    <<"session_id">> => <<"Session identifier">>,
                    <<"encrypted_message">> => <<"Base64-encoded encrypted message">>,
                    <<"iv">> => <<"Initialization vector used during encryption">>
                },
                <<"returns">> => #{
                    <<"decrypted_message">> => <<"Original plaintext message">>
                }
            }
        }
    },
    {ok, #{<<"status">> => 200, <<"body">> => InfoBody}}.

%% @doc Creates a new private conversation session with a unique ID and AES key.
%%
%% This function performs the following operations:
%% 1. Generates a unique session ID using crypto random bytes
%% 2. Creates a new 256-bit AES key for the session
%% 3. Optionally accepts a human-readable session name
%% 4. Stores the session information in the node's configuration
%% 5. Returns the session ID and creation timestamp
%%
%% @param _M1 Ignored parameter
%% @param M2 May contain session configuration like name
%% @param Opts A map of configuration options
%% @returns {ok, Map} containing session_id and creation details, or
%% {error, Binary} on failure
-spec create_session(M1 :: term(), M2 :: term(), Opts :: map()) -> 
    {ok, map()} | {error, binary()}.
create_session(_M1, M2, Opts) ->
    ?event(priv_session, {create_session, start}),
    
    % Generate a unique session ID
    SessionID = base64:encode(crypto:strong_rand_bytes(16)),
    
    % Generate a 256-bit AES key for this session
    SessionKey = crypto:strong_rand_bytes(32),
    
    % Get optional session name
    SessionName = case M2 of
        #{<<"session_name">> := Name} when is_binary(Name) -> Name;
        _ -> <<"Unnamed Session">>
    end,
    
    % Create session metadata
    CreatedAt = erlang:system_time(second),
    SessionData = #{
        session_id => SessionID,
        session_key => SessionKey,
        session_name => SessionName,
        created_at => CreatedAt,
        message_count => 0
    },
    
    % Store the session in the node's configuration
    PrivSessions = hb_opts:get(priv_sessions, #{}, Opts),
    UpdatedSessions = maps:put(SessionID, SessionData, PrivSessions),
    
    hb_http_server:set_opts(Opts#{
        priv_sessions => UpdatedSessions
    }),
    
    ?event(priv_session, {create_session, complete, SessionID}),
    
    {ok, #{
        <<"status">> => 200,
        <<"body">> => #{
            <<"session_id">> => SessionID,
            <<"session_name">> => SessionName,
            <<"created_at">> => CreatedAt,
            <<"message">> => <<"Private session created successfully">>
        }
    }}.

%% @doc Retrieves the encryption key for a specific session.
%%
%% This function performs the following operations:
%% 1. Validates that the session ID exists
%% 2. Retrieves the session's AES key
%% 3. Generates a fresh IV for encryption operations
%% 4. Returns the key and IV in base64 format
%%
%% @param _M1 Ignored parameter
%% @param M2 Map containing session_id
%% @param Opts A map of configuration options
%% @returns {ok, Map} containing the session key and IV, or
%% {error, Binary} if session not found
-spec get_key(M1 :: term(), M2 :: term(), Opts :: map()) -> 
    {ok, map()} | {error, binary()}.
get_key(_M1, M2, Opts) ->
    ?event(priv_session, {get_key, start}),
    
    SessionID = case M2 of
        #{<<"session_id">> := ID} -> ID;
        _ -> hb_opts:get(<<"session_id">>, undefined, Opts)
    end,
    
    case SessionID of
        undefined ->
            {error, <<"Session ID is required">>};
        _ ->
            PrivSessions = hb_opts:get(priv_sessions, #{}, Opts),
            case maps:find(SessionID, PrivSessions) of
                {ok, #{session_key := SessionKey}} ->
                    % Generate a fresh IV for this encryption operation
                    IV = crypto:strong_rand_bytes(16),
                    
                    ?event(priv_session, {get_key, success, SessionID}),
                    
                    {ok, #{
                        <<"status">> => 200,
                        <<"body">> => #{
                            <<"session_id">> => SessionID,
                            <<"key">> => base64:encode(SessionKey),
                            <<"iv">> => base64:encode(IV)
                        }
                    }};
                error ->
                    ?event(priv_session, {get_key, not_found, SessionID}),
                    {error, <<"Session not found">>}
            end
    end.

%% @doc Encrypts a message for a specific session.
%%
%% This function performs the following operations:
%% 1. Validates that the session exists
%% 2. Retrieves the session's AES key
%% 3. Generates a fresh IV for this encryption
%% 4. Encrypts the message using AES-256-GCM
%% 5. Returns the encrypted message and IV in base64 format
%% 6. Increments the session's message count
%%
%% @param _M1 Ignored parameter
%% @param M2 Map containing session_id and message
%% @param Opts A map of configuration options
%% @returns {ok, Map} containing encrypted message and IV, or
%% {error, Binary} if session not found or encryption fails
-spec encrypt(M1 :: term(), M2 :: term(), Opts :: map()) -> 
    {ok, map()} | {error, binary()}.
encrypt(_M1, M2, Opts) ->
    ?event(priv_session, {encrypt, start}),
    
    SessionID = case M2 of
        #{<<"session_id">> := ID} -> ID;
        _ -> hb_opts:get(<<"session_id">>, undefined, Opts)
    end,
    
    Message = case M2 of
        #{<<"message">> := Msg} -> Msg;
        _ -> hb_opts:get(<<"message">>, undefined, Opts)
    end,
    
    case {SessionID, Message} of
        {undefined, _} ->
            {error, <<"Session ID is required">>};
        {_, undefined} ->
            {error, <<"Message is required">>};
        {_, _} ->
            PrivSessions = hb_opts:get(priv_sessions, #{}, Opts),
            case maps:find(SessionID, PrivSessions) of
                {ok, SessionData = #{session_key := SessionKey}} ->
                    % Generate a fresh IV for this encryption
                    IV = crypto:strong_rand_bytes(16),
                    
                    % Convert message to binary if it's not already
                    MessageBin = case is_binary(Message) of
                        true -> Message;
                        false -> hb_util:bin(Message)
                    end,
                    
                    % Encrypt using AES-256-GCM
                    {EncryptedMessage, Tag} = crypto:crypto_one_time_aead(
                        aes_256_gcm,
                        SessionKey,
                        IV,
                        MessageBin,
                        <<>>,
                        true
                    ),
                    
                    % Increment message count
                    UpdatedSessionData = SessionData#{
                        message_count => maps:get(message_count, SessionData, 0) + 1
                    },
                    UpdatedSessions = maps:put(SessionID, UpdatedSessionData, PrivSessions),
                    
                    hb_http_server:set_opts(Opts#{
                        priv_sessions => UpdatedSessions
                    }),
                    
                    ?event(priv_session, {encrypt, success, SessionID}),
                    
                    {ok, #{
                        <<"status">> => 200,
                        <<"body">> => #{
                            <<"session_id">> => SessionID,
                            <<"encrypted_message">> => 
                                base64:encode(<<EncryptedMessage/binary, Tag/binary>>),
                            <<"iv">> => base64:encode(IV)
                        }
                    }};
                error ->
                    ?event(priv_session, {encrypt, not_found, SessionID}),
                    {error, <<"Session not found">>}
            end
    end.

%% @doc Decrypts a message for a specific session.
%%
%% This function performs the following operations:
%% 1. Validates that the session exists
%% 2. Retrieves the session's AES key
%% 3. Decodes the encrypted message and IV from base64
%% 4. Separates the ciphertext from the authentication tag
%% 5. Decrypts the message using AES-256-GCM
%% 6. Returns the original plaintext message
%%
%% @param _M1 Ignored parameter
%% @param M2 Map containing session_id, encrypted_message, and iv
%% @param Opts A map of configuration options
%% @returns {ok, Map} containing the decrypted message, or
%% {error, Binary} if session not found or decryption fails
-spec decrypt(M1 :: term(), M2 :: term(), Opts :: map()) -> 
    {ok, map()} | {error, binary()}.
decrypt(_M1, M2, Opts) ->
    ?event(priv_session, {decrypt, start}),
    
    SessionID = case M2 of
        #{<<"session_id">> := ID} -> ID;
        _ -> hb_opts:get(<<"session_id">>, undefined, Opts)
    end,
    
    EncryptedMessage = case M2 of
        #{<<"encrypted_message">> := Msg} -> Msg;
        _ -> hb_opts:get(<<"encrypted_message">>, undefined, Opts)
    end,
    
    IV = case M2 of
        #{<<"iv">> := IVVal} -> IVVal;
        _ -> hb_opts:get(<<"iv">>, undefined, Opts)
    end,
    
    case {SessionID, EncryptedMessage, IV} of
        {undefined, _, _} ->
            {error, <<"Session ID is required">>};
        {_, undefined, _} ->
            {error, <<"Encrypted message is required">>};
        {_, _, undefined} ->
            {error, <<"IV is required">>};
        {_, _, _} ->
            PrivSessions = hb_opts:get(priv_sessions, #{}, Opts),
            case maps:find(SessionID, PrivSessions) of
                {ok, #{session_key := SessionKey}} ->
                    try
                        % Decode the encrypted message and IV
                        Combined = base64:decode(EncryptedMessage),
                        IVBin = base64:decode(IV),
                        
                        % Separate ciphertext and authentication tag
                        CipherLen = byte_size(Combined) - 16,
                        <<Ciphertext:CipherLen/binary, Tag:16/binary>> = Combined,
                        
                        % Decrypt using AES-256-GCM
                        DecryptedMessage = crypto:crypto_one_time_aead(
                            aes_256_gcm,
                            SessionKey,
                            IVBin,
                            Ciphertext,
                            <<>>,
                            Tag,
                            false
                        ),
                        
                        ?event(priv_session, {decrypt, success, SessionID}),
                        
                        {ok, #{
                            <<"status">> => 200,
                            <<"body">> => #{
                                <<"session_id">> => SessionID,
                                <<"decrypted_message">> => DecryptedMessage
                            }
                        }}
                    catch
                        Error:Reason ->
                            ?event(priv_session, {decrypt, error, SessionID, Error, Reason}),
                            {error, <<"Decryption failed - invalid ciphertext or key">>}
                    end;
                error ->
                    ?event(priv_session, {decrypt, not_found, SessionID}),
                    {error, <<"Session not found">>}
            end
    end.

%% @doc Test function to verify the encrypt/decrypt roundtrip works correctly.
%%
%% This test creates a session, encrypts a message, and then decrypts it,
%% verifying that the original message is recovered.
crypto_roundtrip_test() ->
    % Mock options
    Opts = #{priv_sessions => #{}},
    
    % Create a session
    {ok, #{<<"body">> := #{<<"session_id">> := SessionID}}} = 
        create_session(undefined, #{<<"session_name">> => <<"Test Session">>}, Opts),
    
    % Get updated options with the session
    UpdatedOpts = #{priv_sessions => #{SessionID => #{
        session_key => crypto:strong_rand_bytes(32),
        session_name => <<"Test Session">>,
        created_at => erlang:system_time(second),
        message_count => 0
    }}},
    
    % Test message
    TestMessage = <<"This is a test message for encryption">>,
    
    % Encrypt the message
    {ok, #{<<"body">> := #{
        <<"encrypted_message">> := EncMsg,
        <<"iv">> := IV
    }}} = encrypt(undefined, #{
        <<"session_id">> => SessionID,
        <<"message">> => TestMessage
    }, UpdatedOpts),
    
    % Decrypt the message
    {ok, #{<<"body">> := #{
        <<"decrypted_message">> := DecryptedMessage
    }}} = decrypt(undefined, #{
        <<"session_id">> => SessionID,
        <<"encrypted_message">> => EncMsg,
        <<"iv">> => IV
    }, UpdatedOpts),
    
    % Verify roundtrip
    ?assertEqual(TestMessage, DecryptedMessage).
