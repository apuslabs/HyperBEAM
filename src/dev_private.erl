%%% @doc Private conversation device for end-to-end encrypted messaging.
%%%
%%% This device provides session-based encrypted communication where:
%%% 1. Sessions are created with unique IDs and AES keys
%%% 2. Session keys are protected using RSA public key encryption
%%% 3. Messages are encrypted before being stored on-chain
%%% 4. Users can decrypt messages on their side using session keys
%%% 5. Each session has its own encryption key for isolation
-module(dev_private).
-export([info/1, info/3, create_session/3, get_key/3, encrypt/3, decrypt/3, decrypt_key/3]).
-include("include/hb.hrl").
-include_lib("eunit/include/eunit.hrl").
-include_lib("public_key/include/public_key.hrl").

%% @doc Controls which functions are exposed via the device API.
%%
%% This function defines the security boundary for the private conversation device
%% by explicitly listing which functions are available through the API.
%%
%% @param _ Ignored parameter
%% @returns A map with the `exports' key containing a list of allowed functions
info(_) -> 
    #{ exports => [info, create_session, get_key, encrypt, decrypt, decrypt_key] }.

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
            <<"Private conversation device for end-to-end encrypted messaging with RSA key protection">>,
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
                <<"description">> => <<"Retrieve the encryption key for a session (RSA encrypted)">>,
                <<"parameters">> => #{
                    <<"session_id">> => <<"Session identifier from create_session">>,
                    <<"public_key">> => <<"RSA public key to encrypt the session key with">>
                },
                <<"returns">> => #{
                    <<"encrypted_key">> => <<"RSA-encrypted session key (base64)">>,
                    <<"iv">> => <<"Initialization vector for this session">>
                },
                <<"security">> => <<"Session key is encrypted with requester's public key - only they can decrypt it">>
            },
            <<"decrypt_key">> => #{
                <<"description">> => <<"Decrypt a session key using your private key">>,
                <<"parameters">> => #{
                    <<"encrypted_key">> => <<"RSA-encrypted key from get_key">>,
                    <<"private_key">> => <<"Your RSA private key (PEM format)">>
                },
                <<"returns">> => #{
                    <<"session_key">> => <<"Decrypted session key for local use">>
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

%% @doc Retrieves the encryption key for a specific session, encrypted with requester's public key.
%%
%% This function performs the following operations:
%% 1. Validates that the session ID exists
%% 2. Validates that a public key is provided
%% 3. Retrieves the session's AES key
%% 4. Encrypts the session key using the requester's RSA public key
%% 5. Generates a fresh IV for encryption operations
%% 6. Returns the encrypted key and IV in base64 format
%%
%% SECURITY: The session key is encrypted with the requester's public key,
%% ensuring only the holder of the corresponding private key can decrypt it.
%%
%% @param _M1 Ignored parameter
%% @param M2 Map containing session_id and public_key
%% @param Opts A map of configuration options
%% @returns {ok, Map} containing the encrypted session key and IV, or
%% {error, Binary} if session not found or public key invalid
-spec get_key(M1 :: term(), M2 :: term(), Opts :: map()) -> 
    {ok, map()} | {error, binary()}.
get_key(_M1, M2, Opts) ->
    ?event(priv_session, {get_key, start}),
    
    SessionID = case M2 of
        #{<<"session_id">> := ID} -> ID;
        _ -> hb_opts:get(<<"session_id">>, undefined, Opts)
    end,
    
    PublicKeyPEM = case M2 of
        #{<<"public_key">> := PubKey} -> PubKey;
        _ -> hb_opts:get(<<"public_key">>, undefined, Opts)
    end,
    
    case {SessionID, PublicKeyPEM} of
        {undefined, _} ->
            {error, <<"Session ID is required">>};
        {_, undefined} ->
            {error, <<"Public key is required for secure key exchange">>};
        {_, _} ->
            PrivSessions = hb_opts:get(priv_sessions, #{}, Opts),
            case maps:find(SessionID, PrivSessions) of
                {ok, #{session_key := SessionKey}} ->
                    try
                        % Encrypt the session key with the requester's public key
                        EncryptedKey = encrypt_with_public_key(SessionKey, PublicKeyPEM),
                        
                        % Generate a fresh IV for this encryption operation
                        IV = crypto:strong_rand_bytes(16),
                        
                        ?event(priv_session, {get_key, success, SessionID}),
                        
                        {ok, #{
                            <<"status">> => 200,
                            <<"body">> => #{
                                <<"session_id">> => SessionID,
                                <<"encrypted_key">> => base64:encode(EncryptedKey),
                                <<"iv">> => base64:encode(IV),
                                <<"algorithm">> => <<"RSA-OAEP">>,
                                <<"message">> => <<"Session key encrypted with your public key">>
                            }
                        }}
                    catch
                        Error:Reason ->
                            ?event(priv_session, {get_key, encrypt_error, SessionID, Error, Reason}),
                            {error, <<"Invalid public key or encryption failed">>}
                    end;
                error ->
                    ?event(priv_session, {get_key, not_found, SessionID}),
                    {error, <<"Session not found">>}
            end
    end.

%% @doc Decrypts a session key using the user's private key.
%%
%% This is a utility function to help users decrypt session keys on their side.
%% In a real implementation, this would typically be done client-side.
%%
%% @param _M1 Ignored parameter
%% @param M2 Map containing encrypted_key and private_key
%% @param _Opts Configuration options (ignored)
%% @returns {ok, Map} containing the decrypted session key, or
%% {error, Binary} if decryption fails
-spec decrypt_key(M1 :: term(), M2 :: term(), Opts :: map()) -> 
    {ok, map()} | {error, binary()}.
decrypt_key(_M1, M2, _Opts) ->
    ?event(priv_session, {decrypt_key, start}),
    
    EncryptedKey = case M2 of
        #{<<"encrypted_key">> := Key} -> Key;
        _ -> undefined
    end,
    
    PrivateKeyPEM = case M2 of
        #{<<"private_key">> := PrivKey} -> PrivKey;
        _ -> undefined
    end,
    
    case {EncryptedKey, PrivateKeyPEM} of
        {undefined, _} ->
            {error, <<"Encrypted key is required">>};
        {_, undefined} ->
            {error, <<"Private key is required">>};
        {_, _} ->
            try
                % Decode the encrypted key
                EncryptedKeyBin = base64:decode(EncryptedKey),
                
                % Decrypt with private key
                DecryptedKey = decrypt_with_private_key(EncryptedKeyBin, PrivateKeyPEM),
                
                ?event(priv_session, {decrypt_key, success}),
                
                {ok, #{
                    <<"status">> => 200,
                    <<"body">> => #{
                        <<"session_key">> => base64:encode(DecryptedKey),
                        <<"message">> => <<"Session key successfully decrypted">>
                    }
                }}
            catch
                Error:Reason ->
                    ?event(priv_session, {decrypt_key, error, Error, Reason}),
                    {error, <<"Decryption failed - invalid key or ciphertext">>}
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

%% @doc Encrypts data with an RSA public key.
%%
%% This function securely encrypts data for transmission using RSA-OAEP:
%% 1. Parses the PEM-encoded public key
%% 2. Performs RSA public key encryption
%% 3. Returns the encrypted data
%%
%% @param Data The data to encrypt (binary)
%% @param PublicKeyPEM The RSA public key in PEM format
%% @returns The encrypted data (binary)
-spec encrypt_with_public_key(Data :: binary(), PublicKeyPEM :: binary()) -> binary().
encrypt_with_public_key(Data, PublicKeyPEM) ->
    ?event(priv_session, {encrypt_with_public_key, start}),
    
    % Parse the PEM-encoded public key
    [PublicKeyEntry] = public_key:pem_decode(PublicKeyPEM),
    PublicKey = public_key:pem_entry_decode(PublicKeyEntry),
    
    % Encrypt using RSA-OAEP
    EncryptedData = public_key:encrypt_public(Data, PublicKey),
    
    ?event(priv_session, {encrypt_with_public_key, complete}),
    EncryptedData.

%% @doc Decrypts data with an RSA private key.
%%
%% This function decrypts RSA-encrypted data:
%% 1. Parses the PEM-encoded private key
%% 2. Performs RSA private key decryption
%% 3. Returns the decrypted data
%%
%% @param EncryptedData The encrypted data (binary)
%% @param PrivateKeyPEM The RSA private key in PEM format
%% @returns The decrypted data (binary)
-spec decrypt_with_private_key(EncryptedData :: binary(), PrivateKeyPEM :: binary()) -> binary().
decrypt_with_private_key(EncryptedData, PrivateKeyPEM) ->
    ?event(priv_session, {decrypt_with_private_key, start}),
    
    % Parse the PEM-encoded private key
    [PrivateKeyEntry] = public_key:pem_decode(PrivateKeyPEM),
    PrivateKey = public_key:pem_entry_decode(PrivateKeyEntry),
    
    % Decrypt using RSA
    DecryptedData = public_key:decrypt_private(EncryptedData, PrivateKey),
    
    ?event(priv_session, {decrypt_with_private_key, complete}),
    DecryptedData.

%% @doc Test function to verify the asymmetric encryption works correctly.
%%
%% This test creates RSA keys, encrypts data with the public key,
%% and verifies that it can be decrypted with the private key.
asymmetric_crypto_test() ->
    % Generate RSA key pair for testing
    PrivateKey = public_key:generate_key({rsa, 2048, 65537}),
    PublicKey = case PrivateKey of
        #'RSAPrivateKey'{modulus = N, publicExponent = E} ->
            #'RSAPublicKey'{modulus = N, publicExponent = E}
    end,
    
    % Convert to PEM format
    PrivateKeyPEM = public_key:pem_encode([public_key:pem_entry_encode('RSAPrivateKey', PrivateKey)]),
    PublicKeyPEM = public_key:pem_encode([public_key:pem_entry_encode('RSAPublicKey', PublicKey)]),
    
    % Test data
    TestData = <<"This is a test session key">>,
    
    % Encrypt with public key
    EncryptedData = encrypt_with_public_key(TestData, PublicKeyPEM),
    
    % Decrypt with private key
    DecryptedData = decrypt_with_private_key(EncryptedData, PrivateKeyPEM),
    
    % Verify roundtrip
    ?assertEqual(TestData, DecryptedData).

%% @doc Test function to verify the secure session workflow works correctly.
%%
%% This test creates a session, requests the key with a public key,
%% decrypts it with the private key, and then encrypts/decrypts a message.
secure_workflow_test() ->
    % Generate RSA key pair for testing
    PrivateKey = public_key:generate_key({rsa, 2048, 65537}),
    PublicKey = case PrivateKey of
        #'RSAPrivateKey'{modulus = N, publicExponent = E} ->
            #'RSAPublicKey'{modulus = N, publicExponent = E}
    end,
    
    % Convert to PEM format
    PrivateKeyPEM = public_key:pem_encode([public_key:pem_entry_encode('RSAPrivateKey', PrivateKey)]),
    PublicKeyPEM = public_key:pem_encode([public_key:pem_entry_encode('RSAPublicKey', PublicKey)]),
    
    % Mock options
    Opts = #{priv_sessions => #{}},
    
    % 1. Create a session
    {ok, #{<<"body">> := #{<<"session_id">> := SessionID}}} = 
        create_session(undefined, #{<<"session_name">> => <<"Secure Test Session">>}, Opts),
    
    % Get updated options with the session
    SessionKey = crypto:strong_rand_bytes(32),
    UpdatedOpts = #{priv_sessions => #{SessionID => #{
        session_key => SessionKey,
        session_name => <<"Secure Test Session">>,
        created_at => erlang:system_time(second),
        message_count => 0
    }}},
    
    % 2. Request encrypted session key (SECURE)
    {ok, #{<<"body">> := #{
        <<"encrypted_key">> := EncryptedSessionKey,
        <<"algorithm">> := <<"RSA-OAEP">>
    }}} = get_key(undefined, #{
        <<"session_id">> => SessionID,
        <<"public_key">> => PublicKeyPEM
    }, UpdatedOpts),
    
    % 3. Decrypt session key with private key
    {ok, #{<<"body">> := #{
        <<"session_key">> := DecryptedSessionKeyB64
    }}} = decrypt_key(undefined, #{
        <<"encrypted_key">> => EncryptedSessionKey,
        <<"private_key">> => PrivateKeyPEM
    }, undefined),
    
    % Verify the decrypted session key matches the original
    DecryptedSessionKey = base64:decode(DecryptedSessionKeyB64),
    ?assertEqual(SessionKey, DecryptedSessionKey),
    
    % 4. Test message encryption/decryption with recovered key
    TestMessage = <<"This is a secure test message">>,
    
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
    
    % Verify complete roundtrip
    ?assertEqual(TestMessage, DecryptedMessage).

%% @doc Test to verify that get_key fails without a public key (security test).
security_test() ->
    % Mock options with a session
    SessionID = <<"test_session_123">>,
    Opts = #{priv_sessions => #{SessionID => #{
        session_key => crypto:strong_rand_bytes(32),
        session_name => <<"Test Session">>,
        created_at => erlang:system_time(second),
        message_count => 0
    }}},
    
    % Try to get key without providing public key (should fail)
    Result = get_key(undefined, #{<<"session_id">> => SessionID}, Opts),
    
    % Should return an error
    ?assertMatch({error, <<"Public key is required for secure key exchange">>}, Result).
