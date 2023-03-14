// Copyright 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "triton/core/tritonserver.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _COMPILING_TRITONBACKEND
#if defined(_MSC_VER)
#define TRITONBACKEND_DECLSPEC __declspec(dllexport)
#define TRITONBACKEND_ISPEC __declspec(dllimport)
#elif defined(__GNUC__)
#define TRITONBACKEND_DECLSPEC __attribute__((__visibility__("default")))
#define TRITONBACKEND_ISPEC
#else
#define TRITONBACKEND_DECLSPEC
#define TRITONBACKEND_ISPEC
#endif
#else
#if defined(_MSC_VER)
#define TRITONBACKEND_DECLSPEC __declspec(dllimport)
#define TRITONBACKEND_ISPEC __declspec(dllexport)
#else
#define TRITONBACKEND_DECLSPEC
#define TRITONBACKEND_ISPEC
#endif
#endif

struct TRITONBACKEND_MemoryManager;
struct TRITONBACKEND_Input;
struct TRITONBACKEND_Output;
struct TRITONBACKEND_State;
struct TRITONBACKEND_Request;
struct TRITONBACKEND_ResponseFactory;
struct TRITONBACKEND_Response;
struct TRITONBACKEND_Backend;
struct TRITONBACKEND_Model;
struct TRITONBACKEND_ModelInstance;
struct TRITONBACKEND_BackendAttribute;
struct TRITONBACKEND_Batcher;

///
/// TRITONBACKEND API Version
///
/// The TRITONBACKEND API is versioned with major and minor version
/// numbers. Any change to the API that does not impact backwards
/// compatibility (for example, adding a non-required function)
/// increases the minor version number. Any change that breaks
/// backwards compatibility (for example, deleting or changing the
/// behavior of a function) increases the major version number. A
/// backend should check that the API version used to compile the
/// backend is compatible with the API version of the Triton server
/// that it is running in. This is typically done by code similar to
/// the following which makes sure that the major versions are equal
/// and that the minor version of Triton is >= the minor version used
/// to build the backend.
///
///   uint32_t api_version_major, api_version_minor;
///   TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor);
///   if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
///       (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
///     return TRITONSERVER_ErrorNew(
///       TRITONSERVER_ERROR_UNSUPPORTED,
///       "triton backend API version does not support this backend");
///   }
///
#define TRITONBACKEND_API_VERSION_MAJOR 1
#define TRITONBACKEND_API_VERSION_MINOR 12

/// Get the TRITONBACKEND API version supported by Triton. This value
/// can be compared against the TRITONBACKEND_API_VERSION_MAJOR and
/// TRITONBACKEND_API_VERSION_MINOR used to build the backend to
/// ensure that Triton is compatible with the backend.
///
/// \param major Returns the TRITONBACKEND API major version supported
/// by Triton.
/// \param minor Returns the TRITONBACKEND API minor version supported
/// by Triton.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_ApiVersion(
    uint32_t* major, uint32_t* minor);

/// TRITONBACKEND_ArtifactType
///
/// The ways that the files that make up a backend or model are
/// communicated to the backend.
///
///   TRITONBACKEND_ARTIFACT_FILESYSTEM: The model or backend
///     artifacts are made available to Triton via a locally
///     accessible filesystem. The backend can access these files
///     using an appropriate system API.
///
typedef enum TRITONBACKEND_artifacttype_enum {
  TRITONBACKEND_ARTIFACT_FILESYSTEM
} TRITONBACKEND_ArtifactType;


///
/// TRITONBACKEND_MemoryManager
///
/// Object representing an memory manager that is capable of
/// allocating and otherwise managing different memory types. For
/// improved performance Triton maintains pools for GPU and CPU-pinned
/// memory and the memory manager allows backends to access those
/// pools.
///

/// Allocate a contiguous block of memory of a specific type using a
/// memory manager. Two error codes have specific interpretations for
/// this function:
///
///   TRITONSERVER_ERROR_UNSUPPORTED: Indicates that Triton is
///     incapable of allocating the requested memory type and memory
///     type ID. Requests for the memory type and ID will always fail
///     no matter 'byte_size' of the request.
///
///   TRITONSERVER_ERROR_UNAVAILABLE: Indicates that Triton can
///      allocate the memory type and ID but that currently it cannot
///      allocate a contiguous block of memory of the requested
///      'byte_size'.
///
/// \param manager The memory manager.
/// \param buffer Returns the allocated memory.
/// \param memory_type The type of memory to allocate.
/// \param memory_type_id The ID associated with the memory type to
/// allocate. For GPU memory this indicates the device ID of the GPU
/// to allocate from.
/// \param byte_size The size of memory to allocate, in bytes.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_MemoryManagerAllocate(
    TRITONBACKEND_MemoryManager* manager, void** buffer,
    const TRITONSERVER_MemoryType memory_type, const int64_t memory_type_id,
    const uint64_t byte_size);

/// Free a buffer that was previously allocated with
/// TRITONBACKEND_MemoryManagerAllocate. The call must provide the
/// same values for 'memory_type' and 'memory_type_id' as were used
/// when the buffer was allocate or else the behavior is undefined.
///
/// \param manager The memory manager.
/// \param buffer The allocated memory buffer to free.
/// \param memory_type The type of memory of the buffer.
/// \param memory_type_id The ID associated with the memory type of
/// the buffer.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_MemoryManagerFree(
    TRITONBACKEND_MemoryManager* manager, void* buffer,
    const TRITONSERVER_MemoryType memory_type, const int64_t memory_type_id);

///
/// TRITONBACKEND_Input
///
/// Object representing an input tensor.
///

/// Get the name and properties of an input tensor. The returned
/// strings and other properties are owned by the input, not the
/// caller, and so should not be modified or freed.
///
/// \param input The input tensor.
/// \param name If non-nullptr, returns the tensor name.
/// \param datatype If non-nullptr, returns the tensor datatype.
/// \param shape If non-nullptr, returns the tensor shape.
/// \param dim_count If non-nullptr, returns the number of dimensions
/// in the tensor shape.
/// \param byte_size If non-nullptr, returns the size of the available
/// data for the tensor, in bytes. This size reflects the actual data
/// available, and does not necessarily match what is
/// expected/required for the tensor given its shape and datatype. It
/// is the responsibility of the backend to handle mismatches in these
/// sizes appropriately.
/// \param buffer_count If non-nullptr, returns the number of buffers
/// holding the contents of the tensor. These buffers are accessed
/// using TRITONBACKEND_InputBuffer.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_InputProperties(
    TRITONBACKEND_Input* input, const char** name,
    TRITONSERVER_DataType* datatype, const int64_t** shape,
    uint32_t* dims_count, uint64_t* byte_size, uint32_t* buffer_count);

/// Get the name and properties of an input tensor associated with a given
/// host policy. If there are no input buffers for the specified  host policy,
/// the properties of the fallback input buffers are returned. The returned
/// strings and other properties are owned by the input, not the caller, and so
/// should not be modified or freed.
///
/// \param input The input tensor.
/// \param host_policy_name The host policy name. Fallback input properties
/// will be return if nullptr is provided.
/// \param name If non-nullptr, returns the tensor name.
/// \param datatype If non-nullptr, returns the tensor datatype.
/// \param shape If non-nullptr, returns the tensor shape.
/// \param dim_count If non-nullptr, returns the number of dimensions
/// in the tensor shape.
/// \param byte_size If non-nullptr, returns the size of the available
/// data for the tensor, in bytes. This size reflects the actual data
/// available, and does not necessarily match what is
/// expected/required for the tensor given its shape and datatype. It
/// is the responsibility of the backend to handle mismatches in these
/// sizes appropriately.
/// \param buffer_count If non-nullptr, returns the number of buffers
/// holding the contents of the tensor. These buffers are accessed
/// using TRITONBACKEND_InputBufferForHostPolicy.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_InputPropertiesForHostPolicy(
    TRITONBACKEND_Input* input, const char* host_policy_name, const char** name,
    TRITONSERVER_DataType* datatype, const int64_t** shape,
    uint32_t* dims_count, uint64_t* byte_size, uint32_t* buffer_count);

/// Get a buffer holding (part of) the tensor data for an input. For a
/// given input the number of buffers composing the input are found
/// from 'buffer_count' returned by TRITONBACKEND_InputProperties. The
/// returned buffer is owned by the input and so should not be
/// modified or freed by the caller. The lifetime of the buffer
/// matches that of the input and so the buffer should not be accessed
/// after the input tensor object is released.
///
/// \param input The input tensor.
/// \param index The index of the buffer. Must be 0 <= index <
/// buffer_count, where buffer_count is the value returned by
/// TRITONBACKEND_InputProperties.
/// \param buffer Returns a pointer to a contiguous block of data for
/// the named input.
/// \param buffer_byte_size Returns the size, in bytes, of 'buffer'.
/// \param memory_type Acts as both input and output. On input gives
/// the buffer memory type preferred by the function caller.  Returns
/// the actual memory type of 'buffer'.
/// \param memory_type_id Acts as both input and output. On input
/// gives the buffer memory type id preferred by the function caller.
/// Returns the actual memory type id of 'buffer'.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_InputBuffer(
    TRITONBACKEND_Input* input, const uint32_t index, const void** buffer,
    uint64_t* buffer_byte_size, TRITONSERVER_MemoryType* memory_type,
    int64_t* memory_type_id);

/// Get a buffer holding (part of) the tensor data for an input for a specific
/// host policy. If there are no input buffers specified for this host policy,
/// the fallback input buffer is returned.
/// For a given input the number of buffers composing the input are found
/// from 'buffer_count' returned by TRITONBACKEND_InputPropertiesForHostPolicy.
/// The returned buffer is owned by the input and so should not be modified or
/// freed by the caller. The lifetime of the buffer matches that of the input
/// and so the buffer should not be accessed after the input tensor object is
/// released.
///
/// \param input The input tensor.
/// \param host_policy_name The host policy name. Fallback input buffer
/// will be return if nullptr is provided.
/// \param index The index of the buffer. Must be 0 <= index <
/// buffer_count, where buffer_count is the value returned by
/// TRITONBACKEND_InputPropertiesForHostPolicy.
/// \param buffer Returns a pointer to a contiguous block of data for
/// the named input.
/// \param buffer_byte_size Returns the size, in bytes, of 'buffer'.
/// \param memory_type Acts as both input and output. On input gives
/// the buffer memory type preferred by the function caller.  Returns
/// the actual memory type of 'buffer'.
/// \param memory_type_id Acts as both input and output. On input
/// gives the buffer memory type id preferred by the function caller.
/// Returns the actual memory type id of 'buffer'.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_InputBufferForHostPolicy(
    TRITONBACKEND_Input* input, const char* host_policy_name,
    const uint32_t index, const void** buffer, uint64_t* buffer_byte_size,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id);

/// Get the buffer attributes associated with the given input buffer. For a
/// given input the number of buffers composing the input are found from
/// 'buffer_count' returned by TRITONBACKEND_InputProperties. The returned
/// 'buffer_attributes' is owned by the input and so should not be modified or
/// freed by the caller. The lifetime of the 'buffer_attributes' matches that of
/// the input and so the 'buffer_attributes' should not be accessed after the
/// input tensor object is released.
///
/// \param input The input tensor.
/// \param index The index of the buffer. Must be 0 <= index < buffer_count,
/// where buffer_count is the value returned by TRITONBACKEND_InputProperties.
/// \param buffer Returns a pointer to a contiguous block of data for
/// the named input.
/// \param buffer_attributes Returns the attributes for the given buffer.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_InputBufferAttributes(
    TRITONBACKEND_Input* input, const uint32_t index, const void** buffer,
    TRITONSERVER_BufferAttributes** buffer_attributes);

///
/// TRITONBACKEND_Output
///
/// Object representing a response output tensor.
///

/// Get a buffer to use to hold the tensor data for the output. The
/// returned buffer is owned by the output and so should not be freed
/// by the caller. The caller can and should fill the buffer with the
/// output data for the tensor. The lifetime of the buffer matches
/// that of the output and so the buffer should not be accessed after
/// the output tensor object is released.
///
/// \param buffer Returns a pointer to a buffer where the contents of
/// the output tensor should be placed.
/// \param buffer_byte_size The size, in bytes, of the buffer required
/// by the caller.
/// \param memory_type Acts as both input and output. On input gives
/// the buffer memory type preferred by the caller.  Returns the
/// actual memory type of 'buffer'.
/// \param memory_type_id Acts as both input and output. On input
/// gives the buffer memory type id preferred by the caller. Returns
/// the actual memory type id of 'buffer'.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_OutputBuffer(
    TRITONBACKEND_Output* output, void** buffer,
    const uint64_t buffer_byte_size, TRITONSERVER_MemoryType* memory_type,
    int64_t* memory_type_id);

/// Get the buffer attributes associated with the given output buffer. The
/// returned 'buffer_attributes' is owned by the output and so should not be
/// modified or freed by the caller. The lifetime of the 'buffer_attributes'
/// matches that of the output and so the 'buffer_attributes' should not be
/// accessed after the output tensor object is released. This function must be
/// called after the TRITONBACKEND_OutputBuffer otherwise it might contain
/// incorrect data.
///
/// \param output The output tensor.
/// \param buffer_attributes Returns the attributes for the output buffer.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_OutputBufferAttributes(
    TRITONBACKEND_Output* output,
    TRITONSERVER_BufferAttributes** buffer_attributes);

///
/// TRITONBACKEND_Request
///
/// Object representing an inference request.
///

/// Get the ID of the request. Can be nullptr if request doesn't have
/// an ID. The returned string is owned by the request, not the
/// caller, and so should not be modified or freed.
///
/// \param request The inference request.
/// \param id Returns the ID.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_RequestId(
    TRITONBACKEND_Request* request, const char** id);

/// Get the correlation ID of the request if it is an unsigned integer.
/// Zero indicates that the request does not have a correlation ID.
/// Returns failure if correlation ID for given request is not an unsigned
/// integer.
///
/// \param request The inference request.
/// \param id Returns the correlation ID.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_RequestCorrelationId(
    TRITONBACKEND_Request* request, uint64_t* id);

/// Get the correlation ID of the request if it is a string.
/// Empty string indicates that the request does not have a correlation ID.
/// Returns error if correlation ID for given request is not a string.
///
/// \param request The inference request.
/// \param id Returns the correlation ID.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_RequestCorrelationIdString(
    TRITONBACKEND_Request* request, const char** id);

/// Get the flag(s) associated with a request. On return 'flags' holds
/// a bitwise-or of all flag values, see TRITONSERVER_RequestFlag for
/// available flags.
///
/// \param request The inference request.
/// \param flags Returns the flags.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_RequestFlags(
    TRITONBACKEND_Request* request, uint32_t* flags);

/// Get the number of parameters specified in the inference request.
///
/// \param request The inference request.
/// \param count Returns the number of parameters.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_RequestParameterCount(
    TRITONBACKEND_Request* request, uint32_t* count);

/// Get a request parameters by index. The order of parameters in a given
/// request is not necessarily consistent with other requests, even if
/// the requests are in the same batch. As a result, you can not
/// assume that an index obtained from one request will point to the
/// same parameter in a different request.
///
/// The lifetime of the returned parameter object matches that of the
/// request and so the parameter object should not be accessed after the
/// request object is released.
///
/// \param request The inference request.
/// \param index The index of the parameter. Must be 0 <= index <
/// count, where count is the value returned by
/// TRITONBACKEND_RequestParameterCount.
/// \param key Returns the key of the parameter.
/// \param type Returns the type of the parameter.
/// \param vvalue Returns a pointer to the parameter value.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_RequestParameter(
    TRITONBACKEND_Request* request, const uint32_t index, const char** key,
    TRITONSERVER_ParameterType* type, const void** vvalue);

/// Get the number of input tensors specified in the request.
///
/// \param request The inference request.
/// \param count Returns the number of input tensors.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_RequestInputCount(
    TRITONBACKEND_Request* request, uint32_t* count);

/// Get the name of an input tensor. The caller does not own
/// the returned string and must not modify or delete it. The lifetime
/// of the returned string extends only as long as 'request'.
///
/// \param request The inference request.
/// \param index The index of the input tensor. Must be 0 <= index <
/// count, where count is the value returned by
/// TRITONBACKEND_RequestInputCount.
/// \param input_name Returns the name of the input tensor
/// corresponding to the index.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_RequestInputName(
    TRITONBACKEND_Request* request, const uint32_t index,
    const char** input_name);

/// Get a named request input. The lifetime of the returned input
/// object matches that of the request and so the input object should
/// not be accessed after the request object is released.
///
/// \param request The inference request.
/// \param name The name of the input.
/// \param input Returns the input corresponding to the name.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_RequestInput(
    TRITONBACKEND_Request* request, const char* name,
    TRITONBACKEND_Input** input);

/// Get a request input by index. The order of inputs in a given
/// request is not necessarily consistent with other requests, even if
/// the requests are in the same batch. As a result, you can not
/// assume that an index obtained from one request will point to the
/// same input in a different request.
///
/// The lifetime of the returned input object matches that of the
/// request and so the input object should not be accessed after the
/// request object is released.
///
/// \param request The inference request.
/// \param index The index of the input tensor. Must be 0 <= index <
/// count, where count is the value returned by
/// TRITONBACKEND_RequestInputCount.
/// \param input Returns the input corresponding to the index.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_RequestInputByIndex(
    TRITONBACKEND_Request* request, const uint32_t index,
    TRITONBACKEND_Input** input);

/// Get the number of output tensors requested to be returned in the
/// request.
///
/// \param request The inference request.
/// \param count Returns the number of output tensors.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_RequestOutputCount(
    TRITONBACKEND_Request* request, uint32_t* count);

/// Get the name of a requested output tensor. The caller does not own
/// the returned string and must not modify or delete it. The lifetime
/// of the returned string extends only as long as 'request'.
///
/// \param request The inference request.
/// \param index The index of the requested output tensor. Must be 0
/// <= index < count, where count is the value returned by
/// TRITONBACKEND_RequestOutputCount.
/// \param output_name Returns the name of the requested output tensor
/// corresponding to the index.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_RequestOutputName(
    TRITONBACKEND_Request* request, const uint32_t index,
    const char** output_name);

/// Returns the preferred memory type and memory type ID of the output buffer
/// for the request. As much as possible, Triton will attempt to return
/// the same memory_type and memory_type_id values that will be returned by
/// the subsequent call to TRITONBACKEND_OutputBuffer, however, the backend must
/// be capable of handling cases where the values differ.
///
/// \param request The request.
/// \param name The name of the output tensor. This is optional
/// and it should be set to nullptr to indicate that the tensor name has
/// not determined.
/// \param byte_size The expected size of the buffer. This is optional
/// and it should be set to nullptr to indicate that the byte size has
/// not determined.
/// \param memory_type Acts as both input and output. On input gives
/// the memory type preferred by the caller. Returns memory type preferred
/// by Triton, taken account of the caller preferred type.
/// \param memory_type_id Acts as both input and output. On input gives
/// the memory type ID preferred by the caller. Returns memory type ID preferred
/// by Triton, taken account of the caller preferred type ID.
/// \return a TRITONSERVER_Error object if a failure occurs.
/// A TRITONSERVER_ERROR_UNAVAILABLE error indicates that the properties are not
/// available, other error codes indicate an error.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_RequestOutputBufferProperties(
    TRITONBACKEND_Request* request, const char* name, size_t* byte_size,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id);

/// Release the request. The request should be released when it is no
/// longer needed by the backend. If this call returns with an error
/// (i.e. non-nullptr) then the request was not released and ownership
/// remains with the backend. If this call returns with success, the
/// 'request' object is no longer owned by the backend and must not be
/// used. Any tensor names, data types, shapes, input tensors,
/// etc. returned by TRITONBACKEND_Request* functions for this request
/// are no longer valid. If a persistent copy of that data is required
/// it must be created before calling this function.
///
/// \param request The inference request.
/// \param release_flags Flags indicating what type of request release
/// should be performed. \see TRITONSERVER_RequestReleaseFlag. \see
/// TRITONSERVER_InferenceRequestReleaseFn_t.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_RequestRelease(
    TRITONBACKEND_Request* request, uint32_t release_flags);

///
/// TRITONBACKEND_ResponseFactory
///
/// Object representing an inference response factory. Using a
/// response factory is not required; instead a response can be
/// generated directly from a TRITONBACKEND_Request object using
/// TRITONBACKEND_ResponseNew(). A response factory allows a request
/// to be released before all responses have been sent. Releasing a
/// request as early as possible releases all input tensor data and
/// therefore may be desirable in some cases.

/// Create the response factory associated with a request.
///
/// \param factory Returns the new response factory.
/// \param request The inference request.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_ResponseFactoryNew(
    TRITONBACKEND_ResponseFactory** factory, TRITONBACKEND_Request* request);

/// Destroy a response factory.
///
/// \param factory The response factory.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_ResponseFactoryDelete(
    TRITONBACKEND_ResponseFactory* factory);

/// Send response flags without a corresponding response.
///
/// \param factory The response factory.
/// \param send_flags Flags to send. \see
/// TRITONSERVER_ResponseCompleteFlag. \see
/// TRITONSERVER_InferenceResponseCompleteFn_t.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ResponseFactorySendFlags(
    TRITONBACKEND_ResponseFactory* factory, const uint32_t send_flags);

///
/// TRITONBACKEND_Response
///
/// Object representing an inference response. For a given request,
/// the backend must carefully manage the lifecycle of responses
/// generated for that request to ensure that the output tensor
/// buffers are allocated correctly. When a response is created with
/// TRITONBACKEND_ResponseNew or TRITONBACKEND_ResponseNewFromFactory,
/// all the outputs and corresponding buffers must be created for that
/// response using TRITONBACKEND_ResponseOutput and
/// TRITONBACKEND_OutputBuffer *before* another response is created
/// for the request. For a given response, outputs can be created in
/// any order but they must be created sequentially/sychronously (for
/// example, the backend cannot use multiple threads to simultaneously
/// add multiple outputs to a response).
///
/// The above requirement applies only to responses being generated
/// for a given request. The backend may generate responses in
/// parallel on multiple threads as long as those responses are for
/// different requests.
///
/// This order of response creation must be strictly followed. But,
/// once response(s) are created they do not need to be sent
/// immediately, nor do they need to be sent in the order they were
/// created. The backend may even delete a created response instead of
/// sending it by using TRITONBACKEND_ResponseDelete.

/// Create a response for a request.
///
/// \param response Returns the new response.
/// \param request The request.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_ResponseNew(
    TRITONBACKEND_Response** response, TRITONBACKEND_Request* request);

/// Create a response using a factory.
///
/// \param response Returns the new response.
/// \param factory The response factory.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_ResponseNewFromFactory(
    TRITONBACKEND_Response** response, TRITONBACKEND_ResponseFactory* factory);

/// Destroy a response. It is not necessary to delete a response if
/// TRITONBACKEND_ResponseSend is called as that function transfers
/// ownership of the response object to Triton.
///
/// \param response The response.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_ResponseDelete(
    TRITONBACKEND_Response* response);

/// Set a string parameter in the response.
///
/// \param response The response.
/// \param name The name of the parameter.
/// \param value The value of the parameter.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ResponseSetStringParameter(
    TRITONBACKEND_Response* response, const char* name, const char* value);

/// Set an integer parameter in the response.
///
/// \param response The response.
/// \param name The name of the parameter.
/// \param value The value of the parameter.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ResponseSetIntParameter(
    TRITONBACKEND_Response* response, const char* name, const int64_t value);

/// Set a boolean parameter in the response.
///
/// \param response The response.
/// \param name The name of the parameter.
/// \param value The value of the parameter.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ResponseSetBoolParameter(
    TRITONBACKEND_Response* response, const char* name, const bool value);

/// Create an output tensor in the response. The lifetime of the
/// returned output tensor object matches that of the response and so
/// the output tensor object should not be accessed after the response
/// object is deleted.
///
/// \param response The response.
/// \param output Returns the new response output.
/// \param name The name of the output tensor.
/// \param datatype The datatype of the output tensor.
/// \param shape The shape of the output tensor.
/// \param dims_count The number of dimensions in the output tensor
/// shape.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_ResponseOutput(
    TRITONBACKEND_Response* response, TRITONBACKEND_Output** output,
    const char* name, const TRITONSERVER_DataType datatype,
    const int64_t* shape, const uint32_t dims_count);

/// Send a response. Calling this function transfers ownership of the
/// response object to Triton. The caller must not access or delete
/// the response object after calling this function.
///
/// \param response The response.
/// \param send_flags Flags associated with the response. \see
/// TRITONSERVER_ResponseCompleteFlag. \see
/// TRITONSERVER_InferenceResponseCompleteFn_t.
/// \param error The TRITONSERVER_Error to send if the response is an
/// error, or nullptr if the response is successful.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_ResponseSend(
    TRITONBACKEND_Response* response, const uint32_t send_flags,
    TRITONSERVER_Error* error);

///
/// TRITONBACKEND_State
///
/// Object representing a state.
///

/// Create a state in the request. The returned state object is only valid
/// before the TRITONBACKEND_StateUpdate is called. The state should not be
/// freed by the caller. If TRITONBACKEND_StateUpdate is not called, the
/// lifetime of the state matches the lifetime of the request. If the state name
/// does not exist in the "state" section of the model configuration, the state
/// will not be created and an error will be returned. If this function is
/// called when sequence batching is not enabled or there is no 'states' section
/// in the sequence batching section of the model configuration, this call will
/// return an error.
///
/// \param state Returns the new state.
/// \param request The request.
/// \param name The name of the state.
/// \param datatype The datatype of the state.
/// \param shape The shape of the state.
/// \param dims_count The number of dimensions in the state shape.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_StateNew(
    TRITONBACKEND_State** state, TRITONBACKEND_Request* request,
    const char* name, const TRITONSERVER_DataType datatype,
    const int64_t* shape, const uint32_t dims_count);

/// Update the state for the sequence. Calling this function will replace the
/// state stored for this seqeunce in Triton with 'state' provided in the
/// function argument. If this function is called when sequence batching is not
/// enabled or there is no 'states' section in the sequence batching section of
/// the model configuration, this call will return an error. The backend is not
/// required to call this function. If the backend doesn't call
/// TRITONBACKEND_StateUpdate function, this particular state for the sequence
/// will not be updated and the next inference request in the sequence will use
/// the same state as the current inference request.
///
/// \param state The state.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_StateUpdate(
    TRITONBACKEND_State* state);

/// Get a buffer to use to hold the tensor data for the state. The returned
/// buffer is owned by the state and so should not be freed by the caller. The
/// caller can and should fill the buffer with the state data. The buffer must
/// not be accessed by the backend after TRITONBACKEND_StateUpdate is called.
/// The caller should fill the buffer before calling TRITONBACKEND_StateUpdate.
///
/// \param state The state.
/// \param buffer Returns a pointer to a buffer where the contents of the state
/// should be placed.
/// \param buffer_byte_size The size, in bytes, of the buffer required
/// by the caller.
/// \param memory_type Acts as both input and output. On input gives
/// the buffer memory type preferred by the caller.  Returns the
/// actual memory type of 'buffer'.
/// \param memory_type_id Acts as both input and output. On input
/// gives the buffer memory type id preferred by the caller. Returns
/// the actual memory type id of 'buffer'.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_StateBuffer(
    TRITONBACKEND_State* state, void** buffer, const uint64_t buffer_byte_size,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id);

/// Get the buffer attributes associated with the given state buffer.
/// The returned 'buffer_attributes' is owned by the state and so should not be
/// modified or freed by the caller. The lifetime of the 'buffer_attributes'
/// matches that of the state.
///
/// \param state The state.
/// \param buffer_attributes Returns the buffer attributes for the given state.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_StateBufferAttributes(
    TRITONBACKEND_State* state,
    TRITONSERVER_BufferAttributes** buffer_attributes);

///
/// TRITONBACKEND_Backend
///
/// Object representing a backend.
///

/// TRITONBACKEND_ExecutionPolicy
///
/// Types of execution policy that can be implemented by a backend.
///
///   TRITONBACKEND_EXECUTION_BLOCKING: An instance of the model
///     blocks in TRITONBACKEND_ModelInstanceExecute until it is ready
///     to handle another inference. Upon returning from
///     TRITONBACKEND_ModelInstanceExecute, Triton may immediately
///     call TRITONBACKEND_ModelInstanceExecute for the same instance
///     to execute a new batch of requests. Thus, most backends using
///     this policy will not return from
///     TRITONBACKEND_ModelInstanceExecute until all responses have
///     been sent and all requests have been released. This is the
///     default execution policy.
///
///   TRITONBACKEND_EXECUTION_DEVICE_BLOCKING: An instance, A, of the
///     model blocks in TRITONBACKEND_ModelInstanceExecute if the
///     device associated with the instance is unable to handle
///     another inference. Even if another instance, B, associated
///     with the device, is available and ready to perform an
///     inference, Triton will not invoke
///     TRITONBACKEND_ModeInstanceExecute for B until A returns from
///     TRITONBACKEND_ModelInstanceExecute. Triton will not be blocked
///     from calling TRITONBACKEND_ModelInstanceExecute for instance
///     C, which is associated with a different device than A and B,
///     even if A or B has not returned from
///     TRITONBACKEND_ModelInstanceExecute. This execution policy is
///     typically used by a backend that can cooperatively execute
///     multiple model instances on the same device.
///
typedef enum TRITONBACKEND_execpolicy_enum {
  TRITONBACKEND_EXECUTION_BLOCKING,
  TRITONBACKEND_EXECUTION_DEVICE_BLOCKING
} TRITONBACKEND_ExecutionPolicy;

/// Get the name of the backend. The caller does not own the returned
/// string and must not modify or delete it. The lifetime of the
/// returned string extends only as long as 'backend'.
///
/// \param backend The backend.
/// \param name Returns the name of the backend.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_BackendName(
    TRITONBACKEND_Backend* backend, const char** name);

/// Get the backend configuration.  The 'backend_config' message is
/// owned by Triton and should not be modified or freed by the caller.
///
/// The backend configuration, as JSON, is:
///
///   {
///     "cmdline" : {
///       "<setting>" : "<value>",
///       ...
///     }
///   }
///
/// \param backend The backend.
/// \param backend_config Returns the backend configuration as a message.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_BackendConfig(
    TRITONBACKEND_Backend* backend, TRITONSERVER_Message** backend_config);

/// Get the execution policy for this backend. By default the
/// execution policy is TRITONBACKEND_EXECUTION_BLOCKING.
///
/// \param backend The backend.
/// \param policy Returns the execution policy.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_BackendExecutionPolicy(
    TRITONBACKEND_Backend* backend, TRITONBACKEND_ExecutionPolicy* policy);

/// Set the execution policy for this backend. By default the
/// execution policy is TRITONBACKEND_EXECUTION_BLOCKING. Triton reads
/// the backend's execution policy after calling
/// TRITONBACKEND_Initialize, so to be recognized changes to the
/// execution policy must be made in TRITONBACKEND_Initialize.
/// Also, note that if using sequence batcher for the model, Triton will
/// use TRITONBACKEND_EXECUTION_BLOCKING policy irrespective of the
/// policy specified by this setter function.
///
/// \param backend The backend.
/// \param policy The execution policy.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_BackendSetExecutionPolicy(
    TRITONBACKEND_Backend* backend, TRITONBACKEND_ExecutionPolicy policy);

/// Get the location of the files that make up the backend
/// implementation. This location contains the backend shared library
/// and any other files located with the shared library. The
/// 'location' communicated depends on how the backend is being
/// communicated to Triton as indicated by 'artifact_type'.
///
///   TRITONBACKEND_ARTIFACT_FILESYSTEM: The backend artifacts are
///     made available to Triton via the local filesytem. 'location'
///     returns the full path to the directory containing this
///     backend's artifacts. The returned string is owned by Triton,
///     not the caller, and so should not be modified or freed.
///
/// \param backend The backend.
/// \param artifact_type Returns the artifact type for the backend.
/// \param path Returns the location.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_BackendArtifacts(
    TRITONBACKEND_Backend* backend, TRITONBACKEND_ArtifactType* artifact_type,
    const char** location);

/// Get the memory manager associated with a backend.
///
/// \param backend The backend.
/// \param manager Returns the memory manager.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_BackendMemoryManager(
    TRITONBACKEND_Backend* backend, TRITONBACKEND_MemoryManager** manager);

/// Get the user-specified state associated with the backend. The
/// state is completely owned and managed by the backend.
///
/// \param backend The backend.
/// \param state Returns the user state, or nullptr if no user state.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_BackendState(
    TRITONBACKEND_Backend* backend, void** state);

/// Set the user-specified state associated with the backend. The
/// state is completely owned and managed by the backend.
///
/// \param backend The backend.
/// \param state The user state, or nullptr if no user state.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_BackendSetState(
    TRITONBACKEND_Backend* backend, void* state);

///
/// TRITONBACKEND_Model
///
/// Object representing a model implemented using the backend.
///

/// Get the name of the model. The returned string is owned by the
/// model object, not the caller, and so should not be modified or
/// freed.
///
/// \param model The model.
/// \param name Returns the model name.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_ModelName(
    TRITONBACKEND_Model* model, const char** name);

/// Get the version of the model.
///
/// \param model The model.
/// \param version Returns the model version.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_ModelVersion(
    TRITONBACKEND_Model* model, uint64_t* version);

/// Get the location of the files that make up the model. The
/// 'location' communicated depends on how the model is being
/// communicated to Triton as indicated by 'artifact_type'.
///
///   TRITONBACKEND_ARTIFACT_FILESYSTEM: The model artifacts are made
///     available to Triton via the local filesytem. 'location'
///     returns the full path to the directory in the model repository
///     that contains this model's artifacts. The returned string is
///     owned by Triton, not the caller, and so should not be modified
///     or freed.
///
/// \param model The model.
/// \param artifact_type Returns the artifact type for the model.
/// \param path Returns the location.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_ModelRepository(
    TRITONBACKEND_Model* model, TRITONBACKEND_ArtifactType* artifact_type,
    const char** location);

/// Get the model configuration. The caller takes ownership of the
/// message object and must call TRITONSERVER_MessageDelete to release
/// the object. The configuration is available via this call even
/// before the model is loaded and so can be used in
/// TRITONBACKEND_ModelInitialize. TRITONSERVER_ServerModelConfig
/// returns equivalent information but is not useable until after the
/// model loads.
///
/// \param model The model.
/// \param config_version The model configuration will be returned in
/// a format matching this version. If the configuration cannot be
/// represented in the requested version's format then an error will
/// be returned. Currently only version 1 is supported.
/// \param model_config Returns the model configuration as a message.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_ModelConfig(
    TRITONBACKEND_Model* model, const uint32_t config_version,
    TRITONSERVER_Message** model_config);

/// Whether the backend should attempt to auto-complete the model configuration.
/// If true, the model should fill the inputs, outputs, and max batch size in
/// the model configuration if incomplete. If the model configuration is
/// changed,  the new configuration must be reported to Triton using
/// TRITONBACKEND_ModelSetConfig.
///
/// \param model The model.
/// \param auto_complete_config Returns whether the backend should auto-complete
/// the model configuration.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelAutoCompleteConfig(
    TRITONBACKEND_Model* model, bool* auto_complete_config);

/// Set the model configuration in Triton server. This API should only be called
/// when the backend implements the auto-completion of model configuration
/// and TRITONBACKEND_ModelAutoCompleteConfig returns true in
/// auto_complete_config. Only the inputs, outputs, max batch size, and
/// scheduling choice can be changed. A caveat being scheduling choice can only
/// be changed if none is previously set. Any other changes to the model
/// configuration will be ignored by Triton. This function can only be called
/// from TRITONBACKEND_ModelInitialize, calling in any other context will result
/// in an error being returned. Additionally, Triton server can add some of the
/// missing fields in the provided config with this call. The backend must get
/// the complete configuration again by using TRITONBACKEND_ModelConfig.
/// TRITONBACKEND_ModelSetConfig does not take ownership of the message object
/// and so the caller should call TRITONSERVER_MessageDelete to release the
/// object once the function returns.
///
/// \param model The model.
/// \param config_version The format version of the model configuration.
/// If the configuration is not represented in the version's format
/// then an error will be returned. Currently only version 1 is supported.
/// \param model_config The updated model configuration as a message.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_ModelSetConfig(
    TRITONBACKEND_Model* model, const uint32_t config_version,
    TRITONSERVER_Message* model_config);

/// Get the TRITONSERVER_Server object that this model is being served
/// by.
///
/// \param model The model.
/// \param server Returns the server.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_ModelServer(
    TRITONBACKEND_Model* model, TRITONSERVER_Server** server);

/// Get the backend used by the model.
///
/// \param model The model.
/// \param model Returns the backend object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_ModelBackend(
    TRITONBACKEND_Model* model, TRITONBACKEND_Backend** backend);

/// Get the user-specified state associated with the model. The
/// state is completely owned and managed by the backend.
///
/// \param model The model.
/// \param state Returns the user state, or nullptr if no user state.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_ModelState(
    TRITONBACKEND_Model* model, void** state);

/// Set the user-specified state associated with the model. The
/// state is completely owned and managed by the backend.
///
/// \param model The model.
/// \param state The user state, or nullptr if no user state.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_ModelSetState(
    TRITONBACKEND_Model* model, void* state);

///
/// TRITONBACKEND_ModelInstance
///
/// Object representing a model instance implemented using the
/// backend.
///

/// Get the name of the model instance. The returned string is owned by the
/// model object, not the caller, and so should not be modified or
/// freed.
///
/// \param instance The model instance.
/// \param name Returns the instance name.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_ModelInstanceName(
    TRITONBACKEND_ModelInstance* instance, const char** name);

/// Get the kind of the model instance.
///
/// \param instance The model instance.
/// \param kind Returns the instance kind.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_ModelInstanceKind(
    TRITONBACKEND_ModelInstance* instance,
    TRITONSERVER_InstanceGroupKind* kind);

/// Get the device ID of the model instance.
///
/// \param instance The model instance.
/// \param device_id Returns the instance device ID.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_ModelInstanceDeviceId(
    TRITONBACKEND_ModelInstance* instance, int32_t* device_id);

/// Get the host policy setting.  The 'host_policy' message is
/// owned by Triton and should not be modified or freed by the caller.
///
/// The host policy setting, as JSON, is:
///
///   {
///     "<host_policy>" : {
///       "<setting>" : "<value>",
///       ...
///     }
///   }
///
/// \param instance The model instance.
/// \param host_policy Returns the host policy setting as a message.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceHostPolicy(
    TRITONBACKEND_ModelInstance* instance, TRITONSERVER_Message** host_policy);

/// Whether the model instance is passive.
///
/// \param instance The model instance.
/// \param is_passive Returns true if the instance is passive, false otherwise
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_ModelInstanceIsPassive(
    TRITONBACKEND_ModelInstance* instance, bool* is_passive);

/// Get the number of optimization profiles to be loaded for the instance.
///
/// \param instance The model instance.
/// \param count Returns the number of optimization profiles.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceProfileCount(
    TRITONBACKEND_ModelInstance* instance, uint32_t* count);

/// Get the name of optimization profile. The caller does not own
/// the returned string and must not modify or delete it. The lifetime
/// of the returned string extends only as long as 'instance'.
///
/// \param instance The model instance.
/// \param index The index of the optimization profile. Must be 0
/// <= index < count, where count is the value returned by
/// TRITONBACKEND_ModelInstanceProfileCount.
/// \param profile_name Returns the name of the optimization profile
/// corresponding to the index.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceProfileName(
    TRITONBACKEND_ModelInstance* instance, const uint32_t index,
    const char** profile_name);

/// Get the number of secondary devices configured for the instance.
///
/// \param instance The model instance.
/// \param count Returns the number of secondary devices.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceSecondaryDeviceCount(
    TRITONBACKEND_ModelInstance* instance, uint32_t* count);

/// Get the properties of indexed secondary device. The returned
/// strings and other properties are owned by the instance, not the
/// caller, and so should not be modified or freed.
///
/// \param instance The model instance.
/// \param index The index of the secondary device. Must be 0
/// <= index < count, where count is the value returned by
/// TRITONBACKEND_ModelInstanceSecondaryDeviceCount.
/// \param kind Returns the kind of secondary device corresponding
/// to the index.
/// \param id Returns the id of secondary device corresponding to the index.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceSecondaryDeviceProperties(
    TRITONBACKEND_ModelInstance* instance, uint32_t index, const char** kind,
    int64_t* id);

/// Get the model associated with a model instance.
///
/// \param instance The model instance.
/// \param backend Returns the model object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_ModelInstanceModel(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Model** model);

/// Get the user-specified state associated with the model
/// instance. The state is completely owned and managed by the
/// backend.
///
/// \param instance The model instance.
/// \param state Returns the user state, or nullptr if no user state.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_ModelInstanceState(
    TRITONBACKEND_ModelInstance* instance, void** state);

/// Set the user-specified state associated with the model
/// instance. The state is completely owned and managed by the
/// backend.
///
/// \param instance The model instance.
/// \param state The user state, or nullptr if no user state.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_ModelInstanceSetState(
    TRITONBACKEND_ModelInstance* instance, void* state);

/// Record statistics for an inference request.
///
/// Set 'success' true to indicate that the inference request
/// completed successfully. In this case all timestamps should be
/// non-zero values reported in nanoseconds and should be collected
/// using std::chrono::steady_clock::now().time_since_epoch() or the equivalent.
/// Set 'success' to false to indicate that the inference request failed
/// to complete successfully. In this case all timestamps values are
/// ignored.
///
/// For consistency of measurement across different backends, the
/// timestamps should be collected at the following points during
/// TRITONBACKEND_ModelInstanceExecute.
///
///   TRITONBACKEND_ModelInstanceExecute()
///     CAPTURE TIMESPACE (exec_start_ns)
///     < process input tensors to prepare them for inference
///       execution, including copying the tensors to/from GPU if
///       necessary>
///     CAPTURE TIMESPACE (compute_start_ns)
///     < perform inference computations to produce outputs >
///     CAPTURE TIMESPACE (compute_end_ns)
///     < allocate output buffers and extract output tensors, including
///       copying the tensors to/from GPU if necessary>
///     CAPTURE TIMESPACE (exec_end_ns)
///     return
///
/// Note that these statistics are associated with a valid
/// TRITONBACKEND_Request object and so must be reported before the
/// request is released. For backends that release the request before
/// all response(s) are sent, these statistics cannot capture
/// information about the time required to produce the response.
///
/// \param instance The model instance.
/// \param request The inference request that statistics are being
/// reported for.
/// \param success True if the inference request completed
/// successfully, false if it failed to complete.
/// \param exec_start_ns Timestamp for the start of execution.
/// \param compute_start_ns Timestamp for the start of execution
/// computations.
/// \param compute_end_ns Timestamp for the end of execution
/// computations.
/// \param exec_end_ns Timestamp for the end of execution.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceReportStatistics(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request* request,
    const bool success, const uint64_t exec_start_ns,
    const uint64_t compute_start_ns, const uint64_t compute_end_ns,
    const uint64_t exec_end_ns);

/// Record statistics for the execution of an entire batch of
/// inference requests.
///
/// All timestamps should be non-zero values reported in nanoseconds
/// and should be collected using
/// std::chrono::steady_clock::now().time_since_epoch() or the equivalent.
/// See TRITONBACKEND_ModelInstanceReportStatistics for more information about
/// the timestamps.
///
/// 'batch_size' is the sum of the batch sizes for the individual
/// requests that were delivered together in the call to
/// TRITONBACKEND_ModelInstanceExecute. For example, if three requests
/// are passed to TRITONBACKEND_ModelInstanceExecute and those
/// requests have batch size 1, 2, and 3; then 'batch_size' should be
/// set to 6.
///
/// \param instance The model instance.
/// \param batch_size Combined batch size of all the individual
/// requests executed in the batch.
/// \param exec_start_ns Timestamp for the start of execution.
/// \param compute_start_ns Timestamp for the start of execution
/// computations.
/// \param compute_end_ns Timestamp for the end of execution
/// computations.
/// \param exec_end_ns Timestamp for the end of execution.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceReportBatchStatistics(
    TRITONBACKEND_ModelInstance* instance, const uint64_t batch_size,
    const uint64_t exec_start_ns, const uint64_t compute_start_ns,
    const uint64_t compute_end_ns, const uint64_t exec_end_ns);

///
/// The following functions can be implemented by a backend. Functions
/// indicated as required must be implemented or the backend will fail
/// to load.
///

/// Initialize a backend. This function is optional, a backend is not
/// required to implement it. This function is called once when a
/// backend is loaded to allow the backend to initialize any state
/// associated with the backend. A backend has a single state that is
/// shared across all models that use the backend.
///
/// \param backend The backend.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_ISPEC TRITONSERVER_Error* TRITONBACKEND_Initialize(
    TRITONBACKEND_Backend* backend);

/// Finalize for a backend. This function is optional, a backend is
/// not required to implement it. This function is called once, just
/// before the backend is unloaded. All state associated with the
/// backend should be freed and any threads created for the backend
/// should be exited/joined before returning from this function.
///
/// \param backend The backend.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_ISPEC TRITONSERVER_Error* TRITONBACKEND_Finalize(
    TRITONBACKEND_Backend* backend);

/// Initialize for a model. This function is optional, a backend is
/// not required to implement it. This function is called once when a
/// model that uses the backend is loaded to allow the backend to
/// initialize any state associated with the model. The backend should
/// also examine the model configuration to determine if the
/// configuration is suitable for the backend. Any errors reported by
/// this function will prevent the model from loading.
///
/// \param model The model.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_ISPEC TRITONSERVER_Error* TRITONBACKEND_ModelInitialize(
    TRITONBACKEND_Model* model);

/// Finalize for a model. This function is optional, a backend is not
/// required to implement it. This function is called once for a
/// model, just before the model is unloaded from Triton. All state
/// associated with the model should be freed and any threads created
/// for the model should be exited/joined before returning from this
/// function.
///
/// \param model The model.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_ISPEC TRITONSERVER_Error* TRITONBACKEND_ModelFinalize(
    TRITONBACKEND_Model* model);

/// Initialize for a model instance. This function is optional, a
/// backend is not required to implement it. This function is called
/// once when a model instance is created to allow the backend to
/// initialize any state associated with the instance.
///
/// \param instance The model instance.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_ISPEC TRITONSERVER_Error* TRITONBACKEND_ModelInstanceInitialize(
    TRITONBACKEND_ModelInstance* instance);

/// Finalize for a model instance. This function is optional, a
/// backend is not required to implement it. This function is called
/// once for an instance, just before the corresponding model is
/// unloaded from Triton. All state associated with the instance
/// should be freed and any threads created for the instance should be
/// exited/joined before returning from this function.
///
/// \param instance The model instance.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_ISPEC TRITONSERVER_Error* TRITONBACKEND_ModelInstanceFinalize(
    TRITONBACKEND_ModelInstance* instance);

/// Execute a batch of one or more requests on a model instance. This
/// function is required. Triton will not perform multiple
/// simultaneous calls to this function for a given model 'instance';
/// however, there may be simultaneous calls for different model
/// instances (for the same or different models).
///
/// If an error is returned the ownership of the request objects
/// remains with Triton and the backend must not retain references to
/// the request objects or access them in any way.
///
/// If success is returned, ownership of the request objects is
/// transferred to the backend and it is then responsible for creating
/// responses and releasing the request objects. Note that even though
/// ownership of the request objects is transferred to the backend, the
/// ownership of the buffer holding request pointers is returned back
/// to Triton upon return from TRITONBACKEND_ModelInstanceExecute. If
/// any request objects need to be maintained beyond
/// TRITONBACKEND_ModelInstanceExecute, then the pointers must be copied
/// out of the array within TRITONBACKEND_ModelInstanceExecute.
///
/// \param instance The model instance.
/// \param requests The requests.
/// \param request_count The number of requests in the batch.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_ISPEC TRITONSERVER_Error* TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count);

/// Query the backend for different model attributes. This function is optional,
/// a backend is not required to implement it. The backend is also not required
/// to set all backend attribute listed. This function is called when
/// Triton requires further backend / model information to perform operations.
/// This function may be called multiple times within the lifetime of the
/// backend (between TRITONBACKEND_Initialize and TRITONBACKEND_Finalize).
/// The backend may return error to indicate failure to set the backend
/// attributes, and the attributes specified in the same function call will be
/// ignored. Triton will update the specified attributes if 'nullptr' is
/// returned.
///
/// \param backend The backend.
/// \param backend_attributes Return the backend attribute.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_ISPEC TRITONSERVER_Error* TRITONBACKEND_GetBackendAttribute(
    TRITONBACKEND_Backend* backend,
    TRITONBACKEND_BackendAttribute* backend_attributes);

/// TRITONBACKEND_BackendAttribute
///
/// API to modify attributes associated with a backend.
///

/// Add the preferred instance group of the backend. This function
/// can be called multiple times to cover different instance group kinds that
/// the backend supports, given the priority order that the first call describes
/// the most preferred group. In the case where instance group are not
/// explicitly provided, Triton will use this attribute to create model
/// deployment that aligns more with the backend preference.
///
/// \param backend_attributes The backend attributes object.
/// \param kind The kind of the instance group.
/// \param count The number of instances per device. Triton default will be used
/// if 0 is provided.
/// \param device_ids The devices where instances should be available. Triton
/// default will be used if 'nullptr' is provided.
/// \param id_count The number of devices in 'device_ids'.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_BackendAttributeAddPreferredInstanceGroup(
    TRITONBACKEND_BackendAttribute* backend_attributes,
    const TRITONSERVER_InstanceGroupKind kind, const uint64_t count,
    const uint64_t* device_ids, const uint64_t id_count);

/// TRITONBACKEND Batching
///
/// API to add custom batching strategy
///
/// The following functions can be implemented by a backend to add custom
/// batching conditionals on top of the existing Triton batching strategy. The
/// functions are optional but all or none must be implemented.
///

/// Create a new batcher for use with custom batching. This is called during
/// model loading. The batcher will point to a user-defined data structure that
/// holds read-only data used for custom batching.
///
/// \param batcher User-defined placeholder for backend to store and
/// retrieve information about the batching strategy for this
/// model.RITONBACKEND_ISPEC return a TRITONSERVER_Error indicating success or
/// failure. \param model The backend model for which Triton is forming a batch.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_ISPEC TRITONSERVER_Error* TRITONBACKEND_ModelBatcherInitialize(
    TRITONBACKEND_Batcher** batcher, TRITONBACKEND_Model* model);

/// Free memory associated with batcher. This is called during model unloading.
///
/// \param batcher User-defined placeholder for backend to store and
/// retrieve information about the batching strategy for this model.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_ISPEC TRITONSERVER_Error* TRITONBACKEND_ModelBatcherFinalize(
    TRITONBACKEND_Batcher* batcher);

/// Check whether a request should be added to the pending model batch.
///
/// \param request The request to be added to the pending batch.
/// \param userp The placeholder for backend to store and retrieve information
/// about this pending batch. When the callback returns, this should reflect
/// the latest batch information.
/// \param should_include The pointer to be updated on whether the request
/// should be included in the batch.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_ISPEC TRITONSERVER_Error* TRITONBACKEND_ModelBatchIncludeRequest(
    TRITONBACKEND_Request* request, void* userp, bool* should_include);

/// Callback to be invoked when Triton has begun forming a batch.
///
/// \param batcher The read-only placeholder for backend to retrieve
// information about the batching strategy for this model.
/// \param userp The placeholder for backend to store and retrieve information
/// about this pending batch.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_ISPEC TRITONSERVER_Error* TRITONBACKEND_ModelBatchInitialize(
    const TRITONBACKEND_Batcher* batcher, void** userp);

/// Callback to be invoked when Triton has finishing forming a batch.
///
/// \param userp The placeholder for backend to store and retrieve information
/// about this pending batch.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_ISPEC TRITONSERVER_Error* TRITONBACKEND_ModelBatchFinalize(
    void* userp);

#ifdef __cplusplus
}
#endif
