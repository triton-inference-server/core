// Copyright 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES{} LOSS OF USE, DATA, OR
// PROFITS{} OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#if defined(_MSC_VER)
#define TRITONAPI_DECLSPEC __declspec(dllexport)
#elif defined(__GNUC__)
#define TRITONAPI_DECLSPEC __attribute__((__visibility__("default")))
#else
#define TRITONAPI_DECLSPEC
#endif

extern "C" {
TRITONAPI_DECLSPEC void
TRITONSERVER_ApiVersion()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_DataTypeString()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_StringToDataType()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_DataTypeByteSize()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_MemoryTypeString()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ParameterTypeString()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ParameterNew()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ParameterBytesNew()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ParameterDelete()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InstanceGroupKindString()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_LogIsEnabled()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_LogMessage()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ErrorNew()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ErrorDelete()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ErrorCode()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ErrorCodeString()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ErrorMessage()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ResponseAllocatorNew()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ResponseAllocatorSetQueryFunction()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ResponseAllocatorSetBufferAttributesFunction()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ResponseAllocatorDelete()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestIsCancelled()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestCancel()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_MessageNewFromSerializedJson()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_MessageDelete()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_MessageSerializeToJson()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_MetricsDelete()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_MetricsFormatted()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceTraceLevelString()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceTraceActivityString()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceTraceNew()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceTraceTensorNew()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceTraceDelete()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceTraceId()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceTraceParentId()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceTraceModelName()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceTraceModelVersion()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceTraceRequestId()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceTraceSpawnChildTrace()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceTraceSetContext()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceTraceContext()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestNew()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestDelete()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestId()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestSetId()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestFlags()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestSetFlags()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_InferenceRequestTimeoutMicroseconds()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestCorrelationId()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestCorrelationIdString()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestSetCorrelationId()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestSetCorrelationIdString()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestPriority()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestPriorityUInt64()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestSetPriority()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestSetPriorityUInt64()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestTimeoutMicroseconds()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestSetTimeoutMicroseconds()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestAddInput()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestAddRawInput()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestRemoveInput()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestRemoveAllInputs()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestAppendInputData()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestRemoveAllInputData()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestAddRequestedOutput()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestRemoveRequestedOutput()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestRemoveAllRequestedOutputs()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestSetReleaseCallback()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestSetResponseCallback()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceResponseDelete()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceResponseError()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceResponseModel()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceResponseId()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceResponseParameterCount()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceResponseParameter()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceResponseOutputCount()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceResponseOutput()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceResponseOutputClassificationLabel()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsNew()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsDelete()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetServerId()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetModelRepositoryPath()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetModelControlMode()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetStartupModel()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetStrictModelConfig()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetRateLimiterMode()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsAddRateLimiterResource()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetPinnedMemoryPoolByteSize()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetCudaMemoryPoolByteSize()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetCudaVirtualAddressSize()
{
}
// Deprecated. See TRITONSERVER_ServerOptionsSetCacheConfig instead.
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetResponseCacheByteSize()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetCacheConfig()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetCacheDirectory()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetExitOnError()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetStrictReadiness()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetExitTimeout()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetBufferManagerThreadCount()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetModelLoadThreadCount()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetModelLoadRetryCount()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetModelNamespacing()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetLogFile()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetLogInfo()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetLogWarn()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetLogError()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetLogVerbose()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetLogFormat()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetMetrics()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetGpuMetrics()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetCpuMetrics()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetMetricsInterval()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetBackendDirectory()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetRepoAgentDirectory()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetModelLoadDeviceLimit()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetBackendConfig()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetHostPolicy()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestAppendInputDataWithBufferAttributes()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_BufferAttributesNew()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_BufferAttributesDelete()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_BufferAttributesSetMemoryTypeId()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_BufferAttributesSetMemoryType()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_BufferAttributesSetCudaIpcHandle()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_BufferAttributesSetByteSize()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_BufferAttributesMemoryTypeId()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_BufferAttributesMemoryType()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_BufferAttributesCudaIpcHandle()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_BufferAttributesByteSize()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerNew()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerDelete()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerStop()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerSetExitTimeout()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerPollModelRepository()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerIsLive()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerIsReady()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerModelIsReady()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerModelBatchProperties()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerModelTransactionProperties()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerMetadata()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerModelMetadata()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerModelStatistics()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerModelConfig()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerModelIndex()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerRegisterModelRepository()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerUnregisterModelRepository()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerLoadModel()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerLoadModelWithParameters()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerUnloadModel()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerUnloadModelAndDependents()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerMetrics()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_ServerInferAsync()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ApiVersion()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_MemoryManagerAllocate()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_MemoryManagerFree()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_InputProperties()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_InputPropertiesForHostPolicy()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_InputBuffer()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_InputBufferForHostPolicy()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_InputBufferAttributes()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_OutputBuffer()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_OutputBufferAttributes()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_RequestId()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ResponseFactoryIsCancelled()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_RequestIsCancelled()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_RequestCorrelationId()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_RequestCorrelationIdString()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_RequestFlags()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_RequestInputCount()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_RequestInputName()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_RequestInput()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_RequestInputByIndex()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_RequestOutputCount()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_RequestOutputName()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_RequestOutputBufferProperties()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_RequestRelease()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_RequestTrace()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestSetBoolParameter()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestSetDoubleParameter()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestSetIntParameter()
{
}
TRITONAPI_DECLSPEC void
TRITONSERVER_InferenceRequestSetStringParameter()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_RequestParameter()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_RequestParameterCount()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ResponseFactoryNew()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ResponseFactoryDelete()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ResponseFactorySendFlags()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ResponseNew()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ResponseNewFromFactory()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ResponseDelete()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ResponseSetStringParameter()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ResponseSetIntParameter()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ResponseSetBoolParameter()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ResponseSetDoubleParameter()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ResponseOutput()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ResponseSend()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_StateNew()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_StateUpdate()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_StateBuffer()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_StateBufferAttributes()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_BackendName()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_BackendConfig()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_BackendExecutionPolicy()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_BackendSetExecutionPolicy()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_BackendArtifacts()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_BackendMemoryManager()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_BackendState()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_BackendSetState()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelName()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelVersion()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelRepository()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelConfig()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelAutoCompleteConfig()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelSetConfig()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelServer()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelBackend()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelState()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelSetState()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelReportMemoryUsage()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelInstanceName()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelInstanceKind()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelInstanceDeviceId()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelInstanceHostPolicy()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelInstanceIsPassive()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelInstanceProfileCount()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelInstanceProfileName()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelInstanceSecondaryDeviceCount()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelInstanceSecondaryDeviceProperties()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelInstanceModel()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelInstanceState()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelInstanceSetState()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelInstanceReportMemoryUsage()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelInstanceReportStatistics()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelInstanceResponseStatisticsNew()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelInstanceResponseStatisticsSetModelInstance()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelInstanceResponseStatisticsSetResponseFactory()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelInstanceResponseStatisticsSetResponseStart()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelInstanceResponseStatisticsSetComputeOutputStart()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelInstanceResponseStatisticsSetResponseEnd()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelInstanceResponseStatisticsSetError()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelInstanceResponseStatisticsDelete()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelInstanceReportResponseStatistics()
{
}
TRITONAPI_DECLSPEC void
TRITONBACKEND_ModelInstanceReportBatchStatistics()
{
}
TRITONAPI_DECLSPEC void
TRITONREPOAGENT_ApiVersion()
{
}
TRITONAPI_DECLSPEC void
TRITONREPOAGENT_ModelRepositoryLocation()
{
}

TRITONAPI_DECLSPEC void
TRITONREPOAGENT_ModelRepositoryLocationAcquire()
{
}

TRITONAPI_DECLSPEC void
TRITONREPOAGENT_ModelRepositoryLocationRelease()
{
}

TRITONAPI_DECLSPEC void
TRITONREPOAGENT_ModelRepositoryUpdate()
{
}

TRITONAPI_DECLSPEC void
TRITONREPOAGENT_ModelParameterCount()
{
}

TRITONAPI_DECLSPEC void
TRITONREPOAGENT_ModelParameter()
{
}

TRITONAPI_DECLSPEC void
TRITONREPOAGENT_ModelConfig()
{
}

TRITONAPI_DECLSPEC void
TRITONREPOAGENT_ModelState()
{
}

TRITONAPI_DECLSPEC void
TRITONREPOAGENT_ModelSetState()
{
}

TRITONAPI_DECLSPEC void
TRITONREPOAGENT_State()
{
}

TRITONAPI_DECLSPEC void
TRITONREPOAGENT_SetState()
{
}

TRITONAPI_DECLSPEC void
TRITONSERVER_MetricFamilyNew()
{
}

TRITONAPI_DECLSPEC void
TRITONSERVER_MetricFamilyDelete()
{
}

TRITONAPI_DECLSPEC void
TRITONSERVER_MetricNew()
{
}

TRITONAPI_DECLSPEC void
TRITONSERVER_MetricDelete()
{
}

TRITONAPI_DECLSPEC void
TRITONSERVER_MetricValue()
{
}

TRITONAPI_DECLSPEC void
TRITONSERVER_MetricIncrement()
{
}

TRITONAPI_DECLSPEC void
TRITONSERVER_MetricSet()
{
}

TRITONAPI_DECLSPEC void
TRITONSERVER_GetMetricKind()
{
}

TRITONAPI_DECLSPEC void
TRITONSERVER_ServerOptionsSetMetricsConfig()
{
}

TRITONAPI_DECLSPEC void
TRITONBACKEND_BackendAttributeAddPreferredInstanceGroup()
{
}

TRITONAPI_DECLSPEC void
TRITONBACKEND_BackendAttributeSetParallelModelInstanceLoading()
{
}

TRITONAPI_DECLSPEC void
TRITONCACHE_ApiVersion()
{
}

TRITONAPI_DECLSPEC void
TRITONCACHE_CacheEntryBufferCount()
{
}

TRITONAPI_DECLSPEC void
TRITONCACHE_CacheEntryAddBuffer()
{
}

TRITONAPI_DECLSPEC void
TRITONCACHE_CacheEntryGetBuffer()
{
}

TRITONAPI_DECLSPEC void
TRITONCACHE_CacheEntrySetBuffer()
{
}

TRITONAPI_DECLSPEC void
TRITONCACHE_CacheInitialize()
{
}

TRITONAPI_DECLSPEC void
TRITONCACHE_CacheFinalize()
{
}

TRITONAPI_DECLSPEC void
TRITONCACHE_CacheInsert()
{
}

TRITONAPI_DECLSPEC void
TRITONCACHE_CacheLookup()
{
}

TRITONAPI_DECLSPEC void
TRITONCACHE_Copy()
{
}

} /* extern "C" */
