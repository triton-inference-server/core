

// Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <triton/core/tritonserver.h>

// This binding is merely used to map Triton C API into Python equivalent,
// and therefore, the naming will be the same as the one used in corresponding
// sections. However, there are a few exceptions to better transit to Python:
// Structs:
//  * Triton structs are encapsulated in a thin wrapper to isolate raw pointer
//    operations which is not supported in pure Python.
//  * Trival getters and setters are grouped to be a Python class property.
//    However, this creates asymmetry that some APIs are called like function
//    while some like member variables. So I am open to expose getter / setter
//    if it may be more intuitive.
//  * The wrapper is only served as communication between Python and C, it will
//    be unwrapped when control reaches C API and the C struct will be wrapped
//    when control reaches Python side. Python binding user should respect the
//    "ownership" and lifetime of the wrapper in the same way as described in
//    the C API. Python binding user must not assume the same C struct will
//    always be referred through the same wrapper object.
// Enums:
//  * In C API, the enum values are prefixed by the enum name. The Python
//    equivalent is an enum class and thus the prefix is removed to avoid
//    duplication, i.e. Python user may specify a value by
//    'TRITONSERVER_ResponseCompleteFlag.FINAL'.
// Functions / Callbacks:
//  * Output parameters are converted to return value. APIs that return an error
//    will be thrown as an exception. The same applies to callbacks.
//  ** Note that in the C API, the inference response may carry an error object
//     that represent an inference failure. The equivalent Python API will raise
//     the corresponding exception if the response contains error object.
//  * The function parameters and return values are exposed in Python style,
//    for example, object pointer becomes py::object, C array and length
//    condenses into Python array.

namespace py = pybind11;
namespace triton { namespace core { namespace python {

#define DISALLOW_COPY(TypeName) TypeName(const TypeName&) = delete;
#define DISALLOW_ASSIGN(TypeName) void operator=(const TypeName&) = delete;
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  DISALLOW_COPY(TypeName)                  \
  DISALLOW_ASSIGN(TypeName)

// [FIXME] GIL

// base exception for all Triton error code
struct TritonError : public std::runtime_error {
  explicit TritonError(const std::string& what) : std::runtime_error(what) {}
};

// triton::core::python exceptions map 1:1 to TRITONSERVER_Error_Code.
struct Unknown : public TritonError {
  explicit Unknown(const std::string& what) : TritonError(what) {}
};
struct Internal : public TritonError {
  explicit Internal(const std::string& what) : TritonError(what) {}
};
struct NotFound : public TritonError {
  explicit NotFound(const std::string& what) : TritonError(what) {}
};
struct InvalidArgument : public TritonError {
  explicit InvalidArgument(const std::string& what) : TritonError(what) {}
};
struct Unavailable : public TritonError {
  explicit Unavailable(const std::string& what) : TritonError(what) {}
};
struct Unsupported : public TritonError {
  explicit Unsupported(const std::string& what) : TritonError(what) {}
};
struct AlreadyExists : public TritonError {
  explicit AlreadyExists(const std::string& what) : TritonError(what) {}
};

TRITONSERVER_Error*
CreateTRITONSERVER_ErrorFrom(const py::error_already_set& ex)
{
  // Reserved lookup to get Python type of the exceptions,
  // 'TRITONSERVER_ERROR_UNKNOWN' is the fallback error code.
  // static auto uk = py::module::import("triton_bindings").attr("Unknown");
  static auto it = py::module::import("triton_bindings").attr("Internal");
  static auto nf = py::module::import("triton_bindings").attr("NotFound");
  static auto ia =
      py::module::import("triton_bindings").attr("InvalidArgument");
  static auto ua = py::module::import("triton_bindings").attr("Unavailable");
  static auto us = py::module::import("triton_bindings").attr("Unsupported");
  static auto ae = py::module::import("triton_bindings").attr("AlreadyExists");
  TRITONSERVER_Error_Code code = TRITONSERVER_ERROR_UNKNOWN;
  if (ex.matches(it.ptr())) {
    code = TRITONSERVER_ERROR_INTERNAL;
  } else if (ex.matches(nf.ptr())) {
    code = TRITONSERVER_ERROR_NOT_FOUND;
  } else if (ex.matches(ia.ptr())) {
    code = TRITONSERVER_ERROR_INVALID_ARG;
  } else if (ex.matches(ua.ptr())) {
    code = TRITONSERVER_ERROR_UNAVAILABLE;
  } else if (ex.matches(us.ptr())) {
    code = TRITONSERVER_ERROR_UNSUPPORTED;
  } else if (ex.matches(ae.ptr())) {
    code = TRITONSERVER_ERROR_ALREADY_EXISTS;
  }
  return TRITONSERVER_ErrorNew(code, ex.what());
}

void
ThrowIfError(TRITONSERVER_Error* err)
{
  if (err == nullptr) {
    return;
  }
  std::shared_ptr<TRITONSERVER_Error> managed_err(
      err, TRITONSERVER_ErrorDelete);
  std::string msg = TRITONSERVER_ErrorMessage(err);
  switch (TRITONSERVER_ErrorCode(err)) {
    case TRITONSERVER_ERROR_INTERNAL:
      throw Internal(std::move(msg));
    case TRITONSERVER_ERROR_NOT_FOUND:
      throw NotFound(std::move(msg));
    case TRITONSERVER_ERROR_INVALID_ARG:
      throw InvalidArgument(std::move(msg));
    case TRITONSERVER_ERROR_UNAVAILABLE:
      throw Unavailable(std::move(msg));
    case TRITONSERVER_ERROR_UNSUPPORTED:
      throw Unsupported(std::move(msg));
    case TRITONSERVER_ERROR_ALREADY_EXISTS:
      throw AlreadyExists(std::move(msg));
    default:
      throw Unknown(std::move(msg));
  }
}

void
LogIfError(TRITONSERVER_Error* err)
{
  if (err == nullptr) {
    return;
  }
  std::shared_ptr<TRITONSERVER_Error> managed_err(
      err, TRITONSERVER_ErrorDelete);
  py::print(TRITONSERVER_ErrorMessage(err));
}

class PyParameter {
 public:
  PyParameter(
      TRITONSERVER_Parameter* parameter,
      const bool owned /* [FIXME] check if needed*/)
      : parameter_(parameter_), owned_(owned)
  {
  }

  PyParameter(
      const char* name, TRITONSERVER_ParameterType type, const void* value)
      : parameter_(TRITONSERVER_ParameterNew(name, type, value)), owned_(true)
  {
  }

  PyParameter(const char* name, const void* byte_ptr, uint64_t size)
      : parameter_(TRITONSERVER_ParameterBytesNew(name, byte_ptr, size)),
        owned_(true)
  {
  }

  ~PyParameter()
  {
    if (owned_ && parameter_) {
      TRITONSERVER_ParameterDelete(parameter_);
    }
  }

  DISALLOW_COPY_AND_ASSIGN(PyParameter);

  // Use internally when interacting with C APIs that takes ownership
  struct TRITONSERVER_Parameter* Release()
  {
    owned_ = false;
    return parameter_;
  }

  struct TRITONSERVER_Parameter* Ptr() const { return parameter_; }

 private:
  struct TRITONSERVER_Parameter* parameter_{nullptr};
  // [FIXME] may need to transfer ownership
  bool owned_{false};
};

class PyBufferAttributes {
 public:
  // [WIP] need this?
  explicit PyBufferAttributes()
  {
    ThrowIfError(TRITONSERVER_BufferAttributesNew(&buffer_attributes_));
    owned_ = true;
  }

  PyBufferAttributes(
      TRITONSERVER_BufferAttributes* ba,
      const bool owned /* [FIXME] check if needed*/)
      : buffer_attributes_(ba), owned_(owned)
  {
  }

  // Use internally when interacting with C APIs that takes ownership
  TRITONSERVER_BufferAttributes* Release()
  {
    owned_ = false;
    return buffer_attributes_;
  }

  TRITONSERVER_BufferAttributes* Ptr() { return buffer_attributes_; }

  ~PyBufferAttributes()
  {
    if (owned_ && buffer_attributes_) {
      LogIfError(TRITONSERVER_BufferAttributesDelete(buffer_attributes_));
    }
  }

  void SetMemoryTypeId(int64_t memory_type_id)
  {
    ThrowIfError(TRITONSERVER_BufferAttributesSetMemoryTypeId(
        buffer_attributes_, memory_type_id));
  }

  void SetMemoryType(TRITONSERVER_MemoryType memory_type)
  {
    ThrowIfError(TRITONSERVER_BufferAttributesSetMemoryType(
        buffer_attributes_, memory_type));
  }

  void SetCudaIpcHandle(size_t cuda_ipc_handle)
  {
    ThrowIfError(TRITONSERVER_BufferAttributesSetCudaIpcHandle(
        buffer_attributes_, reinterpret_cast<void*>(cuda_ipc_handle)));
  }

  void SetByteSize(size_t byte_size)
  {
    ThrowIfError(TRITONSERVER_BufferAttributesSetByteSize(
        buffer_attributes_, byte_size));
  }

  // Define methods to get buffer attribute fields
  int64_t MemoryTypeId()
  {
    int64_t memory_type_id = 0;
    ThrowIfError(TRITONSERVER_BufferAttributesMemoryTypeId(
        buffer_attributes_, &memory_type_id));
    return memory_type_id;
  }

  TRITONSERVER_MemoryType MemoryType()
  {
    TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
    ThrowIfError(TRITONSERVER_BufferAttributesMemoryType(
        buffer_attributes_, &memory_type));
    return memory_type;
  }

  size_t CudaIpcHandle()
  {
    void* cuda_ipc_handle = nullptr;
    ThrowIfError(TRITONSERVER_BufferAttributesCudaIpcHandle(
        buffer_attributes_, &cuda_ipc_handle));
    return reinterpret_cast<size_t>(cuda_ipc_handle);
  }

  size_t ByteSize()
  {
    size_t byte_size;
    ThrowIfError(
        TRITONSERVER_BufferAttributesByteSize(buffer_attributes_, &byte_size));
    return byte_size;
  }

 private:
  struct TRITONSERVER_BufferAttributes* buffer_attributes_{nullptr};
  bool owned_{false};
};

class PyResponseAllocator {
 public:
  struct CallbackResource {
    CallbackResource(py::object a, py::object uo)
        : allocator(a), user_object(uo)
    {
    }
    // Storing the py::object of PyResponseAllocator to have convenient access
    // to callbacks.
    py::object allocator;
    py::object user_object;
  };
  using AllocFn = std::function<
      std::tuple<size_t, py::object, TRITONSERVER_MemoryType, int64_t>(
          py::object, std::string, size_t, TRITONSERVER_MemoryType, int64_t,
          py::object)>;
  using ReleaseFn = std::function<void(
      py::object, size_t, py::object, size_t, TRITONSERVER_MemoryType,
      int64_t)>;
  using StartFn = std::function<void(py::object, py::object)>;

  // size as input, optional?
  using QueryFn = std::function<std::tuple<TRITONSERVER_MemoryType, int64_t>(
      py::object, py::object, std::string, std::optional<size_t>,
      TRITONSERVER_MemoryType, int64_t)>;
  using BufferAttributesFn = std::function<py::object(
      py::object, std::string, py::object, py::object, py::object)>;

  DISALLOW_COPY_AND_ASSIGN(PyResponseAllocator);

  // Use internally when interacting with C APIs that takes ownership
  TRITONSERVER_ResponseAllocator* Release()
  {
    owned_ = false;
    return allocator_;
  }

  PyResponseAllocator(AllocFn alloc, ReleaseFn release, StartFn start)
      : alloc_fn_(alloc), release_fn_(release), start_fn_(start)
  {
    ThrowIfError(TRITONSERVER_ResponseAllocatorNew(
        &allocator_, PyTritonAllocFn, PyTritonReleaseFn, PyTritonStartFn));
    owned_ = true;
  }

  // Below implements the Triton callbacks, note that when registering the
  // callbacks in Triton, an wrapped 'CallbackResource' must be used to bridge
  // the gap between the Python API and C API.
  static TRITONSERVER_Error* PyTritonAllocFn(
      struct TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
      size_t byte_size, TRITONSERVER_MemoryType memory_type,
      int64_t memory_type_id, void* userp, void** buffer, void** buffer_userp,
      TRITONSERVER_MemoryType* actual_memory_type,
      int64_t* actual_memory_type_id)
  {
    TRITONSERVER_Error* err = nullptr;
    auto cr = reinterpret_cast<CallbackResource*>(userp);
    try {
      auto res = cr->allocator.cast<PyResponseAllocator*>()->alloc_fn_(
          cr->allocator, tensor_name, byte_size, memory_type, memory_type_id,
          cr->user_object);
      *buffer = reinterpret_cast<void*>(std::get<0>(res));
      {
        // In C API usage, its typical to allocate user object within the
        // callback and place the release logic in release callback. The same
        // logic can't trivially ported to Python as user object is scoped,
        // therefore the binding needs to wrap the object to ensure the user
        // object will not be garbage collected until after release callback.
        *buffer_userp = new CallbackResource(cr->allocator, std::get<1>(res));
      }
      *actual_memory_type = std::get<2>(res);
      *actual_memory_type_id = std::get<3>(res);
    }
    catch (py::error_already_set& ex) {
      err = CreateTRITONSERVER_ErrorFrom(ex);
    }
    return err;
  }

  static TRITONSERVER_Error* PyTritonReleaseFn(
      struct TRITONSERVER_ResponseAllocator* allocator, void* buffer,
      void* buffer_userp, size_t byte_size, TRITONSERVER_MemoryType memory_type,
      int64_t memory_type_id)
  {
    TRITONSERVER_Error* err = nullptr;
    auto cr = reinterpret_cast<CallbackResource*>(buffer_userp);
    try {
      cr->allocator.cast<PyResponseAllocator*>()->release_fn_(
          cr->allocator, reinterpret_cast<size_t>(buffer), cr->user_object,
          byte_size, memory_type, memory_type_id);
    }
    catch (py::error_already_set& ex) {
      err = CreateTRITONSERVER_ErrorFrom(ex);
    }
    // Done with CallbackResource
    delete cr;
    return err;
  }

  static TRITONSERVER_Error* PyTritonStartFn(
      struct TRITONSERVER_ResponseAllocator* allocator, void* userp)
  {
    TRITONSERVER_Error* err = nullptr;
    auto cr = reinterpret_cast<CallbackResource*>(userp);
    try {
      cr->allocator.cast<PyResponseAllocator*>()->start_fn_(
          cr->allocator, cr->user_object);
    }
    catch (py::error_already_set& ex) {
      err = CreateTRITONSERVER_ErrorFrom(ex);
    }
    return err;
  }

  static TRITONSERVER_Error* PyTritonQueryFn(
      struct TRITONSERVER_ResponseAllocator* allocator, void* userp,
      const char* tensor_name, size_t* byte_size,
      TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id)
  {
    TRITONSERVER_Error* err = nullptr;
    auto cr = reinterpret_cast<CallbackResource*>(userp);
    try {
      std::optional<size_t> bs;
      if (byte_size) {
        bs = *byte_size;
      }
      auto res = cr->allocator.cast<PyResponseAllocator*>()->query_fn_(
          cr->allocator, cr->user_object, tensor_name, bs, *memory_type,
          *memory_type_id);
      *memory_type = std::get<0>(res);
      *memory_type_id = std::get<1>(res);
    }
    catch (py::error_already_set& ex) {
      err = CreateTRITONSERVER_ErrorFrom(ex);
    }
    return err;
  }

  static TRITONSERVER_Error* PyTritonBufferAttributesFn(
      struct TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
      struct TRITONSERVER_BufferAttributes* buffer_attributes, void* userp,
      void* buffer_userp)
  {
    TRITONSERVER_Error* err = nullptr;
    auto cr = reinterpret_cast<CallbackResource*>(userp);
    auto bcr = reinterpret_cast<CallbackResource*>(buffer_userp);
    PyBufferAttributes pba{buffer_attributes, false /* owned_ */};
    try {
      // Python version of BufferAttributes callback has return value
      // to be the filled buffer attributes. The callback implementation
      // should modify the passed PyBufferAttributes object and return it.
      // However, the implementation may construct new PyBufferAttributes
      // which requires additional checking to properly return the attributes
      // through C API.
      auto res =
          cr->allocator.cast<PyResponseAllocator*>()->buffer_attributes_fn_(
              cr->allocator, tensor_name,
              py::cast(pba, py::return_value_policy::reference),
              cr->user_object, bcr->user_object);
      // Copy if 'res' is new object, otherwise the attributes have been set.
      auto res_pba = res.cast<PyBufferAttributes*>();
      if (res_pba->Ptr() != buffer_attributes) {
        pba.SetMemoryTypeId(res_pba->MemoryTypeId());
        pba.SetMemoryType(res_pba->MemoryType());
        pba.SetCudaIpcHandle(res_pba->CudaIpcHandle());
        pba.SetByteSize(res_pba->ByteSize());
      }
    }
    catch (py::error_already_set& ex) {
      err = CreateTRITONSERVER_ErrorFrom(ex);
    }
    return err;
  }

  void SetBufferAttributesFunction(BufferAttributesFn baf)
  {
    buffer_attributes_fn_ = baf;
    ThrowIfError(TRITONSERVER_ResponseAllocatorSetBufferAttributesFunction(
        allocator_, PyTritonBufferAttributesFn));
  }

  void SetQueryFunction(QueryFn qf)
  {
    query_fn_ = qf;
    ThrowIfError(TRITONSERVER_ResponseAllocatorSetQueryFunction(
        allocator_, PyTritonQueryFn));
  }

  ~PyResponseAllocator()
  {
    if (owned_ && allocator_) {
      // Only log error in destructor.
      LogIfError(TRITONSERVER_ResponseAllocatorDelete(allocator_));
    }
  }

  //  private:
  TRITONSERVER_ResponseAllocator* allocator_{nullptr};
  // [FIXME] may need to transfer ownership
  bool owned_{false};

  AllocFn alloc_fn_{nullptr};
  ReleaseFn release_fn_{nullptr};
  StartFn start_fn_{nullptr};
  QueryFn query_fn_{nullptr};
  BufferAttributesFn buffer_attributes_fn_{nullptr};
};

class PyMessage {
 public:
  explicit PyMessage(const std::string& serialized_json) : owned_(true)
  {
    ThrowIfError(TRITONSERVER_MessageNewFromSerializedJson(
        &message_, serialized_json.c_str(), serialized_json.size()));
  }

  PyMessage(
      struct TRITONSERVER_Message* m,
      const bool owned /* [FIXME] check if needed*/)
      : message_(m), owned_(owned)
  {
  }

  ~PyMessage()
  {
    if (owned_ && message_) {
      // Only log error in destructor.
      LogIfError(TRITONSERVER_MessageDelete(message_));
    }
  }

  std::string SerializeToJson()
  {
    const char* base = nullptr;
    size_t byte_size = 0;
    ThrowIfError(
        TRITONSERVER_MessageSerializeToJson(message_, &base, &byte_size));
    return std::string(base, byte_size);
  }

 private:
  struct TRITONSERVER_Message* message_{nullptr};
  bool owned_{false};
};

class PyMetrics {
 public:
  explicit PyMetrics(struct TRITONSERVER_Metrics* metrics, bool owned)
      : metrics_(metrics), owned_(owned)
  {
  }

  ~PyMetrics()
  {
    if (owned_ && metrics_) {
      // Only log error in destructor.
      LogIfError(TRITONSERVER_MetricsDelete(metrics_));
    }
  }

  std::string Formatted(TRITONSERVER_MetricFormat format)
  {
    const char* base = nullptr;
    size_t byte_size = 0;
    ThrowIfError(
        TRITONSERVER_MetricsFormatted(metrics_, format, &base, &byte_size));
    return std::string(base, byte_size);
  }

 private:
  struct TRITONSERVER_Metrics* metrics_{nullptr};
  bool owned_{false};
};

class PyTrace {
 public:
  using TimestampActivityFn = std::function<void(
      py::object, TRITONSERVER_InferenceTraceActivity, uint64_t, py::object)>;
  using TensorActivityFn = std::function<void(
      py::object, TRITONSERVER_InferenceTraceActivity, std::string,
      TRITONSERVER_DataType, size_t, size_t, py::array_t<int64_t>,
      TRITONSERVER_MemoryType, int64_t, py::object)>;
  using ReleaseFn = std::function<void(py::object, py::object)>;

  struct CallbackResource {
    CallbackResource(
        TimestampActivityFn ts, TensorActivityFn t, ReleaseFn r, py::object uo)
        : timestamp_fn(ts), tensor_fn(t), release_fn(r), user_object(uo)
    {
    }
    TimestampActivityFn timestamp_fn{nullptr};
    TensorActivityFn tensor_fn{nullptr};
    ReleaseFn release_fn{nullptr};
    py::object user_object;
    // The trace API will use the same 'trace_userp' for all traces associated
    // with the request, and becasue there is no guarantee that the root trace
    // must be released last, need to track all trace seen / released to
    // determine whether this CallbackResource may be released.
    std::set<uintptr_t> seen_traces;
  };

  DISALLOW_COPY_AND_ASSIGN(PyTrace);

  // Use internally when interacting with C APIs that takes ownership
  // TRITONSERVER_InferenceTrace* Release()
  // {
  //   owned_ = false;
  //   return trace_;
  // }

  PyTrace(
      TRITONSERVER_InferenceTraceLevel level, uint64_t parent_id,
      TimestampActivityFn timestamp, ReleaseFn release, py::object user_object)
      : owned_(true), callback_resource_(new CallbackResource(
                          timestamp, nullptr, release, user_object))
  {
    ThrowIfError(TRITONSERVER_InferenceTraceNew(
        &trace_, level, parent_id, PyTritonTraceTimestampActivityFn,
        PyTritonTraceRelease, callback_resource_.get()));
  }

  PyTrace(
      TRITONSERVER_InferenceTraceLevel level, uint64_t parent_id,
      TimestampActivityFn timestamp, TensorActivityFn tensor, ReleaseFn release,
      py::object user_object)
      : owned_(true), callback_resource_(new CallbackResource(
                          timestamp, tensor, release, user_object))
  {
    ThrowIfError(TRITONSERVER_InferenceTraceTensorNew(
        &trace_, level, parent_id, PyTritonTraceTimestampActivityFn,
        PyTritonTraceTensorActivityFn, PyTritonTraceRelease,
        callback_resource_.get()));
  }

  PyTrace(
      struct TRITONSERVER_InferenceTrace* t,
      const bool owned /* [FIXME] check if needed*/)
      : trace_(t), owned_(owned)
  {
  }

  ~PyTrace()
  {
    if (owned_ && trace_) {
      // Only log error in destructor.
      LogIfError(TRITONSERVER_InferenceTraceDelete(trace_));
    }
  }
  CallbackResource* ReleaseCallbackResource()
  {
    return callback_resource_.release();
  }

  uint64_t Id()
  {
    uint64_t val = 0;
    ThrowIfError(TRITONSERVER_InferenceTraceId(trace_, &val));
    return val;
  }

  uint64_t ParentId()
  {
    uint64_t val = 0;
    ThrowIfError(TRITONSERVER_InferenceTraceParentId(trace_, &val));
    return val;
  }

  std::string ModelName()
  {
    const char* val = nullptr;
    ThrowIfError(TRITONSERVER_InferenceTraceModelName(trace_, &val));
    return val;
  }

  int64_t ModelVersion()
  {
    int64_t val = 0;
    ThrowIfError(TRITONSERVER_InferenceTraceModelVersion(trace_, &val));
    return val;
  }

  std::string RequestId()
  {
    const char* val = nullptr;
    ThrowIfError(TRITONSERVER_InferenceTraceRequestId(trace_, &val));
    return val;
  }

  // Below implements the Triton callbacks, note that when registering the
  // callbacks in Triton, an wrapped 'CallbackResource' must be used to bridge
  // the gap between the Python API and C API.
  static void PyTritonTraceTimestampActivityFn(
      struct TRITONSERVER_InferenceTrace* trace,
      TRITONSERVER_InferenceTraceActivity activity, uint64_t timestamp_ns,
      void* userp)
  {
    // Note that 'trace' associated with the activity is not necessary the
    // root trace captured in Callback Resource, so need to always wrap 'trace'
    // in PyTrace for the Python callabck to interact with the correct trace.
    PyTrace pt(trace, false /* owned */);
    auto cr = reinterpret_cast<CallbackResource*>(userp);
    cr->seen_traces.insert(reinterpret_cast<uintptr_t>(trace));
    cr->timestamp_fn(
        py::cast(pt, py::return_value_policy::reference), activity,
        timestamp_ns, cr->user_object);
  }

  static void PyTritonTraceTensorActivityFn(
      struct TRITONSERVER_InferenceTrace* trace,
      TRITONSERVER_InferenceTraceActivity activity, const char* name,
      TRITONSERVER_DataType datatype, const void* base, size_t byte_size,
      const int64_t* shape, uint64_t dim_count,
      TRITONSERVER_MemoryType memory_type, int64_t memory_type_id, void* userp)
  {
    // See 'PyTritonTraceTimestampActivityFn' for 'pt' explanation.
    PyTrace pt(trace, false /* owned */);
    auto cr = reinterpret_cast<CallbackResource*>(userp);
    cr->seen_traces.insert(reinterpret_cast<uintptr_t>(trace));
    cr->tensor_fn(
        py::cast(pt, py::return_value_policy::reference), activity, name,
        datatype, reinterpret_cast<size_t>(base), byte_size,
        py::array_t<int64_t>({dim_count}, {}, shape), memory_type,
        memory_type_id, cr->user_object);
  }

  static void PyTritonTraceRelease(
      struct TRITONSERVER_InferenceTrace* trace, void* userp)
  {
    // See 'PyTritonTraceTimestampActivityFn' for 'pt' explanation.
    // wrap in shared_ptr to transfer ownership to Python
    auto managed_pt = std::make_shared<PyTrace>(trace, true /* owned */);
    auto cr = reinterpret_cast<CallbackResource*>(userp);
    cr->release_fn(py::cast(managed_pt), cr->user_object);
    cr->seen_traces.erase(reinterpret_cast<uintptr_t>(trace));
    if (cr->seen_traces.empty()) {
      delete cr;
    }
  }

  //  private:
  TRITONSERVER_InferenceTrace* trace_{nullptr};
  // [FIXME] may need to transfer ownership
  bool owned_{false};
  std::unique_ptr<CallbackResource> callback_resource_{nullptr};
};

class PyInferenceResponse {
 public:
  using CompleteFn = std::function<void(py::object, uint32_t, py::object)>;
  struct CallbackResource {
    // During 'TRITONSERVER_InferenceRequestSetResponseCallback', a
    // PyResponseAllocator::CallbackResource is allocated and passed as
    // 'response_allocator_userp', which is used during any output buffer
    // allocation of the requests. However, unlike other 'userp', there is no
    // dedicated release callback to signal that the allocator resource may be
    // released. So we deduce the point of time is deduced based on the
    // following: 'TRITONSERVER_InferenceResponseCompleteFn_t' invoked with
    // 'TRITONSERVER_RESPONSE_COMPLETE_FINAL' flag indicates there is no more
    // responses to be generated and so does output allocation, therefore
    // 'allocator_resource' may be released as part of releasing
    // 'PyInferenceResponse::CallbackResource'
    PyResponseAllocator::CallbackResource* allocator_resource;
    py::object user_object;
  };

  PyInferenceResponse(TRITONSERVER_InferenceResponse* response, bool owned)
      : response_(response), owned_(owned)
  {
  }

  ~PyInferenceResponse()
  {
    if (owned_ && response_) {
      // Only log error in destructor.
      LogIfError(TRITONSERVER_InferenceResponseDelete(response_));
    }
  }

  void ThrowIfResponseError()
  {
    ThrowIfError(TRITONSERVER_InferenceResponseError(response_));
  }

  std::tuple<std::string, int64_t> Model()
  {
    const char* model_name = nullptr;
    int64_t model_version = 0;
    ThrowIfError(TRITONSERVER_InferenceResponseModel(
        response_, &model_name, &model_version));
    return {model_name, model_version};
  }

  std::string Id()
  {
    const char* val = nullptr;
    ThrowIfError(TRITONSERVER_InferenceResponseId(response_, &val));
    return val;
  }

  uint32_t ParameterCount(uint32_t* count)
  {
    uint32_t val = 0;
    ThrowIfError(TRITONSERVER_InferenceResponseParameterCount(response_, &val));
    return val;
  }

  std::tuple<std::string, TRITONSERVER_ParameterType, py::object> Parameter(
      uint32_t index)
  {
    const char* name = nullptr;
    TRITONSERVER_ParameterType type = TRITONSERVER_PARAMETER_STRING;
    const void* value = nullptr;
    ThrowIfError(TRITONSERVER_InferenceResponseParameter(
        response_, index, &name, &type, &value));
    py::object py_value;
    switch (type) {
      case TRITONSERVER_PARAMETER_STRING:
        py_value = py::str(reinterpret_cast<const char*>(value));
        break;
      case TRITONSERVER_PARAMETER_INT:
        py_value = py::int_(*reinterpret_cast<const int*>(value));
        break;
      case TRITONSERVER_PARAMETER_BOOL:
        py_value = py::bool_(*reinterpret_cast<const bool*>(value));
        break;
      default:
        throw Unsupported(
            std::string("Unexpected type '") +
            TRITONSERVER_ParameterTypeString(type) +
            "' received as response parameter");
        break;
    }
    return {name, type, py_value};
  }

  uint32_t OutputCount()
  {
    uint32_t val = 0;
    ThrowIfError(TRITONSERVER_InferenceResponseOutputCount(response_, &val));
    return val;
  }

  std::tuple<
      std::string, TRITONSERVER_DataType, py::array_t<int64_t>, size_t, size_t,
      TRITONSERVER_MemoryType, int64_t, py::object>
  Output(uint32_t index)
  {
    const char* name = nullptr;
    TRITONSERVER_DataType datatype = TRITONSERVER_TYPE_INVALID;
    const int64_t* shape = nullptr;
    uint64_t dim_count = 0;
    const void* base = nullptr;
    size_t byte_size = 0;
    TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t memory_type_id = 0;
    void* userp = nullptr;
    ThrowIfError(TRITONSERVER_InferenceResponseOutput(
        response_, index, &name, &datatype, &shape, &dim_count, &base,
        &byte_size, &memory_type, &memory_type_id, &userp));
    return {name,
            datatype,
            py::array_t<int64_t>({dim_count}, {}, shape),
            reinterpret_cast<size_t>(base),
            byte_size,
            memory_type,
            memory_type_id,
            reinterpret_cast<PyResponseAllocator::CallbackResource*>(userp)
                ->user_object};
  }

  std::string OutputClassificationLabel(uint32_t index, size_t class_index)
  {
    const char* val = nullptr;
    ThrowIfError(TRITONSERVER_InferenceResponseOutputClassificationLabel(
        response_, index, class_index, &val));
    return val;
  }

 private:
  TRITONSERVER_InferenceResponse* response_;
  bool owned_;
};

// forward declaration
class PyServer;

class PyInferenceRequest {
 public:
  using ReleaseFn = std::function<void(py::object, uint32_t, py::object)>;

  // Defer definition until PyServer is defined
  PyInferenceRequest(
      PyServer& server, const std::string& model_name,
      const int64_t model_version);

  PyInferenceRequest(
      struct TRITONSERVER_InferenceRequest* r,
      const bool owned /* [FIXME] check if needed*/)
      : request_(r), owned_(owned)
  {
  }

  ~PyInferenceRequest()
  {
    if (owned_ && request_) {
      // Only log error in destructor.
      LogIfError(TRITONSERVER_InferenceRequestDelete(request_));
    }
  }

  struct CallbackResource {
    CallbackResource(ReleaseFn r, py::object uo)
        : release_fn(r), user_object(uo)
    {
    }
    ReleaseFn release_fn;
    py::object user_object;
  };


  void SetReleaseCallback(ReleaseFn release, py::object user_object)
  {
    request_callback_resource_.reset(
        new CallbackResource(release, user_object));
    ThrowIfError(TRITONSERVER_InferenceRequestSetReleaseCallback(
        request_, PyTritonRequestReleaseCallback,
        request_callback_resource_.get()));
  }

  static void PyTritonRequestReleaseCallback(
      struct TRITONSERVER_InferenceRequest* request, const uint32_t flags,
      void* userp)
  {
    auto managed_pt =
        std::make_shared<PyInferenceRequest>(request, true /* owned */);
    auto cr = reinterpret_cast<CallbackResource*>(userp);
    cr->release_fn(py::cast(managed_pt), flags, cr->user_object);
    delete cr;
  }

  // [WIP] below
  void SetResponseCallback(
      py::object allocator, py::object allocater_user_object,
      PyInferenceResponse::CompleteFn response, py::object response_user_object)
  {
    // request_callback_resource_.reset(new CallbackResource(release,
    // user_object));
    // ThrowIfError(TRITONSERVER_InferenceRequestSetReleaseCallback(
    //     request_, PyTritonRequestReleaseCallback,
    //     request_callback_resource_.get()));
  }

  // Trivial setters / getters
  void SetId(const std::string& id)
  {
    ThrowIfError(TRITONSERVER_InferenceRequestSetId(request_, id.c_str()));
  }
  std::string Id()
  {
    const char* val = nullptr;
    ThrowIfError(TRITONSERVER_InferenceRequestId(request_, &val));
    return val;
  }

  void SetFlags(uint32_t flags)
  {
    ThrowIfError(TRITONSERVER_InferenceRequestSetFlags(request_, flags));
  }

  uint32_t Flags()
  {
    uint32_t val = 0;
    ThrowIfError(TRITONSERVER_InferenceRequestFlags(request_, &val));
    return val;
  }

  void SetCorrelationId(uint64_t correlation_id)
  {
    ThrowIfError(TRITONSERVER_InferenceRequestSetCorrelationId(
        request_, correlation_id));
  }
  uint64_t CorrelationId()
  {
    uint64_t val = 0;
    ThrowIfError(TRITONSERVER_InferenceRequestCorrelationId(request_, &val));
    return val;
  }
  void SetCorrelationIdString(const std::string& correlation_id)
  {
    ThrowIfError(TRITONSERVER_InferenceRequestSetCorrelationIdString(
        request_, correlation_id.c_str()));
  }
  std::string CorrelationIdString()
  {
    const char* val = nullptr;
    ThrowIfError(
        TRITONSERVER_InferenceRequestCorrelationIdString(request_, &val));
    return val;
  }

  void SetPriority(uint32_t priority)
  {
    ThrowIfError(TRITONSERVER_InferenceRequestSetPriority(request_, priority));
  }
  void SetPriorityUint64(uint64_t priority)
  {
    ThrowIfError(
        TRITONSERVER_InferenceRequestSetPriorityUInt64(request_, priority));
  }
  uint32_t Priority()
  {
    uint32_t val = 0;
    ThrowIfError(TRITONSERVER_InferenceRequestPriority(request_, &val));
    return val;
  }
  uint64_t PriorityUint64()
  {
    uint64_t val = 0;
    ThrowIfError(TRITONSERVER_InferenceRequestPriorityUInt64(request_, &val));
    return val;
  }

  void SetTimeoutMicroseconds(uint64_t timeout_us)
  {
    ThrowIfError(TRITONSERVER_InferenceRequestSetTimeoutMicroseconds(
        request_, timeout_us));
  }
  uint64_t TimeoutMicroseconds()
  {
    uint64_t val = 0;
    ThrowIfError(
        TRITONSERVER_InferenceRequestTimeoutMicroseconds(request_, &val));
    return val;
  }

  void AddInput(
      const std::string& name, TRITONSERVER_DataType data_type,
      std::vector<int64_t> shape)
  {
    ThrowIfError(TRITONSERVER_InferenceRequestAddInput(
        request_, name.c_str(), data_type, shape.data(), shape.size()));
  }
  void AddRawInput(const std::string& name)
  {
    ThrowIfError(
        TRITONSERVER_InferenceRequestAddRawInput(request_, name.c_str()));
  }
  void RemoveInput(const std::string& name)
  {
    ThrowIfError(
        TRITONSERVER_InferenceRequestRemoveInput(request_, name.c_str()));
  }
  void RemoveAllInputs()
  {
    ThrowIfError(TRITONSERVER_InferenceRequestRemoveAllInputs(request_));
  }
  void AppendInputData(
      const std::string& name, size_t base, size_t byte_size,
      TRITONSERVER_MemoryType memory_type, int64_t memory_type_id)
  {
    ThrowIfError(TRITONSERVER_InferenceRequestAppendInputData(
        request_, name.c_str(), reinterpret_cast<const char*>(base), byte_size,
        memory_type, memory_type_id));
  }
  void AppendInputDataWithHostPolicy(
      const std::string name, size_t base, size_t byte_size,
      TRITONSERVER_MemoryType memory_type, int64_t memory_type_id,
      const std::string& host_policy_name)
  {
    ThrowIfError(TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(
        request_, name.c_str(), reinterpret_cast<const char*>(base), byte_size,
        memory_type, memory_type_id, host_policy_name.c_str()));
  }
  void AppendInputDataWithBufferAttributes(
      const std::string& name, size_t base,
      PyBufferAttributes* buffer_attributes)
  {
    ThrowIfError(
        TRITONSERVER_InferenceRequestAppendInputDataWithBufferAttributes(
            request_, name.c_str(), reinterpret_cast<const char*>(base),
            buffer_attributes->Ptr()));
  }
  void RemoveAllInputData(const std::string& name)
  {
    ThrowIfError(TRITONSERVER_InferenceRequestRemoveAllInputData(
        request_, name.c_str()));
  }

  void AddRequestedOutput(const std::string& name)
  {
    ThrowIfError(TRITONSERVER_InferenceRequestAddRequestedOutput(
        request_, name.c_str()));
  }
  void RemoveRequestedOutput(const std::string& name)
  {
    ThrowIfError(TRITONSERVER_InferenceRequestRemoveRequestedOutput(
        request_, name.c_str()));
  }
  void RemoveAllRequestedOutputs()
  {
    ThrowIfError(
        TRITONSERVER_InferenceRequestRemoveAllRequestedOutputs(request_));
  }

  void SetStringParameter(const std::string& key, const std::string& value)
  {
    ThrowIfError(TRITONSERVER_InferenceRequestSetStringParameter(
        request_, key.c_str(), value.c_str()));
  }
  void SetIntParameter(const std::string& key, int64_t value)
  {
    ThrowIfError(TRITONSERVER_InferenceRequestSetIntParameter(
        request_, key.c_str(), value));
  }
  void SetBoolParameter(const std::string& key, bool value)
  {
    ThrowIfError(TRITONSERVER_InferenceRequestSetBoolParameter(
        request_, key.c_str(), value));
  }

 private:
  struct TRITONSERVER_InferenceRequest* request_{nullptr};
  bool owned_{false};
  std::unique_ptr<CallbackResource> request_callback_resource_{nullptr};
  std::unique_ptr<PyResponseAllocator::CallbackResource>
      allocator_callback_resource_{nullptr};
  std::unique_ptr<PyInferenceResponse::CallbackResource>
      response_callback_resource_{nullptr};
};

class PyServerOptions {
 public:
  PyServerOptions() : owned_(true)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsNew(&options_));
  }

  ~PyServerOptions()
  {
    if (owned_ && options_)
      // Only log error in destructor.
      LogIfError(TRITONSERVER_ServerOptionsDelete(options_));
  }

  struct TRITONSERVER_ServerOptions* Ptr() { return options_; }

  void SetServerId(const std::string& server_id)
  {
    ThrowIfError(
        TRITONSERVER_ServerOptionsSetServerId(options_, server_id.c_str()));
  }

  void SetModelRepositoryPath(const std::string& model_repository_path)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetModelRepositoryPath(
        options_, model_repository_path.c_str()));
  }

  void SetModelControlMode(TRITONSERVER_ModelControlMode mode)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetModelControlMode(options_, mode));
  }

  void SetStartupModel(const std::string& model_name)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetStartupModel(
        options_, model_name.c_str()));
  }

  void SetStrictModelConfig(bool strict)
  {
    ThrowIfError(
        TRITONSERVER_ServerOptionsSetStrictModelConfig(options_, strict));
  }
  void SetRateLimiterMode(TRITONSERVER_RateLimitMode mode)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetRateLimiterMode(options_, mode));
  }

  void AddRateLimiterResource(
      const std::string& resource_name, size_t resource_count, int device)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsAddRateLimiterResource(
        options_, resource_name.c_str(), resource_count, device));
  }

  void SetPinnedMemoryPoolByteSize(uint64_t size)
  {
    ThrowIfError(
        TRITONSERVER_ServerOptionsSetPinnedMemoryPoolByteSize(options_, size));
  }

  void SetCudaMemoryPoolByteSize(int gpu_device, uint64_t size)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetCudaMemoryPoolByteSize(
        options_, gpu_device, size));
  }
  void SetResponseCacheByteSize(uint64_t size)
  {
    ThrowIfError(
        TRITONSERVER_ServerOptionsSetResponseCacheByteSize(options_, size));
  }

  void SetCacheConfig(
      const std::string& cache_name, const std::string& config_json)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetCacheConfig(
        options_, cache_name.c_str(), config_json.c_str()));
  }

  void SetCacheDirectory(const std::string& cache_dir)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetCacheDirectory(
        options_, cache_dir.c_str()));
  }

  void SetMinSupportedComputeCapability(double cc)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability(
        options_, cc));
  }

  void SetExitOnError(bool exit)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetExitOnError(options_, exit));
  }

  void SetStrictReadiness(bool strict)
  {
    ThrowIfError(
        TRITONSERVER_ServerOptionsSetStrictReadiness(options_, strict));
  }

  void SetExitTimeout(unsigned int timeout)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetExitTimeout(options_, timeout));
  }
  void SetBufferManagerThreadCount(unsigned int thread_count)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetBufferManagerThreadCount(
        options_, thread_count));
  }

  void SetModelLoadThreadCount(unsigned int thread_count)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetModelLoadThreadCount(
        options_, thread_count));
  }

  void SetModelNamespacing(bool enable_namespace)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetModelNamespacing(
        options_, enable_namespace));
  }

  void SetLogFile(const std::string& file)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetLogFile(options_, file.c_str()));
  }

  void SetLogInfo(bool log)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetLogInfo(options_, log));
  }

  void SetLogWarn(bool log)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetLogWarn(options_, log));
  }

  void SetLogError(bool log)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetLogError(options_, log));
  }

  void SetLogFormat(TRITONSERVER_LogFormat format)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetLogFormat(options_, format));
  }

  void SetLogVerbose(int level)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetLogVerbose(options_, level));
  }
  void SetMetrics(bool metrics)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetMetrics(options_, metrics));
  }

  void SetGpuMetrics(bool gpu_metrics)
  {
    ThrowIfError(
        TRITONSERVER_ServerOptionsSetGpuMetrics(options_, gpu_metrics));
  }

  void SetCpuMetrics(bool cpu_metrics)
  {
    ThrowIfError(
        TRITONSERVER_ServerOptionsSetCpuMetrics(options_, cpu_metrics));
  }

  void SetMetricsInterval(uint64_t metrics_interval_ms)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetMetricsInterval(
        options_, metrics_interval_ms));
  }

  void SetBackendDirectory(const std::string& backend_dir)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetBackendDirectory(
        options_, backend_dir.c_str()));
  }

  void SetRepoAgentDirectory(const std::string& repoagent_dir)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetRepoAgentDirectory(
        options_, repoagent_dir.c_str()));
  }

  void SetModelLoadDeviceLimit(
      TRITONSERVER_InstanceGroupKind kind, int device_id, double fraction)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetModelLoadDeviceLimit(
        options_, kind, device_id, fraction));
  }

  void SetBackendConfig(
      const std::string& backend_name, const std::string& setting,
      const std::string& value)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetBackendConfig(
        options_, backend_name.c_str(), setting.c_str(), value.c_str()));
  }

  void SetHostPolicy(
      const std::string& policy_name, const std::string& setting,
      const std::string& value)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetHostPolicy(
        options_, policy_name.c_str(), setting.c_str(), value.c_str()));
  }

  void SetMetricsConfig(
      const std::string& name, const std::string& setting,
      const std::string& value)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetMetricsConfig(
        options_, name.c_str(), setting.c_str(), value.c_str()));
  }

 private:
  struct TRITONSERVER_ServerOptions* options_{nullptr};
  bool owned_{false};
};

class PyServer {
 public:
  struct TRITONSERVER_Server* Ptr() { return server_; }

  PyServer(PyServerOptions& options) : owned_(true)
  {
    ThrowIfError(TRITONSERVER_ServerNew(&server_, options.Ptr()));
  }

  ~PyServer()
  {
    if (owned_ && server_) {
      // Only log error in destructor.
      LogIfError(TRITONSERVER_ServerDelete(server_));
    }
  }

  void Stop() const { ThrowIfError(TRITONSERVER_ServerStop(server_)); }

  void RegisterModelRepository(
      const std::string& repository_path,
      const std::vector<std::shared_ptr<PyParameter>>& name_mapping) const
  {
    std::vector<const struct TRITONSERVER_Parameter*> params;
    for (const auto& nm : name_mapping) {
      params.emplace_back(nm->Ptr());
    }
    ThrowIfError(TRITONSERVER_ServerRegisterModelRepository(
        server_, repository_path.c_str(), params.data(), params.size()));
  }

  void UnregisterModelRepository(const std::string& repository_path) const
  {
    ThrowIfError(TRITONSERVER_ServerUnregisterModelRepository(
        server_, repository_path.c_str()));
  }

  void PollModelRepository() const
  {
    ThrowIfError(TRITONSERVER_ServerPollModelRepository(server_));
  }

  bool IsLive() const
  {
    bool live;
    ThrowIfError(TRITONSERVER_ServerIsLive(server_, &live));
    return live;
  }

  bool IsReady() const
  {
    bool ready;
    ThrowIfError(TRITONSERVER_ServerIsReady(server_, &ready));
    return ready;
  }

  bool ModelIsReady(const std::string& model_name, int64_t model_version) const
  {
    bool ready;
    ThrowIfError(TRITONSERVER_ServerModelIsReady(
        server_, model_name.c_str(), model_version, &ready));
    return ready;
  }

  std::tuple<uint32_t, size_t> ModelBatchProperties(
      const std::string& model_name, int64_t model_version) const
  {
    uint32_t flags;
    void* voidp;
    ThrowIfError(TRITONSERVER_ServerModelBatchProperties(
        server_, model_name.c_str(), model_version, &flags, &voidp));
    return {flags, reinterpret_cast<size_t>(voidp)};
  }

  std::tuple<uint32_t, size_t> ModelTransactionProperties(
      const std::string& model_name, int64_t model_version) const
  {
    uint32_t txn_flags;
    void* voidp;
    ThrowIfError(TRITONSERVER_ServerModelTransactionProperties(
        server_, model_name.c_str(), model_version, &txn_flags, &voidp));
    return {txn_flags, reinterpret_cast<size_t>(voidp)};
  }

  std::shared_ptr<PyMessage> Metadata() const
  {
    struct TRITONSERVER_Message* server_metadata;
    ThrowIfError(TRITONSERVER_ServerMetadata(server_, &server_metadata));
    return std::make_shared<PyMessage>(server_metadata, true /* owned */);
  }

  std::shared_ptr<PyMessage> ModelMetadata(
      const std::string& model_name, int64_t model_version) const
  {
    struct TRITONSERVER_Message* model_metadata;
    ThrowIfError(TRITONSERVER_ServerModelMetadata(
        server_, model_name.c_str(), model_version, &model_metadata));
    return std::make_shared<PyMessage>(model_metadata, true /* owned */);
  }

  std::shared_ptr<PyMessage> ModelStatistics(
      const std::string& model_name, int64_t model_version) const
  {
    struct TRITONSERVER_Message* model_stats;
    ThrowIfError(TRITONSERVER_ServerModelStatistics(
        server_, model_name.c_str(), model_version, &model_stats));
    return std::make_shared<PyMessage>(model_stats, true /* owned */);
  }

  std::shared_ptr<PyMessage> ModelConfig(
      const std::string& model_name, int64_t model_version,
      uint32_t config_version = 1) const
  {
    struct TRITONSERVER_Message* model_config;
    ThrowIfError(TRITONSERVER_ServerModelConfig(
        server_, model_name.c_str(), model_version, config_version,
        &model_config));
    return std::make_shared<PyMessage>(model_config, true /* owned */);
  }

  std::shared_ptr<PyMessage> ModelIndex(uint32_t flags) const
  {
    struct TRITONSERVER_Message* model_index;
    ThrowIfError(TRITONSERVER_ServerModelIndex(server_, flags, &model_index));
    return std::make_shared<PyMessage>(model_index, true /* owned */);
  }

  void LoadModel(const std::string& model_name)
  {
    ThrowIfError(TRITONSERVER_ServerLoadModel(server_, model_name.c_str()));
  }

  void LoadModelWithParameters(
      const std::string& model_name,
      const std::vector<std::shared_ptr<PyParameter>>& parameters) const
  {
    std::vector<const struct TRITONSERVER_Parameter*> params;
    for (const auto& p : parameters) {
      params.emplace_back(p->Ptr());
    }
    ThrowIfError(TRITONSERVER_ServerLoadModelWithParameters(
        server_, model_name.c_str(), params.data(), params.size()));
  }

  void UnloadModel(const std::string& model_name)
  {
    ThrowIfError(TRITONSERVER_ServerUnloadModel(server_, model_name.c_str()));
  }

  void UnloadModelAndDependents(const std::string& model_name)
  {
    ThrowIfError(TRITONSERVER_ServerUnloadModelAndDependents(
        server_, model_name.c_str()));
  }

  std::shared_ptr<PyMetrics> Metrics() const
  {
    struct TRITONSERVER_Metrics* metrics;
    ThrowIfError(TRITONSERVER_ServerMetrics(server_, &metrics));
    return std::make_shared<PyMetrics>(metrics, true /* owned */);
  }

  // [WIP] TRITONSERVER_ServerInferAsync

 private:
  struct TRITONSERVER_Server* server_{nullptr};
  bool owned_{false};
};

class PyMetricFamily {
 public:
  PyMetricFamily(
      TRITONSERVER_MetricKind kind, const std::string& name,
      const std::string& description)
      : owned_(true)
  {
    TRITONSERVER_MetricFamilyNew(
        &family_, kind, name.c_str(), description.c_str());
  }

  ~PyMetricFamily()
  {
    if (owned_ && family_) {
      // Only log error in destructor.
      LogIfError(TRITONSERVER_MetricFamilyDelete(family_));
    }
  }

  TRITONSERVER_MetricFamily* Ptr() const { return family_; }

 private:
  struct TRITONSERVER_MetricFamily* family_{nullptr};
  bool owned_{false};
};

class PyMetric {
 public:
  PyMetric(
      PyMetricFamily& family,
      const std::vector<std::shared_ptr<PyParameter>>& labels)
      : owned_(true)
  {
    std::vector<const struct TRITONSERVER_Parameter*> params;
    for (const auto& label : labels) {
      params.emplace_back(label->Ptr());
    }
    ThrowIfError(TRITONSERVER_MetricNew(
        &metric_, family.Ptr(), params.data(), params.size()));
  }

  ~PyMetric()
  {
    if (owned_ && metric_) {
      // Only log error in destructor.
      LogIfError(TRITONSERVER_MetricDelete(metric_));
    }
  }

  struct TRITONSERVER_Metric* Ptr() const { return metric_; }

  double Value() const
  {
    double val = 0;
    ThrowIfError(TRITONSERVER_MetricValue(metric_, &val));
    return val;
  }

  void Increment(double val) const
  {
    ThrowIfError(TRITONSERVER_MetricIncrement(metric_, val));
  }

  void SetValue(double val) const
  {
    ThrowIfError(TRITONSERVER_MetricSet(metric_, val));
  }

  TRITONSERVER_MetricKind Kind() const
  {
    TRITONSERVER_MetricKind val = TRITONSERVER_METRIC_KIND_COUNTER;
    ThrowIfError(TRITONSERVER_GetMetricKind(metric_, &val));
    return val;
  }

 private:
  struct TRITONSERVER_Metric* metric_{nullptr};
  bool owned_{false};
};

// Deferred definitions..
PyInferenceRequest::PyInferenceRequest(
    PyServer& server, const std::string& model_name,
    const int64_t model_version)
    : owned_(true)
{
  ThrowIfError(TRITONSERVER_InferenceRequestNew(
      &request_, server.Ptr(), model_name.c_str(), model_version));
}

// [FIXME] module name?
PYBIND11_MODULE(triton_bindings, m)
{
  m.doc() = "Python bindings for Triton Inference Server";
  // [FIXME] testing field
  // ========================================================
  auto mt = m.def_submodule("testing", "For testing purpose");
  // ========================================================

  // [FIXME] if dynamic linking, should have version check here as well to
  // make sure the binding is compatible with the Triton library loaded
  m.def("api_version", []() {
    uint32_t major = 0, minor = 0;
    ThrowIfError(TRITONSERVER_ApiVersion(&major, &minor));
    return py::make_tuple(major, minor);
  });

  // TRITONSERVER_Error... converted to 'TritonError' exception
  // Implement exception inheritance in PyBind:
  // https://github.com/jagerman/pybind11/blob/master/tests/test_exceptions.cpp#L149-L152
  auto te = pybind11::register_exception<TritonError>(m, "TritonError");
  pybind11::register_exception<Unknown>(m, "Unknown", te.ptr());
  pybind11::register_exception<Internal>(m, "Internal", te.ptr());
  pybind11::register_exception<NotFound>(m, "NotFound", te.ptr());
  pybind11::register_exception<InvalidArgument>(m, "InvalidArgument", te.ptr());
  pybind11::register_exception<Unavailable>(m, "Unavailable", te.ptr());
  pybind11::register_exception<Unsupported>(m, "Unsupported", te.ptr());
  pybind11::register_exception<AlreadyExists>(m, "AlreadyExists", te.ptr());

  // TRITONSERVER_DataType
  py::enum_<TRITONSERVER_DataType>(m, "TRITONSERVER_DataType")
      .value("INVALID", TRITONSERVER_TYPE_INVALID)
      .value("BOOL", TRITONSERVER_TYPE_BOOL)
      .value("UINT8", TRITONSERVER_TYPE_UINT8)
      .value("UINT16", TRITONSERVER_TYPE_UINT16)
      .value("UINT32", TRITONSERVER_TYPE_UINT32)
      .value("UINT64", TRITONSERVER_TYPE_UINT64)
      .value("INT8", TRITONSERVER_TYPE_INT8)
      .value("INT16", TRITONSERVER_TYPE_INT16)
      .value("INT32", TRITONSERVER_TYPE_INT32)
      .value("INT64", TRITONSERVER_TYPE_INT64)
      .value("FP16", TRITONSERVER_TYPE_FP16)
      .value("FP32", TRITONSERVER_TYPE_FP32)
      .value("FP64", TRITONSERVER_TYPE_FP64)
      .value("BYTES", TRITONSERVER_TYPE_BYTES)
      .value("BF16", TRITONSERVER_TYPE_BF16);
  // helper functions
  m.def("TRITONSERVER_DataTypeString", [](TRITONSERVER_DataType datatype) {
    return TRITONSERVER_DataTypeString(datatype);
  });
  m.def("TRITONSERVER_StringToDataType", [](const char* dtype) {
    return TRITONSERVER_StringToDataType(dtype);
  });
  m.def("TRITONSERVER_DataTypeByteSize", [](TRITONSERVER_DataType datatype) {
    return TRITONSERVER_DataTypeByteSize(datatype);
  });

  // TRITONSERVER_MemoryType
  py::enum_<TRITONSERVER_MemoryType>(m, "TRITONSERVER_MemoryType")
      .value("CPU", TRITONSERVER_MEMORY_CPU)
      .value("CPU_PINNED", TRITONSERVER_MEMORY_CPU_PINNED)
      .value("GPU", TRITONSERVER_MEMORY_GPU);
  // helper functions
  m.def("TRITONSERVER_MemoryTypeString", [](TRITONSERVER_MemoryType memtype) {
    return TRITONSERVER_MemoryTypeString(memtype);
  });

  // TRITONSERVER_ParameterType
  py::enum_<TRITONSERVER_ParameterType>(m, "TRITONSERVER_ParameterType")
      .value("STRING", TRITONSERVER_PARAMETER_STRING)
      .value("INT", TRITONSERVER_PARAMETER_INT)
      .value("BOOL", TRITONSERVER_PARAMETER_BOOL)
      .value("BYTES", TRITONSERVER_PARAMETER_BYTES);
  // helper functions
  m.def(
      "TRITONSERVER_ParameterTypeString",
      [](TRITONSERVER_ParameterType paramtype) {
        return TRITONSERVER_ParameterTypeString(paramtype);
      });
  // TRITONSERVER_Parameter
  py::class_<PyParameter, std::shared_ptr<PyParameter>>(
      m, "TRITONSERVER_Parameter")
      .def(py::init<const char*, TRITONSERVER_ParameterType, const void*>())
      .def(py::init([](const char* name, py::bytes bytes) {
        // [FIXME] does not own 'bytes' in the same way as C API, but can also
        // hold 'bytes' to make sure it will not be invalidated while in use.
        // i.e. safe to perform
        //   a = triton_bindings.TRITONSERVER_Parameter("abc", b'abc')
        //   # 'a' still points to valid buffer at this line.
        // Note that even holding 'bytes', it is the user's responsibility not
        // to modify 'bytes' while the parameter is in use.
        py::buffer_info info(py::buffer(bytes).request());
        return std::make_unique<PyParameter>(name, info.ptr, info.size);
      }));

  // TRITONSERVER_InstanceGroupKind
  py::enum_<TRITONSERVER_InstanceGroupKind>(m, "TRITONSERVER_InstanceGroupKind")
      .value("AUTO", TRITONSERVER_INSTANCEGROUPKIND_AUTO)
      .value("CPU", TRITONSERVER_INSTANCEGROUPKIND_CPU)
      .value("GPU", TRITONSERVER_INSTANCEGROUPKIND_GPU)
      .value("MODEL", TRITONSERVER_INSTANCEGROUPKIND_MODEL);
  m.def(
      "TRITONSERVER_InstanceGroupKindString",
      [](TRITONSERVER_InstanceGroupKind kind) {
        return TRITONSERVER_InstanceGroupKindString(kind);
      });

  // TRITONSERVER_Log
  py::enum_<TRITONSERVER_LogLevel>(m, "TRITONSERVER_LogLevel")
      .value("INFO", TRITONSERVER_LOG_INFO)
      .value("WARN", TRITONSERVER_LOG_WARN)
      .value("ERROR", TRITONSERVER_LOG_ERROR)
      .value("VERBOSE", TRITONSERVER_LOG_VERBOSE);

  py::enum_<TRITONSERVER_LogFormat>(m, "TRITONSERVER_LogFormat")
      .value("DEFAULT", TRITONSERVER_LOG_DEFAULT)
      .value("ISO8601", TRITONSERVER_LOG_ISO8601);

  m.def("TRITONSERVER_LogIsEnabled", [](TRITONSERVER_LogLevel level) {
    return TRITONSERVER_LogIsEnabled(level);
  });
  m.def(
      "TRITONSERVER_LogMessage",
      [](TRITONSERVER_LogLevel level, const char* filename, const int line,
         const char* msg) {
        ThrowIfError(TRITONSERVER_LogMessage(level, filename, line, msg));
      });

  py::class_<PyBufferAttributes>(m, "TRITONSERVER_BufferAttributes")
      .def(py::init<>())
      .def_property(
          "memory_type_id", &PyBufferAttributes::MemoryTypeId,
          &PyBufferAttributes::SetMemoryTypeId)
      .def_property(
          "memory_type", &PyBufferAttributes::MemoryType,
          &PyBufferAttributes::SetMemoryType)
      .def_property(
          "cuda_ipc_handle", &PyBufferAttributes::CudaIpcHandle,
          &PyBufferAttributes::SetCudaIpcHandle)
      .def_property(
          "byte_size", &PyBufferAttributes::ByteSize,
          &PyBufferAttributes::SetByteSize);

  py::class_<PyResponseAllocator>(m, "TRITONSERVER_ResponseAllocator")
      .def(
          py::init<
              PyResponseAllocator::AllocFn, PyResponseAllocator::ReleaseFn,
              PyResponseAllocator::StartFn>(),
          py::arg("alloc_function"), py::arg("release_function"),
          py::arg("start_function").none(true))
      .def(
          "set_buffer_attributes_function",
          &PyResponseAllocator::SetBufferAttributesFunction,
          py::arg("buffer_attributes_function"))
      .def(
          "set_query_function", &PyResponseAllocator::SetQueryFunction,
          py::arg("query_function"))
      // ========================================================
      .def(
          "invoke_allocator",
          [](py::object alloc, py::object user_object) {
            py::print("abc");
            void* buffer = nullptr;
            void* buffer_userp = nullptr;
            TRITONSERVER_MemoryType actual_memory_type =
                TRITONSERVER_MEMORY_CPU;
            int64_t actual_memory_type_id = 0;
            auto callback_resource =
                new PyResponseAllocator::CallbackResource(alloc, user_object);
            ThrowIfError(PyResponseAllocator::PyTritonAllocFn(
                alloc.cast<PyResponseAllocator*>()->allocator_, "abc", 10,
                TRITONSERVER_MEMORY_CPU_PINNED, 1, callback_resource, &buffer,
                &buffer_userp, &actual_memory_type, &actual_memory_type_id));
            py::handle bu = static_cast<PyResponseAllocator::CallbackResource*>(
                                buffer_userp)
                                ->user_object;
            return py::make_tuple(
                reinterpret_cast<size_t>(buffer), bu, actual_memory_type,
                actual_memory_type_id);
          })
      .def(
          "invoke_query",
          [](py::object alloc, py::object user_object) {
            void* buffer = nullptr;
            void* buffer_userp = nullptr;
            TRITONSERVER_MemoryType actual_memory_type =
                TRITONSERVER_MEMORY_CPU;
            int64_t actual_memory_type_id = 0;
            auto callback_resource =
                new PyResponseAllocator::CallbackResource(alloc, user_object);
            ThrowIfError(PyResponseAllocator::PyTritonAllocFn(
                alloc.cast<PyResponseAllocator*>()->allocator_, "abc", 10,
                TRITONSERVER_MEMORY_CPU_PINNED, 1, callback_resource, &buffer,
                &buffer_userp, &actual_memory_type, &actual_memory_type_id));
            py::handle bu = static_cast<PyResponseAllocator::CallbackResource*>(
                                buffer_userp)
                                ->user_object;
            return py::make_tuple(
                reinterpret_cast<size_t>(buffer), bu, actual_memory_type,
                actual_memory_type_id);
          })
      // ========================================================
      ;

  // TRITONSERVER_Message
  py::class_<PyMessage, std::shared_ptr<PyMessage>>(m, "TRITONSERVER_Message")
      .def(py::init<const std::string&>())
      .def("serialize_to_json", &PyMessage::SerializeToJson);

  // TRITONSERVER_Metrics
  py::enum_<TRITONSERVER_MetricFormat>(m, "TRITONSERVER_MetricFormat")
      .value("PROMETHEUS", TRITONSERVER_METRIC_PROMETHEUS);
  py::class_<PyMetrics, std::shared_ptr<PyMetrics>>(m, "TRITONSERVER_Metrics")
      .def("formatted", &PyMetrics::Formatted);

  // TRITONSERVER_InferenceTrace
  py::enum_<TRITONSERVER_InferenceTraceLevel>(
      m, "TRITONSERVER_InferenceTraceLevel")
      .value("DISABLED", TRITONSERVER_TRACE_LEVEL_DISABLED)
      .value("MIN", TRITONSERVER_TRACE_LEVEL_MIN)
      .value("MAX", TRITONSERVER_TRACE_LEVEL_MAX)
      .value("TIMESTAMPS", TRITONSERVER_TRACE_LEVEL_TIMESTAMPS)
      .value("TENSORS", TRITONSERVER_TRACE_LEVEL_TENSORS)
      .export_values();
  m.def(
      "TRITONSERVER_InferenceTraceLevelString",
      &TRITONSERVER_InferenceTraceLevelString);
  py::enum_<TRITONSERVER_InferenceTraceActivity>(
      m, "TRITONSERVER_InferenceTraceActivity")
      .value("REQUEST_START", TRITONSERVER_TRACE_REQUEST_START)
      .value("QUEUE_START", TRITONSERVER_TRACE_QUEUE_START)
      .value("COMPUTE_START", TRITONSERVER_TRACE_COMPUTE_START)
      .value("COMPUTE_INPUT_END", TRITONSERVER_TRACE_COMPUTE_INPUT_END)
      .value("COMPUTE_OUTPUT_START", TRITONSERVER_TRACE_COMPUTE_OUTPUT_START)
      .value("COMPUTE_END", TRITONSERVER_TRACE_COMPUTE_END)
      .value("REQUEST_END", TRITONSERVER_TRACE_REQUEST_END)
      .value("TENSOR_QUEUE_INPUT", TRITONSERVER_TRACE_TENSOR_QUEUE_INPUT)
      .value("TENSOR_BACKEND_INPUT", TRITONSERVER_TRACE_TENSOR_BACKEND_INPUT)
      .value("TENSOR_BACKEND_OUTPUT", TRITONSERVER_TRACE_TENSOR_BACKEND_OUTPUT)
      .export_values();
  m.def(
      "TRITONSERVER_InferenceTraceActivityString",
      &TRITONSERVER_InferenceTraceActivityString);
  py::class_<PyTrace, std::shared_ptr<PyTrace>>(
      m, "TRITONSERVER_InferenceTrace")
      .def(
          py::init<
              TRITONSERVER_InferenceTraceLevel, uint64_t,
              PyTrace::TimestampActivityFn, PyTrace::TensorActivityFn,
              PyTrace::ReleaseFn, py::object>(),
          py::arg("level"), py::arg("parent_id"), py::arg("activity_function"),
          py::arg("tensor_activity_function"), py::arg("release_function"),
          py::arg("trace_userp"))
      .def(
          py::init<
              TRITONSERVER_InferenceTraceLevel, uint64_t,
              PyTrace::TimestampActivityFn, PyTrace::ReleaseFn, py::object>(),
          py::arg("level"), py::arg("parent_id"), py::arg("activity_function"),
          py::arg("release_function"), py::arg("trace_userp"))
      .def_property_readonly("id", &PyTrace::Id)
      .def_property_readonly("parent_id", &PyTrace::ParentId)
      .def_property_readonly("model_name", &PyTrace::ModelName)
      .def_property_readonly("model_version", &PyTrace::ModelVersion)
      .def_property_readonly("request_id", &PyTrace::RequestId);

  // TRITONSERVER_InferenceRequest
  py::enum_<TRITONSERVER_RequestFlag>(m, "TRITONSERVER_RequestFlag")
      .value("SEQUENCE_START", TRITONSERVER_REQUEST_FLAG_SEQUENCE_START)
      .value("SEQUENCE_END", TRITONSERVER_REQUEST_FLAG_SEQUENCE_END)
      .export_values();
  py::enum_<TRITONSERVER_RequestReleaseFlag>(
      m, "TRITONSERVER_RequestReleaseFlag")
      .value("ALL", TRITONSERVER_REQUEST_RELEASE_ALL)
      .export_values();

  py::class_<PyInferenceRequest>(m, "TRITONSERVER_InferenceRequest")
      .def(py::init<PyServer&, const std::string&, int64_t>())
      .def("set_release_callback", &PyInferenceRequest::SetReleaseCallback)
      .def("set_response_callback", &PyInferenceRequest::SetResponseCallback)
      .def_property("id", &PyInferenceRequest::Id, &PyInferenceRequest::SetId)
      .def_property(
          "flags", &PyInferenceRequest::Flags, &PyInferenceRequest::SetFlags)
      .def_property(
          "correlation_id", &PyInferenceRequest::CorrelationId,
          &PyInferenceRequest::SetCorrelationId)
      .def_property(
          "correlation_id_string", &PyInferenceRequest::CorrelationIdString,
          &PyInferenceRequest::SetCorrelationIdString)
      .def_property(
          "priority", &PyInferenceRequest::Priority,
          &PyInferenceRequest::SetPriority)
      .def_property(
          "priority_uint64", &PyInferenceRequest::PriorityUint64,
          &PyInferenceRequest::SetPriorityUint64)
      .def_property(
          "timeout_microseconds", &PyInferenceRequest::TimeoutMicroseconds,
          &PyInferenceRequest::SetTimeoutMicroseconds)
      .def("add_input", &PyInferenceRequest::AddInput)
      .def("add_raw_input", &PyInferenceRequest::AddRawInput)
      .def("remove_input", &PyInferenceRequest::RemoveInput)
      .def("remove_all_inputs", &PyInferenceRequest::RemoveAllInputs)
      .def("append_input_data", &PyInferenceRequest::AppendInputData)
      .def(
          "append_input_data_with_host_policy",
          &PyInferenceRequest::AppendInputDataWithHostPolicy)
      .def(
          "append_input_data_with_buffer_attributes",
          &PyInferenceRequest::AppendInputDataWithBufferAttributes)
      .def("remove_all_input_data", &PyInferenceRequest::RemoveAllInputData)
      .def("add_requested_output", &PyInferenceRequest::AddRequestedOutput)
      .def(
          "remove_requested_output", &PyInferenceRequest::RemoveRequestedOutput)
      .def(
          "remove_all_requested_outputs",
          &PyInferenceRequest::RemoveAllRequestedOutputs)
      .def("set_string_parameter", &PyInferenceRequest::SetStringParameter)
      .def("set_int_parameter", &PyInferenceRequest::SetIntParameter)
      .def("set_bool_parameter", &PyInferenceRequest::SetBoolParameter);

  // TRITONSERVER_InferenceResponse
  py::enum_<TRITONSERVER_ResponseCompleteFlag>(
      m, "TRITONSERVER_ResponseCompleteFlag")
      .value("FINAL", TRITONSERVER_RESPONSE_COMPLETE_FINAL)
      .export_values();
  py::class_<PyInferenceResponse>(m, "TRITONSERVER_InferenceResponse")
      .def(
          "throw_if_response_error", &PyInferenceResponse::ThrowIfResponseError)
      .def("model", &PyInferenceResponse::Model)
      .def("id", &PyInferenceResponse::Id)
      .def("parameter_count", &PyInferenceResponse::ParameterCount)
      .def("parameter", &PyInferenceResponse::Parameter)
      .def("output_count", &PyInferenceResponse::OutputCount)
      .def("output", &PyInferenceResponse::Output)
      .def(
          "output_classification_label",
          &PyInferenceResponse::OutputClassificationLabel);

  // TRITONSERVER_ServerOptions
  py::enum_<TRITONSERVER_ModelControlMode>(m, "TRITONSERVER_ModelControlMode")
      .value("NONE", TRITONSERVER_MODEL_CONTROL_NONE)
      .value("POLL", TRITONSERVER_MODEL_CONTROL_POLL)
      .value("EXPLICIT", TRITONSERVER_MODEL_CONTROL_EXPLICIT);
  py::enum_<TRITONSERVER_RateLimitMode>(m, "TRITONSERVER_RateLimitMode")
      .value("OFF", TRITONSERVER_RATE_LIMIT_OFF)
      .value("EXEC_COUNT", TRITONSERVER_RATE_LIMIT_EXEC_COUNT);
  py::class_<PyServerOptions>(m, "TRITONSERVER_ServerOptions")
      .def(py::init<>())
      .def("set_server_id", &PyServerOptions::SetServerId)
      .def(
          "set_model_repository_path", &PyServerOptions::SetModelRepositoryPath)
      .def("set_model_control_mode", &PyServerOptions::SetModelControlMode)
      .def("set_startup_model", &PyServerOptions::SetStartupModel)
      .def("set_strict_model_config", &PyServerOptions::SetStrictModelConfig)
      .def("set_rate_limiter_mode", &PyServerOptions::SetRateLimiterMode)
      .def(
          "add_rate_limiter_resource", &PyServerOptions::AddRateLimiterResource)
      .def(
          "set_pinned_memory_pool_byte_size",
          &PyServerOptions::SetPinnedMemoryPoolByteSize)
      .def(
          "set_cuda_memory_pool_byte_size",
          &PyServerOptions::SetCudaMemoryPoolByteSize)
      .def(
          "set_response_cache_byte_size",
          &PyServerOptions::SetResponseCacheByteSize)
      .def("set_cache_config", &PyServerOptions::SetCacheConfig)
      .def("set_cache_directory", &PyServerOptions::SetCacheDirectory)
      .def(
          "set_min_supported_compute_capability",
          &PyServerOptions::SetMinSupportedComputeCapability)
      .def("set_exit_on_error", &PyServerOptions::SetExitOnError)
      .def("set_strict_readiness", &PyServerOptions::SetStrictReadiness)
      .def("set_exit_timeout", &PyServerOptions::SetExitTimeout)
      .def(
          "set_buffer_manager_thread_count",
          &PyServerOptions::SetBufferManagerThreadCount)
      .def(
          "set_model_load_thread_count",
          &PyServerOptions::SetModelLoadThreadCount)
      .def("set_model_namespacing", &PyServerOptions::SetModelNamespacing)
      .def("set_log_file", &PyServerOptions::SetLogFile)
      .def("set_log_info", &PyServerOptions::SetLogInfo)
      .def("set_log_warn", &PyServerOptions::SetLogWarn)
      .def("set_log_error", &PyServerOptions::SetLogError)
      .def("set_log_format", &PyServerOptions::SetLogFormat)
      .def("set_log_verbose", &PyServerOptions::SetLogVerbose)
      .def("set_metrics", &PyServerOptions::SetMetrics)
      .def("set_gpu_metrics", &PyServerOptions::SetGpuMetrics)
      .def("set_cpu_metrics", &PyServerOptions::SetCpuMetrics)
      .def("set_metrics_interval", &PyServerOptions::SetMetricsInterval)
      .def("set_backend_directory", &PyServerOptions::SetBackendDirectory)
      .def("set_repo_agent_directory", &PyServerOptions::SetRepoAgentDirectory)
      .def(
          "set_model_load_device_limit",
          &PyServerOptions::SetModelLoadDeviceLimit)
      .def("set_backend_config", &PyServerOptions::SetBackendConfig)
      .def("set_host_policy", &PyServerOptions::SetHostPolicy)
      .def("set_metrics_config", &PyServerOptions::SetMetricsConfig);

  // TRITONSERVER_Server
  py::class_<PyServer>(m, "TRITONSERVER_Server")
      .def(py::init<PyServerOptions&>())
      .def("stop", &PyServer::Stop)
      .def("register_model_repository", &PyServer::RegisterModelRepository)
      .def("unregister_model_repository", &PyServer::UnregisterModelRepository)
      .def("poll_model_repository", &PyServer::PollModelRepository)
      .def("poll_model_repository", &PyServer::PollModelRepository)
      .def("is_live", &PyServer::IsLive)
      .def("is_ready", &PyServer::IsReady)
      .def("model_is_ready", &PyServer::ModelIsReady)
      .def("model_batch_properties", &PyServer::ModelBatchProperties)
      .def(
          "model_transaction_properties", &PyServer::ModelTransactionProperties)
      .def("metadata", &PyServer::Metadata)
      .def("model_metadata", &PyServer::ModelMetadata)
      .def("model_statistics", &PyServer::ModelStatistics)
      .def("model_config", &PyServer::ModelConfig)
      .def("model_index", &PyServer::ModelIndex)
      .def("load_model", &PyServer::LoadModel)
      .def("load_model_with_parameters", &PyServer::LoadModelWithParameters)
      .def("unload_model", &PyServer::UnloadModel)
      .def("unload_model_and_dependents", &PyServer::UnloadModelAndDependents)
      .def("metrics", &PyServer::Metrics);

  // TRITONSERVER_MetricFamily
  py::class_<PyMetricFamily>(m, "TRITONSERVER_MetricFamily")
      .def(py::init<
           TRITONSERVER_MetricKind, const std::string&, const std::string&>());

  // TRITONSERVER_Metric
  py::class_<PyMetric>(m, "TRITONSERVER_Metric")
      .def(py::init<
           PyMetricFamily&, const std::vector<std::shared_ptr<PyParameter>>&>())
      .def("value", &PyMetric::Value)
      .def("increment", &PyMetric::Increment)
      .def("set_value", &PyMetric::SetValue)
      .def("kind", &PyMetric::Kind);
}

}}}  // namespace triton::core::python
