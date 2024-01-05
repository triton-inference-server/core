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

#include <iostream>

// This binding is merely used to map Triton C API into Python equivalent,
// and therefore, the naming will be the same as the one used in corresponding
// sections. However, there are a few exceptions to better transit to Python:
// Structs:
//  * Triton structs are encapsulated in a thin wrapper to isolate raw pointer
//    operations which is not supported in pure Python. A thin 'PyWrapper' base
//    class is defined with common utilities
//  * Trivial getters and setters are grouped to be a Python class property.
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

// Macro used by PyWrapper
#define DISALLOW_COPY(TypeName) TypeName(const TypeName&) = delete;
#define DISALLOW_ASSIGN(TypeName) void operator=(const TypeName&) = delete;
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  DISALLOW_COPY(TypeName)                  \
  DISALLOW_ASSIGN(TypeName)
#define DESTRUCTOR_WITH_LOG(TypeName, DeleteFunction)    \
  ~TypeName()                                            \
  {                                                      \
    if (owned_ && triton_object_) {                      \
      auto err__ = (DeleteFunction(triton_object_));     \
      if (err__) {                                       \
        std::shared_ptr<TRITONSERVER_Error> managed_err( \
            err__, TRITONSERVER_ErrorDelete);            \
        py::print(TRITONSERVER_ErrorMessage(err__));     \
      }                                                  \
  }}
// base exception for all Triton error code
struct TritonError : public std::runtime_error {
  explicit TritonError(const std::string& what) : std::runtime_error(what) {}
};

// triton::core::python exceptions map 1:1 to TRITONSERVER_Error_Code.
struct UnknownError : public TritonError {
  explicit UnknownError(const std::string& what) : TritonError(what) {}
};
struct InternalError : public TritonError {
  explicit InternalError(const std::string& what) : TritonError(what) {}
};
struct NotFoundError : public TritonError {
  explicit NotFoundError(const std::string& what) : TritonError(what) {}
};
struct InvalidArgumentError : public TritonError {
  explicit InvalidArgumentError(const std::string& what) : TritonError(what) {}
};
struct UnavailableError : public TritonError {
  explicit UnavailableError(const std::string& what) : TritonError(what) {}
};
struct UnsupportedError : public TritonError {
  explicit UnsupportedError(const std::string& what) : TritonError(what) {}
};
struct AlreadyExistsError : public TritonError {
  explicit AlreadyExistsError(const std::string& what) : TritonError(what) {}
};

TRITONSERVER_Error*
CreateTRITONSERVER_ErrorFrom(const py::error_already_set& ex)
{
  // Reserved lookup to get Python type of the exceptions,
  // 'TRITONSERVER_ERROR_UNKNOWN' is the fallback error code.
  // static auto uk =
  // py::module::import("triton_bindings").attr("UnknownError");
  static auto it = py::module::import("triton_bindings").attr("InternalError");
  static auto nf = py::module::import("triton_bindings").attr("NotFoundError");
  static auto ia =
      py::module::import("triton_bindings").attr("InvalidArgumentError");
  static auto ua =
      py::module::import("triton_bindings").attr("UnavailableError");
  static auto us =
      py::module::import("triton_bindings").attr("UnsupportedError");
  static auto ae =
      py::module::import("triton_bindings").attr("AlreadyExistsError");
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
      throw InternalError(std::move(msg));
    case TRITONSERVER_ERROR_NOT_FOUND:
      throw NotFoundError(std::move(msg));
    case TRITONSERVER_ERROR_INVALID_ARG:
      throw InvalidArgumentError(std::move(msg));
    case TRITONSERVER_ERROR_UNAVAILABLE:
      throw UnavailableError(std::move(msg));
    case TRITONSERVER_ERROR_UNSUPPORTED:
      throw UnsupportedError(std::move(msg));
    case TRITONSERVER_ERROR_ALREADY_EXISTS:
      throw AlreadyExistsError(std::move(msg));
    default:
      throw UnknownError(std::move(msg));
  }
}

template <typename TritonStruct>
class PyWrapper {
 public:
  explicit PyWrapper(TritonStruct* triton_object, bool owned)
      : triton_object_(triton_object), owned_(owned)
  {
  }
  PyWrapper() = default;
  // Destructor will be defined per specialization for now as a few
  // Triton object delete functions have different signatures, which
  // requires a function wrapper to generalize the destructor.

  // Use internally to get the pointer of the underlying Triton object
  TritonStruct* Ptr() { return triton_object_; }

  DISALLOW_COPY_AND_ASSIGN(PyWrapper);

 protected:
  TritonStruct* triton_object_{nullptr};
  bool owned_{false};
};

class PyParameter : public PyWrapper<struct TRITONSERVER_Parameter> {
 public:
  explicit PyParameter(struct TRITONSERVER_Parameter* p, const bool owned)
      : PyWrapper(p, owned)
  {
  }

  PyParameter(const char* name, const std::string& val)
      : PyWrapper(
            TRITONSERVER_ParameterNew(
                name, TRITONSERVER_PARAMETER_STRING, val.c_str()),
            true)
  {
  }

  PyParameter(const char* name, int64_t val)
      : PyWrapper(
            TRITONSERVER_ParameterNew(name, TRITONSERVER_PARAMETER_INT, &val),
            true)
  {
  }

  PyParameter(const char* name, bool val)
      : PyWrapper(
            TRITONSERVER_ParameterNew(name, TRITONSERVER_PARAMETER_BOOL, &val),
            true)
  {
  }

  PyParameter(const char* name, const void* byte_ptr, uint64_t size)
      : PyWrapper(TRITONSERVER_ParameterBytesNew(name, byte_ptr, size), true)
  {
  }

  ~PyParameter()
  {
    if (owned_ && triton_object_) {
      TRITONSERVER_ParameterDelete(triton_object_);
    }
  }
};

class PyBufferAttributes
    : public PyWrapper<struct TRITONSERVER_BufferAttributes> {
 public:
  DESTRUCTOR_WITH_LOG(PyBufferAttributes, TRITONSERVER_BufferAttributesDelete);

  PyBufferAttributes()
  {
    ThrowIfError(TRITONSERVER_BufferAttributesNew(&triton_object_));
    owned_ = true;
  }

  explicit PyBufferAttributes(
      struct TRITONSERVER_BufferAttributes* ba, const bool owned)
      : PyWrapper(ba, owned)
  {
  }

  void SetMemoryTypeId(int64_t memory_type_id)
  {
    ThrowIfError(TRITONSERVER_BufferAttributesSetMemoryTypeId(
        triton_object_, memory_type_id));
  }

  void SetMemoryType(TRITONSERVER_MemoryType memory_type)
  {
    ThrowIfError(TRITONSERVER_BufferAttributesSetMemoryType(
        triton_object_, memory_type));
  }

  void SetCudaIpcHandle(uintptr_t cuda_ipc_handle)
  {
    ThrowIfError(TRITONSERVER_BufferAttributesSetCudaIpcHandle(
        triton_object_, reinterpret_cast<void*>(cuda_ipc_handle)));
  }

  void SetByteSize(size_t byte_size)
  {
    ThrowIfError(
        TRITONSERVER_BufferAttributesSetByteSize(triton_object_, byte_size));
  }

  // Define methods to get buffer attribute fields
  int64_t MemoryTypeId()
  {
    int64_t memory_type_id = 0;
    ThrowIfError(TRITONSERVER_BufferAttributesMemoryTypeId(
        triton_object_, &memory_type_id));
    return memory_type_id;
  }

  TRITONSERVER_MemoryType MemoryType()
  {
    TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
    ThrowIfError(
        TRITONSERVER_BufferAttributesMemoryType(triton_object_, &memory_type));
    return memory_type;
  }

  uintptr_t CudaIpcHandle()
  {
    void* cuda_ipc_handle = nullptr;
    ThrowIfError(TRITONSERVER_BufferAttributesCudaIpcHandle(
        triton_object_, &cuda_ipc_handle));
    return reinterpret_cast<uintptr_t>(cuda_ipc_handle);
  }

  size_t ByteSize()
  {
    size_t byte_size;
    ThrowIfError(
        TRITONSERVER_BufferAttributesByteSize(triton_object_, &byte_size));
    return byte_size;
  }
};

class PyResponseAllocator
    : public PyWrapper<struct TRITONSERVER_ResponseAllocator> {
 public:
  DESTRUCTOR_WITH_LOG(
      PyResponseAllocator, TRITONSERVER_ResponseAllocatorDelete);

  // Callback resource that holds Python user provided buffer and
  // Triton C callback wrappers. This struct will be used for both
  // 'allocator_userp' and 'buffer_userp'
  struct CallbackResource {
    CallbackResource(const py::object& a, const py::object& uo)
        : allocator(a), user_object(uo)
    {
    }
    // Storing the py::object of PyResponseAllocator to have convenient access
    // to callbacks.
    py::object allocator;
    py::object user_object;
  };
  using AllocFn = std::function<
      std::tuple<uintptr_t, py::object, TRITONSERVER_MemoryType, int64_t>(
          py::object, std::string, size_t, TRITONSERVER_MemoryType, int64_t,
          py::object)>;
  using ReleaseFn = std::function<void(
      py::object, uintptr_t, py::object, size_t, TRITONSERVER_MemoryType,
      int64_t)>;
  using StartFn = std::function<void(py::object, py::object)>;

  // size as input, optional?
  using QueryFn = std::function<std::tuple<TRITONSERVER_MemoryType, int64_t>(
      py::object, py::object, std::string, std::optional<size_t>,
      TRITONSERVER_MemoryType, int64_t)>;
  using BufferAttributesFn = std::function<py::object(
      py::object, std::string, py::object, py::object, py::object)>;

  PyResponseAllocator(AllocFn alloc, ReleaseFn release)
      : alloc_fn_(alloc), release_fn_(release)
  {
    ThrowIfError(TRITONSERVER_ResponseAllocatorNew(
        &triton_object_, PyTritonAllocFn, PyTritonReleaseFn, nullptr));
    owned_ = true;
  }

  PyResponseAllocator(AllocFn alloc, ReleaseFn release, StartFn start)
      : alloc_fn_(alloc), release_fn_(release), start_fn_(start)
  {
    ThrowIfError(TRITONSERVER_ResponseAllocatorNew(
        &triton_object_, PyTritonAllocFn, PyTritonReleaseFn, PyTritonStartFn));
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
    py::gil_scoped_acquire gil;
    struct TRITONSERVER_Error* err = nullptr;
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
    py::gil_scoped_acquire gil;
    struct TRITONSERVER_Error* err = nullptr;
    auto cr = reinterpret_cast<CallbackResource*>(buffer_userp);
    try {
      cr->allocator.cast<PyResponseAllocator*>()->release_fn_(
          cr->allocator, reinterpret_cast<uintptr_t>(buffer), cr->user_object,
          byte_size, memory_type, memory_type_id);
    }
    catch (py::error_already_set& ex) {
      err = CreateTRITONSERVER_ErrorFrom(ex);
    }
    // Done with CallbackResource associated with this buffer
    delete cr;
    return err;
  }

  static TRITONSERVER_Error* PyTritonStartFn(
      struct TRITONSERVER_ResponseAllocator* allocator, void* userp)
  {
    py::gil_scoped_acquire gil;
    struct TRITONSERVER_Error* err = nullptr;
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
    py::gil_scoped_acquire gil;
    struct TRITONSERVER_Error* err = nullptr;
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
    py::gil_scoped_acquire gil;
    struct TRITONSERVER_Error* err = nullptr;
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
        triton_object_, PyTritonBufferAttributesFn));
  }

  void SetQueryFunction(QueryFn qf)
  {
    query_fn_ = qf;
    ThrowIfError(TRITONSERVER_ResponseAllocatorSetQueryFunction(
        triton_object_, PyTritonQueryFn));
  }

 private:
  AllocFn alloc_fn_{nullptr};
  ReleaseFn release_fn_{nullptr};
  StartFn start_fn_{nullptr};
  QueryFn query_fn_{nullptr};
  BufferAttributesFn buffer_attributes_fn_{nullptr};
};

class PyMessage : public PyWrapper<struct TRITONSERVER_Message> {
 public:
  DESTRUCTOR_WITH_LOG(PyMessage, TRITONSERVER_MessageDelete);

  PyMessage(const std::string& serialized_json)
  {
    ThrowIfError(TRITONSERVER_MessageNewFromSerializedJson(
        &triton_object_, serialized_json.c_str(), serialized_json.size()));
    owned_ = true;
  }

  explicit PyMessage(struct TRITONSERVER_Message* m, const bool owned)
      : PyWrapper(m, owned)
  {
  }

  std::string SerializeToJson()
  {
    const char* base = nullptr;
    size_t byte_size = 0;
    ThrowIfError(
        TRITONSERVER_MessageSerializeToJson(triton_object_, &base, &byte_size));
    return std::string(base, byte_size);
  }
};

class PyMetrics : public PyWrapper<struct TRITONSERVER_Metrics> {
 public:
  DESTRUCTOR_WITH_LOG(PyMetrics, TRITONSERVER_MetricsDelete);

  explicit PyMetrics(struct TRITONSERVER_Metrics* metrics, bool owned)
      : PyWrapper(metrics, owned)
  {
  }

  std::string Formatted(TRITONSERVER_MetricFormat format)
  {
    const char* base = nullptr;
    size_t byte_size = 0;
    ThrowIfError(TRITONSERVER_MetricsFormatted(
        triton_object_, format, &base, &byte_size));
    return std::string(base, byte_size);
  }
};

class PyTrace : public PyWrapper<struct TRITONSERVER_InferenceTrace> {
 public:
  DESTRUCTOR_WITH_LOG(PyTrace, TRITONSERVER_InferenceTraceDelete);

  using TimestampActivityFn = std::function<void(
      py::object, TRITONSERVER_InferenceTraceActivity, uint64_t, py::object)>;
  using TensorActivityFn = std::function<void(
      py::object, TRITONSERVER_InferenceTraceActivity, std::string,
      TRITONSERVER_DataType, uintptr_t, size_t, py::array_t<int64_t>,
      TRITONSERVER_MemoryType, int64_t, py::object)>;
  using ReleaseFn = std::function<void(std::shared_ptr<PyTrace>, py::object)>;

  struct CallbackResource {
    CallbackResource(
        TimestampActivityFn ts, TensorActivityFn t, ReleaseFn r,
        const py::object& uo)
        : timestamp_fn(ts), tensor_fn(t), release_fn(r), user_object(uo)
    {
    }
    TimestampActivityFn timestamp_fn{nullptr};
    TensorActivityFn tensor_fn{nullptr};
    ReleaseFn release_fn{nullptr};
    py::object user_object;
    // The trace API will use the same 'trace_userp' for all traces associated
    // with the request, and because there is no guarantee that the root trace
    // must be released last, need to track all trace seen / released to
    // determine whether this CallbackResource may be released.
    std::set<uintptr_t> seen_traces;
  };

  // Use internally when interacting with C APIs that takes ownership,
  // this function will also release the ownership of the callback resource
  // because once the ownership is transferred, the callback resource
  // will be accessed in the callback pipeline and should not be tied to the
  // PyWrapper's lifecycle. The callback resource will be released in the
  // Triton C callback wrapper.
  struct TRITONSERVER_InferenceTrace* Release()
  {
    owned_ = false;
    callback_resource_.release();
    return triton_object_;
  }

  PyTrace(
      int level, uint64_t parent_id, TimestampActivityFn timestamp,
      ReleaseFn release, const py::object& user_object)
      : callback_resource_(
            new CallbackResource(timestamp, nullptr, release, user_object))
  {
    ThrowIfError(TRITONSERVER_InferenceTraceNew(
        &triton_object_, static_cast<TRITONSERVER_InferenceTraceLevel>(level),
        parent_id, PyTritonTraceTimestampActivityFn, PyTritonTraceRelease,
        callback_resource_.get()));
    owned_ = true;
  }

  PyTrace(
      int level, uint64_t parent_id, TimestampActivityFn timestamp,
      TensorActivityFn tensor, ReleaseFn release, const py::object& user_object)
      : callback_resource_(
            new CallbackResource(timestamp, tensor, release, user_object))
  {
    ThrowIfError(TRITONSERVER_InferenceTraceTensorNew(
        &triton_object_, static_cast<TRITONSERVER_InferenceTraceLevel>(level),
        parent_id, PyTritonTraceTimestampActivityFn,
        PyTritonTraceTensorActivityFn, PyTritonTraceRelease,
        callback_resource_.get()));
    owned_ = true;
  }

  explicit PyTrace(struct TRITONSERVER_InferenceTrace* t, const bool owned)
      : PyWrapper(t, owned)
  {
  }

  CallbackResource* ReleaseCallbackResource()
  {
    return callback_resource_.release();
  }

  uint64_t Id()
  {
    uint64_t val = 0;
    ThrowIfError(TRITONSERVER_InferenceTraceId(triton_object_, &val));
    return val;
  }

  uint64_t ParentId()
  {
    uint64_t val = 0;
    ThrowIfError(TRITONSERVER_InferenceTraceParentId(triton_object_, &val));
    return val;
  }

  std::string ModelName()
  {
    const char* val = nullptr;
    ThrowIfError(TRITONSERVER_InferenceTraceModelName(triton_object_, &val));
    return val;
  }

  int64_t ModelVersion()
  {
    int64_t val = 0;
    ThrowIfError(TRITONSERVER_InferenceTraceModelVersion(triton_object_, &val));
    return val;
  }

  std::string RequestId()
  {
    const char* val = nullptr;
    ThrowIfError(TRITONSERVER_InferenceTraceRequestId(triton_object_, &val));
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
    py::gil_scoped_acquire gil;
    // Note that 'trace' associated with the activity is not necessary the
    // root trace captured in Callback Resource, so need to always wrap 'trace'
    // in PyTrace for the Python callback to interact with the correct trace.
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
    py::gil_scoped_acquire gil;
    // See 'PyTritonTraceTimestampActivityFn' for 'pt' explanation.
    PyTrace pt(trace, false /* owned */);
    auto cr = reinterpret_cast<CallbackResource*>(userp);
    cr->seen_traces.insert(reinterpret_cast<uintptr_t>(trace));
    cr->tensor_fn(
        py::cast(pt, py::return_value_policy::reference), activity, name,
        datatype, reinterpret_cast<uintptr_t>(base), byte_size,
        py::array_t<int64_t>(dim_count, shape), memory_type, memory_type_id,
        cr->user_object);
  }

  static void PyTritonTraceRelease(
      struct TRITONSERVER_InferenceTrace* trace, void* userp)
  {
    py::gil_scoped_acquire gil;
    // See 'PyTritonTraceTimestampActivityFn' for 'pt' explanation.
    // wrap in shared_ptr to transfer ownership to Python
    auto managed_pt = std::make_shared<PyTrace>(trace, true /* owned */);
    auto cr = reinterpret_cast<CallbackResource*>(userp);
    cr->release_fn(managed_pt, cr->user_object);
    cr->seen_traces.erase(reinterpret_cast<uintptr_t>(trace));
    if (cr->seen_traces.empty()) {
      delete cr;
    }
  }

 private:
  std::unique_ptr<CallbackResource> callback_resource_{nullptr};
};

class PyInferenceResponse
    : public PyWrapper<struct TRITONSERVER_InferenceResponse> {
 public:
  DESTRUCTOR_WITH_LOG(
      PyInferenceResponse, TRITONSERVER_InferenceResponseDelete);

  using CompleteFn = std::function<void(py::object, uint32_t, py::object)>;
  struct CallbackResource {
    CallbackResource(
        CompleteFn c, PyResponseAllocator::CallbackResource* a,
        const py::object& u)
        : complete_fn(c), allocator_resource(a), user_object(u)
    {
    }
    CompleteFn complete_fn;
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

  explicit PyInferenceResponse(
      struct TRITONSERVER_InferenceResponse* response, bool owned)
      : PyWrapper(response, owned)
  {
  }


  void ThrowIfResponseError()
  {
    ThrowIfError(TRITONSERVER_InferenceResponseError(triton_object_));
  }

  std::tuple<std::string, int64_t> Model()
  {
    const char* model_name = nullptr;
    int64_t model_version = 0;
    ThrowIfError(TRITONSERVER_InferenceResponseModel(
        triton_object_, &model_name, &model_version));
    return {model_name, model_version};
  }

  std::string Id()
  {
    const char* val = nullptr;
    ThrowIfError(TRITONSERVER_InferenceResponseId(triton_object_, &val));
    return val;
  }

  uint32_t ParameterCount()
  {
    uint32_t val = 0;
    ThrowIfError(
        TRITONSERVER_InferenceResponseParameterCount(triton_object_, &val));
    return val;
  }

  std::tuple<std::string, TRITONSERVER_ParameterType, py::object> Parameter(
      uint32_t index)
  {
    const char* name = nullptr;
    TRITONSERVER_ParameterType type = TRITONSERVER_PARAMETER_STRING;
    const void* value = nullptr;
    ThrowIfError(TRITONSERVER_InferenceResponseParameter(
        triton_object_, index, &name, &type, &value));
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
        throw UnsupportedError(
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
    ThrowIfError(
        TRITONSERVER_InferenceResponseOutputCount(triton_object_, &val));
    return val;
  }

  std::tuple<
      std::string, TRITONSERVER_DataType, py::array_t<int64_t>, uintptr_t,
      size_t, TRITONSERVER_MemoryType, int64_t, py::object>
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
        triton_object_, index, &name, &datatype, &shape, &dim_count, &base,
        &byte_size, &memory_type, &memory_type_id, &userp));
    return {
        name,
        datatype,
        py::array_t<int64_t>(dim_count, shape),
        reinterpret_cast<uintptr_t>(base),
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
        triton_object_, index, class_index, &val));
    return (val == nullptr) ? "" : val;
  }
};

// forward declaration
class PyServer;

class PyInferenceRequest
    : public PyWrapper<struct TRITONSERVER_InferenceRequest> {
 public:
  DESTRUCTOR_WITH_LOG(PyInferenceRequest, TRITONSERVER_InferenceRequestDelete);

  using ReleaseFn = std::function<void(
      std::shared_ptr<PyInferenceRequest>, uint32_t, py::object)>;

  // Defer definition until PyServer is defined
  PyInferenceRequest(
      PyServer& server, const std::string& model_name,
      const int64_t model_version);

  explicit PyInferenceRequest(
      struct TRITONSERVER_InferenceRequest* r, const bool owned)
      : PyWrapper(r, owned)
  {
  }


  // Use internally when interacting with C APIs that takes ownership,
  // this function will also release the ownership of the callback resource
  // because once the ownership is transferred, the callback resource
  // will be accessed in the callback pipeline and should not be tied to the
  // PyWrapper's lifecycle. The callback resource will be released in the
  // Triton C callback wrapper.
  struct TRITONSERVER_InferenceRequest* Release()
  {
    // Note that Release() doesn't change ownership as the
    // same PyInferenceRequest will be passed along the life cycle.
    allocator_callback_resource_.release();
    response_callback_resource_.release();
    return triton_object_;
  }

  struct CallbackResource {
    CallbackResource(ReleaseFn r, const py::object& uo)
        : release_fn(r), user_object(uo)
    {
    }
    ReleaseFn release_fn;
    py::object user_object;
    // Unsafe handling to ensure the same PyInferenceRequest object
    // goes through the request release cycle. This is due to
    // a 'keep_alive' relationship is built between 'PyInferenceRequest'
    // and 'PyServer': a request is associated with a server and the server
    // should be kept alive until all associated requests is properly released.
    // And here we exploit the 'keep_alive' utility in PyBind to guarantee so.
    // See PyServer::InferAsync on how this field is set to avoid potential
    // circular inclusion.
    std::shared_ptr<PyInferenceRequest> request;
  };


  void SetReleaseCallback(ReleaseFn release, const py::object& user_object)
  {
    request_callback_resource_.reset(
        new CallbackResource(release, user_object));
    ThrowIfError(TRITONSERVER_InferenceRequestSetReleaseCallback(
        triton_object_, PyTritonRequestReleaseCallback,
        request_callback_resource_.get()));
  }

  static void PyTritonRequestReleaseCallback(
      struct TRITONSERVER_InferenceRequest* request, const uint32_t flags,
      void* userp)
  {
    py::gil_scoped_acquire gil;
    auto cr = reinterpret_cast<CallbackResource*>(userp);
    cr->release_fn(cr->request, flags, cr->user_object);
    delete cr;
  }

  void SetResponseCallback(
      const py::object& allocator, const py::object& allocater_user_object,
      PyInferenceResponse::CompleteFn response,
      const py::object& response_user_object)
  {
    allocator_callback_resource_.reset(
        new PyResponseAllocator::CallbackResource(
            allocator, allocater_user_object));
    response_callback_resource_.reset(new PyInferenceResponse::CallbackResource(
        response, allocator_callback_resource_.get(), response_user_object));
    ThrowIfError(TRITONSERVER_InferenceRequestSetResponseCallback(
        triton_object_, allocator.cast<PyResponseAllocator*>()->Ptr(),
        allocator_callback_resource_.get(), PyTritonResponseCompleteCallback,
        response_callback_resource_.get()));
  }
  static void PyTritonResponseCompleteCallback(
      struct TRITONSERVER_InferenceResponse* response, const uint32_t flags,
      void* userp)
  {
    py::gil_scoped_acquire gil;
    auto managed_pt =
        std::make_shared<PyInferenceResponse>(response, true /* owned */);
    auto cr = reinterpret_cast<PyInferenceResponse::CallbackResource*>(userp);
    if (response == nullptr) {
      cr->complete_fn(py::none(), flags, cr->user_object);
    } else {
      cr->complete_fn(py::cast(managed_pt), flags, cr->user_object);
    }
    if (flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) {
      delete cr->allocator_resource;
      delete cr;
    }
  }

  // Trivial setters / getters
  void SetId(const std::string& id)
  {
    ThrowIfError(
        TRITONSERVER_InferenceRequestSetId(triton_object_, id.c_str()));
  }
  std::string Id()
  {
    const char* val = nullptr;
    ThrowIfError(TRITONSERVER_InferenceRequestId(triton_object_, &val));
    return val;
  }

  void SetFlags(uint32_t flags)
  {
    ThrowIfError(TRITONSERVER_InferenceRequestSetFlags(triton_object_, flags));
  }

  uint32_t Flags()
  {
    uint32_t val = 0;
    ThrowIfError(TRITONSERVER_InferenceRequestFlags(triton_object_, &val));
    return val;
  }

  void SetCorrelationId(uint64_t correlation_id)
  {
    ThrowIfError(TRITONSERVER_InferenceRequestSetCorrelationId(
        triton_object_, correlation_id));
  }
  uint64_t CorrelationId()
  {
    uint64_t val = 0;
    ThrowIfError(
        TRITONSERVER_InferenceRequestCorrelationId(triton_object_, &val));
    return val;
  }
  void SetCorrelationIdString(const std::string& correlation_id)
  {
    ThrowIfError(TRITONSERVER_InferenceRequestSetCorrelationIdString(
        triton_object_, correlation_id.c_str()));
  }
  std::string CorrelationIdString()
  {
    const char* val = nullptr;
    ThrowIfError(
        TRITONSERVER_InferenceRequestCorrelationIdString(triton_object_, &val));
    return val;
  }

  void SetPriority(uint32_t priority)
  {
    ThrowIfError(
        TRITONSERVER_InferenceRequestSetPriority(triton_object_, priority));
  }
  void SetPriorityUint64(uint64_t priority)
  {
    ThrowIfError(TRITONSERVER_InferenceRequestSetPriorityUInt64(
        triton_object_, priority));
  }
  uint32_t Priority()
  {
    uint32_t val = 0;
    ThrowIfError(TRITONSERVER_InferenceRequestPriority(triton_object_, &val));
    return val;
  }
  uint64_t PriorityUint64()
  {
    uint64_t val = 0;
    ThrowIfError(
        TRITONSERVER_InferenceRequestPriorityUInt64(triton_object_, &val));
    return val;
  }

  void SetTimeoutMicroseconds(uint64_t timeout_us)
  {
    ThrowIfError(TRITONSERVER_InferenceRequestSetTimeoutMicroseconds(
        triton_object_, timeout_us));
  }
  uint64_t TimeoutMicroseconds()
  {
    uint64_t val = 0;
    ThrowIfError(
        TRITONSERVER_InferenceRequestTimeoutMicroseconds(triton_object_, &val));
    return val;
  }

  void AddInput(
      const std::string& name, TRITONSERVER_DataType data_type,
      std::vector<int64_t> shape)
  {
    ThrowIfError(TRITONSERVER_InferenceRequestAddInput(
        triton_object_, name.c_str(), data_type, shape.data(), shape.size()));
  }
  void AddRawInput(const std::string& name)
  {
    ThrowIfError(
        TRITONSERVER_InferenceRequestAddRawInput(triton_object_, name.c_str()));
  }
  void RemoveInput(const std::string& name)
  {
    ThrowIfError(
        TRITONSERVER_InferenceRequestRemoveInput(triton_object_, name.c_str()));
  }
  void RemoveAllInputs()
  {
    ThrowIfError(TRITONSERVER_InferenceRequestRemoveAllInputs(triton_object_));
  }
  void AppendInputData(
      const std::string& name, uintptr_t base, size_t byte_size,
      TRITONSERVER_MemoryType memory_type, int64_t memory_type_id)
  {
    ThrowIfError(TRITONSERVER_InferenceRequestAppendInputData(
        triton_object_, name.c_str(), reinterpret_cast<const char*>(base),
        byte_size, memory_type, memory_type_id));
  }
  void AppendInputDataWithHostPolicy(
      const std::string name, uintptr_t base, size_t byte_size,
      TRITONSERVER_MemoryType memory_type, int64_t memory_type_id,
      const std::string& host_policy_name)
  {
    ThrowIfError(TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(
        triton_object_, name.c_str(), reinterpret_cast<const char*>(base),
        byte_size, memory_type, memory_type_id, host_policy_name.c_str()));
  }
  void AppendInputDataWithBufferAttributes(
      const std::string& name, uintptr_t base,
      PyBufferAttributes* buffer_attributes)
  {
    ThrowIfError(
        TRITONSERVER_InferenceRequestAppendInputDataWithBufferAttributes(
            triton_object_, name.c_str(), reinterpret_cast<const char*>(base),
            buffer_attributes->Ptr()));
  }
  void RemoveAllInputData(const std::string& name)
  {
    ThrowIfError(TRITONSERVER_InferenceRequestRemoveAllInputData(
        triton_object_, name.c_str()));
  }

  void AddRequestedOutput(const std::string& name)
  {
    ThrowIfError(TRITONSERVER_InferenceRequestAddRequestedOutput(
        triton_object_, name.c_str()));
  }
  void RemoveRequestedOutput(const std::string& name)
  {
    ThrowIfError(TRITONSERVER_InferenceRequestRemoveRequestedOutput(
        triton_object_, name.c_str()));
  }
  void RemoveAllRequestedOutputs()
  {
    ThrowIfError(
        TRITONSERVER_InferenceRequestRemoveAllRequestedOutputs(triton_object_));
  }

  void SetStringParameter(const std::string& key, const std::string& value)
  {
    ThrowIfError(TRITONSERVER_InferenceRequestSetStringParameter(
        triton_object_, key.c_str(), value.c_str()));
  }
  void SetIntParameter(const std::string& key, int64_t value)
  {
    ThrowIfError(TRITONSERVER_InferenceRequestSetIntParameter(
        triton_object_, key.c_str(), value));
  }
  void SetBoolParameter(const std::string& key, bool value)
  {
    ThrowIfError(TRITONSERVER_InferenceRequestSetBoolParameter(
        triton_object_, key.c_str(), value));
  }
  void Cancel()
  {
    ThrowIfError(TRITONSERVER_InferenceRequestCancel(triton_object_));
  }


 public:
  std::unique_ptr<CallbackResource> request_callback_resource_{nullptr};

 private:
  std::unique_ptr<PyResponseAllocator::CallbackResource>
      allocator_callback_resource_{nullptr};
  std::unique_ptr<PyInferenceResponse::CallbackResource>
      response_callback_resource_{nullptr};
};

class PyServerOptions : public PyWrapper<struct TRITONSERVER_ServerOptions> {
 public:
  DESTRUCTOR_WITH_LOG(PyServerOptions, TRITONSERVER_ServerOptionsDelete);
  PyServerOptions()
  {
    ThrowIfError(TRITONSERVER_ServerOptionsNew(&triton_object_));
    owned_ = true;
  }

  void SetServerId(const std::string& server_id)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetServerId(
        triton_object_, server_id.c_str()));
  }

  void SetModelRepositoryPath(const std::string& model_repository_path)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetModelRepositoryPath(
        triton_object_, model_repository_path.c_str()));
  }

  void SetModelControlMode(TRITONSERVER_ModelControlMode mode)
  {
    ThrowIfError(
        TRITONSERVER_ServerOptionsSetModelControlMode(triton_object_, mode));
  }

  void SetStartupModel(const std::string& model_name)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetStartupModel(
        triton_object_, model_name.c_str()));
  }

  void SetStrictModelConfig(bool strict)
  {
    ThrowIfError(
        TRITONSERVER_ServerOptionsSetStrictModelConfig(triton_object_, strict));
  }
  void SetRateLimiterMode(TRITONSERVER_RateLimitMode mode)
  {
    ThrowIfError(
        TRITONSERVER_ServerOptionsSetRateLimiterMode(triton_object_, mode));
  }

  void AddRateLimiterResource(
      const std::string& resource_name, size_t resource_count, int device)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsAddRateLimiterResource(
        triton_object_, resource_name.c_str(), resource_count, device));
  }

  void SetPinnedMemoryPoolByteSize(uint64_t size)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetPinnedMemoryPoolByteSize(
        triton_object_, size));
  }

  void SetCudaMemoryPoolByteSize(int gpu_device, uint64_t size)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetCudaMemoryPoolByteSize(
        triton_object_, gpu_device, size));
  }
  void SetResponseCacheByteSize(uint64_t size)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetResponseCacheByteSize(
        triton_object_, size));
  }

  void SetCacheConfig(
      const std::string& cache_name, const std::string& config_json)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetCacheConfig(
        triton_object_, cache_name.c_str(), config_json.c_str()));
  }

  void SetCacheDirectory(const std::string& cache_dir)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetCacheDirectory(
        triton_object_, cache_dir.c_str()));
  }

  void SetMinSupportedComputeCapability(double cc)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability(
        triton_object_, cc));
  }

  void SetExitOnError(bool exit)
  {
    ThrowIfError(
        TRITONSERVER_ServerOptionsSetExitOnError(triton_object_, exit));
  }

  void SetStrictReadiness(bool strict)
  {
    ThrowIfError(
        TRITONSERVER_ServerOptionsSetStrictReadiness(triton_object_, strict));
  }

  void SetExitTimeout(unsigned int timeout)
  {
    ThrowIfError(
        TRITONSERVER_ServerOptionsSetExitTimeout(triton_object_, timeout));
  }
  void SetBufferManagerThreadCount(unsigned int thread_count)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetBufferManagerThreadCount(
        triton_object_, thread_count));
  }

  void SetModelLoadThreadCount(unsigned int thread_count)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetModelLoadThreadCount(
        triton_object_, thread_count));
  }

  void SetModelLoadRetryCount(unsigned int retry_count)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetModelLoadRetryCount(
        triton_object_, retry_count));
  }

  void SetModelNamespacing(bool enable_namespace)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetModelNamespacing(
        triton_object_, enable_namespace));
  }

  void SetLogFile(const std::string& file)
  {
    ThrowIfError(
        TRITONSERVER_ServerOptionsSetLogFile(triton_object_, file.c_str()));
  }

  void SetLogInfo(bool log)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetLogInfo(triton_object_, log));
  }

  void SetLogWarn(bool log)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetLogWarn(triton_object_, log));
  }

  void SetLogError(bool log)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetLogError(triton_object_, log));
  }

  void SetLogFormat(TRITONSERVER_LogFormat format)
  {
    ThrowIfError(
        TRITONSERVER_ServerOptionsSetLogFormat(triton_object_, format));
  }

  void SetLogVerbose(int level)
  {
    ThrowIfError(
        TRITONSERVER_ServerOptionsSetLogVerbose(triton_object_, level));
  }
  void SetMetrics(bool metrics)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetMetrics(triton_object_, metrics));
  }

  void SetGpuMetrics(bool gpu_metrics)
  {
    ThrowIfError(
        TRITONSERVER_ServerOptionsSetGpuMetrics(triton_object_, gpu_metrics));
  }

  void SetCpuMetrics(bool cpu_metrics)
  {
    ThrowIfError(
        TRITONSERVER_ServerOptionsSetCpuMetrics(triton_object_, cpu_metrics));
  }

  void SetMetricsInterval(uint64_t metrics_interval_ms)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetMetricsInterval(
        triton_object_, metrics_interval_ms));
  }

  void SetBackendDirectory(const std::string& backend_dir)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetBackendDirectory(
        triton_object_, backend_dir.c_str()));
  }

  void SetRepoAgentDirectory(const std::string& repoagent_dir)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetRepoAgentDirectory(
        triton_object_, repoagent_dir.c_str()));
  }

  void SetModelLoadDeviceLimit(
      TRITONSERVER_InstanceGroupKind kind, int device_id, double fraction)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetModelLoadDeviceLimit(
        triton_object_, kind, device_id, fraction));
  }

  void SetBackendConfig(
      const std::string& backend_name, const std::string& setting,
      const std::string& value)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetBackendConfig(
        triton_object_, backend_name.c_str(), setting.c_str(), value.c_str()));
  }

  void SetHostPolicy(
      const std::string& policy_name, const std::string& setting,
      const std::string& value)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetHostPolicy(
        triton_object_, policy_name.c_str(), setting.c_str(), value.c_str()));
  }

  void SetMetricsConfig(
      const std::string& name, const std::string& setting,
      const std::string& value)
  {
    ThrowIfError(TRITONSERVER_ServerOptionsSetMetricsConfig(
        triton_object_, name.c_str(), setting.c_str(), value.c_str()));
  }
};

class PyServer : public PyWrapper<struct TRITONSERVER_Server> {
 public:
  DESTRUCTOR_WITH_LOG(PyServer, TRITONSERVER_ServerDelete);

  PyServer(PyServerOptions& options)
  {
    ThrowIfError(TRITONSERVER_ServerNew(&triton_object_, options.Ptr()));
    owned_ = true;
  }

  void Stop() const { ThrowIfError(TRITONSERVER_ServerStop(triton_object_)); }

  void RegisterModelRepository(
      const std::string& repository_path,
      const std::vector<std::shared_ptr<PyParameter>>& name_mapping) const
  {
    std::vector<const struct TRITONSERVER_Parameter*> params;
    for (const auto& nm : name_mapping) {
      params.emplace_back(nm->Ptr());
    }
    ThrowIfError(TRITONSERVER_ServerRegisterModelRepository(
        triton_object_, repository_path.c_str(), params.data(), params.size()));
  }

  void UnregisterModelRepository(const std::string& repository_path) const
  {
    ThrowIfError(TRITONSERVER_ServerUnregisterModelRepository(
        triton_object_, repository_path.c_str()));
  }

  void PollModelRepository() const
  {
    ThrowIfError(TRITONSERVER_ServerPollModelRepository(triton_object_));
  }

  bool IsLive() const
  {
    bool live;
    ThrowIfError(TRITONSERVER_ServerIsLive(triton_object_, &live));
    return live;
  }

  bool IsReady() const
  {
    bool ready;
    ThrowIfError(TRITONSERVER_ServerIsReady(triton_object_, &ready));
    return ready;
  }

  bool ModelIsReady(const std::string& model_name, int64_t model_version) const
  {
    bool ready;
    ThrowIfError(TRITONSERVER_ServerModelIsReady(
        triton_object_, model_name.c_str(), model_version, &ready));
    return ready;
  }

  std::tuple<uint32_t, uintptr_t> ModelBatchProperties(
      const std::string& model_name, int64_t model_version) const
  {
    uint32_t flags;
    void* voidp;
    ThrowIfError(TRITONSERVER_ServerModelBatchProperties(
        triton_object_, model_name.c_str(), model_version, &flags, &voidp));
    return {flags, reinterpret_cast<uintptr_t>(voidp)};
  }

  std::tuple<uint32_t, uintptr_t> ModelTransactionProperties(
      const std::string& model_name, int64_t model_version) const
  {
    uint32_t txn_flags;
    void* voidp;
    ThrowIfError(TRITONSERVER_ServerModelTransactionProperties(
        triton_object_, model_name.c_str(), model_version, &txn_flags, &voidp));
    return {txn_flags, reinterpret_cast<uintptr_t>(voidp)};
  }

  std::shared_ptr<PyMessage> Metadata() const
  {
    struct TRITONSERVER_Message* server_metadata;
    ThrowIfError(TRITONSERVER_ServerMetadata(triton_object_, &server_metadata));
    return std::make_shared<PyMessage>(server_metadata, true /* owned */);
  }

  std::shared_ptr<PyMessage> ModelMetadata(
      const std::string& model_name, int64_t model_version) const
  {
    struct TRITONSERVER_Message* model_metadata;
    ThrowIfError(TRITONSERVER_ServerModelMetadata(
        triton_object_, model_name.c_str(), model_version, &model_metadata));
    return std::make_shared<PyMessage>(model_metadata, true /* owned */);
  }

  std::shared_ptr<PyMessage> ModelStatistics(
      const std::string& model_name, int64_t model_version) const
  {
    struct TRITONSERVER_Message* model_stats;
    ThrowIfError(TRITONSERVER_ServerModelStatistics(
        triton_object_, model_name.c_str(), model_version, &model_stats));
    return std::make_shared<PyMessage>(model_stats, true /* owned */);
  }

  std::shared_ptr<PyMessage> ModelConfig(
      const std::string& model_name, int64_t model_version,
      uint32_t config_version = 1) const
  {
    struct TRITONSERVER_Message* model_config;
    ThrowIfError(TRITONSERVER_ServerModelConfig(
        triton_object_, model_name.c_str(), model_version, config_version,
        &model_config));
    return std::make_shared<PyMessage>(model_config, true /* owned */);
  }

  std::shared_ptr<PyMessage> ModelIndex(uint32_t flags) const
  {
    struct TRITONSERVER_Message* model_index;
    ThrowIfError(
        TRITONSERVER_ServerModelIndex(triton_object_, flags, &model_index));
    return std::make_shared<PyMessage>(model_index, true /* owned */);
  }

  void LoadModel(const std::string& model_name)
  {
    // load model is blocking, ensure to release GIL
    py::gil_scoped_release release;
    ThrowIfError(
        TRITONSERVER_ServerLoadModel(triton_object_, model_name.c_str()));
  }

  void LoadModelWithParameters(
      const std::string& model_name,
      const std::vector<std::shared_ptr<PyParameter>>& parameters) const
  {
    std::vector<const struct TRITONSERVER_Parameter*> params;
    for (const auto& p : parameters) {
      params.emplace_back(p->Ptr());
    }
    // load model is blocking, ensure to release GIL
    py::gil_scoped_release release;
    ThrowIfError(TRITONSERVER_ServerLoadModelWithParameters(
        triton_object_, model_name.c_str(), params.data(), params.size()));
  }

  void UnloadModel(const std::string& model_name)
  {
    ThrowIfError(
        TRITONSERVER_ServerUnloadModel(triton_object_, model_name.c_str()));
  }

  void UnloadModelAndDependents(const std::string& model_name)
  {
    ThrowIfError(TRITONSERVER_ServerUnloadModelAndDependents(
        triton_object_, model_name.c_str()));
  }

  std::shared_ptr<PyMetrics> Metrics() const
  {
    struct TRITONSERVER_Metrics* metrics;
    ThrowIfError(TRITONSERVER_ServerMetrics(triton_object_, &metrics));
    return std::make_shared<PyMetrics>(metrics, true /* owned */);
  }

  void InferAsync(
      const std::shared_ptr<PyInferenceRequest>& request, PyTrace& trace)
  {
    // Extra handling to avoid circular inclusion:
    //   request -> request_callback_resource_ -> request
    // 1. extract 'request_callback_resource_' out and provide
    //    scoped handler to place resource back to request if not released,
    //    TRITONSERVER_ServerInferAsync failed in other words.
    // 2. add 'request' into resource so request release callback can access it.
    // 3. call TRITONSERVER_ServerInferAsync.
    // 4. release the extracted resource if TRITONSERVER_ServerInferAsync
    //    returns.
    static auto resource_handler =
        [](PyInferenceRequest::CallbackResource* cr) {
          if (cr != nullptr) {
            cr->request->request_callback_resource_.reset(cr);
            cr->request.reset();
          }
        };
    std::unique_ptr<
        PyInferenceRequest::CallbackResource, decltype(resource_handler)>
        scoped_rh(
            request->request_callback_resource_.release(), resource_handler);
    scoped_rh->request = request;

    ThrowIfError(TRITONSERVER_ServerInferAsync(
        triton_object_, request->Ptr(), trace.Ptr()));
    // Ownership of the internal C object is transferred.
    scoped_rh.release();
    request->Release();
    trace.Release();
  }

  void InferAsync(const std::shared_ptr<PyInferenceRequest>& request)
  {
    static auto resource_handler =
        [](PyInferenceRequest::CallbackResource* cr) {
          if (cr != nullptr) {
            cr->request->request_callback_resource_.reset(cr);
            cr->request.reset();
          }
        };
    std::unique_ptr<
        PyInferenceRequest::CallbackResource, decltype(resource_handler)>
        scoped_rh(
            request->request_callback_resource_.release(), resource_handler);
    scoped_rh->request = request;

    ThrowIfError(
        TRITONSERVER_ServerInferAsync(triton_object_, request->Ptr(), nullptr));
    // Ownership of the internal C object is transferred.
    scoped_rh.release();
    request->Release();
  }
};

class PyMetricFamily : public PyWrapper<struct TRITONSERVER_MetricFamily> {
 public:
  DESTRUCTOR_WITH_LOG(PyMetricFamily, TRITONSERVER_MetricFamilyDelete);

  PyMetricFamily(
      TRITONSERVER_MetricKind kind, const std::string& name,
      const std::string& description)
  {
    TRITONSERVER_MetricFamilyNew(
        &triton_object_, kind, name.c_str(), description.c_str());
    owned_ = true;
  }
};

class PyMetric : public PyWrapper<struct TRITONSERVER_Metric> {
 public:
  DESTRUCTOR_WITH_LOG(PyMetric, TRITONSERVER_MetricDelete);
  PyMetric(
      PyMetricFamily& family,
      const std::vector<std::shared_ptr<PyParameter>>& labels)
  {
    std::vector<const struct TRITONSERVER_Parameter*> params;
    for (const auto& label : labels) {
      params.emplace_back(label->Ptr());
    }
    ThrowIfError(TRITONSERVER_MetricNew(
        &triton_object_, family.Ptr(), params.data(), params.size()));
    owned_ = true;
  }

  double Value() const
  {
    double val = 0;
    ThrowIfError(TRITONSERVER_MetricValue(triton_object_, &val));
    return val;
  }

  void Increment(double val) const
  {
    ThrowIfError(TRITONSERVER_MetricIncrement(triton_object_, val));
  }

  void SetValue(double val) const
  {
    ThrowIfError(TRITONSERVER_MetricSet(triton_object_, val));
  }

  TRITONSERVER_MetricKind Kind() const
  {
    TRITONSERVER_MetricKind val = TRITONSERVER_METRIC_KIND_COUNTER;
    ThrowIfError(TRITONSERVER_GetMetricKind(triton_object_, &val));
    return val;
  }
};

// Deferred definitions..
PyInferenceRequest::PyInferenceRequest(
    PyServer& server, const std::string& model_name,
    const int64_t model_version)
{
  ThrowIfError(TRITONSERVER_InferenceRequestNew(
      &triton_object_, server.Ptr(), model_name.c_str(), model_version));
  owned_ = true;
}

// [FIXME] module name?
PYBIND11_MODULE(triton_bindings, m)
{
  m.doc() = "Python bindings for Triton Inference Server";

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
  pybind11::register_exception<UnknownError>(m, "UnknownError", te.ptr());
  pybind11::register_exception<InternalError>(m, "InternalError", te.ptr());
  pybind11::register_exception<NotFoundError>(m, "NotFoundError", te.ptr());
  pybind11::register_exception<InvalidArgumentError>(
      m, "InvalidArgumentError", te.ptr());
  pybind11::register_exception<UnavailableError>(
      m, "UnavailableError", te.ptr());
  pybind11::register_exception<UnsupportedError>(
      m, "UnsupportedError", te.ptr());
  pybind11::register_exception<AlreadyExistsError>(
      m, "AlreadyExistsError", te.ptr());

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
      // Python bytes can be consumed by function accepting string, so order
      // the py::bytes constructor before string to ensure correct overload
      // constructor is used
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
      }))
      .def(py::init<const char*, const std::string&>())
      .def(py::init<const char*, int64_t>())
      .def(py::init<const char*, bool>());

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
          py::arg("start_function"))
      .def(
          py::init<
              PyResponseAllocator::AllocFn, PyResponseAllocator::ReleaseFn>(),
          py::arg("alloc_function"), py::arg("release_function"))
      .def(
          "set_buffer_attributes_function",
          &PyResponseAllocator::SetBufferAttributesFunction,
          py::arg("buffer_attributes_function"))
      .def(
          "set_query_function", &PyResponseAllocator::SetQueryFunction,
          py::arg("query_function"));

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
              int, uint64_t, PyTrace::TimestampActivityFn,
              PyTrace::TensorActivityFn, PyTrace::ReleaseFn,
              const py::object&>(),
          py::arg("level"), py::arg("parent_id"), py::arg("activity_function"),
          py::arg("tensor_activity_function"), py::arg("release_function"),
          py::arg("trace_userp"))
      .def(
          py::init<
              int, uint64_t, PyTrace::TimestampActivityFn, PyTrace::ReleaseFn,
              const py::object&>(),
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

  py::class_<PyInferenceRequest, std::shared_ptr<PyInferenceRequest>>(
      m, "TRITONSERVER_InferenceRequest")
      .def(
          py::init<PyServer&, const std::string&, int64_t>(),
          py::keep_alive<1, 2>())
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
      .def("set_bool_parameter", &PyInferenceRequest::SetBoolParameter)
      .def("cancel", &PyInferenceRequest::Cancel);

  // TRITONSERVER_InferenceResponse
  py::enum_<TRITONSERVER_ResponseCompleteFlag>(
      m, "TRITONSERVER_ResponseCompleteFlag")
      .value("FINAL", TRITONSERVER_RESPONSE_COMPLETE_FINAL)
      .export_values();
  py::class_<PyInferenceResponse, std::shared_ptr<PyInferenceResponse>>(
      m, "TRITONSERVER_InferenceResponse")
      .def(
          "throw_if_response_error", &PyInferenceResponse::ThrowIfResponseError)
      .def_property_readonly("model", &PyInferenceResponse::Model)
      .def_property_readonly("id", &PyInferenceResponse::Id)
      .def_property_readonly(
          "parameter_count", &PyInferenceResponse::ParameterCount)
      .def("parameter", &PyInferenceResponse::Parameter)
      .def_property_readonly("output_count", &PyInferenceResponse::OutputCount)
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
      .def(
          "set_model_load_retry_count",
          &PyServerOptions::SetModelLoadRetryCount)
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
  py::enum_<TRITONSERVER_ModelBatchFlag>(m, "TRITONSERVER_ModelBatchFlag")
      .value("UNKNOWN", TRITONSERVER_BATCH_UNKNOWN)
      .value("FIRST_DIM", TRITONSERVER_BATCH_FIRST_DIM)
      .export_values();
  py::enum_<TRITONSERVER_ModelIndexFlag>(m, "TRITONSERVER_ModelIndexFlag")
      .value("READY", TRITONSERVER_INDEX_FLAG_READY)
      .export_values();
  py::enum_<TRITONSERVER_ModelTxnPropertyFlag>(
      m, "TRITONSERVER_ModelTxnPropertyFlag")
      .value("ONE_TO_ONE", TRITONSERVER_TXN_ONE_TO_ONE)
      .value("DECOUPLED", TRITONSERVER_TXN_DECOUPLED)
      .export_values();
  py::class_<PyServer>(m, "TRITONSERVER_Server")
      .def(py::init<PyServerOptions&>())
      .def("stop", &PyServer::Stop)
      .def("register_model_repository", &PyServer::RegisterModelRepository)
      .def("unregister_model_repository", &PyServer::UnregisterModelRepository)
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
      .def("metrics", &PyServer::Metrics)
      .def(
          "infer_async",
          py::overload_cast<
              const std::shared_ptr<PyInferenceRequest>&, PyTrace&>(
              &PyServer::InferAsync))
      .def(
          "infer_async",
          py::overload_cast<const std::shared_ptr<PyInferenceRequest>&>(
              &PyServer::InferAsync));

  // TRITONSERVER_MetricKind
  py::enum_<TRITONSERVER_MetricKind>(m, "TRITONSERVER_MetricKind")
      .value("COUNTER", TRITONSERVER_METRIC_KIND_COUNTER)
      .value("GAUGE", TRITONSERVER_METRIC_KIND_GAUGE);
  // TRITONSERVER_MetricFamily
  py::class_<PyMetricFamily>(m, "TRITONSERVER_MetricFamily")
      .def(py::init<
           TRITONSERVER_MetricKind, const std::string&, const std::string&>());
  // TRITONSERVER_Metric
  py::class_<PyMetric>(m, "TRITONSERVER_Metric")
      .def(
          py::init<
              PyMetricFamily&,
              const std::vector<std::shared_ptr<PyParameter>>&>(),
          py::keep_alive<1, 2>())
      .def_property_readonly("value", &PyMetric::Value)
      .def("increment", &PyMetric::Increment)
      .def("set_value", &PyMetric::SetValue)
      .def_property_readonly("kind", &PyMetric::Kind);
}

}}}  // namespace triton::core::python
