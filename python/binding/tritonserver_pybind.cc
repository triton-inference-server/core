

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
//  * Triton structs are encapsulated in a thin wrapper to isolate raw pointer
//    operations which is not supported in pure Python.
//  * Output parameters are converted to return value, so the APIs return a
//    tuple of (error, *ret_vals), where user will check if `error is None`
//    before examining the return values. The same applies to callbacks

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

// [WIP] to convertor and helper function
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
  TRITONSERVER_Parameter* Release()
  {
    owned_ = false;
    return parameter_;
  }

 private:
  TRITONSERVER_Parameter* parameter_{nullptr};
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
      owned_ = false;
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
    // [WIP] error -> exception in general
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
      TRITONSERVER_ResponseAllocatorDelete(allocator_);
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
  struct CallbackResource {
    CallbackResource(py::object uo) : user_object(uo) {}
    // 'trace' is not initalized until a later point, see Capture()
    // for reasoning.
    py::object trace;
    py::object user_object;
    // The trace API will use the same 'trace_userp' for all traces associated
    // with the request, and becasue there is no guarantee that the root trace
    // must be released last, need to track all trace seen / released to
    // determine whether this CallbackResource may be released.
    std::set<uintptr_t> seen_traces;
  };
  using TimestampActivityFn = std::function<void(
      py::object, TRITONSERVER_InferenceTraceActivity, uint64_t, py::object)>;
  using TensorActivityFn = std::function<void(
      py::object, TRITONSERVER_InferenceTraceActivity, std::string,
      TRITONSERVER_DataType, size_t, size_t, py::array_t<int64_t>,
      TRITONSERVER_MemoryType, int64_t, py::object)>;
  // [FIXME] mean different things in Python binding..
  using ReleaseFn = std::function<void(py::object, py::object)>;

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
      : owned_(true), timestamp_fn_(timestamp), release_fn_(release),
        callback_resource_(new CallbackResource(user_object))
  {
    ThrowIfError(TRITONSERVER_InferenceTraceNew(
        &trace_, level, parent_id, PyTritonTraceTimestampActivityFn,
        PyTritonTraceRelease, callback_resource_.get()));
  }

  PyTrace(
      TRITONSERVER_InferenceTraceLevel level, uint64_t parent_id,
      TimestampActivityFn timestamp, TensorActivityFn tensor, ReleaseFn release,
      py::object user_object)
      : owned_(true), timestamp_fn_(timestamp), tensor_fn_(tensor),
        release_fn_(release),
        callback_resource_(new CallbackResource(user_object))
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
      TRITONSERVER_InferenceTraceDelete(trace_);
    }
  }
  // Capture the py::object representation ('trace') of 'this' into
  // CallbackResource, by doing so the 'trace' Python object will be kept
  // alive due to non-zero reference count even if Python user drops all
  // reference on the Python side.
  // This function will be called in the binding equivalent of
  // 'TRITONSERVER_ServerInferAsync' to follow the behavior that the API
  // unconditionally takes ownership of 'trace' and only releases through
  // release callback, which implies the user may drop all references and
  // expect the same object to be passed into release callback.
  // Note that the capture must be separated from object construction to comply
  // with regular Python object lifecycle until 'TRITONSERVER_ServerInferAsync'.
  void Capture(py::object trace)
  {
    if (callback_resource_) {
      callback_resource_->trace = trace;
    }
  }
  void InternalRelease()
  {
    if (callback_resource_) {
      callback_resource_.reset();
    }
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
    cr->trace.cast<PyTrace*>()->timestamp_fn_(
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
    cr->trace.cast<PyTrace*>()->tensor_fn_(
        py::cast(pt, py::return_value_policy::reference), activity, name,
        datatype, reinterpret_cast<size_t>(base), byte_size,
        py::array_t<int64_t>({dim_count}, {}, shape), memory_type,
        memory_type_id, cr->user_object);
  }

  static void PyTritonTraceRelease(
      struct TRITONSERVER_InferenceTrace* trace, void* userp)
  {
    // See 'PyTritonTraceTimestampActivityFn' for 'pt' explanation.
    PyTrace pt(trace, true /* owned */);
    auto cr = reinterpret_cast<CallbackResource*>(userp);
    cr->trace.cast<PyTrace*>()->release_fn_(
        py::cast(pt, py::return_value_policy::reference), cr->user_object);
    cr->seen_traces.erase(reinterpret_cast<uintptr_t>(trace));
    if (cr->seen_traces.empty()) {
      auto py_trace = cr->trace;
      py_trace.cast<PyTrace*>()->InternalRelease();
    }
  }

  //  private:
  TRITONSERVER_InferenceTrace* trace_{nullptr};
  // [FIXME] may need to transfer ownership
  bool owned_{false};

  TimestampActivityFn timestamp_fn_{nullptr};
  TensorActivityFn tensor_fn_{nullptr};
  ReleaseFn release_fn_{nullptr};
  std::unique_ptr<CallbackResource> callback_resource_{nullptr};
};


class PyInferenceRequest {
 public:
  // [WIP] PyServer ...
  PyInferenceRequest(
      TRITONSERVER_Server* server, const char* model_name,
      const int64_t model_version)
      : owned_(true)
  {
    ThrowIfError(TRITONSERVER_InferenceRequestNew(
        &request_, server, model_name, model_version));
  }

  ~PyInferenceRequest()
  {
    if (owned_ && request_) {
      TRITONSERVER_InferenceRequestDelete(request_);
    }
  }

  struct CallbackResource {
    CallbackResource(py::object r, py::object uo) : request(r), user_object(uo)
    {
    }
    py::object request;
    py::object user_object;
  }

  using ReleaseFn = std::function<void(py::object, uint32_t, py::object)>;

  void SetReleaseCallback(ReleaseFn release)
  {
    ThrowIfError(TRITONSERVER_InferenceRequestSetReleaseCallback(request_, ))
  }

  static TRITONSERVER_Error* PyTritonRequestReleaseCallback(
      struct TRITONSERVER_InferenceRequest* request, const uint32_t flags,
      void* userp)
  {
    auto cr = reinterpret_cast<CallbackResource*>(userp);
    cr->request.insert(reinterpret_cast<uintptr_t>(trace));
    cr->request.cast<PyInferenceRequest*>()->release_fn_(
        cr->request, flags, cr->user_object);
    delete cr;
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

  void SetFlag(uint32_t flags)
  {
    ThrowIfError(TRITONSERVER_InferenceRequestSetFlags(request_, flags));
  }

  uint32_t Flag()
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
    uint32_t val = 0;
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
  TRITONSERVER_InferenceRequest* request_{nullptr};
  bool owned_{false};
};

// [WIP] NOTE: 'response_allocator_userp' in
// TRITONSERVER_InferenceRequestSetResponseCallback is not keeping alive
// internally as there is no clear exit point on when the reference can be
// dropped, in constrast to 'response_userp' where the exit point is clear
// to be the ResponseComplete invocation with FINAL flag.

// [FIXME] testing field
// ========================================================
class Allocator {
 public:
  // WAR: PyBind optional_cast deduction seems to be wrong in the case of
  // std::tuple<std::optional<CustomType>, ...> which requires non-const copy
  // constructor to be provided for CustomType. Copy constructor should be
  // avoided in the wrapper class and thus py::object is specified as return
  // type. And all QueryFn usage will assume the actual return type is
  //   std::tuple<
  //     std::optional<PyError>, TRITONSERVER_MemoryType, int64_t>
  using QueryFn = std::function<py::object(
      std::string, TRITONSERVER_MemoryType, int64_t, size_t*)>;

  DISALLOW_COPY_AND_ASSIGN(Allocator);

  Allocator(QueryFn func) : func_(func) {}
  QueryFn func_{nullptr};
};
// [FIXME] end of testing field
// ========================================================

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
  py::class_<PyParameter>(m, "TRITONSERVER_Parameter")
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
  py::class_<PyMessage>(m, "TRITONSERVER_Message")
      .def(py::init<const std::string&>())
      .def("serialize_to_json", &PyMessage::SerializeToJson);

  // TRITONSERVER_Metrics
  py::enum_<TRITONSERVER_MetricFormat>(m, "TRITONSERVER_MetricFormat")
      .value("PROMETHEUS", TRITONSERVER_METRIC_PROMETHEUS);
  py::class_<PyMetrics>(m, "TRITONSERVER_Metrics")
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
      .def_property_readonly("request_id", &PyTrace::RequestId)
      // ========================================================
      .def(
          "capture",
          [](py::object trace) { trace.cast<PyTrace*>()->Capture(trace); })
      .def("release", &PyTrace::InternalRelease)
      // ========================================================
      ;
}

}}}  // namespace triton::core::python
