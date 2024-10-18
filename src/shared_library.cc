// Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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

#include "shared_library.h"

#include "filesystem/api.h"
#include "mutex"
#include "triton/common/logging.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif

#ifdef _WIN32
// suppress the min and max definitions in Windef.h.
#define NOMINMAX
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

namespace triton { namespace core {

static std::mutex mu_;

Status
SharedLibrary::Acquire(std::unique_ptr<SharedLibrary>* slib)
{
  mu_.lock();
  slib->reset(new SharedLibrary());
  return Status::Success;
}

SharedLibrary::~SharedLibrary()
{
  mu_.unlock();
}

Status
SharedLibrary::SetLibraryDirectory(const std::string& path)
{
#ifdef _WIN32
  LOG_VERBOSE(1) << "SetLibraryDirectory: path = " << path;
  if (!SetDllDirectory(path.c_str())) {
    LPSTR err_buffer = nullptr;
    size_t size = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPSTR)&err_buffer, 0, NULL);
    std::string errstr(err_buffer, size);
    LocalFree(err_buffer);

    return Status(
        Status::Code::NOT_FOUND,
        "unable to set dll path " + path + ": " + errstr);
  }
#endif

  return Status::Success;
}

Status
SharedLibrary::ResetLibraryDirectory()
{
#ifdef _WIN32
  LOG_VERBOSE(1) << "ResetLibraryDirectory";
  if (!SetDllDirectory(NULL)) {
    LPSTR err_buffer = nullptr;
    size_t size = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPSTR)&err_buffer, 0, NULL);
    std::string errstr(err_buffer, size);
    LocalFree(err_buffer);

    return Status(
        Status::Code::NOT_FOUND, "unable to reset dll path: " + errstr);
  }
#endif

  return Status::Success;
}

Status
SharedLibrary::OpenLibraryHandle(const std::string& path, void** handle)
{
  LOG_VERBOSE(1) << "OpenLibraryHandle: " << path;

#ifdef TRITON_ENABLE_GPU
  // This call is to prevent a deadlock issue with dlopening backend shared
  // libraries and calling CUDA APIs in other threads. Since CUDA API also
  // dlopens some libraries and dlopen has an internal lock, it can create a
  // deadlock. Intentionally ignore the CUDA_ERROR (if any) for containers
  // running on a CPU.
  int device_count;
  cudaGetDeviceCount(&device_count);
#endif

#ifdef _WIN32
  // Need to put shared library directory on the DLL path so that any
  // dependencies of the shared library are found
  const std::string library_dir = DirName(path);
  RETURN_IF_ERROR(SetLibraryDirectory(library_dir));

  // HMODULE is typedef of void*
  // https://docs.microsoft.com/en-us/windows/win32/winprog/windows-data-types
  LOG_VERBOSE(1) << "OpenLibraryHandle: path = " << path;
  *handle = LoadLibrary(path.c_str());

  // Remove the dll path added above... do this unconditionally before
  // check for failure in dll load.
  RETURN_IF_ERROR(ResetLibraryDirectory());

  if (*handle == nullptr) {
    LPSTR err_buffer = nullptr;
    size_t size = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPSTR)&err_buffer, 0, NULL);
    std::string errstr(err_buffer, size);
    LocalFree(err_buffer);

    return Status(
        Status::Code::NOT_FOUND, "unable to load shared library: " + errstr);
  }
#else
  *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (*handle == nullptr) {
    return Status(
        Status::Code::NOT_FOUND,
        "unable to load shared library: " + std::string(dlerror()));
  }
#endif

  return Status::Success;
}

Status
SharedLibrary::CloseLibraryHandle(void* handle)
{
  if (handle != nullptr) {
#ifdef _WIN32
    if (FreeLibrary((HMODULE)handle) == 0) {
      LPSTR err_buffer = nullptr;
      size_t size = FormatMessageA(
          FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
              FORMAT_MESSAGE_IGNORE_INSERTS,
          NULL, GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
          (LPSTR)&err_buffer, 0, NULL);
      std::string errstr(err_buffer, size);
      LocalFree(err_buffer);
      return Status(
          Status::Code::INTERNAL, "unable to unload shared library: " + errstr);
    }
#else
    if (dlclose(handle) != 0) {
      return Status(
          Status::Code::INTERNAL,
          "unable to unload shared library: " + std::string(dlerror()));
    }
#endif
  }

  return Status::Success;
}

Status
SharedLibrary::GetEntrypoint(
    void* handle, const std::string& name, const bool optional, void** befn)
{
  *befn = nullptr;

#ifdef _WIN32
  void* fn = GetProcAddress((HMODULE)handle, name.c_str());
  if ((fn == nullptr) && !optional) {
    LPSTR err_buffer = nullptr;
    size_t size = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPSTR)&err_buffer, 0, NULL);
    std::string errstr(err_buffer, size);
    LocalFree(err_buffer);
    return Status(
        Status::Code::NOT_FOUND,
        "unable to find '" + name +
            "' entrypoint in custom library: " + errstr);
  }
#else
  dlerror();
  void* fn = dlsym(handle, name.c_str());
  const char* dlsym_error = dlerror();
  if (dlsym_error != nullptr) {
    if (optional) {
      return Status::Success;
    }

    std::string errstr(dlsym_error);  // need copy as dlclose overwrites
    return Status(
        Status::Code::NOT_FOUND, "unable to find required entrypoint '" + name +
                                     "' in shared library: " + errstr);
  }

  if (fn == nullptr) {
    if (optional) {
      return Status::Success;
    }

    return Status(
        Status::Code::NOT_FOUND,
        "unable to find required entrypoint '" + name + "' in shared library");
  }
#endif

  *befn = fn;
  return Status::Success;
}

Status
SharedLibrary::AddAdditionalDependencyDir(
    const std::string& additional_path, std::wstring& original_path)
{
#ifdef _WIN32
  const std::wstring PATH(L"Path");

  if (additional_path.back() != ';') {
    return Status(
        Status::Code::INVALID_ARG,
        "backend config parameter \"additional-dependency-dirs\" is malformed. "
        "Each additional path provided should terminate with a ';'.");
  }

  DWORD len = GetEnvironmentVariableW(PATH.c_str(), NULL, 0);
  if (len > 0) {
    original_path.resize(len);
    GetEnvironmentVariableW(PATH.c_str(), &original_path[0], len);
  } else {
    original_path = L"";
  }

  LOG_VERBOSE(1) << "Environment before extending PATH: "
                 << std::string(original_path.begin(), original_path.end());

  std::wstring updated_path_value =
      std::wstring(additional_path.begin(), additional_path.end());
  updated_path_value += original_path;

  if (!SetEnvironmentVariableW(PATH.c_str(), updated_path_value.c_str())) {
    LPSTR err_buffer = nullptr;
    size_t size = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPSTR)&err_buffer, 0, NULL);
    std::string errstr(err_buffer, size);
    LocalFree(err_buffer);
    return Status(
        Status::Code::INTERNAL,
        "failed to append user-provided directory to PATH " + errstr);
  }

  if (LOG_VERBOSE_IS_ON(1)) {
    std::wstring path_after;
    len = GetEnvironmentVariableW(PATH.c_str(), NULL, 0);
    if (len > 0) {
      path_after.resize(len);
      GetEnvironmentVariableW(PATH.c_str(), &path_after[0], len);
    }
    LOG_VERBOSE(1) << "Environment after extending PATH: "
                   << std::string(path_after.begin(), path_after.end());
  }
#else
  LOG_WARNING
      << "The parameter \"additional-dependency-dirs\" has been specified but "
         "is not supported for Linux. It is currently a Windows-only feature. "
         "No change to the environment will take effect.";
#endif
  return Status::Success;
}

Status
SharedLibrary::RemoveAdditionalDependencyDir(const std::wstring& original_path)
{
#ifdef _WIN32
  const std::wstring PATH(L"Path");
  if (!SetEnvironmentVariableW(PATH.c_str(), original_path.c_str())) {
    LPSTR err_buffer = nullptr;
    size_t size = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPSTR)&err_buffer, 0, NULL);
    std::string errstr(err_buffer, size);
    LocalFree(err_buffer);
    return Status(
        Status::Code::INTERNAL,
        "failed to restore PATH to its original configuration " + errstr);
  }

  if (LOG_VERBOSE_IS_ON(1)) {
    std::wstring path_after;
    DWORD len = GetEnvironmentVariableW(PATH.c_str(), NULL, 0);
    if (len > 0) {
      path_after.resize(len);
      GetEnvironmentVariableW(PATH.c_str(), &path_after[0], len);
    }
    LOG_VERBOSE(1) << "Environment after restoring PATH: "
                   << std::string(path_after.begin(), path_after.end());
  }
#endif
  return Status::Success;
}

}}  // namespace triton::core
