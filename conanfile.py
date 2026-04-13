from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMakeDeps, cmake_layout
from conan.errors import ConanInvalidConfiguration


class TritonCoreConan(ConanFile):
    name = "triton-core"
    version = "2.68.0"
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "enable_gpu":     [True, False],
        "enable_metrics": [True, False],
        "headers_only":   [True, False],
    }
    default_options = {
        "enable_gpu":     True,
        "enable_metrics": True,
        "headers_only":   False,
    }

    def validate(self):
        if self.settings.os != "Linux":
            raise ConanInvalidConfiguration("triton-core only supports Linux")

    def requirements(self):
        if not self.options.headers_only:
            self.requires("protobuf/3.21.12")
            self.requires("re2/20230301")
            self.requires("rapidjson/cci.20230929")
            if self.options.enable_gpu:
                self.requires("cnmem/1.0.0")
                self.requires("dcgm/4.5.3")
            if self.options.enable_metrics:
                self.requires("prometheus-cpp/1.2.4")

    def configure(self):
        if not self.options.headers_only:
            if self.options.enable_metrics:
                self.options["prometheus-cpp"].shared = False
                self.options["prometheus-cpp"].with_pull = False
                self.options["prometheus-cpp"].with_push = False

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["TRITON_CORE_HEADERS_ONLY"]      = self.options.headers_only
        tc.variables["TRITON_ENABLE_GPU"]             = self.options.enable_gpu
        tc.variables["TRITON_ENABLE_METRICS"]         = self.options.enable_metrics
        tc.variables["TRITON_SKIP_THIRD_PARTY_FETCH"] = True
        tc.generate()
        CMakeDeps(self).generate()
