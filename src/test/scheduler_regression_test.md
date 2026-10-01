# Scheduler regression tests

Configure the core build with its usual dependencies, then run:

```sh
cmake --build BUILD --target scheduler_regression_test
ctest --test-dir BUILD/test -R SchedulerRegression --output-on-failure
```

Install GDB before configuration to enable the accounting test. It fills a
one-request queue with a batch of eight, rejects seven requests, drains the
admitted batch, and admits one more request. The GDB script checks the queue
counter at each checkpoint and verifies that the later admission remains below
the preferred batch size of eight. The fixture uses public core factories;
GDB reads private accounting state without production test hooks.

The removal tests block a slot check, remove its instance or model, and require
the check to return false. CTest bounds a stuck removal with a 30-second timeout.
The capacity and lookup tests read the queue map while another thread registers
and removes two instances for 2,000 iterations.

Use a separate build with ThreadSanitizer to detect map races:

```sh
cmake -S src -B BUILD_TSAN <dependency options> \
  -DCMAKE_CXX_FLAGS="-fsanitize=thread -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread"
cmake --build BUILD_TSAN --target scheduler_regression_test
TSAN_OPTIONS=halt_on_error=1 ctest --test-dir BUILD_TSAN/test \
  -R 'SchedulerRegression.(capacity|lookup|instance-removal|model-removal)' \
  --output-on-failure
```

The test target compiles the core sources with the core's include paths,
definitions, and dependencies because its shared library hides C++ symbols.
The backend fixture creates passive CPU instances and performs no inference.
