// Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "latency_histogram.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>

namespace triton { namespace core {

namespace {

std::string
EscapeLabel(const std::string& value)
{
  std::string out;
  out.reserve(value.size());
  for (const char c : value) {
    switch (c) {
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '"':
        out += "\\\"";
        break;
      default:
        out.push_back(c);
        break;
    }
  }
  return out;
}

std::string
FormatLe(double boundary)
{
  if (std::isfinite(boundary) && boundary >= 0.0 &&
      boundary == std::floor(boundary) &&
      boundary < static_cast<double>(std::numeric_limits<uint64_t>::max())) {
    return std::to_string(static_cast<uint64_t>(boundary));
  }
  std::ostringstream os;
  os.precision(std::numeric_limits<double>::max_digits10);
  os << boundary;
  return os.str();
}

bool
ExpositionDeclaresFamily(const std::string& text, const std::string& name)
{
  const std::string type_mark = "# TYPE " + name + " ";
  const std::string help_mark = "# HELP " + name + " ";
  if (text.find(type_mark) != std::string::npos ||
      text.find(help_mark) != std::string::npos) {
    return true;
  }
  size_t pos = 0;
  while ((pos = text.find(name, pos)) != std::string::npos) {
    const bool at_line_start = (pos == 0) || (text[pos - 1] == '\n');
    const size_t end = pos + name.size();
    const bool at_token_end = (end == text.size()) || (text[end] == '{') ||
                              (text[end] == ' ') || (text[end] == '\n');
    if (at_line_start && at_token_end) {
      return true;
    }
    pos = end;
  }
  return false;
}

size_t
ShardForThisThread()
{
  thread_local const size_t shard = [] {
    const size_t hash =
        std::hash<std::thread::id>{}(std::this_thread::get_id());
    return hash % ShardedLatencyHistogram::kShardCount;
  }();
  return shard;
}

std::vector<double>
NormalizeBoundaries(std::vector<double> boundaries)
{
  boundaries.erase(
      std::remove_if(
          boundaries.begin(), boundaries.end(),
          [](double value) { return !std::isfinite(value); }),
      boundaries.end());
  std::sort(boundaries.begin(), boundaries.end());
  boundaries.erase(
      std::unique(boundaries.begin(), boundaries.end()), boundaries.end());
  if (boundaries.empty()) {
    return DefaultLatencyHistogramBoundaries();
  }
  return boundaries;
}

}  // namespace

bool
IsReservedLatencyHistogramName(const std::string& name)
{
  static constexpr const char* kNames[] = {
      LatencyHistogramName::kRequest,       LatencyHistogramName::kQueue,
      LatencyHistogramName::kComputeInput,  LatencyHistogramName::kComputeInfer,
      LatencyHistogramName::kComputeOutput, LatencyHistogramName::kCacheHit,
      LatencyHistogramName::kCacheMiss};
  for (const char* reserved : kNames) {
    if (name == reserved) {
      return true;
    }
  }
  return false;
}

std::vector<double>
DefaultLatencyHistogramBoundaries()
{
  std::vector<double> buckets;
  double boundary = 1.0;
  constexpr double kMaxUs = 60.0 * 1000.0 * 1000.0;
  constexpr double kRatio = 1.2;
  while (boundary < kMaxUs) {
    buckets.push_back(boundary);
    const double next = boundary * kRatio;
    if (!(next > boundary)) {
      break;
    }
    boundary = next;
  }
  if (buckets.empty() || buckets.back() < kMaxUs) {
    buckets.push_back(kMaxUs);
  }
  return buckets;
}

LatencyMetricToggles
ResolveLatencyMetricToggles(
    const std::vector<std::pair<std::string, std::string>>& settings)
{
  LatencyMetricToggles toggles;
  for (const auto& setting : settings) {
    if (setting.first == "histogram_latencies" && setting.second == "true") {
      toggles.first_response_histogram = true;
    } else if (setting.first == "summary_latencies") {
      const std::string& value = setting.second;
      if (value == "false" || value == "none") {
        toggles.summary_level = SummaryLatencyLevel::kNone;
        toggles.rejected_summary_latencies.clear();
      } else if (value == "simplified") {
        toggles.summary_level = SummaryLatencyLevel::kSimplified;
        toggles.rejected_summary_latencies.clear();
      } else if (value == "true" || value == "verbose") {
        toggles.summary_level = SummaryLatencyLevel::kVerbose;
        toggles.rejected_summary_latencies.clear();
      } else {
        toggles.rejected_summary_latencies = value;
      }
    }
  }
  return toggles;
}

ShardedLatencyHistogram::ShardedLatencyHistogram(
    std::string name, std::string help,
    std::map<std::string, std::string> labels, std::vector<double> boundaries)
    : name_(std::move(name)), help_(std::move(help)),
      labels_(std::move(labels)),
      boundaries_(NormalizeBoundaries(std::move(boundaries))),
      shards_(std::make_unique<Shard[]>(kShardCount)),
      bucket_count_(boundaries_.size() + 1)
{
  for (size_t shard = 0; shard < kShardCount; ++shard) {
    shards_[shard].buckets =
        std::make_unique<std::atomic<uint64_t>[]>(bucket_count_);
  }
}

size_t
ShardedLatencyHistogram::BucketIndex(uint64_t value_us) const
{
  const auto it = std::lower_bound(
      boundaries_.begin(), boundaries_.end(), static_cast<double>(value_us));
  return static_cast<size_t>(it - boundaries_.begin());
}

void
ShardedLatencyHistogram::Observe(uint64_t value_us) noexcept
{
  Shard& shard = shards_[ShardForThisThread()];
  shard.buckets[BucketIndex(value_us)].fetch_add(1, std::memory_order_relaxed);
  shard.sum.fetch_add(value_us, std::memory_order_relaxed);
}

ShardedLatencyHistogram::Snapshot
ShardedLatencyHistogram::Collect() const
{
  Snapshot snapshot;
  snapshot.boundaries = boundaries_;
  std::vector<uint64_t> raw(bucket_count_, 0);
  for (size_t shard = 0; shard < kShardCount; ++shard) {
    snapshot.sum += shards_[shard].sum.load(std::memory_order_relaxed);
    // Count is the sum of these same bucket loads. A separate counter can
    // move between the two reads and make a finite bucket exceed +Inf.
    for (size_t bucket = 0; bucket < bucket_count_; ++bucket) {
      raw[bucket] +=
          shards_[shard].buckets[bucket].load(std::memory_order_relaxed);
    }
  }
  for (uint64_t observations : raw) {
    snapshot.count += observations;
  }

  snapshot.cumulative.resize(boundaries_.size());
  uint64_t running = 0;
  for (size_t bucket = 0; bucket < boundaries_.size(); ++bucket) {
    running += raw[bucket];
    snapshot.cumulative[bucket] = running;
  }
  return snapshot;
}

void
ShardedLatencyHistogram::AppendSamples(std::string* out) const
{
  const Snapshot snapshot = Collect();
  std::string base_labels;
  for (const auto& label : labels_) {
    if (!base_labels.empty()) {
      base_labels += ",";
    }
    base_labels += label.first;
    base_labels += "=\"";
    base_labels += EscapeLabel(label.second);
    base_labels += "\"";
  }

  auto write_sample = [&](const std::string& suffix, const std::string& extra,
                          uint64_t value) {
    *out += name_;
    *out += suffix;
    if (base_labels.empty() && extra.empty()) {
      *out += " ";
    } else {
      *out += "{";
      *out += base_labels;
      if (!extra.empty()) {
        if (!base_labels.empty()) {
          *out += ",";
        }
        *out += extra;
      }
      *out += "} ";
    }
    *out += std::to_string(value);
    *out += "\n";
  };

  for (size_t i = 0; i < snapshot.boundaries.size(); ++i) {
    write_sample(
        "_bucket", "le=\"" + FormatLe(snapshot.boundaries[i]) + "\"",
        snapshot.cumulative[i]);
  }
  write_sample("_bucket", "le=\"+Inf\"", snapshot.count);
  write_sample("_sum", "", snapshot.sum);
  write_sample("_count", "", snapshot.count);
}

void
ShardedLatencyHistogram::AppendPrometheusText(std::string* out) const
{
  *out += "# HELP ";
  *out += name_;
  *out += " ";
  *out += help_;
  *out += "\n# TYPE ";
  *out += name_;
  *out += " histogram\n";
  AppendSamples(out);
}

LatencyHistogramRegistry&
LatencyHistogramRegistry::Instance()
{
  static LatencyHistogramRegistry registry;
  return registry;
}

void
LatencyHistogramRegistry::Register(
    std::shared_ptr<ShardedLatencyHistogram> histogram)
{
  std::lock_guard<std::mutex> lock(mu_);
  histograms_.push_back(std::move(histogram));
}

void
LatencyHistogramRegistry::Unregister(const ShardedLatencyHistogram* histogram)
{
  std::lock_guard<std::mutex> lock(mu_);
  histograms_.erase(
      std::remove_if(
          histograms_.begin(), histograms_.end(),
          [histogram](const std::shared_ptr<ShardedLatencyHistogram>& entry) {
            return entry.get() == histogram;
          }),
      histograms_.end());
}

std::string
LatencyHistogramRegistry::Serialize(
    const std::string& already_exposed, std::vector<std::string>* omitted) const
{
  std::vector<std::shared_ptr<ShardedLatencyHistogram>> histograms;
  {
    std::lock_guard<std::mutex> lock(mu_);
    histograms = histograms_;
  }
  std::map<std::string, std::vector<std::shared_ptr<ShardedLatencyHistogram>>>
      families;
  std::vector<std::string> family_order;
  for (const auto& histogram : histograms) {
    auto& family = families[histogram->Name()];
    if (family.empty()) {
      family_order.push_back(histogram->Name());
    }
    family.push_back(histogram);
  }

  std::string out;
  for (const auto& name : family_order) {
    if (ExpositionDeclaresFamily(already_exposed, name)) {
      if (omitted != nullptr) {
        omitted->push_back(name);
      }
      continue;
    }
    const auto& family = families[name];
    out += "# HELP ";
    out += name;
    out += " ";
    out += family.front()->Help();
    out += "\n# TYPE ";
    out += name;
    out += " histogram\n";
    for (const auto& histogram : family) {
      histogram->AppendSamples(&out);
    }
  }
  return out;
}

void
LatencyHistogramRegistry::Clear()
{
  std::lock_guard<std::mutex> lock(mu_);
  histograms_.clear();
}

}}  // namespace triton::core
