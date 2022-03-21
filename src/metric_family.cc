#include "metric_family.h"
#include <limits>

namespace triton { namespace core {

//
// Implementation for TRITONSERVER_MetricFamily.
//
MetricFamily::MetricFamily(
    TRITONSERVER_MetricKind kind, const char* name, const char* description,
    std::shared_ptr<prometheus::Registry> registry)
{
  // TODO: Check correctness of void* cast and lifetimes here
  switch (kind) {
    case TRITONSERVER_METRIC_KIND_COUNTER:
      family_ = reinterpret_cast<void*>(&prometheus::BuildCounter()
                                             .Name(name)
                                             .Help(description)
                                             .Register(*registry));
      break;
    case TRITONSERVER_METRIC_KIND_GAUGE:
      family_ = reinterpret_cast<void*>(&prometheus::BuildGauge()
                                             .Name(name)
                                             .Help(description)
                                             .Register(*registry));
      break;
    default:
      // TODO: Error unsupported kind
      family_ = nullptr;
      break;
  }

  kind_ = kind;
}

MetricFamily::~MetricFamily()
{
  // NOTE: registry->Remove() not added until until prometheus-cpp v1.0 which
  // we do not currently install

  /*
  // Remove family from registry
  switch (kind_) {
    case TRITONSERVER_METRIC_KIND_COUNTER:
      registry_->Remove(*reinterpret_cast<prometheus::Family<prometheus::Counter>*>(family_));
      break;
    case TRITONSERVER_METRIC_KIND_GAUGE:
      registry_->Remove(*reinterpret_cast<prometheus::Family<prometheus::Gauge>*>(family_));
      break;
    default:
      // TODO: Error unsupported kind
      break;
  }*/
}

// TODO: Figure out or remove
/*prometheus::MetricType
MetricFamily::PrometheusType()
{
  switch (kind_) {
    case TRITONSERVER_METRIC_KIND_COUNTER:
      return prometheus::MetricType::Counter;
    case TRITONSERVER_METRIC_KIND_GAUGE:
      return prometheus::MetricType::Gauge;
  }

  // Unsupported type
  return prometheus::MetricType::Untyped;
}

// TODO: Slicker way to do this with generics/templates or passing type?
template<typename T>
prometheus::Family<T>
MetricFamily::PrometheusFamily()
{
  return reinterpret_cast<prometheus::Family<T>*>(family_);
}

prometheus::Family<prometheus::MetricType>
MetricFamily::PrometheusFamily(prometheus::MetricType T)
{
  return reinterpret_cast<prometheus::Family<T>*>(family_);
}

prometheus::Family<prometheus::MetricType>*
MetricFamily::PrometheusFamily() {
  switch (kind_) {
    case TRITONSERVER_METRIC_KIND_COUNTER:
      return
reinterpret_cast<prometheus::Family<prometheus::Counter>*>(family_); case
TRITONSERVER_METRIC_KIND_GAUGE: return
reinterpret_cast<prometheus::Family<prometheus::Gauge>*>(family_);
  }

  // Unsupported type
  return nullptr;
}
*/

//
// Implementation for TRITONSERVER_Metric.
//
Metric::Metric(
    TRITONSERVER_MetricFamily* family, TRITONSERVER_Parameter** labels,
    int num_labels)
{
  family_ = reinterpret_cast<MetricFamily*>(family);
  kind_ = family_->Kind();

  // TODO: Cleanup family_ ptr names
  switch (kind_) {
    case TRITONSERVER_METRIC_KIND_COUNTER: {
      auto counter_family_ptr =
          reinterpret_cast<prometheus::Family<prometheus::Counter>*>(
              family_->Family());
      // TODO: Use labels
      auto counter_ptr = &counter_family_ptr->Add({});
      metric_ = reinterpret_cast<void*>(counter_ptr);
      break;
    }
    case TRITONSERVER_METRIC_KIND_GAUGE: {
      auto gauge_family_ptr =
          reinterpret_cast<prometheus::Family<prometheus::Gauge>*>(
              family_->Family());
      // TODO: Use labels
      auto gauge_ptr = &gauge_family_ptr->Add({});
      metric_ = reinterpret_cast<void*>(gauge_ptr);
      break;
    }
    default:
      // TODO: LOG ERROR
      break;
  }

  // TODO: Figure out or remove
  // const auto prometheus_type = family_->PrometheusType();
  // auto prometheus_family = family_->PrometheusFamily(prometheus_type);
}

Metric::~Metric()
{
  // TODO: Cleanup family_ ptr names
  switch (kind_) {
    case TRITONSERVER_METRIC_KIND_COUNTER: {
      auto counter_family_ptr =
          reinterpret_cast<prometheus::Family<prometheus::Counter>*>(
              family_->Family());
      auto counter_ptr = reinterpret_cast<prometheus::Counter*>(metric_);
      counter_family_ptr->Remove(counter_ptr);
      break;
    }
    case TRITONSERVER_METRIC_KIND_GAUGE: {
      auto gauge_family_ptr =
          reinterpret_cast<prometheus::Family<prometheus::Gauge>*>(
              family_->Family());
      auto gauge_ptr = reinterpret_cast<prometheus::Gauge*>(metric_);
      gauge_family_ptr->Remove(gauge_ptr);
      break;
    }
    default:
      // TODO: LOG ERROR
      break;
  }
}

TRITONSERVER_Error*
Metric::Value(double* value)
{
  switch (kind_) {
    case TRITONSERVER_METRIC_KIND_COUNTER: {
      auto counter_ptr = reinterpret_cast<prometheus::Counter*>(metric_);
      *value = counter_ptr->Value();
      return nullptr;  // Success
    }
    case TRITONSERVER_METRIC_KIND_GAUGE: {
      auto gauge_ptr = reinterpret_cast<prometheus::Gauge*>(metric_);
      *value = gauge_ptr->Value();
      return nullptr;  // Success
    }
    default:
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "Unsupported TRITONSERVER_MetricKind");
  }
}

TRITONSERVER_Error*
Metric::Increment(double value)
{
  switch (kind_) {
    case TRITONSERVER_METRIC_KIND_COUNTER: {
      auto counter_ptr = reinterpret_cast<prometheus::Counter*>(metric_);
      counter_ptr->Increment(value);
      return nullptr;  // Success
    }
    case TRITONSERVER_METRIC_KIND_GAUGE: {
      auto gauge_ptr = reinterpret_cast<prometheus::Gauge*>(metric_);
      gauge_ptr->Increment(value);
      return nullptr;  // Success
    }
    default:
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "Unsupported TRITONSERVER_MetricKind");
  }
}

TRITONSERVER_Error*
Metric::Decrement(double value)
{
  switch (kind_) {
    case TRITONSERVER_METRIC_KIND_COUNTER: {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "TRITONSERVER_METRIC_KIND_COUNTER does not support Decrement");
    }
    case TRITONSERVER_METRIC_KIND_GAUGE: {
      auto gauge_ptr = reinterpret_cast<prometheus::Gauge*>(metric_);
      gauge_ptr->Decrement(value);
      return nullptr;  // Success
    }
    default:
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "Unsupported TRITONSERVER_MetricKind");
  }
}

TRITONSERVER_Error*
Metric::Set(double value)
{
  switch (kind_) {
    case TRITONSERVER_METRIC_KIND_COUNTER: {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "TRITONSERVER_METRIC_KIND_COUNTER does not support Set");
    }
    case TRITONSERVER_METRIC_KIND_GAUGE: {
      auto gauge_ptr = reinterpret_cast<prometheus::Gauge*>(metric_);
      gauge_ptr->Set(value);
      return nullptr;  // Success
    }
    default:
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "Unsupported TRITONSERVER_MetricKind");
  }
}

}}  // namespace triton::core
