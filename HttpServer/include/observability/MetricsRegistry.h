#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace http::observability
{

using Labels = std::map<std::string, std::string>;

class MetricsRegistry
{
public:
    void increment(const std::string& name, Labels labels = {},
                   std::uint64_t amount = 1);
    void setGauge(const std::string& name, Labels labels, double value);
    void addGauge(const std::string& name, Labels labels, double delta);
    void observe(const std::string& name, Labels labels, double value);
    std::string renderPrometheus() const;

private:
    struct MetricKey
    {
        std::string name;
        Labels labels;
        bool operator<(const MetricKey& other) const noexcept;
    };
    struct Histogram
    {
        std::vector<std::uint64_t> buckets;
        std::uint64_t count{0};
        double sum{0.0};
    };

    static const std::vector<double>& histogramBounds();
    mutable std::mutex mutex_;
    std::map<MetricKey, std::uint64_t> counters_;
    std::map<MetricKey, double> gauges_;
    std::map<MetricKey, Histogram> histograms_;
};

} // namespace http::observability
