#include "observability/MetricsRegistry.h"

#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace http::observability
{
namespace
{

std::string escapeLabel(const std::string& value)
{
    std::string result;
    result.reserve(value.size());
    for (char ch : value)
    {
        if (ch == '\\' || ch == '"') result.push_back('\\');
        if (ch == '\n') result += "\\n";
        else result.push_back(ch);
    }
    return result;
}

std::string renderLabels(const Labels& labels,
                         const std::string& extraKey = {},
                         const std::string& extraValue = {})
{
    if (labels.empty() && extraKey.empty()) return {};
    std::ostringstream out;
    out << '{';
    bool first = true;
    for (const auto& [key, value] : labels)
    {
        if (!first) out << ',';
        out << key << "=\"" << escapeLabel(value) << '"';
        first = false;
    }
    if (!extraKey.empty())
    {
        if (!first) out << ',';
        out << extraKey << "=\"" << escapeLabel(extraValue) << '"';
    }
    out << '}';
    return out.str();
}

void validateName(const std::string& name)
{
    if (name.empty() || name.find_first_not_of(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_:") !=
            std::string::npos)
    {
        throw std::invalid_argument("invalid metric name");
    }
}

} // namespace

bool MetricsRegistry::MetricKey::operator<(const MetricKey& other) const noexcept
{
    if (name != other.name) return name < other.name;
    return labels < other.labels;
}

const std::vector<double>& MetricsRegistry::histogramBounds()
{
    static const std::vector<double> bounds{
        0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1,
        0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0};
    return bounds;
}

void MetricsRegistry::increment(const std::string& name, Labels labels,
                                std::uint64_t amount)
{
    validateName(name);
    std::lock_guard<std::mutex> lock(mutex_);
    counters_[{name, std::move(labels)}] += amount;
}

void MetricsRegistry::setGauge(const std::string& name, Labels labels, double value)
{
    validateName(name);
    if (!std::isfinite(value)) throw std::invalid_argument("gauge must be finite");
    std::lock_guard<std::mutex> lock(mutex_);
    gauges_[{name, std::move(labels)}] = value;
}

void MetricsRegistry::addGauge(const std::string& name, Labels labels, double delta)
{
    validateName(name);
    if (!std::isfinite(delta)) throw std::invalid_argument("gauge delta must be finite");
    std::lock_guard<std::mutex> lock(mutex_);
    gauges_[{name, std::move(labels)}] += delta;
}

void MetricsRegistry::observe(const std::string& name, Labels labels, double value)
{
    validateName(name);
    if (!std::isfinite(value) || value < 0.0)
        throw std::invalid_argument("histogram observation must be finite and nonnegative");
    std::lock_guard<std::mutex> lock(mutex_);
    Histogram& histogram = histograms_[{name, std::move(labels)}];
    if (histogram.buckets.empty())
        histogram.buckets.assign(histogramBounds().size(), 0);
    for (std::size_t index = 0; index < histogramBounds().size(); ++index)
        if (value <= histogramBounds()[index]) ++histogram.buckets[index];
    ++histogram.count;
    histogram.sum += value;
}

std::string MetricsRegistry::renderPrometheus() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream out;
    out << std::setprecision(17);
    for (const auto& [key, value] : counters_)
        out << key.name << renderLabels(key.labels) << ' ' << value << '\n';
    for (const auto& [key, value] : gauges_)
        out << key.name << renderLabels(key.labels) << ' ' << value << '\n';
    for (const auto& [key, histogram] : histograms_)
    {
        for (std::size_t index = 0; index < histogramBounds().size(); ++index)
            out << key.name << "_bucket"
                << renderLabels(key.labels, "le", std::to_string(histogramBounds()[index]))
                << ' ' << histogram.buckets[index] << '\n';
        out << key.name << "_bucket"
            << renderLabels(key.labels, "le", "+Inf") << ' '
            << histogram.count << '\n';
        out << key.name << "_sum" << renderLabels(key.labels) << ' '
            << histogram.sum << '\n';
        out << key.name << "_count" << renderLabels(key.labels) << ' '
            << histogram.count << '\n';
    }
    return out.str();
}

} // namespace http::observability
