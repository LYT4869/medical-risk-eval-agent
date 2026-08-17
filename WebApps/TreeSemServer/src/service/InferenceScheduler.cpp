#include "service/InferenceScheduler.h"

namespace treesem
{
namespace service
{
InferenceScheduler::InferenceScheduler(std::size_t workerCount,
                                       std::size_t queueCapacity)
    : BlockingTaskScheduler(
          workerCount, queueCapacity, predictionSchedulerErrors())
{}

} // namespace service
} // namespace treesem
