#pragma once

#include "service/BlockingTaskScheduler.h"

namespace treesem
{
namespace service
{

class InferenceScheduler : public BlockingTaskScheduler
{
public:
    InferenceScheduler(std::size_t workerCount, std::size_t queueCapacity);
};

} // namespace service
} // namespace treesem
