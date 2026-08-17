#pragma once

#include <string>

#include "model/ModelTypes.h"

namespace treesem
{
namespace api
{

class PredictionJsonCodec
{
public:
    static model::ModelInput parseInput(const std::string& requestBody);
    static std::string serializeResult(const model::ModelResult& result);
};

} // namespace api
} // namespace treesem
