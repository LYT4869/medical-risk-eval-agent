#pragma once

#include <string>

#include "model/ModelTypes.h"

namespace treesem
{
namespace serialization
{

class ModelResultJsonCodec
{
public:
    static std::string encode(const model::ModelResult& result);
    static model::ModelResult decode(const std::string& value);
};

} // namespace serialization
} // namespace treesem
