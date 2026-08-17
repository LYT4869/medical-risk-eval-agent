#include <filesystem>
#include <iostream>

#include "api/PredictionJsonCodec.h"
#include "infrastructure/model/OnnxTreeSemModelService.h"

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        return 2;
    }
    treesem::infrastructure::OnnxTreeSemModelService service{
        std::filesystem::path(argv[1])};
    std::cout << '[';
    for (std::size_t index = 0; index < service.bundle().referenceRows(); ++index)
    {
        if (index != 0)
        {
            std::cout << ',';
        }
        std::cout << treesem::api::PredictionJsonCodec::serializeResult(
            service.predict(treesem::model::SampleIndexInput{
                static_cast<std::int64_t>(index)}));
    }
    std::cout << "]\n";
}
