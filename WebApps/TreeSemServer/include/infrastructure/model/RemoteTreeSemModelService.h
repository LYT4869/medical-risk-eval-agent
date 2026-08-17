#pragma once

#include "client/IModelAdapterClient.h"
#include "model/IModelService.h"

namespace treesem
{
namespace infrastructure
{

class RemoteTreeSemModelService : public model::IModelService
{
public:
    explicit RemoteTreeSemModelService(const client::IModelAdapterClient& client);

    model::ModelResult predict(const model::ModelInput& input) const override;

private:
    const client::IModelAdapterClient& client_;
};

} // namespace infrastructure
} // namespace treesem
