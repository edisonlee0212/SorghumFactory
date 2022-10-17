#pragma once
#include <sorghum_factory_export.h>

using namespace UniEngine;
namespace PlantArchitect {

class CBTFGroup : public IAsset{
public:
  std::vector<AssetRef> m_cBTFs;
  void OnInspect() override;
  void CollectAssetRef(std::vector<AssetRef> &list) override;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;
  AssetRef GetRandom() const;
};
}