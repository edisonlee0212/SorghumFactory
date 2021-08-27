#pragma once
#include <sorghum_factory_export.h>

using namespace UniEngine;
namespace SorghumFactory {

class SORGHUM_FACTORY_API SorghumParameters : public IAsset {
public:
  void OnGui();

  void Serialize(YAML::Emitter &out);
  void Deserialize(const YAML::Node &in);

  int m_leafCount = 8;
};
} // namespace PlantFactory