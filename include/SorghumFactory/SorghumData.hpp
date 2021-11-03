#pragma once
#include <sorghum_factory_export.h>

#include <SorghumProceduralDescriptor.hpp>
using namespace UniEngine;
namespace SorghumFactory {
class SORGHUM_FACTORY_API SorghumData : public IPrivateComponent {
public:
  glm::vec3 m_gravityDirection = glm::vec3(0, -1, 0);
  bool m_meshGenerated = false;
  int m_segmentAmount = 2;
  int m_step = 4;

  AssetRef m_parameters;
  void OnCreate() override;
  void OnDestroy() override;
  void OnInspect() override;
  void ExportModel(const std::string &filename,
                   const bool &includeFoliage = true) const;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;

  void ApplyParameters();
  void CollectAssetRef(std::vector<AssetRef> &list) override;
  void GenerateGeometry(bool segmentedMask);
};
} // namespace PlantFactory
