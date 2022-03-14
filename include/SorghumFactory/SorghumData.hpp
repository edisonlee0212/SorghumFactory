#pragma once
#include <sorghum_factory_export.h>
#include "ProceduralSorghum.hpp"
#include <SorghumStateGenerator.hpp>
using namespace UniEngine;
namespace SorghumFactory {
enum class SorghumMode{
  ProceduralSorghum,
  SorghumStateGenerator
};

class SORGHUM_FACTORY_API SorghumData : public IPrivateComponent {
  float m_currentTime = 1.0f;
  unsigned m_recordedVersion = 0;
  friend class SorghumLayer;
public:
  int m_mode = (int)SorghumMode::ProceduralSorghum;
  glm::vec3 m_gravityDirection = glm::vec3(0, -1, 0);
  bool m_meshGenerated = false;
  AssetRef m_descriptor;
  int m_seed = 0;
  void OnCreate() override;
  void OnDestroy() override;
  void OnInspect() override;
  void SetTime(float time);
  void ExportModel(const std::string &filename,
                   const bool &includeFoliage = true) const;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;
  void CollectAssetRef(std::vector<AssetRef> &list) override;
  void GenerateGeometry();
  void ApplyGeometry(bool seperated = true, bool includeStem = true, bool segmentedMask = false);
};
} // namespace PlantFactory
