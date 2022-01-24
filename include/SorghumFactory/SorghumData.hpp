#pragma once
#include <sorghum_factory_export.h>
#include "ProceduralSorghum.hpp"
#include <SorghumStateGenerator.hpp>
using namespace UniEngine;
namespace SorghumFactory {
class SORGHUM_FACTORY_API SorghumData : public IPrivateComponent {
  float m_currentTime = 1.0f;
  unsigned m_recordedVersion = 0;
  friend class SorghumLayer;
public:
  glm::vec3 m_gravityDirection = glm::vec3(0, -1, 0);
  bool m_meshGenerated = false;
  ProceduralSorghumState m_state;
  AssetRef m_proceduralSorghum;
  int m_stemSubdivisionAmount = 16;
  int m_leafSubdivisionAmount = 16;
  void OnCreate() override;
  void OnDestroy() override;
  void OnInspect() override;
  void SetTime(float time);
  void ExportModel(const std::string &filename,
                   const bool &includeFoliage = true) const;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;
  void PreparePinnacleMesh(const glm::vec3 &center, std::vector<Vertex>& vertices, std::vector<glm::uvec3>& triangles);
  void Apply();
  void CollectAssetRef(std::vector<AssetRef> &list) override;
  void GenerateGeometry();
  void ApplyGeometry(bool seperated = true, bool includeStem = true, bool segmentedMask = false);
  void LateUpdate() override;
};
} // namespace PlantFactory
