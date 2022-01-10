#pragma once
#include <sorghum_factory_export.h>
#include "ProceduralSorghumGrowthDescriptor.hpp"
#include <SorghumProceduralDescriptor.hpp>
using namespace UniEngine;
namespace SorghumFactory {
class SORGHUM_FACTORY_API SorghumData : public IPrivateComponent {
public:
  glm::vec3 m_gravityDirection = glm::vec3(0, -1, 0);
  bool m_meshGenerated = false;
  ProceduralSorghumState m_state;
  AssetRef m_parameters;
  int m_stemSubdivisionAmount = 16;
  int m_leafSubdivisionAmount = 16;
  void OnCreate() override;
  void OnDestroy() override;
  void OnInspect() override;
  void ExportModel(const std::string &filename,
                   const bool &includeFoliage = true) const;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;
  void PreparePinnacleMesh(const glm::vec3 &center, std::vector<Vertex>& vertices, std::vector<glm::uvec3>& triangles);
  void ApplyParameters(float time);
  void CollectAssetRef(std::vector<AssetRef> &list) override;
  void GenerateGeometry(bool includeStem = false);
  void ApplyGeometry(bool seperated = true, bool includeStem = false, bool segmentedMask = false);
};
} // namespace PlantFactory
