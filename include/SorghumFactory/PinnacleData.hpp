#pragma once
#include "ProceduralSorghum.hpp"
#include <SorghumStateGenerator.hpp>
#include <sorghum_factory_export.h>

using namespace UniEngine;
namespace SorghumFactory {
class SORGHUM_FACTORY_API PinnacleData : public IPrivateComponent {
public:
  std::vector<Vertex> m_vertices;
  std::vector<glm::uvec3> m_triangles;
  void FormPinnacle(const SorghumStatePair & sorghumStatePair);
  void OnInspect() override;
  void OnDestroy() override;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;
};
} // namespace SorghumFactory