#pragma once
#include <Volume.hpp>
using namespace UniEngine;
namespace PlantFactory {
class CubeVolume : public Volume {
public:
  void ApplyMeshRendererBounds();
  void OnCreate() override;
  bool m_displayPoints = false;
  bool m_displayBounds = false;
  Bound m_minMaxBound;
  void OnGui() override;
  bool InVolume(const glm::vec3 &position) override;
  glm::vec3 GetRandomPoint() override;
};
} // namespace PlantFactory