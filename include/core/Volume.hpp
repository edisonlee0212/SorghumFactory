#pragma once
using namespace UniEngine;
namespace PlantFactory {
class Volume : public IPrivateComponent {
public:
  bool m_asObstacle = false;
  virtual glm::vec3 GetRandomPoint() = 0;
  virtual bool InVolume(const glm::vec3 &position) = 0;
};
} // namespace PlantFactory
