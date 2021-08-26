#pragma once
#include <sorghum_factory_export.h>

using namespace UniEngine;
namespace SorghumFactory {
class SORGHUM_FACTORY_API Volume : public IPrivateComponent {
public:
  bool m_asObstacle = false;
  virtual glm::vec3 GetRandomPoint() = 0;
  virtual bool InVolume(const GlobalTransform& globalTransform, const glm::vec3 &position) = 0;
  virtual bool InVolume(const glm::vec3 &position) = 0;
};
} // namespace PlantFactory
