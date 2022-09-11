#pragma once
#include <sorghum_factory_export.h>

using namespace UniEngine;
namespace PlantArchitect {
class SORGHUM_FACTORY_API IVolume : public IPrivateComponent {
public:
  bool m_asObstacle = false;
  virtual glm::vec3 GetRandomPoint() = 0;
  virtual bool InVolume(const GlobalTransform& globalTransform, const glm::vec3 &position) = 0;
  virtual bool InVolume(const glm::vec3 &position) = 0;
};

class SORGHUM_FACTORY_API SphericalVolume : public IVolume{
public:
  glm::vec3 m_radius = glm::vec3(1.0f);
  glm::vec3 GetRandomPoint() override;
  bool InVolume(const GlobalTransform &globalTransform,
                const glm::vec3 &position) override;
  bool InVolume(const glm::vec3 &position) override;
};

} // namespace PlantFactory
