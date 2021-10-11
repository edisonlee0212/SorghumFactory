#pragma once

#include <sorghum_factory_export.h>
#ifdef RAYTRACERFACILITY
#include <CUDAModule.hpp>
#endif
using namespace UniEngine;
namespace SorghumFactory {
class SORGHUM_FACTORY_API TriangleIlluminationEstimator
    : public IPrivateComponent {
public:
  std::vector<Entity> m_entities;
  std::vector<glm::mat4> m_probeTransforms;
  std::vector<glm::vec4> m_probeColors;
  std::vector<float> m_triangleAreas;
#ifdef RAYTRACERFACILITY
  std::vector<RayTracerFacility::LightSensor<float>> m_lightProbes;
  void CalculateIllumination(
      const RayTracerFacility::IlluminationEstimationProperties &properties =
          RayTracerFacility::IlluminationEstimationProperties());
#endif
  float m_totalArea = 0.0f;
  float m_totalEnergy = 0.0f;
  float m_radiantFlux = 0.0f;
  void OnInspect() override;
};
} // namespace SorghumFactory
