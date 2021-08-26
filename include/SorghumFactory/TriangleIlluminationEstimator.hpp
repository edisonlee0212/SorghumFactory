#pragma once
#include <sorghum_factory_export.h>

#include <CUDAModule.hpp>
using namespace UniEngine;
namespace PlantFactory {
class SORGHUM_FACTORY_API TriangleIlluminationEstimator : public IPrivateComponent {
public:
  std::vector<Entity> m_entities;
  std::vector<glm::mat4> m_probeTransforms;
  std::vector<glm::vec4> m_probeColors;
  std::vector<float> m_triangleAreas;
  std::vector<RayTracerFacility::LightSensor<float>> m_lightProbes;
  float m_totalArea = 0.0f;
  float m_totalEnergy = 0.0f;
  float m_radiantFlux = 0.0f;
  void OnGui() override;
  void CalculateIllumination(
      const RayTracerFacility::IlluminationEstimationProperties &properties =
          RayTracerFacility::IlluminationEstimationProperties());

  void Clone(const std::shared_ptr<IPrivateComponent> &target) override;
};
} // namespace PlantFactory
