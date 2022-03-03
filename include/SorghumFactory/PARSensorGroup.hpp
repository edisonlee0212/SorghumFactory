#pragma once
#ifdef RAYTRACERFACILITY
#include <sorghum_factory_export.h>

#include <CUDAModule.hpp>
using namespace UniEngine;
using namespace RayTracerFacility;
namespace SorghumFactory {
class SORGHUM_FACTORY_API PARSensorGroup : public IAsset {
public:
  std::vector<IlluminationSampler<float>> m_samplers;
  void CalculateIllumination(const RayProperties& rayProperties, int seed, float pushNormalDistance);
  void OnInspect();
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;
};
} // namespace SorghumFactory
#endif