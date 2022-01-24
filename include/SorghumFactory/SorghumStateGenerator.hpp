#pragma once
#include "CurveDescriptor.hpp"
#include <sorghum_factory_export.h>
#include "ProceduralSorghum.hpp"
using namespace UniEngine;
namespace SorghumFactory {

class SORGHUM_FACTORY_API SorghumStateGenerator : public IAsset {
public:
  bool m_hasPinnacle;
  SingleDistribution<glm::vec3> m_pinnacleSize;
  SingleDistribution<int> m_pinnacleSeedAmount;
  SingleDistribution<float> m_pinnacleSeedRadius;

  glm::vec3 m_stemDirection;
  SingleDistribution<float> m_stemLength;
  SingleDistribution<float> m_stemWidth;

  int m_leafAmount;
  SingleDistribution<float> m_firstLeafStartingPoint;
  SingleDistribution<float> m_lastLeafEndingPoint;

  MixedDistribution<float> m_leafRollAngle;
  MixedDistribution<float> m_leafBranchingAngle;
  MixedDistribution<float> m_leafBending;
  MixedDistribution<float> m_leafBendingAcceleration;
  MixedDistribution<float> m_leafWaviness;
  MixedDistribution<float> m_leafWavinessFrequency;
  MixedDistribution<float> m_leafLength;
  MixedDistribution<float> m_leafWidth;


  SorghumStateGenerator();
  void OnInspect() override;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;
  [[nodiscard]] ProceduralSorghumState Generate(unsigned int seed);
};
} // namespace SorghumFactory