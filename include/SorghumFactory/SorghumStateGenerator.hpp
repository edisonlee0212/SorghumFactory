#pragma once
#include "CurveDescriptor.hpp"
#include <sorghum_factory_export.h>
#include "ProceduralSorghum.hpp"
using namespace UniEngine;
namespace SorghumFactory {

class SORGHUM_FACTORY_API SorghumStateGenerator : public IAsset {
  unsigned m_version = 0;
public:
  //Pinnacle
  bool m_hasPinnacle;
  SingleDistribution<glm::vec3> m_pinnacleSize;
  SingleDistribution<float> m_pinnacleSeedAmount;
  SingleDistribution<float> m_pinnacleSeedRadius;
  //Stem
  glm::vec3 m_stemDirection;
  SingleDistribution<float> m_stemLength;
  SingleDistribution<float> m_stemWidth;
  //Leaf
  SingleDistribution<float> m_leafAmount;

  MixedDistribution<float> m_leafStartingPoint;
  MixedDistribution<float> m_leafCurling;
  MixedDistribution<float> m_leafRollAngle;
  MixedDistribution<float> m_leafBranchingAngle;
  MixedDistribution<float> m_leafBending;
  MixedDistribution<float> m_leafBendingAcceleration;
  MixedDistribution<float> m_leafWaviness;
  MixedDistribution<float> m_leafWavinessFrequency;
  MixedDistribution<float> m_leafPeriodStart;
  MixedDistribution<float> m_leafLength;
  MixedDistribution<float> m_leafWidth;

  //Finer control
  UniEngine::Curve m_widthAlongStem;
  UniEngine::Curve m_widthAlongLeaf;
  UniEngine::Curve m_wavinessAlongLeaf;
  [[nodiscard]] unsigned GetVersion() const;
  void OnCreate() override;
  void OnInspect() override;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;
  [[nodiscard]] ProceduralSorghumState Generate(unsigned int seed);
};
} // namespace SorghumFactory