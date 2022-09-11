#pragma once
#include "CurveDescriptor.hpp"
#include <sorghum_factory_export.h>
#include "ProceduralSorghum.hpp"
using namespace UniEngine;
namespace PlantArchitect {

class SORGHUM_FACTORY_API SorghumStateGenerator : public IAsset {
  unsigned m_version = 0;
public:
  //Panicle
  SingleDistribution<glm::vec2> m_panicleSize;
  SingleDistribution<float> m_panicleSeedAmount;
  SingleDistribution<float> m_panicleSeedRadius;
  //Stem
  SingleDistribution<float> m_stemTiltAngle;
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
  MixedDistribution<float> m_leafBendingSmoothness;
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
  [[nodiscard]] SorghumState Generate(unsigned int seed);
};
} // namespace PlantArchitect