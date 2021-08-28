#pragma once
#include <sorghum_factory_export.h>

using namespace UniEngine;
namespace SorghumFactory {

class SORGHUM_FACTORY_API SorghumParameters : public IAsset {
public:
  SorghumParameters();
  void OnGui();

  void Serialize(YAML::Emitter &out);
  void Deserialize(const YAML::Node &in);

  int m_leafCount = 8;
  float m_stemLength = 4.0f;
  float m_firstLeafStartingPoint = 0.2f;
  float m_leafLengthBase = 5.0f;
  float m_branchingAngle = 30.0f;
  float m_branchingAngleVariance = 1.0f;
  BezierCubic2D m_leafLength;
  float m_gravitropism = 2;
  float m_gravitropismFactor = 0.5;
};
} // namespace PlantFactory