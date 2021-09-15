#pragma once
#include <sorghum_factory_export.h>

using namespace UniEngine;
namespace SorghumFactory {
struct SORGHUM_FACTORY_API SorghumStemDescriptor{
  glm::vec3 m_direction = glm::vec3(0, 1, 0);
  float m_length = 8;
  void OnInspect();
  void Serialize(YAML::Emitter &out);
  void Deserialize(const YAML::Node &in);
};

struct SORGHUM_FACTORY_API SorghumLeafDescriptor {
  int m_leafIndex;
  float m_leafStartingPoint;
  float m_leafLength;
  float m_branchingAngle;
  float m_rollAngle;
  float m_gravitropism;
  float m_gravitropismFactor;
  void OnInspect();
  void Serialize(YAML::Emitter &out);
  void Deserialize(const YAML::Node &in);
};

class SORGHUM_FACTORY_API SorghumProceduralDescriptor : public IAsset {
public:
  SorghumProceduralDescriptor();
  void OnInspect() override;
  void L1ToBase();
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;


#pragma region L1 (No variance, better control of single leaf)
  bool      m_l1Locked = true;
  int       m_l1LeafCount = 8;
  float     m_l1StemLength = 4.0f;
  float     m_l1FirstLeafStartingPoint = 0.2f;

  float     m_l1LeafLengthMax = 5.0f;
  Bezier2D  m_l1LeafLengthDistribution;

  float     m_l1LeafLengthVarianceMax = 0.0f;
  Bezier2D  m_l1LeafLengthVarianceDistribution;

  float     m_l1BranchingAngleMax = 30;
  Bezier2D  m_l1BranchingAngleDistribution;

  float     m_l1BranchingAngleVarianceMax = 0.0f;
  Bezier2D  m_l1BranchingAngleVarianceDistribution;

  float     m_l1RollAngleVarianceMax = 0.0f;
  Bezier2D  m_l1RollAngleVarianceDistribution;

  float     m_l1GravitropismMax = 2;
  Bezier2D  m_l1GravitropismDistribution;

  float     m_l1GravitropismVarianceMax = 0.0f;
  Bezier2D  m_l1GravitropismVarianceDistribution;

  float     m_l1GravitropismFactorMax = 0.5;
  Bezier2D  m_l1GravitropismFactorDistribution;

  float     m_l1GravitropismFactorVarianceMax = 0.0f;
  Bezier2D  m_l1GravitropismFactorVarianceDistribution;
#pragma endregion

#pragma region Base
  bool m_baseLocked = true;
  SorghumStemDescriptor m_stemDescriptor;
  std::vector<SorghumLeafDescriptor> m_leafDescriptors;
#pragma endregion

};
} // namespace PlantFactory