#pragma once
#include <sorghum_factory_export.h>

using namespace UniEngine;
namespace SorghumFactory {
struct SORGHUM_FACTORY_API SorghumStemDescriptor{
  glm::vec3 m_direction = glm::vec3(0, 1, 0);
  float m_length = 8;
  float m_stemWidth = 0.1f;
  bool OnInspect();
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

  float m_stemWidth = 0.1f;
  float m_leafMaxWidth = 0.2f;
  float m_leafWidthDecreaseStart = 0.5;
  bool OnInspect();
  void Serialize(YAML::Emitter &out);
  void Deserialize(const YAML::Node &in);
};

class SORGHUM_FACTORY_API SorghumProceduralDescriptor : public IAsset {
  void L1ToBase();
public:
  SorghumProceduralDescriptor();
  void OnInspect() override;
  void Ready();
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;

  int m_cascadeIndex = 1;
#pragma region L1 (No variance, better control of single leaf)
  float     m_l1StemWidthMax = 0.1f;
  UniEngine::Curve m_l1StemWidthDistribution;
  float     m_l1LeafWidthMax = 0.2f;
  UniEngine::Curve  m_l1LeafWidthDistribution;

  UniEngine::Curve  m_l1LeafLengthDecreaseStartingPointDistribution;

  int       m_l1LeafCount = 8;
  float     m_l1StemLength = 3.5f;
  float     m_l1FirstLeafStartingPoint = 0.3f;

  float     m_l1LeafLengthMax = 10.0f;
  UniEngine::Curve  m_l1LeafLengthDistribution;

  float     m_l1LeafLengthVarianceMax = 0.0f;
  UniEngine::Curve  m_l1LeafLengthVarianceDistribution;

  float     m_l1BranchingAngleMax = 50;
  UniEngine::Curve  m_l1BranchingAngleDistribution;

  float     m_l1BranchingAngleVarianceMax = 0.0f;
  UniEngine::Curve  m_l1BranchingAngleVarianceDistribution;

  float     m_l1RollAngleVarianceMax = 10.0f;
  UniEngine::Curve  m_l1RollAngleVarianceDistribution;

  float     m_l1GravitropismMax = 4;
  UniEngine::Curve  m_l1GravitropismDistribution;

  float     m_l1GravitropismVarianceMax = 0.0f;
  UniEngine::Curve  m_l1GravitropismVarianceDistribution;

  float     m_l1GravitropismFactorMax = 1.0f;
  UniEngine::Curve  m_l1GravitropismFactorDistribution;

  float     m_l1GravitropismFactorVarianceMax = 0.0f;
  UniEngine::Curve  m_l1GravitropismFactorVarianceDistribution;
#pragma endregion

#pragma region Base
  SorghumStemDescriptor m_stemDescriptor;
  std::vector<SorghumLeafDescriptor> m_leafDescriptors;
#pragma endregion

};
} // namespace PlantFactory