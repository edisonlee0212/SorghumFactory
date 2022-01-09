#pragma once
#include "SorghumProceduralDescriptor.hpp"
#include <sorghum_factory_export.h>

using namespace UniEngine;
namespace SorghumFactory {
struct ProceduralPinnacleGrowthState {
  bool m_active = false;
  glm::vec3 m_pinnacleSize = glm::vec3(0, 0, 0);
  int m_seedAmount = 1200;
  float m_seedRadius = 0.02;
  void OnInspect() const;
};
struct ProceduralStemGrowthState {
  glm::vec3 m_direction = {0, 0, 0};
  float m_length = 0;
};
struct ProceduralLeafGrowthState {
  bool m_active = false;
  float m_startingPoint = 0;
  float m_length = 0;
  float m_width = 0;
  float m_rollAngle = 0;
  float m_branchingAngle = 0;
  glm::vec2 m_bending = {0, 0};
  void OnInspect() const;
};
struct ProceduralSorghumState {
  float m_time = 0;

  ProceduralPinnacleGrowthState m_pinnacle;
  ProceduralStemGrowthState m_stem;
  std::vector<ProceduralLeafGrowthState> m_leaves;
  void OnInspect() const;
};

struct ProceduralPinnacleGrowthDescriptor {
  float m_startTime = 0;
  float m_endTime = 0;
  glm::vec3 m_maxPinnacleSize = glm::vec3(0, 0, 0);
  UniEngine::Curve m_pinnacleSizeCurve =
      UniEngine::Curve(0.0f, 1.0f, {0, 0}, {1, 1});
  int m_maxSeedAmount = 0;
  UniEngine::Curve m_seedAmountCurve =
      UniEngine::Curve(0.0f, 1.0f, {0, 0}, {1, 1});
  float m_maxSeedRadius = 0;
  UniEngine::Curve m_seedRadiusCurve =
      UniEngine::Curve(0.0f, 1.0f, {0, 0}, {1, 1});
  [[nodiscard]] ProceduralPinnacleGrowthState Get(float time,
                                                  unsigned seed) const;
  bool OnInspect();
  void Serialize(YAML::Emitter &out);
  void Deserialize(const YAML::Node &in);
};
struct ProceduralStemGrowthDescriptor {
  float m_endTime = 0;

  glm::vec3 m_direction = glm::vec3(0, 1, 0);
  float m_maxLength = 0;
  UniEngine::Curve m_lengthCurve = UniEngine::Curve(0.0f, 1.0f, {0, 0}, {1, 1});
  [[nodiscard]] ProceduralStemGrowthState Get(float time, unsigned seed) const;
  bool OnInspect();
  void Serialize(YAML::Emitter &out);
  void Deserialize(const YAML::Node &in);
};
struct ProceduralLeafGrowthDescriptor {
  float m_startTime;
  float m_endTime;

  glm::vec2 m_startingPoint;
  UniEngine::Curve m_startingPointCurve =
      UniEngine::Curve(0.0f, 1.0f, {0, 0}, {1, 1});

  float m_maxLength;
  UniEngine::Curve m_lengthCurve = UniEngine::Curve(0.0f, 1.0f, {0, 0}, {1, 1});
  float m_maxWidth;
  UniEngine::Curve m_widthCurve = UniEngine::Curve(0.0f, 1.0f, {0, 0}, {1, 1});

  float m_rollAngle;
  float m_branchingAngle;
  glm::vec2 m_bending;

  [[nodiscard]] ProceduralLeafGrowthState Get(float time, unsigned seed) const;
  bool OnInspect();
  void Serialize(YAML::Emitter &out);
  void Deserialize(const YAML::Node &in);
};

class ProceduralSorghumGrowthDescriptor : public IAsset {
public:
  unsigned m_seed = 0;
  ProceduralPinnacleGrowthDescriptor m_pinnacle;
  ProceduralStemGrowthDescriptor m_stem;
  std::vector<ProceduralLeafGrowthDescriptor> m_leaves;
  void Set(const ProceduralSorghumState &endState);
  void Set(const std::shared_ptr<SorghumProceduralDescriptor> &endState);
  [[nodiscard]] ProceduralSorghumState Get(float time) const;
  void OnInspect();

  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;
};
} // namespace SorghumFactory