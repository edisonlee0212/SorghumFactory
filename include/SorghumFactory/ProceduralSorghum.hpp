#pragma once
#include <sorghum_factory_export.h>
#include "CurveDescriptor.hpp"
using namespace UniEngine;
namespace SorghumFactory {
#pragma region States
struct ProceduralPinnacleState {
  bool m_active = false;
  glm::vec3 m_pinnacleSize = glm::vec3(0, 0, 0);
  int m_seedAmount = 1200;
  float m_seedRadius = 0.02;
  bool OnInspect();
  void Serialize(YAML::Emitter &out);
  void Deserialize(const YAML::Node &in);
};
struct ProceduralStemState {
  glm::vec3 m_direction = {0, 0, 0};
  CurveDescriptor<float> m_widthAlongStem;
  float m_length = 0;
  [[nodiscard]] glm::vec3 GetPoint(float point) const;
  void Serialize(YAML::Emitter &out);
  void Deserialize(const YAML::Node &in);
  bool OnInspect();
};
struct ProceduralLeafState {
  bool m_active = false;
  int m_index;

  float m_distanceToRoot = 0;
  float m_length = 0;
  CurveDescriptor<float> m_widthAlongLeaf;
  float m_rollAngle = 0;
  float m_branchingAngle = 0;
  glm::vec2 m_bending = {0, 0};
  CurveDescriptor<float> m_wavinessAlongLeaf;
  float m_wavinessFrequency = 0;

  void Serialize(YAML::Emitter &out);
  void Deserialize(const YAML::Node &in);
  bool OnInspect();
};
#pragma endregion
#pragma region Descriptors
struct ProceduralPinnacleDescriptor {
  float m_startTime = 0;
  float m_endTime = 0;
  CurveDescriptor<glm::vec3> m_pinnacleSize;
  CurveDescriptor<int> m_seedAmount;
  CurveDescriptor<float> m_seedRadius;
  [[nodiscard]] ProceduralPinnacleState Get(float time) const;
  bool OnInspect(float maxTime);
  void Serialize(YAML::Emitter &out);
  void Deserialize(const YAML::Node &in);
};
struct ProceduralStemDescriptor {
  float m_endTime = 0;
  glm::vec3 m_direction = glm::vec3(0, 1, 0);
  CurveDescriptor<float> m_length;
  CurveDescriptor<float> m_width;
  UniEngine::Curve m_widthAlongStem;
  [[nodiscard]] ProceduralStemState Get(float time) const;
  bool OnInspect(float maxTime);
  void Serialize(YAML::Emitter &out);
  void Deserialize(const YAML::Node &in);
};
struct ProceduralLeafDescriptor {
  float m_startTime;
  float m_endTime;
  float m_rollAngle;
  float m_branchingAngle;
  glm::vec2 m_bending;
  float m_wavinessFrequency = 0;

  CurveDescriptor<float> m_distanceToRoot;
  CurveDescriptor<float> m_length;
  CurveDescriptor<float> m_width;
  CurveDescriptor<float> m_waviness;
  UniEngine::Curve m_widthAlongLeaf;
  UniEngine::Curve m_wavinessAlongLeaf;

  [[nodiscard]] ProceduralLeafState Get(float time) const;
  bool OnInspect(float maxTime);
  void Serialize(YAML::Emitter &out);
  void Deserialize(const YAML::Node &in);
};
#pragma endregion
struct ProceduralSorghumState {
  float m_time = 0;

  ProceduralPinnacleState m_pinnacle;
  ProceduralStemState m_stem;
  std::vector<ProceduralLeafState> m_leaves;
  bool OnInspect();

  void Serialize(YAML::Emitter &out);
  void Deserialize(const YAML::Node &in);
};
class SorghumStateGenerator;


class ProceduralSorghum : public IAsset {
  unsigned m_version = 0;
public:
  int m_seed = 0;
  ProceduralSorghumState m_endState;
  ProceduralPinnacleDescriptor m_pinnacle;
  ProceduralStemDescriptor m_stem;
  std::vector<ProceduralLeafDescriptor> m_leaves;
  float m_endTime = 0;
  void Set(float endTime, float longestLeafTime);
  [[nodiscard]] ProceduralSorghumState Get(float time) const;
  void OnInspect();
  [[nodiscard]] unsigned GetVersion() const;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;
};

} // namespace SorghumFactory