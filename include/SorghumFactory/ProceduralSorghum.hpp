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
  int m_index;
  float m_distanceToRoot = 0;
  float m_length = 0;
  CurveDescriptor<float> m_widthAlongLeaf;
  float m_rollAngle = 0;
  float m_branchingAngle = 0;
  float m_curling;
  glm::vec2 m_bending = {0, 0};
  CurveDescriptor<float> m_wavinessAlongLeaf;
  glm::vec2 m_wavinessPeriodStart = glm::vec2(0.0f);
  glm::vec2 m_wavinessFrequency = glm::vec2(0.0f);

  void Serialize(YAML::Emitter &out);
  void Deserialize(const YAML::Node &in);
  bool OnInspect();
};
#pragma endregion

class SorghumState{
  friend class ProceduralSorghum;
  unsigned m_version = 0;
  bool OnMenu();
public:
  ProceduralPinnacleState m_pinnacle;
  ProceduralStemState m_stem;
  std::vector<ProceduralLeafState> m_leaves;
  void OnInspect();

  void Serialize(YAML::Emitter &out);
  void Deserialize(const YAML::Node &in);
};
class SorghumStateGenerator;

struct SorghumStatePair{
  SorghumState m_left = SorghumState();
  SorghumState m_right = SorghumState();
  float m_a = 1.0f;
  [[nodiscard]] int SizeOfLeaf();
};

class ProceduralSorghum : public IAsset {
  unsigned m_version = 0;
  friend class SorghumData;
  std::vector<std::pair<float, SorghumState>> m_sorghumStates;
public:
  [[nodiscard]] unsigned GetVersion() const;
  [[nodiscard]] float GetCurrentStartTime() const;
  [[nodiscard]] float GetCurrentEndTime() const;
  void Add(float time, const SorghumState& state);
  void ResetTime(float previousTime, float newTime);
  void Remove(float time);
  [[nodiscard]] SorghumStatePair Get(float time) const;

  void OnInspect() override;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;
};

} // namespace SorghumFactory