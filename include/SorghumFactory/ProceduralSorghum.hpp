#pragma once
#include "CurveDescriptor.hpp"
#include "ICurve.hpp"
#include <sorghum_factory_export.h>
using namespace UniEngine;
namespace SorghumFactory {
#pragma region States
enum class StateMode { Default, CubicBezier };

struct ProceduralPanicleState {
  glm::vec3 m_panicleSize = glm::vec3(0, 0, 0);
  int m_seedAmount = 0;
  float m_seedRadius = 0.002f;
  bool OnInspect();
  void Serialize(YAML::Emitter &out);
  void Deserialize(const YAML::Node &in);
};
struct ProceduralStemState {
  BezierSpline m_spline;
  glm::vec3 m_direction = {0, 1, 0};
  CurveDescriptor<float> m_widthAlongStem;
  float m_length = 0;
  ProceduralStemState();
  [[nodiscard]] glm::vec3 GetPoint(float point) const;
  void Serialize(YAML::Emitter &out);
  void Deserialize(const YAML::Node &in);
  bool OnInspect(int mode);
};
struct ProceduralLeafState {
  BezierSpline m_spline;
  int m_index = 0;
  float m_startingPoint = 0;
  float m_length = 0;
  CurveDescriptor<float> m_widthAlongLeaf;
  float m_rollAngle = 0;
  float m_branchingAngle = 0;
  CurveDescriptor<float>  m_curlingAlongLeaf;
  CurveDescriptor<float>  m_bendingAlongLeaf;
  CurveDescriptor<float> m_wavinessAlongLeaf;
  glm::vec2 m_wavinessPeriodStart = glm::vec2(0.0f);
  glm::vec2 m_wavinessFrequency = glm::vec2(0.0f);
  ProceduralLeafState();
  void Serialize(YAML::Emitter &out);
  void Deserialize(const YAML::Node &in);
  bool OnInspect(int mode);
};
#pragma endregion

class SorghumState {
  friend class ProceduralSorghum;
  unsigned m_version = 0;
public:
  ProceduralPanicleState m_panicle;
  ProceduralStemState m_stem;
  std::vector<ProceduralLeafState> m_leaves;
  bool OnInspect(int mode);

  void Serialize(YAML::Emitter &out);
  void Deserialize(const YAML::Node &in);
};
class SorghumStateGenerator;

struct SorghumStatePair {
  SorghumState m_left = SorghumState();
  SorghumState m_right = SorghumState();
  float m_a = 1.0f;
  int m_mode = (int)StateMode::Default;
  [[nodiscard]] int GetLeafSize() const;
  [[nodiscard]] float GetStemLength() const;
  [[nodiscard]] glm::vec3 GetStemDirection() const;
  [[nodiscard]] glm::vec3 GetStemPoint(float point) const;
};

class ProceduralSorghum : public IAsset {
  unsigned m_version = 0;
  friend class SorghumData;
  std::vector<std::pair<float, SorghumState>> m_sorghumStates;
public:
  int m_mode = (int)StateMode::Default;

  [[nodiscard]] unsigned GetVersion() const;
  [[nodiscard]] float GetCurrentStartTime() const;
  [[nodiscard]] float GetCurrentEndTime() const;
  void Add(float time, const SorghumState &state);
  void ResetTime(float previousTime, float newTime);
  void Remove(float time);
  [[nodiscard]] SorghumStatePair Get(float time) const;

  void OnInspect() override;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;
};

} // namespace SorghumFactory