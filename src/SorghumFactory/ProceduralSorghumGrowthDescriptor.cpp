//
// Created by lllll on 1/8/2022.
//
#include "ProceduralSorghumGrowthDescriptor.hpp"
#include "Utilities.hpp"
using namespace SorghumFactory;
ProceduralPinnacleGrowthState
ProceduralPinnacleGrowthDescriptor::Get(float time, unsigned seed) const {
  ProceduralPinnacleGrowthState state = {};
  if (time < m_startTime) {
    state.m_active = false;
    return state;
  }
  state.m_active = true;
  float a =
      glm::clamp((time - m_startTime) / (m_endTime - m_startTime), 0.0f, 1.0f);
  state.m_pinnacleSize = m_maxPinnacleSize * m_pinnacleSizeCurve.GetValue(a);
  state.m_seedAmount = m_maxSeedAmount * m_seedAmountCurve.GetValue(a);
  state.m_seedRadius = m_maxSeedRadius * m_seedRadiusCurve.GetValue(a);
  return state;
}

ProceduralStemGrowthState
ProceduralStemGrowthDescriptor::Get(float time, unsigned seed) const {
  ProceduralStemGrowthState state = {};
  float a = glm::clamp(time / m_endTime, 0.0f, 1.0f);
  state.m_length = m_maxLength * m_lengthCurve.GetValue(a);
  state.m_direction = m_direction;
  return state;
}
ProceduralLeafGrowthState
ProceduralLeafGrowthDescriptor::Get(float time, unsigned seed) const {
  ProceduralLeafGrowthState state = {};
  if (time < m_startTime) {
    state.m_active = false;
    return state;
  }
  state.m_active = true;
  float a =
      glm::clamp((time - m_startTime) / (m_endTime - m_startTime), 0.0f, 1.0f);
  state.m_startingPoint =
      m_startingPoint.x + (m_startingPoint.y - m_startingPoint.x) *
                              m_startingPointCurve.GetValue(a);
  state.m_length = m_maxLength * m_lengthCurve.GetValue(a);
  state.m_width = m_maxWidth * m_widthCurve.GetValue(a);
  state.m_rollAngle = m_rollAngle;
  state.m_branchingAngle = m_branchingAngle;
  state.m_bending = m_bending;
  return state;
}
ProceduralSorghumState
ProceduralSorghumGrowthDescriptor::Get(float time) const {
  srand(m_seed);
  ProceduralSorghumState state;
  state.m_pinnacle = m_pinnacle.Get(time, m_seed);
  state.m_stem = m_stem.Get(time, m_seed);

  state.m_leaves.resize(m_leaves.size());
  for (int i = 0; i < m_leaves.size(); i++) {
    state.m_leaves[i] = m_leaves[i].Get(time, m_seed);
  }
  return state;
}
void ProceduralSorghumGrowthDescriptor::Set(
    const ProceduralSorghumState &endState) {}

void ProceduralPinnacleGrowthState::OnInspect() const {}
void ProceduralLeafGrowthState::OnInspect() const {}
void ProceduralSorghumState::OnInspect() const {}
void ProceduralSorghumGrowthDescriptor::OnInspect() {
  if (ImGui::TreeNode("Stem"))
    m_stem.OnInspect();
  if (ImGui::TreeNode("Pinnacle"))
    m_pinnacle.OnInspect();
  if (ImGui::TreeNode("Leaves")) {
    for (auto &leaf : m_leaves) {
      leaf.OnInspect();
    }
  }
}
void ProceduralSorghumGrowthDescriptor::Set(
    const std::shared_ptr<SorghumProceduralDescriptor> &endState) {
  srand(m_seed);
  
}

bool ProceduralLeafGrowthDescriptor::OnInspect() {
  bool changed = false;
  if (ImGui::DragFloat("Start time", &m_startTime, 0.01f, 0.0f, m_endTime))
    changed = true;
  if (ImGui::DragFloat("End time", &m_endTime, 0.01f, m_startTime, 999))
    changed = true;

  if (ImGui::DragFloat2("Starting point", &m_startingPoint.x, 0.01f, 0.0f,
                        1.0f))
    changed = true;
  if (m_startingPointCurve.CurveEditor("Starting point"))
    changed = true;

  if (ImGui::DragFloat("Max length", &m_maxLength, 0.01f, 0.0f, 999.0f))
    changed = true;
  if (m_lengthCurve.CurveEditor("Max length"))
    changed = true;

  if (ImGui::DragFloat("Max width", &m_maxWidth, 0.01f, 0.0f, 999.0f))
    changed = true;
  if (m_widthCurve.CurveEditor("Max width"))
    changed = true;

  if (ImGui::DragFloat("Roll angle", &m_rollAngle, 0.01f, 0.0f, 999.0f))
    changed = true;

  if (ImGui::DragFloat("Branching angle", &m_branchingAngle, 0.01f, 0.0f,
                       999.0f))
    changed = true;

  if (ImGui::DragFloat2("Bending", &m_bending.x, 0.01f, 0.0f, 1.0f))
    changed = true;
  return changed;
}

bool ProceduralPinnacleGrowthDescriptor::OnInspect() {
  bool changed = false;
  if (ImGui::DragFloat("Start time", &m_startTime, 0.01f, 0.0f, m_endTime))
    changed = true;
  if (ImGui::DragFloat("End time", &m_endTime, 0.01f, m_startTime, 999))
    changed = true;
  if (ImGui::DragFloat3("Max size", &m_maxPinnacleSize.x, 0.01f, 0.0f, 999.0f))
    changed = true;
  if (m_pinnacleSizeCurve.CurveEditor("Size"))
    changed = true;

  if (ImGui::DragInt("Max seed amount", &m_maxSeedAmount))
    changed = true;
  if (m_seedAmountCurve.CurveEditor("Seed amount"))
    changed = true;

  if (ImGui::DragFloat("Max seed radius", &m_maxSeedRadius, 0.01f, 0.0f,
                       999.0f))
    changed = true;
  if (m_seedRadiusCurve.CurveEditor("Seed radius"))
    changed = true;

  return changed;
}

bool ProceduralStemGrowthDescriptor::OnInspect() {
  bool changed = false;
  if (ImGui::DragFloat("End time", &m_endTime, 0.01f, 0.0f, 999))
    changed = true;

  if (ImGui::DragFloat("Max length", &m_maxLength, 0.01f, 0.0f, 999.0f))
    changed = true;
  if (m_lengthCurve.CurveEditor("Length"))
    changed = true;

  return changed;
}

void ProceduralSorghumGrowthDescriptor::Serialize(YAML::Emitter &out) {}

void ProceduralSorghumGrowthDescriptor::Deserialize(const YAML::Node &in) {}

void ProceduralLeafGrowthDescriptor::Serialize(YAML::Emitter &out) {}

void ProceduralLeafGrowthDescriptor::Deserialize(const YAML::Node &in) {}

void ProceduralPinnacleGrowthDescriptor::Serialize(YAML::Emitter &out) {}

void ProceduralPinnacleGrowthDescriptor::Deserialize(const YAML::Node &in) {}

void ProceduralStemGrowthDescriptor::Serialize(YAML::Emitter &out) {}

void ProceduralStemGrowthDescriptor::Deserialize(const YAML::Node &in) {}
