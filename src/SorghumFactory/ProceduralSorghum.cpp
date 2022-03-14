//
// Created by lllll on 1/8/2022.
//
#include "ProceduralSorghum.hpp"

#include "SorghumData.hpp"
#include "SorghumLayer.hpp"
#include "SorghumStateGenerator.hpp"
#include "Utilities.hpp"
#include <utility>
using namespace SorghumFactory;

SorghumStatePair ProceduralSorghum::Get(float time) const {
  SorghumStatePair retVal;
  if (m_sorghumStates.empty())
    return retVal;
  if (m_sorghumStates.size() == 1) {
    retVal.m_right = m_sorghumStates.begin()->second;
    return retVal;
  }
  auto actualTime = glm::clamp(time, 0.0f, 99999.0f);
  float previousTime = m_sorghumStates.begin()->first;
  SorghumState previousState = m_sorghumStates.begin()->second;
  if (actualTime < previousTime) {
    // Get from zero state to first state.
    retVal.m_right = m_sorghumStates.begin()->second;
    return retVal;
  }
  for (auto it = (++m_sorghumStates.begin()); it != m_sorghumStates.end();
       it++) {
    if (it->first > actualTime) {
      retVal.m_left = previousState;
      retVal.m_right = it->second;
      retVal.m_a = (actualTime - previousTime) / it->first - previousTime;
      return retVal;
    }
  }
  return {(--m_sorghumStates.end())->second, (--m_sorghumStates.end())->second,
          1.0f};
}

bool ProceduralPinnacleState::OnInspect() {
  bool changed = ImGui::Checkbox("Active", &m_active);
  if (ImGui::DragFloat3("Pinnacle size", &m_pinnacleSize.x, 0.01f))
    changed = true;
  if (ImGui::DragInt("Num of seed", &m_seedAmount, 0.01f))
    changed = true;
  if (ImGui::DragFloat("Seed radius", &m_seedRadius, 0.01f))
    changed = true;
  return changed;
}
void ProceduralPinnacleState::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_active" << YAML::Value << m_active;
  out << YAML::Key << "m_pinnacleSize" << YAML::Value << m_pinnacleSize;
  out << YAML::Key << "m_seedAmount" << YAML::Value << m_seedAmount;
  out << YAML::Key << "m_seedRadius" << YAML::Value << m_seedRadius;
}
void ProceduralPinnacleState::Deserialize(const YAML::Node &in) {
  if (in["m_active"])
    m_active = in["m_active"].as<bool>();
  if (in["m_pinnacleSize"])
    m_pinnacleSize = in["m_pinnacleSize"].as<glm::vec3>();
  if (in["m_seedAmount"])
    m_seedAmount = in["m_seedAmount"].as<int>();
  if (in["m_seedRadius"])
    m_seedRadius = in["m_seedRadius"].as<float>();
}
void ProceduralStemState::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_direction" << YAML::Value << m_direction;
  m_widthAlongStem.Serialize("m_widthAlongStem", out);
  out << YAML::Key << "m_length" << YAML::Value << m_length;
}
void ProceduralStemState::Deserialize(const YAML::Node &in) {
  if (in["m_direction"])
    m_direction = in["m_direction"].as<glm::vec3>();
  if (in["m_length"])
    m_length = in["m_length"].as<float>();
  m_widthAlongStem.Deserialize("m_widthAlongStem", in);
}
bool ProceduralStemState::OnInspect() {
  bool changed = ImGui::DragFloat3("Direction", &m_direction.x, 0.01f);
  if (ImGui::DragFloat("Length", &m_length, 0.01f))
    changed = true;
  if (m_widthAlongStem.OnInspect("Width along stem"))
    changed = true;
  return changed;
}
bool ProceduralLeafState::OnInspect() {
  bool changed = false;
  ImGui::Text(("Index: " + std::to_string(m_index)).c_str());
  if (ImGui::DragFloat2("Bending", &m_bending.x, 0.01f, 0.0f, 999.0f))
    changed = true;
  if (ImGui::DragFloat("Length", &m_length, 0.01f, 0.0f, 999.0f))
    changed = true;
  if (ImGui::DragFloat("Dist to root", &m_distanceToRoot, 0.01f, 0.0f, 999.0f))
    changed = true;
  if (m_widthAlongLeaf.OnInspect("Width along leaf"))
    changed = true;
  if (ImGui::DragFloat("Roll angle", &m_rollAngle, 0.01f, 0.0f, 999.0f))
    changed = true;
  if (ImGui::DragFloat("Branching angle", &m_branchingAngle, 0.01f, 0.0f,
                       999.0f))
    changed = true;
  if (m_wavinessAlongLeaf.OnInspect("Waviness along leaf"))
    changed = true;
  if (ImGui::DragFloat2("Waviness frequency", &m_wavinessFrequency.x, 0.01f,
                        0.0f, 999.0f))
    changed = true;
  if (ImGui::DragFloat2("Waviness start period", &m_wavinessPeriodStart.x,
                        0.01f, 0.0f, 999.0f))
    changed = true;
  return changed;
}
void ProceduralLeafState::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_index" << YAML::Value << m_index;
  out << YAML::Key << "m_distanceToRoot" << YAML::Value << m_distanceToRoot;
  out << YAML::Key << "m_length" << YAML::Value << m_length;
  out << YAML::Key << "m_curling" << YAML::Value << m_curling;
  m_widthAlongLeaf.Serialize("m_widthAlongLeaf", out);
  out << YAML::Key << "m_rollAngle" << YAML::Value << m_rollAngle;
  out << YAML::Key << "m_branchingAngle" << YAML::Value << m_branchingAngle;
  out << YAML::Key << "m_bending" << YAML::Value << m_bending;
  m_wavinessAlongLeaf.Serialize("m_wavinessAlongLeaf", out);
  out << YAML::Key << "m_wavinessFrequency" << YAML::Value
      << m_wavinessFrequency;
  out << YAML::Key << "m_wavinessPeriodStart" << YAML::Value
      << m_wavinessPeriodStart;
}
void ProceduralLeafState::Deserialize(const YAML::Node &in) {
  if (in["m_index"])
    m_index = in["m_index"].as<int>();
  if (in["m_distanceToRoot"])
    m_distanceToRoot = in["m_distanceToRoot"].as<float>();
  if (in["m_curling"])
    m_curling = in["m_curling"].as<float>();
  if (in["m_length"])
    m_length = in["m_length"].as<float>();
  if (in["m_rollAngle"])
    m_rollAngle = in["m_rollAngle"].as<float>();
  if (in["m_branchingAngle"])
    m_branchingAngle = in["m_branchingAngle"].as<float>();
  if (in["m_bending"])
    m_bending = in["m_bending"].as<glm::vec2>();
  if (in["m_wavinessFrequency"])
    m_wavinessFrequency = in["m_wavinessFrequency"].as<glm::vec2>();
  if (in["m_wavinessPeriodStart"])
    m_wavinessPeriodStart = in["m_wavinessPeriodStart"].as<glm::vec2>();

  m_widthAlongLeaf.Deserialize("m_widthAlongLeaf", in);
  m_wavinessAlongLeaf.Deserialize("m_wavinessAlongLeaf", in);
}

bool SorghumState::OnMenu() {
  bool changed = false;
  if (ImGui::TreeNode("Stem")) {
    if (m_stem.OnInspect())
      changed = true;
    ImGui::TreePop();
  }
  if (ImGui::TreeNode("Pinnacle")) {
    if (m_pinnacle.OnInspect())
      changed = true;
    ImGui::TreePop();
  }
  if (ImGui::TreeNode("Leaves")) {

    for (auto &leaf : m_leaves) {
      if (ImGui::TreeNode(("Leaf " + std::to_string(leaf.m_index)).c_str())) {
        if (leaf.OnInspect())
          changed = true;
        ImGui::TreePop();
      }
    }
    ImGui::TreePop();
  }
  return changed;
}
void SorghumState::OnInspect() { OnMenu(); }
void SorghumState::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_version" << YAML::Value << m_version;
  out << YAML::Key << "m_pinnacle" << YAML::Value << YAML::BeginMap;
  m_pinnacle.Serialize(out);
  out << YAML::EndMap;
  out << YAML::Key << "m_stem" << YAML::Value << YAML::BeginMap;
  m_stem.Serialize(out);
  out << YAML::EndMap;

  if (!m_leaves.empty()) {
    out << YAML::Key << "m_leaves" << YAML::Value << YAML::BeginSeq;
    for (auto &i : m_leaves) {
      out << YAML::BeginMap;
      i.Serialize(out);
      out << YAML::EndMap;
    }
    out << YAML::EndSeq;
  }
}
void SorghumState::Deserialize(const YAML::Node &in) {
  if (in["m_version"])
    m_version = in["m_version"].as<unsigned>();

  if (in["m_pinnacle"])
    m_pinnacle.Deserialize(in["m_pinnacle"]);

  if (in["m_stem"])
    m_stem.Deserialize(in["m_stem"]);

  if (in["m_leaves"]) {
    for (const auto &i : in["m_leaves"]) {
      ProceduralLeafState leaf;
      leaf.Deserialize(i);
      m_leaves.push_back(leaf);
    }
  }
}

void ProceduralSorghum::OnInspect() {
  if (ImGui::Button("Instantiate")) {
    Application::GetLayer<SorghumLayer>()->CreateSorghum(
        AssetManager::Get<ProceduralSorghum>(GetHandle()));
  }
  bool changed = false;
  if (ImGui::TreeNodeEx("States", ImGuiTreeNodeFlags_DefaultOpen)) {
    float startTime =
        m_sorghumStates.empty() ? 1.0f : m_sorghumStates.begin()->first;
    if (startTime >= 0.01f) {
      if (ImGui::Button("Push new start state")) {
        changed = true;
        if (m_sorghumStates.empty()) {
          Add(0.0f, SorghumState());
        } else {
          Add(0.0f, m_sorghumStates.begin()->second);
        }
      }
    }

    float previousTime = 0.0f;
    int stateIndex = 1;
    for (auto it = m_sorghumStates.begin(); it != m_sorghumStates.end(); ++it) {
      if (ImGui::TreeNode(("State " + std::to_string(stateIndex)).c_str())) {
        if (it != (--m_sorghumStates.end())) {
          auto tit = it;
          ++tit;
          float nextTime = tit->first - 0.01f;
          float currentTime = it->first;
          if (ImGui::SliderFloat("Time", &currentTime, previousTime,
                                 nextTime)) {
            it->first = glm::clamp(currentTime, previousTime, nextTime);
            changed = true;
          }
        } else {
          float currentTime = it->first;
          if (ImGui::DragFloat("Time", &currentTime, 0.01f, previousTime,
                               99999.0f)) {
            it->first = glm::clamp(currentTime, previousTime, 99999.0f);
            changed = true;
          }
        }
        if (it->second.OnMenu()) {
          changed = true;
        }
        ImGui::TreePop();
      }
      previousTime = it->first + 0.01f;
      stateIndex++;
    }

    if (!m_sorghumStates.empty()) {
      if (ImGui::Button("Push new end state")) {
        changed = true;
        float endTime = (--m_sorghumStates.end())->first;
        Add(endTime + 0.01f, (--m_sorghumStates.end())->second);
      }
    }
    ImGui::TreePop();
  }
  if (changed) {
    m_saved = false;
    m_version++;
  }
}

void ProceduralSorghum::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_version" << YAML::Value << m_version;
  out << YAML::Key << "m_sorghumStates" << YAML::Value << YAML::BeginSeq;
  for (auto &state : m_sorghumStates) {
    out << YAML::BeginMap;
    out << YAML::Key << "Time" << YAML::Value << state.first;
    state.second.Serialize(out);
    out << YAML::EndMap;
  }
  out << YAML::EndSeq;
}

void ProceduralSorghum::Deserialize(const YAML::Node &in) {
  if (in["m_version"])
    m_version = in["m_version"].as<unsigned>();
  if (in["m_sorghumStates"]) {
    m_sorghumStates.clear();
    for (const auto &inState : in["m_sorghumStates"]) {
      SorghumState state;
      state.Deserialize(inState);
      m_sorghumStates.emplace_back(inState["Time"].as<float>(), state);
    }
  }
}

void ProceduralSorghum::Add(float time, const SorghumState &state) {
  for (auto it = m_sorghumStates.begin(); it != m_sorghumStates.end(); ++it) {
    if (it->first == time) {
      it->second = state;
      return;
    }
    if (it->first > time) {
      m_sorghumStates.insert(it, {time, state});
      return;
    }
  }
  m_sorghumStates.emplace_back(time, state);
}

void ProceduralSorghum::ResetTime(float previousTime, float newTime) {
  for (auto &i : m_sorghumStates) {
    if (i.first == previousTime) {
      i.first = newTime;
      return;
    }
  }
  UNIENGINE_ERROR("Failed: State at previous time not exists!");
}
void ProceduralSorghum::Remove(float time) {
  for (auto it = m_sorghumStates.begin(); it != m_sorghumStates.end(); ++it) {
    if (it->first == time) {
      m_sorghumStates.erase(it);
      return;
    }
  }
}
float ProceduralSorghum::GetCurrentStartTime() const {
  if (m_sorghumStates.empty()) {
    return 0.0f;
  }
  return m_sorghumStates.begin()->first;
}
float ProceduralSorghum::GetCurrentEndTime() const {
  if (!m_sorghumStates.empty()) {
    return 0.0f;
  }
  return (--m_sorghumStates.end())->first;
}

unsigned ProceduralSorghum::GetVersion() const { return m_version; }

glm::vec3 ProceduralStemState::GetPoint(float point) const {
  return m_direction * point * m_length;
}
int SorghumStatePair::SizeOfLeaf() {
  return (m_left.m_leaves.size() + m_right.m_leaves.size()) * m_a;
}
