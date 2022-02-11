//
// Created by lllll on 1/8/2022.
//
#include "ProceduralSorghum.hpp"
#include "SorghumData.hpp"
#include "SorghumLayer.hpp"
#include "SorghumStateGenerator.hpp"
#include "Utilities.hpp"
using namespace SorghumFactory;
ProceduralPinnacleState ProceduralPinnacleDescriptor::Get(float time) const {
  ProceduralPinnacleState state = {};
  if (time < m_startTime) {
    state.m_active = false;
    return state;
  }
  state.m_active = true;
  float a =
      glm::clamp((time - m_startTime) / (m_endTime - m_startTime), 0.0f, 1.0f);
  state.m_pinnacleSize = m_pinnacleSize.GetValue(a);
  state.m_seedAmount = m_seedAmount.GetValue(a);
  state.m_seedRadius = m_seedRadius.GetValue(a);
  return state;
}

ProceduralStemState ProceduralStemDescriptor::Get(float time) const {
  ProceduralStemState state = {};
  float a = glm::clamp(time / m_endTime, 0.0f, 1.0f);
  state.m_length = m_length.GetValue(a);
  state.m_direction = m_direction;
  state.m_widthAlongStem.m_curve = m_widthAlongStem;
  state.m_widthAlongStem.m_maxValue = m_width.GetValue(a);
  state.m_widthAlongStem.m_minValue = 0.0f;
  return state;
}
ProceduralLeafState ProceduralLeafDescriptor::Get(float time) const {
  ProceduralLeafState state = {};
  if (time < m_startTime) {
    state.m_active = false;
    return state;
  }
  state.m_active = true;
  float a =
      glm::clamp((time - m_startTime) / (m_endTime - m_startTime), 0.0f, 1.0f);
  state.m_distanceToRoot = m_distanceToRoot.GetValue(a);
  state.m_length = m_length.GetValue(a);

  state.m_widthAlongLeaf.m_curve = m_widthAlongLeaf;
  state.m_widthAlongLeaf.m_minValue = 0.0f;
  state.m_widthAlongLeaf.m_maxValue = m_width.GetValue(a);

  state.m_wavinessAlongLeaf.m_curve = m_wavinessAlongLeaf;
  state.m_wavinessAlongLeaf.m_minValue = 0.0f;
  state.m_wavinessAlongLeaf.m_maxValue = m_waviness.GetValue(a);

  state.m_rollAngle = m_rollAngle;
  state.m_branchingAngle = m_branchingAngle;
  state.m_bending = m_bending;

  state.m_wavinessFrequency = m_wavinessFrequency;
  return state;
}
ProceduralSorghumState ProceduralSorghum::Get(float time) const {
  float actualTime = time;
  if (actualTime > m_endTime)
    actualTime = m_endTime;
  srand(m_seed);
  ProceduralSorghumState state;
  state.m_pinnacle = m_pinnacle.Get(actualTime);
  state.m_stem = m_stem.Get(actualTime);
  state.m_leaves.resize(m_leaves.size());
  for (int i = 0; i < m_leaves.size(); i++) {
    state.m_leaves[i] = m_leaves[i].Get(actualTime);
  }
  return state;
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
  bool changed = ImGui::Checkbox("Active", &m_active);
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
  if (ImGui::DragFloat("Waviness frequency", &m_wavinessFrequency, 0.01f, 0.0f,
                       999.0f))
    changed = true;
  return changed;
}
void ProceduralLeafState::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_active" << YAML::Value << m_active;
  out << YAML::Key << "m_index" << YAML::Value << m_index;
  out << YAML::Key << "m_distanceToRoot" << YAML::Value << m_distanceToRoot;
  out << YAML::Key << "m_length" << YAML::Value << m_length;
  m_widthAlongLeaf.Serialize("m_widthAlongLeaf", out);
  out << YAML::Key << "m_rollAngle" << YAML::Value << m_rollAngle;
  out << YAML::Key << "m_branchingAngle" << YAML::Value << m_branchingAngle;
  out << YAML::Key << "m_bending" << YAML::Value << m_bending;
  m_wavinessAlongLeaf.Serialize("m_wavinessAlongLeaf", out);
  out << YAML::Key << "m_wavinessFrequency" << YAML::Value
      << m_wavinessFrequency;
}
void ProceduralLeafState::Deserialize(const YAML::Node &in) {
  if (in["m_active"])
    m_active = in["m_active"].as<bool>();
  if (in["m_index"])
    m_index = in["m_index"].as<int>();
  if (in["m_distanceToRoot"])
    m_distanceToRoot = in["m_distanceToRoot"].as<float>();
  if (in["m_length"])
    m_length = in["m_length"].as<float>();
  if (in["m_rollAngle"])
    m_rollAngle = in["m_rollAngle"].as<float>();
  if (in["m_branchingAngle"])
    m_branchingAngle = in["m_branchingAngle"].as<float>();
  if (in["m_bending"])
    m_bending = in["m_bending"].as<glm::vec2>();
  if (in["m_wavinessFrequency"])
    m_wavinessFrequency = in["m_wavinessFrequency"].as<float>();

  m_widthAlongLeaf.Deserialize("m_widthAlongLeaf", in);
  m_wavinessAlongLeaf.Deserialize("m_wavinessAlongLeaf", in);
}
bool ProceduralSorghumState::OnInspect() {
  bool changed = ImGui::DragFloat("Time", &m_time, 0.01f);
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
    int index = 0;
    for (auto &leaf : m_leaves) {
      if (ImGui::TreeNode(("Leaf " + std::to_string(index)).c_str())) {
        if (leaf.OnInspect())
          changed = true;
        ImGui::TreePop();
      }
      index++;
    }
    ImGui::TreePop();
  }
  return changed;
}
void ProceduralSorghumState::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_time" << YAML::Value << m_time;
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
void ProceduralSorghumState::Deserialize(const YAML::Node &in) {
  if (in["m_time"])
    m_time = in["m_time"].as<float>();

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
  static AssetRef descriptor;
  Editor::DragAndDropButton<SorghumStateGenerator>(
      descriptor, "Drag SPD here to apply end state");
  auto temp = descriptor.Get<SorghumStateGenerator>();
  static float maxTime = 1.0f;
  static float longestLeafTime = 0.3f;
  static bool autoCal = true;
  if (ImGui::DragInt("Seed", &m_seed))
    m_saved = false;
  ImGui::Checkbox("Auto recalculate growth", &autoCal);
  if (temp) {
    m_endState = temp->Generate(m_seed);
    Set(maxTime, longestLeafTime);
    descriptor.Clear();
  }
  if (ImGui::TreeNodeEx("End state", ImGuiTreeNodeFlags_DefaultOpen)) {
    bool changed = m_endState.OnInspect();
    m_saved = m_saved && changed;
    if (autoCal && changed) {
      Set(maxTime, longestLeafTime);
    }
    ImGui::TreePop();
  }
  if (!autoCal) {
    if (ImGui::Button("Calculate growth")) {
      Set(maxTime, longestLeafTime);
    }
  }
  if (ImGui::DragFloat("Max time", &maxTime, 0.01f, 0.0f) && autoCal) {
    Set(maxTime, longestLeafTime);
  }
  if (ImGui::SliderFloat("Time for longest leaf", &longestLeafTime, 0.0f,
                         1.0f) &&
      autoCal) {
    Set(maxTime, longestLeafTime);
  }
  if (ImGui::Button("Instantiate")) {
    Application::GetLayer<SorghumLayer>()->CreateSorghum(
        AssetManager::Get<ProceduralSorghum>(GetHandle()));
  }
  if (ImGui::TreeNode("Growth")) {
    if (ImGui::TreeNode("Stem")) {
      if (m_stem.OnInspect(m_endTime)) {
        m_saved = false;
        m_version++;
      }
      ImGui::TreePop();
    }
    if (ImGui::TreeNode("Pinnacle")) {
      if (m_pinnacle.OnInspect(m_endTime)) {
        m_saved = false;
        m_version++;
      }
      ImGui::TreePop();
    }
    if (ImGui::TreeNode("Leaves")) {
      int index = 0;
      for (auto &leaf : m_leaves) {
        if (ImGui::TreeNode(("Leaf " + std::to_string(index)).c_str())) {
          if (leaf.OnInspect(m_endTime)) {
            m_saved = false;
            m_version++;
          }
          ImGui::TreePop();
        }
        index++;
      }
      ImGui::TreePop();
    }
    ImGui::TreePop();
  }
}

void ProceduralSorghum::Set(float endTime, float longestLeafTime) {
  m_saved = false;
  m_endState.m_time = endTime;
  m_endTime = endTime;
  m_version++;
  m_stem.m_length = {0.0f, m_endState.m_stem.m_length,
                     UniEngine::Curve(0.1f, 1.0f, {0, 0}, {1, 1})};
  m_stem.m_direction = m_endState.m_stem.m_direction;
  m_stem.m_widthAlongStem = m_endState.m_stem.m_widthAlongStem.m_curve;
  m_stem.m_width = {0.0f, m_endState.m_stem.m_widthAlongStem.m_maxValue,
                    UniEngine::Curve(0.1f, 1.0f, {0, 0}, {1, 1})};

  m_pinnacle.m_startTime = m_endState.m_pinnacle.m_active
                               ? endTime - endTime * 0.1f
                               : endTime + 0.1f;
  m_pinnacle.m_endTime = endTime;
  m_pinnacle.m_seedRadius = {0.0f, m_endState.m_pinnacle.m_seedRadius,
                             UniEngine::Curve(0.0f, 1.0f, {0, 0}, {1, 1})};
  m_pinnacle.m_seedAmount = {0, m_endState.m_pinnacle.m_seedAmount,
                             UniEngine::Curve(0.0f, 1.0f, {0, 0}, {1, 1})};
  m_pinnacle.m_pinnacleSize = {glm::vec3(0.0f),
                               m_endState.m_pinnacle.m_pinnacleSize,
                               UniEngine::Curve(0.0f, 1.0f, {0, 0}, {1, 1})};

  int leafSize = (int)m_endState.m_leaves.size();
  m_leaves.resize(leafSize);

  float maxLength = 0.0f;
  float growthTimeForLongestLeaf = endTime * longestLeafTime;

  for (int i = 0; i < leafSize; i++) {
    auto &leaf = m_leaves[i];
    const auto &leafEndState = m_endState.m_leaves[i];
    leaf.m_length = {0.0f, leafEndState.m_length,
                     UniEngine::Curve(0.0f, 1.0f, {0, 0}, {1, 1})};
    if (leaf.m_length.GetValue(1.0f) > maxLength) {
      maxLength = leaf.m_length.GetValue(1.0f);
    }
  }
  float growthTimeForLastLeaf = growthTimeForLongestLeaf *
                                m_endState.m_leaves.back().m_length / maxLength;
  float lastLeafStartTime = endTime - growthTimeForLastLeaf;

  m_stem.m_endTime =
      lastLeafStartTime / (m_endState.m_leaves.back().m_distanceToRoot /
                           m_stem.m_length.GetValue(1.0f));
  float firstLeafStartTime = m_stem.m_endTime *
                             m_endState.m_leaves.front().m_distanceToRoot /
                             m_stem.m_length.GetValue(1.0f);
  float startingTimeStep =
      leafSize <= 1 ? 0.0f
                    : (lastLeafStartTime - firstLeafStartTime) / (leafSize - 1);
  for (int i = 0; i < leafSize; i++) {
    auto &leaf = m_leaves[i];
    const auto &leafEndState = m_endState.m_leaves[i];
    leaf.m_wavinessFrequency = leafEndState.m_wavinessFrequency;
    leaf.m_wavinessAlongLeaf = leafEndState.m_wavinessAlongLeaf.m_curve;
    leaf.m_waviness = {0.0f, leafEndState.m_wavinessAlongLeaf.m_maxValue,
                       UniEngine::Curve(0.1f, 1.0f, {0, 0}, {1, 1})};

    leaf.m_distanceToRoot = {0.0f, leafEndState.m_distanceToRoot * 2.0f,
                             UniEngine::Curve(0.5f, 0.5f, {0, 0}, {1, 1})};

    leaf.m_widthAlongLeaf = leafEndState.m_widthAlongLeaf.m_curve;
    leaf.m_width = {0.0f, leafEndState.m_widthAlongLeaf.m_maxValue,
                    UniEngine::Curve(0.1f, 1.0f, {0, 0}, {1, 1})};

    leaf.m_rollAngle = leafEndState.m_rollAngle;
    leaf.m_branchingAngle = leafEndState.m_branchingAngle;
    leaf.m_bending = leafEndState.m_bending;

    float growthTime = leaf.m_length.GetValue(1.0f) / maxLength *
                       growthTimeForLongestLeaf * endTime;
    leaf.m_startTime = glm::clamp(
        firstLeafStartTime + startingTimeStep * (float)i, 0.0f, endTime);
    leaf.m_endTime = glm::clamp(leaf.m_startTime + growthTime, 0.0f, endTime);
  }
}
bool ProceduralLeafDescriptor::OnInspect(float maxTime) {
  bool changed = ImGui::SliderFloat("Start time", &m_startTime, 0.0f, maxTime);
  if (ImGui::SliderFloat("End time", &m_endTime, 0.0f, maxTime))
    changed = true;
  if (ImGui::DragFloat("Roll angle", &m_rollAngle, 0.01f, 0.0f, 999.0f))
    changed = true;
  if (ImGui::DragFloat("Branching angle", &m_branchingAngle, 0.01f, 0.0f,
                       999.0f))
    changed = true;
  if (ImGui::DragFloat2("Bending", &m_bending.x, 0.01f, 0.0f, 1.0f))
    changed = true;
  if (m_distanceToRoot.OnInspect("Distance to root"))
    changed = true;
  if (m_length.OnInspect("Length"))
    changed = true;
  if (m_width.OnInspect("Width"))
    changed = true;
  if (m_waviness.OnInspect("Waviness"))
    changed = true;
  if (m_widthAlongLeaf.OnInspect("Width along leaf"))
    changed = true;
  if (m_wavinessAlongLeaf.OnInspect("Waviness along leaf"))
    changed = true;
  return changed;
}

bool ProceduralPinnacleDescriptor::OnInspect(float maxTime) {
  bool changed = false;
  if (ImGui::SliderFloat("Start time", &m_startTime, 0.0f, m_endTime))
    changed = true;
  if (ImGui::SliderFloat("End time", &m_endTime, 0.0f, maxTime))
    changed = true;
  if (m_pinnacleSize.OnInspect("Pinnacle size"))
    changed = true;
  if (m_seedAmount.OnInspect("Seed size"))
    changed = true;
  if (m_seedRadius.OnInspect("Seed radius"))
    changed = true;
  return changed;
}

bool ProceduralStemDescriptor::OnInspect(float maxTime) {
  bool changed = false;
  changed = ImGui::DragFloat("End time", &m_endTime, 0.01f, maxTime);
  if (ImGui::DragFloat3("Direction", &m_direction.x, 0.01f))
    changed = true;
  if (m_length.OnInspect("Length"))
    changed = true;
  if (m_width.OnInspect("Width"))
    changed = true;
  return changed;
}

void ProceduralSorghum::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_endTime" << YAML::Value << m_endTime;
  out << YAML::Key << "m_seed" << YAML::Value << m_seed;
  out << YAML::Key << "m_version" << YAML::Value << m_version;
  out << YAML::Key << "m_endState" << YAML::Value << YAML::BeginMap;
  m_endState.Serialize(out);
  out << YAML::EndMap;

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

void ProceduralSorghum::Deserialize(const YAML::Node &in) {
  if (in["m_endTime"])
    m_endTime = in["m_endTime"].as<float>();
  if (in["m_seed"])
    m_seed = in["m_seed"].as<int>();
  if (in["m_version"])
    m_version = in["m_version"].as<unsigned>();
  if (in["m_endState"])
    m_endState.Deserialize(in["m_endState"]);

  if (in["m_pinnacle"])
    m_pinnacle.Deserialize(in["m_pinnacle"]);

  if (in["m_stem"])
    m_stem.Deserialize(in["m_stem"]);

  if (in["m_leaves"]) {
    for (const auto &i : in["m_leaves"]) {
      ProceduralLeafDescriptor leaf;
      leaf.Deserialize(i);
      m_leaves.push_back(leaf);
    }
  }
}
unsigned ProceduralSorghum::GetVersion() const { return m_version; }

void ProceduralPinnacleDescriptor::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_startTime" << YAML::Value << m_startTime;
  out << YAML::Key << "m_endTime" << YAML::Value << m_endTime;
  m_pinnacleSize.Serialize("m_pinnacleSize", out);
  m_seedAmount.Serialize("m_seedAmount", out);
  m_seedRadius.Serialize("m_seedRadius", out);
}

void ProceduralPinnacleDescriptor::Deserialize(const YAML::Node &in) {
  if (in["m_startTime"])
    m_startTime = in["m_startTime"].as<float>();
  if (in["m_endTime"])
    m_endTime = in["m_endTime"].as<float>();
  m_pinnacleSize.Deserialize("m_pinnacleSize", in);
  m_seedAmount.Deserialize("m_seedAmount", in);
  m_seedRadius.Deserialize("m_seedRadius", in);
}

void ProceduralStemDescriptor::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_endTime" << YAML::Value << m_endTime;
  out << YAML::Key << "m_direction" << YAML::Value << m_direction;
  m_length.Serialize("m_length", out);
  m_width.Serialize("m_width", out);

  m_widthAlongStem.UniEngine::ISerializable::Serialize("m_widthAlongStem", out);
}

void ProceduralStemDescriptor::Deserialize(const YAML::Node &in) {
  if (in["m_endTime"])
    m_endTime = in["m_endTime"].as<float>();
  if (in["m_direction"])
    m_direction = in["m_direction"].as<glm::vec3>();
  m_length.Deserialize("m_length", in);
  m_width.Deserialize("m_width", in);

  m_widthAlongStem.UniEngine::ISerializable::Deserialize("m_widthAlongStem", in);
}

void ProceduralLeafDescriptor::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_startTime" << YAML::Value << m_startTime;
  out << YAML::Key << "m_endTime" << YAML::Value << m_endTime;
  out << YAML::Key << "m_rollAngle" << YAML::Value << m_rollAngle;
  out << YAML::Key << "m_branchingAngle" << YAML::Value << m_branchingAngle;
  out << YAML::Key << "m_bending" << YAML::Value << m_bending;
  out << YAML::Key << "m_wavinessFrequency" << YAML::Value
      << m_wavinessFrequency;
  m_distanceToRoot.Serialize("m_distanceToRoot", out);
  m_length.Serialize("m_length", out);
  m_width.Serialize("m_width", out);
  m_waviness.Serialize("m_waviness", out);
  m_widthAlongLeaf.UniEngine::ISerializable::Serialize("m_widthAlongLeaf", out);
  m_wavinessAlongLeaf.UniEngine::ISerializable::Serialize("m_wavinessAlongLeaf", out);
}

void ProceduralLeafDescriptor::Deserialize(const YAML::Node &in) {
  if (in["m_startTime"])
    m_startTime = in["m_startTime"].as<float>();
  if (in["m_endTime"])
    m_endTime = in["m_endTime"].as<float>();
  if (in["m_rollAngle"])
    m_rollAngle = in["m_rollAngle"].as<float>();
  if (in["m_branchingAngle"])
    m_branchingAngle = in["m_branchingAngle"].as<float>();
  if (in["m_bending"])
    m_bending = in["m_bending"].as<glm::vec2>();
  if (in["m_wavinessFrequency"])
    m_wavinessFrequency = in["m_wavinessFrequency"].as<float>();

  m_distanceToRoot.Deserialize("m_distanceToRoot", in);
  m_length.Deserialize("m_length", in);
  m_width.Deserialize("m_width", in);
  m_waviness.Deserialize("m_waviness", in);
  m_widthAlongLeaf.UniEngine::ISerializable::Deserialize("m_widthAlongLeaf", in);
  m_wavinessAlongLeaf.UniEngine::ISerializable::Deserialize("m_wavinessAlongLeaf", in);
}
glm::vec3 ProceduralStemState::GetPoint(float point) const {
  return m_direction * point * m_length;
}
