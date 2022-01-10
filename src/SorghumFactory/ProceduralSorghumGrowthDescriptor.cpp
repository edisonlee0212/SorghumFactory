//
// Created by lllll on 1/8/2022.
//
#include "ProceduralSorghumGrowthDescriptor.hpp"
#include "SorghumData.hpp"
#include "SorghumLayer.hpp"
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
  state.m_startWidth = m_maxStartWidth * m_startWidthCurve.GetValue(a);
  state.m_endWidth = m_maxEndWidth * m_endWidthCurve.GetValue(a);
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
  state.m_maxWidth = m_maxWidth * m_widthCurve.GetValue(a);
  state.m_widthAlongLeafCurve = m_widthAlongLeafCurve;
  state.m_rollAngle = m_rollAngle;
  state.m_branchingAngle = m_branchingAngle;
  state.m_bending = m_bending;
  return state;
}
ProceduralSorghumState
ProceduralSorghumGrowthDescriptor::Get(float time) const {
  float actualTime = time;
  if (actualTime > m_endTime)
    actualTime = m_endTime;
  srand(m_seed);
  ProceduralSorghumState state;
  state.m_pinnacle = m_pinnacle.Get(actualTime, m_seed);
  state.m_stem = m_stem.Get(actualTime, m_seed);
  state.m_leaves.resize(m_leaves.size());
  for (int i = 0; i < m_leaves.size(); i++) {
    state.m_leaves[i] = m_leaves[i].Get(actualTime, m_seed);
  }
  return state;
}

void ProceduralPinnacleGrowthState::OnInspect() {
  ImGui::Text(("Active: " + std::to_string(m_active)).c_str());
  ImGui::Text(("Pinnacle size: " + std::to_string(m_pinnacleSize.x) + ", " +
               std::to_string(m_pinnacleSize.y) + ", " +
               std::to_string(m_pinnacleSize.z))
                  .c_str());
  ImGui::Text(("Seed amount: " + std::to_string(m_seedAmount)).c_str());
  ImGui::Text(("Seed radius: " + std::to_string(m_seedRadius)).c_str());
}
void ProceduralPinnacleGrowthState::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_active" << YAML::Value << m_active;
  out << YAML::Key << "m_pinnacleSize" << YAML::Value << m_pinnacleSize;
  out << YAML::Key << "m_seedAmount" << YAML::Value << m_seedAmount;
  out << YAML::Key << "m_seedRadius" << YAML::Value << m_seedRadius;
}
void ProceduralPinnacleGrowthState::Deserialize(const YAML::Node &in) {
  // if(in[""]) = in[""].as<>();
  if (in["m_active"])
    m_active = in["m_active"].as<bool>();
  if (in["m_pinnacleSize"])
    m_pinnacleSize = in["m_pinnacleSize"].as<glm::vec3>();
  if (in["m_seedAmount"])
    m_seedAmount = in["m_seedAmount"].as<int>();
  if (in["m_seedRadius"])
    m_seedRadius = in["m_seedRadius"].as<float>();
}
void ProceduralStemGrowthState::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_direction" << YAML::Value << m_direction;
  out << YAML::Key << "m_startWidth" << YAML::Value << m_startWidth;
  out << YAML::Key << "m_endWidth" << YAML::Value << m_endWidth;
  out << YAML::Key << "m_length" << YAML::Value << m_length;
}
void ProceduralStemGrowthState::Deserialize(const YAML::Node &in) {
  if (in["m_direction"])
    m_direction = in["m_direction"].as<glm::vec3>();
  if (in["m_startWidth"])
    m_startWidth = in["m_startWidth"].as<float>();
  if (in["m_endWidth"])
    m_endWidth = in["m_endWidth"].as<float>();
  if (in["m_length"])
    m_length = in["m_length"].as<float>();
}
void ProceduralStemGrowthState::OnInspect() {
  ImGui::Text(("Start width: " + std::to_string(m_startWidth)).c_str());
  ImGui::Text(("Direction: " + std::to_string(m_direction.x) + ", " +
               std::to_string(m_direction.y) + ", " +
               std::to_string(m_direction.z))
                  .c_str());
  ImGui::Text(("End width: " + std::to_string(m_endWidth)).c_str());
  ImGui::Text(("Length: " + std::to_string(m_length)).c_str());
}
void ProceduralLeafGrowthState::OnInspect() {
  ImGui::Checkbox("Active", &m_active);
  ImGui::DragFloat2("Bending", &m_bending.x, 0.01f, 0.0f, 999.0f);
  ImGui::Text(("Index: " + std::to_string(m_index)).c_str());
  ImGui::DragFloat("Length", &m_length, 0.01f, 0.0f, 999.0f);
  ImGui::Text(("Starting point: " + std::to_string(m_startingPoint)).c_str());

  ImGui::DragFloat("Max width", &m_maxWidth, 0.01f, 0.0f, 999.0f);
  m_widthAlongLeafCurve.CurveEditor("Width along leaf");

  ImGui::DragFloat("Roll angle", &m_rollAngle, 0.01f, 0.0f, 999.0f);
  ImGui::DragFloat("Branching angle", &m_branchingAngle, 0.01f, 0.0f, 999.0f);
  ImGui::DragFloat("Max waviness", &m_maxWaviness, 0.01f, 0.0f, 999.0f);
  m_wavinessCurve.CurveEditor("Waviness");
  ImGui::DragFloat("Waviness period", &m_wavinessPeriod, 0.01f, 0.0f, 999.0f);
}
void ProceduralLeafGrowthState::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_active" << YAML::Value << m_active;
  out << YAML::Key << "m_index" << YAML::Value << m_index;
  out << YAML::Key << "m_startingPoint" << YAML::Value << m_startingPoint;
  out << YAML::Key << "m_length" << YAML::Value << m_length;
  out << YAML::Key << "m_maxWidth" << YAML::Value << m_maxWidth;
  out << YAML::Key << "m_widthAlongLeafCurve" << YAML::Value << YAML::BeginMap;
  m_widthAlongLeafCurve.Serialize(out);
  out << YAML::EndMap;
  out << YAML::Key << "m_rollAngle" << YAML::Value << m_rollAngle;
  out << YAML::Key << "m_branchingAngle" << YAML::Value << m_branchingAngle;
  out << YAML::Key << "m_bending" << YAML::Value << m_bending;
  out << YAML::Key << "m_maxWaviness" << YAML::Value << m_maxWaviness;
  out << YAML::Key << "m_wavinessCurve" << YAML::Value << YAML::BeginMap;
  m_wavinessCurve.Serialize(out);
  out << YAML::EndMap;
  out << YAML::Key << "m_wavinessPeriod" << YAML::Value << m_wavinessPeriod;
}
void ProceduralLeafGrowthState::Deserialize(const YAML::Node &in) {
  if (in["m_active"])
    m_active = in["m_active"].as<bool>();
  if (in["m_index"])
    m_index = in["m_index"].as<int>();
  if (in["m_startingPoint"])
    m_startingPoint = in["m_startingPoint"].as<float>();
  if (in["m_length"])
    m_length = in["m_length"].as<float>();
  if (in["m_maxWidth"])
    m_maxWidth = in["m_maxWidth"].as<float>();
  if (in["m_widthAlongLeafCurve"])
    m_widthAlongLeafCurve.Deserialize(in["m_widthAlongLeafCurve"]);
  if (in["m_rollAngle"])
    m_rollAngle = in["m_rollAngle"].as<float>();
  if (in["m_branchingAngle"])
    m_branchingAngle = in["m_branchingAngle"].as<float>();
  if (in["m_bending"])
    m_bending = in["m_bending"].as<glm::vec2>();
  if (in["m_maxWaviness"])
    m_maxWaviness = in["m_maxWaviness"].as<float>();
  if (in["m_wavinessCurve"])
    m_wavinessCurve.Deserialize(in["m_wavinessCurve"]);
  if (in["m_wavinessPeriod"])
    m_wavinessPeriod = in["m_wavinessPeriod"].as<float>();
}
void ProceduralSorghumState::OnInspect() {
  ImGui::Text(("Time: " + std::to_string(m_time)).c_str());
  if (ImGui::TreeNode("Stem")) {
    m_stem.OnInspect();
    ImGui::TreePop();
  }
  if (ImGui::TreeNode("Pinnacle")) {
    m_pinnacle.OnInspect();
    ImGui::TreePop();
  }
  if (ImGui::TreeNode("Leaves")) {
    int index = 0;
    for (auto &leaf : m_leaves) {
      if (ImGui::TreeNode(("Leaf " + std::to_string(index)).c_str())) {
        leaf.OnInspect();
        ImGui::TreePop();
      }
      index++;
    }
    ImGui::TreePop();
  }
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
      ProceduralLeafGrowthState leaf;
      leaf.Deserialize(i);
      m_leaves.push_back(leaf);
    }
  }
}
void ProceduralSorghumGrowthDescriptor::OnInspect() {
  static AssetRef descriptor;
  Editor::DragAndDropButton<SorghumProceduralDescriptor>(
      descriptor, "Drag SPD here to apply end state");
  auto temp = descriptor.Get<SorghumProceduralDescriptor>();
  if (temp) {
    Set(temp, 1.0f);
    descriptor.Clear();
  }

  static float time = 1.0f;
  ImGui::DragFloat("Time", &time, 0.01f, 0.0f, m_endTime);
  time = glm::clamp(time, 0.0f, m_endTime);
  if (ImGui::Button("Instantiate")) {
    Application::GetLayer<SorghumLayer>()->CreateSorghum(
        AssetManager::Get<ProceduralSorghumGrowthDescriptor>(GetHandle()));
  }

  if (ImGui::TreeNode("Stem")) {
    m_saved = m_saved && !m_stem.OnInspect();
    ImGui::TreePop();
  }
  if (ImGui::TreeNode("Pinnacle")) {
    m_saved = m_saved && !m_pinnacle.OnInspect();
    ImGui::TreePop();
  }
  if (ImGui::TreeNode("Leaves")) {
    int index = 0;
    for (auto &leaf : m_leaves) {
      if (ImGui::TreeNode(("Leaf " + std::to_string(index)).c_str())) {
        m_saved = m_saved && !leaf.OnInspect();
        ImGui::TreePop();
      }
      index++;
    }
    ImGui::TreePop();
  }
}
void ProceduralSorghumGrowthDescriptor::Set(
    const std::shared_ptr<SorghumProceduralDescriptor> &descriptor,
    float time) {
  srand(m_seed);
  ProceduralSorghumState endState = {};
  endState.m_time = time;
  endState.m_pinnacle.m_active = descriptor->m_pinnacleDescriptor.m_hasPinnacle;
  if (endState.m_pinnacle.m_active) {
    endState.m_pinnacle.m_seedAmount =
        descriptor->m_pinnacleDescriptor.m_seedAmount;
    endState.m_pinnacle.m_pinnacleSize =
        descriptor->m_pinnacleDescriptor.m_pinnacleSize;
    endState.m_pinnacle.m_seedRadius =
        descriptor->m_pinnacleDescriptor.m_seedRadius;
  }
  endState.m_stem.m_direction = descriptor->m_stemDescriptor.m_direction;
  endState.m_stem.m_length = descriptor->m_stemDescriptor.m_length;
  endState.m_stem.m_startWidth =
      descriptor->m_stemDescriptor.m_widthDistribution.GetValue(0) *
      descriptor->m_stemDescriptor.m_widthMax;
  endState.m_stem.m_endWidth =
      descriptor->m_stemDescriptor.m_widthDistribution.GetValue(1) *
      descriptor->m_stemDescriptor.m_widthMax;
  endState.m_leaves.resize(descriptor->m_l1LeafCount);
  for (int i = 0; i < descriptor->m_l1LeafCount; i++) {
    float step = static_cast<float>(i) /
                 (static_cast<float>(descriptor->m_l1LeafCount) - 1.0f);
    auto &leafState = endState.m_leaves[i];

    leafState.m_maxWaviness =
        descriptor->m_l1maxLeafWaviness *
        descriptor->m_l1LeafWavinessDistribution.GetValue(step);
    leafState.m_wavinessPeriod =
        descriptor->m_l1maxLeafWavinessPeriod *
        descriptor->m_l1LeafWavinessPeriodDistribution.GetValue(step);

    leafState.m_maxWidth = descriptor->m_l1LeafWidthMax *
                           descriptor->m_l1LeafWidthDistribution.GetValue(step);
    leafState.m_widthAlongLeafCurve =
        UniEngine::Curve(1.0f, 0.005f, {0, 0}, {1, 1});

    leafState.m_index = i;
    leafState.m_startingPoint = descriptor->m_l1FirstLeafStartingPoint +
                                step * (descriptor->m_l1LastLeafEndingPoint -
                                        descriptor->m_l1FirstLeafStartingPoint);
    leafState.m_length =
        descriptor->m_l1LeafLengthMax *
            descriptor->m_l1LeafLengthDistribution.GetValue(step) +
        glm::gaussRand(
            0.0f,
            descriptor->m_l1LeafLengthVarianceMax *
                descriptor->m_l1LeafLengthVarianceDistribution.GetValue(step));
    leafState.m_branchingAngle =
        descriptor->m_l1BranchingAngleMax *
            descriptor->m_l1BranchingAngleDistribution.GetValue(step) +
        glm::gaussRand(
            0.0f,
            descriptor->m_l1BranchingAngleVarianceMax *
                descriptor->m_l1BranchingAngleVarianceDistribution.GetValue(
                    step));
    leafState.m_rollAngle =
        i % 2 * 180.0f +
        glm::gaussRand(
            0.0f,
            descriptor->m_l1RollAngleVarianceMax *
                descriptor->m_l1RollAngleVarianceDistribution.GetValue(step));
    leafState.m_bending.x =
        descriptor->m_l1GravitropismMax *
            descriptor->m_l1GravitropismDistribution.GetValue(step) +
        glm::gaussRand(
            0.0f, descriptor->m_l1GravitropismVarianceMax *
                      descriptor->m_l1GravitropismVarianceDistribution.GetValue(
                          step));
    leafState.m_bending.y =
        descriptor->m_l1GravitropismFactorMax *
            descriptor->m_l1GravitropismFactorDistribution.GetValue(step) +
        glm::gaussRand(
            0.0f,
            descriptor->m_l1GravitropismFactorVarianceMax *
                descriptor->m_l1GravitropismFactorVarianceDistribution.GetValue(
                    step));
  }
  Set(endState);
}
void ProceduralSorghumGrowthDescriptor::Set(
    const ProceduralSorghumState &endState) {
  float endTime = endState.m_time;
  m_endTime = endTime;

  m_stem.m_endTime = endTime;
  m_stem.m_maxLength = endState.m_stem.m_length;
  m_stem.m_direction = endState.m_stem.m_direction;
  m_stem.m_maxStartWidth = endState.m_stem.m_startWidth;
  m_stem.m_maxEndWidth = endState.m_stem.m_endWidth;
  m_stem.m_lengthCurve = UniEngine::Curve(0.0f, 1.0f, {0, 0}, {1, 1});
  m_stem.m_startWidthCurve = UniEngine::Curve(0.0f, 1.0f, {0, 0}, {1, 1});
  m_stem.m_endWidthCurve = UniEngine::Curve(0.0f, 1.0f, {0, 0}, {1, 1});

  m_pinnacle.m_startTime = endTime - endTime / 5.0f;
  m_pinnacle.m_endTime = endTime;
  m_pinnacle.m_maxSeedRadius = endState.m_pinnacle.m_seedRadius;
  m_pinnacle.m_seedRadiusCurve = UniEngine::Curve(0.0f, 1.0f, {0, 0}, {1, 1});
  m_pinnacle.m_maxSeedAmount = endState.m_pinnacle.m_seedAmount;
  m_pinnacle.m_seedAmountCurve = UniEngine::Curve(0.0f, 1.0f, {0, 0}, {1, 1});
  m_pinnacle.m_maxPinnacleSize = endState.m_pinnacle.m_pinnacleSize;
  m_pinnacle.m_pinnacleSizeCurve = UniEngine::Curve(0.0f, 1.0f, {0, 0}, {1, 1});

  int leafSize = endState.m_leaves.size();
  m_leaves.resize(leafSize);

  float maxLength = 0.0f;
  int maxIndex = 0;
  float growthTimeForLongestLeaf = 0.9f;

  for (int i = 0; i < leafSize; i++) {
    auto &leaf = m_leaves[i];
    const auto &leafEndState = endState.m_leaves[i];
    leaf.m_maxLength = leafEndState.m_length;
    if (leaf.m_maxLength > maxLength) {
      maxIndex = i;
      maxLength = m_leaves[i].m_maxLength;
    }
  }
  float growthTimeForLastLeaf = growthTimeForLongestLeaf * m_leaves.back().m_maxLength / maxLength;
  float startingTimeStep = leafSize <= 1 ? 0.0f : (1.0f - growthTimeForLastLeaf) / (leafSize - 1);
  for (int i = 0; i < leafSize; i++) {
    float step = (i + 1.0f) / leafSize;
    auto &leaf = m_leaves[i];
    const auto &leafEndState = endState.m_leaves[i];

    leaf.m_startingPoint = glm::vec2(leafEndState.m_startingPoint);
    leaf.m_maxWidth = leafEndState.m_maxWidth;

    leaf.m_rollAngle = leafEndState.m_rollAngle;
    leaf.m_branchingAngle = leafEndState.m_branchingAngle;
    leaf.m_bending = leafEndState.m_bending;

    float growthTime = leaf.m_maxLength / maxLength * growthTimeForLongestLeaf * endTime;
    leaf.m_startTime = glm::clamp(startingTimeStep * (float)i * endTime, 0.0f, 1.0f);
    leaf.m_endTime = glm::clamp(leaf.m_startTime + growthTime, 0.0f, 1.0f);
  }
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
  if (m_widthAlongLeafCurve.CurveEditor("Width along leaves"))
    changed = true;

  if (ImGui::DragFloat("Max waviness", &m_maxWaviness, 0.01f, 0.0f, 999.0f))
    changed = true;
  if (m_wavinessCurve.CurveEditor("Waviness"))
    changed = true;
  if (ImGui::DragFloat("Waviness period", &m_wavinessPeriod, 0.01f, 0.0f,
                       999.0f))
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

  if (ImGui::DragFloat("Max start width", &m_maxStartWidth, 0.01f, 0.0f,
                       999.0f))
    changed = true;
  if (m_startWidthCurve.CurveEditor("Max start width"))
    changed = true;

  if (ImGui::DragFloat("Max end width", &m_maxEndWidth, 0.01f, 0.0f, 999.0f))
    changed = true;
  if (m_endWidthCurve.CurveEditor("Max end width"))
    changed = true;

  return changed;
}

void ProceduralSorghumGrowthDescriptor::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_endTime" << YAML::Value << m_endTime;
  out << YAML::Key << "m_seed" << YAML::Value << m_seed;

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

void ProceduralSorghumGrowthDescriptor::Deserialize(const YAML::Node &in) {
  if (in["m_endTime"])
    m_endTime = in["m_endTime"].as<float>();
  if (in["m_seed"])
    m_seed = in["m_seed"].as<unsigned>();

  if (in["m_pinnacle"])
    m_pinnacle.Deserialize(in["m_pinnacle"]);

  if (in["m_stem"])
    m_stem.Deserialize(in["m_stem"]);

  if (in["m_leaves"]) {
    for (const auto &i : in["m_leaves"]) {
      ProceduralLeafGrowthDescriptor leaf;
      leaf.Deserialize(i);
      m_leaves.push_back(leaf);
    }
  }
}

void ProceduralPinnacleGrowthDescriptor::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_startTime" << YAML::Value << m_startTime;
  out << YAML::Key << "m_endTime" << YAML::Value << m_endTime;
  out << YAML::Key << "m_maxPinnacleSize" << YAML::Value << m_maxPinnacleSize;
  out << YAML::Key << "m_maxSeedAmount" << YAML::Value << m_maxSeedAmount;
  out << YAML::Key << "m_maxSeedRadius" << YAML::Value << m_maxSeedRadius;

  out << YAML::Key << "m_pinnacleSizeCurve" << YAML::Value << YAML::BeginMap;
  m_pinnacleSizeCurve.Serialize(out);
  out << YAML::EndMap;
  out << YAML::Key << "m_seedAmountCurve" << YAML::Value << YAML::BeginMap;
  m_seedAmountCurve.Serialize(out);
  out << YAML::EndMap;
  out << YAML::Key << "m_seedRadiusCurve" << YAML::Value << YAML::BeginMap;
  m_seedRadiusCurve.Serialize(out);
  out << YAML::EndMap;
}

void ProceduralPinnacleGrowthDescriptor::Deserialize(const YAML::Node &in) {
  if (in["m_startTime"])
    m_startTime = in["m_startTime"].as<float>();
  if (in["m_endTime"])
    m_endTime = in["m_endTime"].as<float>();
  if (in["m_maxPinnacleSize"])
    m_maxPinnacleSize = in["m_maxPinnacleSize"].as<glm::vec3>();
  if (in["m_maxSeedAmount"])
    m_maxSeedAmount = in["m_maxSeedAmount"].as<int>();
  if (in["m_maxSeedRadius"])
    m_maxSeedRadius = in["m_maxSeedRadius"].as<float>();

  if (in["m_pinnacleSizeCurve"])
    m_pinnacleSizeCurve.Deserialize(in["m_pinnacleSizeCurve"]);
  if (in["m_seedAmountCurve"])
    m_seedAmountCurve.Deserialize(in["m_seedAmountCurve"]);
  if (in["m_seedRadiusCurve"])
    m_seedRadiusCurve.Deserialize(in["m_seedRadiusCurve"]);
}

void ProceduralStemGrowthDescriptor::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_endTime" << YAML::Value << m_endTime;
  out << YAML::Key << "m_direction" << YAML::Value << m_direction;
  out << YAML::Key << "m_maxLength" << YAML::Value << m_maxLength;
  out << YAML::Key << "m_maxStartWidth" << YAML::Value << m_maxStartWidth;
  out << YAML::Key << "m_maxEndWidth" << YAML::Value << m_maxEndWidth;

  out << YAML::Key << "m_lengthCurve" << YAML::Value << YAML::BeginMap;
  m_lengthCurve.Serialize(out);
  out << YAML::EndMap;
  out << YAML::Key << "m_startWidthCurve" << YAML::Value << YAML::BeginMap;
  m_startWidthCurve.Serialize(out);
  out << YAML::EndMap;
  out << YAML::Key << "m_endWidthCurve" << YAML::Value << YAML::BeginMap;
  m_endWidthCurve.Serialize(out);
  out << YAML::EndMap;
}

void ProceduralStemGrowthDescriptor::Deserialize(const YAML::Node &in) {
  if (in["m_endTime"])
    m_endTime = in["m_endTime"].as<float>();
  if (in["m_direction"])
    m_direction = in["m_direction"].as<glm::vec3>();
  if (in["m_maxLength"])
    m_maxLength = in["m_maxLength"].as<float>();
  if (in["m_maxStartWidth"])
    m_maxStartWidth = in["m_maxStartWidth"].as<float>();
  if (in["m_maxEndWidth"])
    m_maxEndWidth = in["m_maxEndWidth"].as<float>();

  if (in["m_lengthCurve"])
    m_lengthCurve.Deserialize(in["m_lengthCurve"]);
  if (in["m_startWidthCurve"])
    m_startWidthCurve.Deserialize(in["m_startWidthCurve"]);
  if (in["m_endWidthCurve"])
    m_endWidthCurve.Deserialize(in["m_endWidthCurve"]);
}
void ProceduralLeafGrowthDescriptor::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_startTime" << YAML::Value << m_startTime;
  out << YAML::Key << "m_endTime" << YAML::Value << m_endTime;
  out << YAML::Key << "m_startingPoint" << YAML::Value << m_startingPoint;
  out << YAML::Key << "m_maxLength" << YAML::Value << m_maxLength;
  out << YAML::Key << "m_maxWidth" << YAML::Value << m_maxWidth;
  out << YAML::Key << "m_rollAngle" << YAML::Value << m_rollAngle;
  out << YAML::Key << "m_branchingAngle" << YAML::Value << m_branchingAngle;
  out << YAML::Key << "m_bending" << YAML::Value << m_bending;

  out << YAML::Key << "m_startingPointCurve" << YAML::Value << YAML::BeginMap;
  m_startingPointCurve.Serialize(out);
  out << YAML::EndMap;

  out << YAML::Key << "m_lengthCurve" << YAML::Value << YAML::BeginMap;
  m_lengthCurve.Serialize(out);
  out << YAML::EndMap;

  out << YAML::Key << "m_widthCurve" << YAML::Value << YAML::BeginMap;
  m_widthCurve.Serialize(out);
  out << YAML::EndMap;
  out << YAML::Key << "m_widthAlongLeafCurve" << YAML::Value << YAML::BeginMap;
  m_widthAlongLeafCurve.Serialize(out);
  out << YAML::EndMap;

  out << YAML::Key << "m_maxWaviness" << YAML::Value << m_maxWaviness;
  out << YAML::Key << "m_wavinessCurve" << YAML::Value << YAML::BeginMap;
  m_wavinessCurve.Serialize(out);
  out << YAML::EndMap;
  out << YAML::Key << "m_wavinessPeriod" << YAML::Value << m_wavinessPeriod;
}

void ProceduralLeafGrowthDescriptor::Deserialize(const YAML::Node &in) {
  if (in["m_startTime"])
    m_startTime = in["m_startTime"].as<float>();
  if (in["m_endTime"])
    m_endTime = in["m_endTime"].as<float>();
  if (in["m_startingPoint"])
    m_startingPoint = in["m_startingPoint"].as<glm::vec2>();
  if (in["m_maxLength"])
    m_maxLength = in["m_maxLength"].as<float>();
  if (in["m_maxWidth"])
    m_maxWidth = in["m_maxWidth"].as<float>();
  if (in["m_rollAngle"])
    m_rollAngle = in["m_rollAngle"].as<float>();
  if (in["m_branchingAngle"])
    m_branchingAngle = in["m_branchingAngle"].as<float>();
  if (in["m_bending"])
    m_bending = in["m_bending"].as<glm::vec2>();

  if (in["m_startingPointCurve"])
    m_startingPointCurve.Deserialize(in["m_startingPointCurve"]);
  if (in["m_lengthCurve"])
    m_lengthCurve.Deserialize(in["m_lengthCurve"]);
  if (in["m_widthCurve"])
    m_widthCurve.Deserialize(in["m_widthCurve"]);
  if (in["m_widthAlongLeafCurve"])
    m_widthAlongLeafCurve.Deserialize(in["m_widthAlongLeafCurve"]);

  if (in["m_maxWaviness"])
    m_maxWaviness = in["m_maxWaviness"].as<float>();
  if (in["m_wavinessCurve"])
    m_wavinessCurve.Deserialize(in["m_wavinessCurve"]);
  if (in["m_wavinessPeriod"])
    m_wavinessPeriod = in["m_wavinessPeriod"].as<float>();
}
glm::vec3 ProceduralStemGrowthState::GetPoint(float point) const {
  return m_direction * point * m_length;
}
