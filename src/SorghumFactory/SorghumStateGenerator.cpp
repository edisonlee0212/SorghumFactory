#include "AssetManager.hpp"
#include "SorghumLayer.hpp"
#include <SorghumStateGenerator.hpp>
#include <rapidxml.hpp>
#include <rapidxml_print.hpp>
#include <rapidxml_utils.hpp>
using namespace SorghumFactory;
SorghumStateGenerator::SorghumStateGenerator() {
  m_hasPinnacle = false;
  m_pinnacleSize.m_mean = glm::vec3(0.2, 0.75, 0.2);
  m_pinnacleSeedAmount.m_mean = 1200;
  m_pinnacleSeedRadius.m_mean = 0.02f;

  m_stemDirection = {0, 1, 0};
  m_stemLength.m_mean = 0.875f;
  m_stemWidth.m_mean = 0.015f;

  m_leafAmount.m_mean = 20.0f;
  m_firstLeafStartingPoint.m_mean = 0.15f;
  m_lastLeafEndingPoint.m_mean = 1.0f;

  m_leafRollAngle.m_mean = {-1.0f, 1.0f,
                            UniEngine::Curve(0.5f, 0.5f, {0, 0}, {1, 1})};
  m_leafRollAngle.m_deviation = {0.0f, 0.0f,
                                 UniEngine::Curve(0.1f, 1.0f, {0, 0}, {1, 1})};

  m_leafBranchingAngle.m_mean = {0.0f, 50.0f,
                                 UniEngine::Curve(0.5f, 0.1f, {0, 0}, {1, 1})};
  m_leafBranchingAngle.m_deviation = {
      0.0f, 0.0f, UniEngine::Curve(0.0f, 1.0f, {0, 0}, {1, 1})};

  m_leafBending.m_mean = {0.0f, 4.0f,
                          UniEngine::Curve(0.5f, 0.5f, {0, 0}, {1, 1})};
  m_leafBending.m_deviation = {0.0f, 0.0f,
                               UniEngine::Curve(0.5f, 0.5f, {0, 0}, {1, 1})};

  m_leafBendingAcceleration.m_mean = {
      0.0f, 1.0f, UniEngine::Curve(0.5f, 0.5f, {0, 0}, {1, 1})};
  m_leafBendingAcceleration.m_deviation = {
      0.0f, 0.0f, UniEngine::Curve(0.5f, 0.5f, {0, 0}, {1, 1})};

  m_leafWaviness.m_mean = {0.0f, 20.0f,
                           UniEngine::Curve(0.5f, 0.5f, {0, 0}, {1, 1})};
  m_leafWaviness.m_deviation = {0.0f, 0.0f,
                                UniEngine::Curve(0.5f, 0.5f, {0, 0}, {1, 1})};

  m_leafWavinessFrequency.m_mean = {
      0.0f, 1.0f, UniEngine::Curve(0.5f, 0.5f, {0, 0}, {1, 1})};
  m_leafWavinessFrequency.m_deviation = {
      0.0f, 0.0f, UniEngine::Curve(0.5f, 0.5f, {0, 0}, {1, 1})};

  m_leafLength.m_mean = {0.0f, 2.55f,
                         UniEngine::Curve(0.333, 0.247, {0, 0}, {1, 1})};
  m_leafLength.m_deviation = {0.0f, 0.0f,
                              UniEngine::Curve(0.5f, 0.5f, {0, 0}, {1, 1})};

  m_leafWidth.m_mean = {0.0f, 0.035f,
                        UniEngine::Curve(0.5f, 0.5f, {0, 0}, {1, 1})};
  m_leafWidth.m_deviation = {0.0f, 0.0f,
                             UniEngine::Curve(0.5f, 0.5f, {0, 0}, {1, 1})};

  m_widthAlongStem = UniEngine::Curve(1.0f, 0.1f, {0, 0}, {1, 1});
  m_widthAlongLeaf = UniEngine::Curve(1.0f, 0.1f, {0, 0}, {1, 1});
  m_wavinessAlongLeaf = UniEngine::Curve(0.5f, 0.5f, {0, 0}, {1, 1});
}
void SorghumStateGenerator::OnInspect() {
  bool changed = ImGui::Checkbox("Pinnacle", &m_hasPinnacle);
  if (m_hasPinnacle) {
    if (ImGui::TreeNodeEx("Pinnacle settings", ImGuiTreeNodeFlags_DefaultOpen)) {
      if (m_pinnacleSize.OnInspect("Size"))
        changed = true;
      if (m_pinnacleSeedAmount.OnInspect("Seed amount"))
        changed = true;
      if (m_pinnacleSeedRadius.OnInspect("Seed radius"))
        changed = true;
      ImGui::TreePop();
    }
  }
  if (ImGui::TreeNodeEx("Stem settings", ImGuiTreeNodeFlags_DefaultOpen)) {
    if (ImGui::DragFloat3("Direction", &m_stemDirection.x))
      changed = true;
    if (m_stemLength.OnInspect("Length"))
      changed = true;
    if (m_stemWidth.OnInspect("Width"))
      changed = true;
    if (ImGui::TreeNode("Stem Details")) {
      if (m_widthAlongStem.OnInspect("Width along stem"))
        changed = true;
      ImGui::TreePop();
    }
    ImGui::TreePop();
  }
  if (ImGui::TreeNodeEx("Leaves settings", ImGuiTreeNodeFlags_DefaultOpen)) {
    if (m_leafAmount.OnInspect("Num of leaves"))
      changed = true;
    if (m_firstLeafStartingPoint.OnInspect("Starting point"))
      changed = true;
    if (m_lastLeafEndingPoint.OnInspect("Ending point"))
      changed = true;
    if (m_leafRollAngle.OnInspect("Roll angle"))
      changed = true;
    if (m_leafBranchingAngle.OnInspect("Branching angle"))
      changed = true;
    if (m_leafBending.OnInspect("Bending"))
      changed = true;
    if (m_leafBendingAcceleration.OnInspect("Bending acceleration"))
      changed = true;
    if (m_leafWaviness.OnInspect("Waviness"))
      changed = true;
    if (m_leafWavinessFrequency.OnInspect("Waviness Frequency"))
      changed = true;
    if (m_leafLength.OnInspect("Length"))
      changed = true;
    if (m_leafWidth.OnInspect("Width"))
      changed = true;
    if (ImGui::TreeNode("Leaf Details")) {
      if (m_widthAlongLeaf.OnInspect("Width along leaf"))
        changed = true;
      if (m_wavinessAlongLeaf.OnInspect("Waviness along leaf"))
        changed = true;
      ImGui::TreePop();
    }
    ImGui::TreePop();
  }
  if (changed) {
    m_saved = false;
    m_version++;
  }
  if (ImGui::Button("Instantiate")) {
    Application::GetLayer<SorghumLayer>()->CreateSorghum(
        AssetManager::Get<SorghumStateGenerator>(GetHandle()));
  }
}
void SorghumStateGenerator::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_version" << YAML::Value << m_version;

  out << YAML::Key << "m_hasPinnacle" << YAML::Value << m_hasPinnacle;
  m_pinnacleSize.Serialize("m_pinnacleSize", out);
  m_pinnacleSeedAmount.Serialize("m_pinnacleSeedAmount", out);
  m_pinnacleSeedRadius.Serialize("m_pinnacleSeedRadius", out);

  out << YAML::Key << "m_stemDirection" << YAML::Value << m_stemDirection;
  m_stemLength.Serialize("m_stemLength", out);
  m_stemWidth.Serialize("m_stemWidth", out);

  m_leafAmount.Serialize("m_leafAmount", out);
  m_firstLeafStartingPoint.Serialize("m_firstLeafStartingPoint", out);
  m_lastLeafEndingPoint.Serialize("m_lastLeafEndingPoint", out);
  m_leafRollAngle.Serialize("m_leafRollAngle", out);
  m_leafBranchingAngle.Serialize("m_leafBranchingAngle", out);
  m_leafBending.Serialize("m_leafBending", out);
  m_leafBendingAcceleration.Serialize("m_leafBendingAcceleration", out);
  m_leafWaviness.Serialize("m_leafWaviness", out);
  m_leafWavinessFrequency.Serialize("m_leafWavinessFrequency", out);
  m_leafLength.Serialize("m_leafLength", out);
  m_leafWidth.Serialize("m_leafWidth", out);

  m_widthAlongStem.UniEngine::ISerializable::Serialize("m_widthAlongStem", out);
  m_widthAlongLeaf.UniEngine::ISerializable::Serialize("m_widthAlongLeaf", out);
  m_wavinessAlongLeaf.UniEngine::ISerializable::Serialize("m_wavinessAlongLeaf", out);
}
void SorghumStateGenerator::Deserialize(const YAML::Node &in) {
  if (in["m_version"])
    m_version = in["m_version"].as<unsigned>();

  if (in["m_hasPinnacle"])
    m_hasPinnacle = in["m_hasPinnacle"].as<bool>();
  m_pinnacleSize.Deserialize("m_pinnacleSize", in);
  m_pinnacleSeedAmount.Deserialize("m_pinnacleSeedAmount", in);
  m_pinnacleSeedRadius.Deserialize("m_pinnacleSeedRadius", in);

  if (in["m_stemDirection"])
    m_stemDirection = in["m_stemDirection"].as<glm::vec3>();
  m_stemLength.Deserialize("m_stemLength", in);
  m_stemWidth.Deserialize("m_stemWidth", in);

  m_leafAmount.Deserialize("m_leafAmount", in);
  m_firstLeafStartingPoint.Deserialize("m_firstLeafStartingPoint", in);
  m_lastLeafEndingPoint.Deserialize("m_lastLeafEndingPoint", in);

  m_leafRollAngle.Deserialize("m_leafRollAngle", in);
  m_leafBranchingAngle.Deserialize("m_leafBranchingAngle", in);
  m_leafBendingAcceleration.Deserialize("m_leafBendingAcceleration", in);
  m_leafWaviness.Deserialize("m_leafWaviness", in);
  m_leafWavinessFrequency.Deserialize("m_leafWavinessFrequency", in);
  m_leafLength.Deserialize("m_leafLength", in);
  m_leafWidth.Deserialize("m_leafWidth", in);

  m_widthAlongStem.UniEngine::ISerializable::Deserialize("m_widthAlongStem", in);
  m_widthAlongLeaf.UniEngine::ISerializable::Deserialize("m_widthAlongLeaf", in);
  m_wavinessAlongLeaf.UniEngine::ISerializable::Deserialize("m_wavinessAlongLeaf", in);
}
ProceduralSorghumState SorghumStateGenerator::Generate(unsigned int seed) {
  srand(seed);
  ProceduralSorghumState endState = {};
  endState.m_pinnacle.m_active = m_hasPinnacle;
  if (endState.m_pinnacle.m_active) {
    endState.m_pinnacle.m_seedAmount = m_pinnacleSeedAmount.GetValue();
    endState.m_pinnacle.m_pinnacleSize = m_pinnacleSize.GetValue();
    endState.m_pinnacle.m_seedRadius = m_pinnacleSeedRadius.GetValue();
  }
  endState.m_stem.m_direction = m_stemDirection;
  endState.m_stem.m_length = m_stemLength.GetValue();
  endState.m_stem.m_widthAlongStem = {
      0.0f, m_stemWidth.GetValue(),
      m_widthAlongStem};
  int leafSize = glm::clamp(m_leafAmount.GetValue(), 2.0f, 128.0f);
  endState.m_leaves.resize(leafSize);
  for (int i = 0; i < leafSize; i++) {
    float step =
        static_cast<float>(i) / (static_cast<float>(leafSize) - 1.0f);
    auto &leafState = endState.m_leaves[i];
    leafState.m_active = true;
    leafState.m_index = i;
    auto firstLeafStartingPoint = m_firstLeafStartingPoint.GetValue();
    leafState.m_distanceToRoot =
        endState.m_stem.m_length *
        (firstLeafStartingPoint +
         step * (m_lastLeafEndingPoint.GetValue() - firstLeafStartingPoint));
    leafState.m_length = m_leafLength.GetValue(step);

    leafState.m_wavinessAlongLeaf = {
        0.0f, m_leafWaviness.GetValue(step) * 2.0f,
        m_wavinessAlongLeaf};
    leafState.m_wavinessFrequency = m_leafWavinessFrequency.GetValue(step);

    leafState.m_widthAlongLeaf = {
        0.0f, m_leafWidth.GetValue(step) * 2.0f,
        m_widthAlongLeaf};

    leafState.m_branchingAngle = m_leafBranchingAngle.GetValue(step);
    leafState.m_rollAngle = (i % 2) * 180.0f + m_leafRollAngle.GetValue(step);
    leafState.m_bending = {m_leafBending.GetValue(step),
                           m_leafBendingAcceleration.GetValue(step)};
  }
  return endState;
}
unsigned SorghumStateGenerator::GetVersion() const { return m_version; }
