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

  m_leafAmount = 10;
  m_firstLeafStartingPoint.m_mean = 0.15f;
  m_lastLeafEndingPoint.m_mean = 1.0f;

  m_leafRollAngle.m_mean = {-1.0f, 1.0f, UniEngine::Curve(0.5f, 0.5f, {0, 0}, {1, 1})};
  m_leafRollAngle.m_deviation = {0.0f, 0.0f, UniEngine::Curve(0.1f, 1.0f, {0, 0}, {1, 1})};

  m_leafBranchingAngle.m_mean = {0.0f, 50.0f, UniEngine::Curve(0.5f, 0.1f, {0, 0}, {1, 1})};
  m_leafBranchingAngle.m_deviation = {0.0f, 0.0f, UniEngine::Curve(0.0f, 1.0f, {0, 0}, {1, 1})};

  m_leafBending.m_mean = {0.0f, 4.0f, UniEngine::Curve(0.5f, 0.5f, {0, 0}, {1, 1})};
  m_leafBending.m_deviation = {0.0f, 0.0f, UniEngine::Curve(0.5f, 0.5f, {0, 0}, {1, 1})};

  m_leafBendingAcceleration.m_mean = {0.0f, 1.0f, UniEngine::Curve(0.5f, 0.5f, {0, 0}, {1, 1})};
  m_leafBendingAcceleration.m_deviation = {0.0f, 0.0f, UniEngine::Curve(0.5f, 0.5f, {0, 0}, {1, 1})};

  m_leafWaviness.m_mean = {0.0f, 20.0f, UniEngine::Curve(0.5f, 0.5f, {0, 0}, {1, 1})};
  m_leafWaviness.m_deviation = {0.0f, 0.0f, UniEngine::Curve(0.5f, 0.5f, {0, 0}, {1, 1})};

  m_leafWavinessFrequency.m_mean = {0.0f, 1.0f, UniEngine::Curve(0.5f, 0.5f, {0, 0}, {1, 1})};
  m_leafWavinessFrequency.m_deviation = {0.0f, 0.0f, UniEngine::Curve(0.5f, 0.5f, {0, 0}, {1, 1})};

  m_leafLength.m_mean = {0.0f, 2.55f, UniEngine::Curve(0.333, 0.247, {0, 0}, {1, 1})};
  m_leafLength.m_deviation = {0.0f, 0.0f, UniEngine::Curve(0.5f, 0.5f, {0, 0}, {1, 1})};

  m_leafWidth.m_mean = {0.0f, 0.035f, UniEngine::Curve(0.5f, 0.5f, {0, 0}, {1, 1})};
  m_leafWidth.m_deviation = {0.0f, 0.0f, UniEngine::Curve(0.5f, 0.5f, {0, 0}, {1, 1})};
}
void SorghumStateGenerator::OnInspect() {
  m_saved = ImGui::Checkbox("Pinnacle", &m_hasPinnacle);
  if (m_hasPinnacle) {
    if (ImGui::TreeNode("Pinnacle settings")) {
      if(m_pinnacleSize.OnInspect("Size")) m_saved = false;
      if(m_pinnacleSeedAmount.OnInspect("Seed amount")) m_saved = false;
      if(m_pinnacleSeedRadius.OnInspect("Seed radius")) m_saved = false;
      ImGui::TreePop();
    }
  }
  if (ImGui::TreeNode("Stem settings")) {
    if(ImGui::DragFloat3("Direction", &m_stemDirection.x)) m_saved = false;
    if(m_stemLength.OnInspect("Length")) m_saved = false;
    if(m_stemWidth.OnInspect("Width")) m_saved = false;
    ImGui::TreePop();
  }
  if (ImGui::TreeNode("Leaves settings")) {
    if(ImGui::DragInt("Num of leaves", &m_leafAmount)) m_saved = false;
    if(m_firstLeafStartingPoint.OnInspect("Starting point")) m_saved = false;
    if(m_lastLeafEndingPoint.OnInspect("Ending point")) m_saved = false;

    if(m_leafRollAngle.OnInspect("Roll angle")) m_saved = false;
    if(m_leafBranchingAngle.OnInspect("Branching angle")) m_saved = false;
    if(m_leafBending.OnInspect("Bending")) m_saved = false;
    if(m_leafBendingAcceleration.OnInspect("Bending acceleration")) m_saved = false;
    if(m_leafWaviness.OnInspect("Waviness")) m_saved = false;
    if(m_leafWavinessFrequency.OnInspect("Waviness Frequency")) m_saved = false;
    if(m_leafLength.OnInspect("Length")) m_saved = false;
    if(m_leafWidth.OnInspect("Width")) m_saved = false;
    ImGui::TreePop();
  }
}
void SorghumStateGenerator::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_hasPinnacle" << YAML::Value << m_hasPinnacle;
  m_pinnacleSize.Serialize("m_pinnacleSize", out);
  m_pinnacleSeedAmount.Serialize("m_pinnacleSeedAmount", out);
  m_pinnacleSeedRadius.Serialize("m_pinnacleSeedRadius", out);

  out << YAML::Key << "m_stemDirection" << YAML::Value << m_stemDirection;
  m_stemLength.Serialize("m_stemLength", out);
  m_stemWidth.Serialize("m_stemWidth", out);

  out << YAML::Key << "m_leafAmount" << YAML::Value << m_leafAmount;
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
}
void SorghumStateGenerator::Deserialize(const YAML::Node &in) {
  if(in["m_hasPinnacle"]) m_hasPinnacle = in["m_hasPinnacle"].as<bool>();
  m_pinnacleSize.Deserialize("m_pinnacleSize", in);
  m_pinnacleSeedAmount.Deserialize("m_pinnacleSeedAmount", in);
  m_pinnacleSeedRadius.Deserialize("m_pinnacleSeedRadius", in);

  if(in["m_stemDirection"]) m_stemDirection = in["m_stemDirection"].as<glm::vec3>();
  m_stemLength.Deserialize("m_stemLength", in);
  m_stemWidth.Deserialize("m_stemWidth", in);

  if(in["m_leafAmount"]) m_leafAmount = in["m_leafAmount"].as<int>();
  m_firstLeafStartingPoint.Deserialize("m_firstLeafStartingPoint", in);
  m_lastLeafEndingPoint.Deserialize("m_lastLeafEndingPoint", in);

  m_leafRollAngle.Deserialize("m_leafRollAngle", in);
  m_leafBranchingAngle.Deserialize("m_leafBranchingAngle", in);
  m_leafBendingAcceleration.Deserialize("m_leafBendingAcceleration", in);
  m_leafWaviness.Deserialize("m_leafWaviness", in);
  m_leafWavinessFrequency.Deserialize("m_leafWavinessFrequency", in);
  m_leafLength.Deserialize("m_leafLength", in);
  m_leafWidth.Deserialize("m_leafWidth", in);
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
      UniEngine::Curve(1.0f, 0.1f, {0, 0}, {1, 1})};
  endState.m_leaves.resize(m_leafAmount);
  for (int i = 0; i < m_leafAmount; i++) {
    float step =
        static_cast<float>(i) / (static_cast<float>(m_leafAmount) - 1.0f);
    auto &leafState = endState.m_leaves[i];
    leafState.m_index = i;
    auto firstLeafStartingPoint = m_firstLeafStartingPoint.GetValue();
    leafState.m_distanceToRoot =
        endState.m_stem.m_length *
        (firstLeafStartingPoint +
         step * (m_lastLeafEndingPoint.GetValue() - firstLeafStartingPoint));
    leafState.m_length = m_leafLength.GetValue(step);

    leafState.m_wavinessAlongLeaf = {
        0.0f, m_leafWaviness.GetValue(step) * 2.0f,
        UniEngine::Curve(0.0f, 1.0f, {0, 0}, {1, 1})};
    leafState.m_wavinessFrequency = m_leafWavinessFrequency.GetValue(step);

    leafState.m_widthAlongLeaf = {
        0.0f, m_leafWidth.GetValue(step) * 2.0f,
        UniEngine::Curve(1.0f, 0.005f, {0, 0}, {1, 1})};

    leafState.m_branchingAngle = m_leafBranchingAngle.GetValue(step);
    leafState.m_rollAngle = (i % 2) * 180.0f + m_leafRollAngle.GetValue(step);
    leafState.m_bending = {m_leafBending.GetValue(step),
                           m_leafBendingAcceleration.GetValue(step)};
  }
  return endState;
}