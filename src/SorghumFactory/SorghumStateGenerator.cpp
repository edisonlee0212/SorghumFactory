#include "AssetManager.hpp"
#include "SorghumLayer.hpp"
#include <SorghumStateGenerator.hpp>
#include <rapidxml.hpp>
#include <rapidxml_print.hpp>
#include <rapidxml_utils.hpp>
using namespace SorghumFactory;

void TipMenu(const std::string &content) {
  if (ImGui::IsItemHovered()) {
    ImGui::BeginTooltip();
    ImGui::TextUnformatted(content.c_str());
    ImGui::EndTooltip();
  }
}

void SorghumStateGenerator::OnInspect() {
  if (!m_saved) {
    ImGui::Text("Warning: Changed not saved!");
    TipMenu("Click \"Save\" above to save the changes!");
  }
  static bool intro = true;
  ImGui::Checkbox("Introduction", &intro);
  if (intro) {
    ImGui::TextWrapped(
        "This is the introduction of the parameter setting interface. "
        "\nFor each parameter, you are allowed to set average and "
        "variance value. \nInstantiate a new sorghum in the scene so you "
        "can preview the changes in real time. \nThe curve editors are "
        "provided for stem/leaf details to allow you have control of "
        "geometric properties along the stem/leaf. It's also provided "
        "for leaf settings to allow you control the distribution of "
        "different leaves from the bottom to top.\nMake sure you Save the "
        "parameters!\nValues are in meters or degrees.");
  }

  bool changed = ImGui::Checkbox("Pinnacle", &m_hasPinnacle);
  TipMenu("Whether the sorghum has pinnacles");
  if (m_hasPinnacle) {
    if (ImGui::TreeNodeEx("Pinnacle settings",
                          ImGuiTreeNodeFlags_DefaultOpen)) {
      TipMenu("The settings for pinnacle. The pinnacle will always be placed "
              "at the tip of the stem.");
      if (m_pinnacleSize.OnInspect("Size", 0.001f, "The size of pinnacle")) {
        changed = true;
      }
      if (m_pinnacleSeedAmount.OnInspect("Seed amount", 1.0f,
                                         "The amount of seeds in the pinnacle"))
        changed = true;
      if (m_pinnacleSeedRadius.OnInspect(
              "Seed radius", 0.001f, "The size of the seed in the pinnacle"))
        changed = true;
      ImGui::TreePop();
    }
  }
  if (ImGui::TreeNodeEx("Stem settings", ImGuiTreeNodeFlags_DefaultOpen)) {
    TipMenu("The settings for stem.");
    /*
    if (ImGui::DragFloat3("Direction", &m_stemDirection.x))
      changed = true;
      */
    if (m_stemLength.OnInspect(
            "Length", 0.01f,
            "The length of the stem, use Ending Point in leaf settings to make "
            "stem taller than top leaf for pinnacle"))
      changed = true;
    if (m_stemWidth.OnInspect("Width", 0.001f,
                              "The overall width of the stem, adjust the width "
                              "along stem in Stem Details"))
      changed = true;
    if (ImGui::TreeNode("Stem Details")) {
      TipMenu("The detailed settings for stem.");
      if (m_widthAlongStem.OnInspect("Width along stem"))
        changed = true;
      ImGui::TreePop();
    }
    ImGui::TreePop();
  }
  if (ImGui::TreeNodeEx("Leaves settings", ImGuiTreeNodeFlags_DefaultOpen)) {
    TipMenu("The settings for leaves.");
    if (m_leafAmount.OnInspect("Num of leaves", 1.0f,
                               "The total amount of leaves"))
      changed = true;

    static MixedDistributionSettings leafStartingPoint = {
        0.01f,
        {0.01f, false, true, ""},
        {0.01f, false, false, ""},
        "The starting point of each leaf along stem. Default each leaf "
        "located uniformly on stem."};

    if (m_leafStartingPoint.OnInspect("Starting point along stem",
                                      leafStartingPoint)) {
      changed = true;
    }
    static MixedDistributionSettings leafRollAngle = {
        0.01f,
        {},
        {},
        "The polar angle of leaf. Normally you should only change the "
        "deviation. Values are in degrees"};
    if (m_leafRollAngle.OnInspect("Roll angle", leafRollAngle))
      changed = true;

    static MixedDistributionSettings leafBranchingAngle = {
        0.01f,
        {},
        {},
        "The branching angle of the leaf. Values are in degrees"};
    if (m_leafBranchingAngle.OnInspect("Branching angle", leafBranchingAngle))
      changed = true;

    static MixedDistributionSettings leafBending = {
        0.01f,
        {},
        {},
        "The bending of the leaf, controls how leaves bend because of "
        "gravity. Positive value results in leaf bending towards the "
        "ground, negative value results in leaf bend towards the sky"};
    if (m_leafBending.OnInspect("Bending", leafBending))
      changed = true;

    static MixedDistributionSettings leafBendingAcceleration = {
        0.01f,
        {},
        {},
        "The changes of bending along the leaf. You can use this to create "
        "S-shaped leaves."};

    if (m_leafBendingAcceleration.OnInspect("Bending acceleration",
                                            leafBendingAcceleration))
      changed = true;

    if (m_leafWaviness.OnInspect("Waviness"))
      changed = true;
    if (m_leafWavinessFrequency.OnInspect("Waviness Frequency"))
      changed = true;
    if (m_leafPeriodStart.OnInspect("Waviness Period Start"))
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
  m_leafStartingPoint.Serialize("m_leafStartingPoint", out);
  m_leafRollAngle.Serialize("m_leafRollAngle", out);
  m_leafBranchingAngle.Serialize("m_leafBranchingAngle", out);
  m_leafBending.Serialize("m_leafBending", out);
  m_leafBendingAcceleration.Serialize("m_leafBendingAcceleration", out);
  m_leafWaviness.Serialize("m_leafWaviness", out);
  m_leafWavinessFrequency.Serialize("m_leafWavinessFrequency", out);
  m_leafPeriodStart.Serialize("m_leafPeriodStart", out);
  m_leafLength.Serialize("m_leafLength", out);
  m_leafWidth.Serialize("m_leafWidth", out);

  m_widthAlongStem.UniEngine::ISerializable::Serialize("m_widthAlongStem", out);
  m_widthAlongLeaf.UniEngine::ISerializable::Serialize("m_widthAlongLeaf", out);
  m_wavinessAlongLeaf.UniEngine::ISerializable::Serialize("m_wavinessAlongLeaf",
                                                          out);
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
  m_leafStartingPoint.Deserialize("m_leafStartingPoint", in);

  m_leafRollAngle.Deserialize("m_leafRollAngle", in);
  m_leafBranchingAngle.Deserialize("m_leafBranchingAngle", in);

  m_leafBending.Deserialize("m_leafBending", in);
  m_leafBendingAcceleration.Deserialize("m_leafBendingAcceleration", in);
  m_leafWaviness.Deserialize("m_leafWaviness", in);
  m_leafWavinessFrequency.Deserialize("m_leafWavinessFrequency", in);
  m_leafPeriodStart.Deserialize("m_leafPeriodStart", in);
  m_leafLength.Deserialize("m_leafLength", in);
  m_leafWidth.Deserialize("m_leafWidth", in);

  m_widthAlongStem.UniEngine::ISerializable::Deserialize("m_widthAlongStem",
                                                         in);
  m_widthAlongLeaf.UniEngine::ISerializable::Deserialize("m_widthAlongLeaf",
                                                         in);
  m_wavinessAlongLeaf.UniEngine::ISerializable::Deserialize(
      "m_wavinessAlongLeaf", in);
}
ProceduralSorghumState SorghumStateGenerator::Generate(unsigned int seed) {
  srand(seed);
  ProceduralSorghumState endState = {};

  endState.m_stem.m_direction = m_stemDirection;
  endState.m_stem.m_length = m_stemLength.GetValue();
  endState.m_stem.m_widthAlongStem = {0.0f, m_stemWidth.GetValue(),
                                      m_widthAlongStem};
  int leafSize = glm::clamp(m_leafAmount.GetValue(), 2.0f, 128.0f);
  endState.m_leaves.resize(leafSize);
  for (int i = 0; i < leafSize; i++) {
    float step = static_cast<float>(i) / (static_cast<float>(leafSize) - 1.0f);
    auto &leafState = endState.m_leaves[i];
    leafState.m_active = true;
    leafState.m_index = i;

    leafState.m_distanceToRoot =
        endState.m_stem.m_length * m_leafStartingPoint.GetValue(step);
    leafState.m_length = m_leafLength.GetValue(step);

    leafState.m_wavinessAlongLeaf = {0.0f, m_leafWaviness.GetValue(step) * 2.0f,
                                     m_wavinessAlongLeaf};
    leafState.m_wavinessFrequency.x = m_leafWavinessFrequency.GetValue(step);
    leafState.m_wavinessFrequency.y = m_leafWavinessFrequency.GetValue(step);

    leafState.m_wavinessPeriodStart.x = m_leafPeriodStart.GetValue(step);
    leafState.m_wavinessPeriodStart.y = m_leafPeriodStart.GetValue(step);

    leafState.m_widthAlongLeaf = {0.0f, m_leafWidth.GetValue(step) * 2.0f,
                                  m_widthAlongLeaf};

    leafState.m_branchingAngle = m_leafBranchingAngle.GetValue(step);
    leafState.m_rollAngle = (i % 2) * 180.0f + m_leafRollAngle.GetValue(step);
    leafState.m_bending = {m_leafBending.GetValue(step),
                           m_leafBendingAcceleration.GetValue(step)};
  }

  endState.m_pinnacle.m_active = m_hasPinnacle;
  if (endState.m_pinnacle.m_active) {
    endState.m_pinnacle.m_seedAmount = m_pinnacleSeedAmount.GetValue();
    endState.m_pinnacle.m_pinnacleSize = m_pinnacleSize.GetValue();
    endState.m_pinnacle.m_seedRadius = m_pinnacleSeedRadius.GetValue();
  }
  return endState;
}
unsigned SorghumStateGenerator::GetVersion() const { return m_version; }
void SorghumStateGenerator::OnCreate() {
  m_hasPinnacle = false;
  m_pinnacleSize.m_mean = glm::vec3(0.01, 0.1, 0.01);
  m_pinnacleSeedAmount.m_mean = 1200;
  m_pinnacleSeedRadius.m_mean = 0.002f;

  m_stemDirection = {0, 1, 0};
  m_stemLength.m_mean = 0.449999988f;
  m_stemLength.m_deviation = 0.150000006f;
  m_stemWidth.m_mean = 0.0140000004;
  m_stemWidth.m_deviation = 0.0f;

  m_leafAmount.m_mean = 9.0f;
  m_leafAmount.m_deviation = 1.0f;

  m_leafStartingPoint.m_mean = {0.0f, 1.0f,
                                UniEngine::Curve(0.1f, 1.0f, {0, 0}, {1, 1})};
  m_leafStartingPoint.m_deviation = {
      0.0f, 1.0f, UniEngine::Curve(0.0f, 0.0f, {0, 0}, {1, 1})};

  m_leafRollAngle.m_mean = {-1.0f, 1.0f,
                            UniEngine::Curve(0.5f, 0.5f, {0, 0}, {1, 1})};
  m_leafRollAngle.m_deviation = {0.0f, 6.0f,
                                 UniEngine::Curve(0.3f, 1.0f, {0, 0}, {1, 1})};

  m_leafBranchingAngle.m_mean = {0.0f, 55.0f,
                                 UniEngine::Curve(0.5f, 0.2f, {0, 0}, {1, 1})};
  m_leafBranchingAngle.m_deviation = {
      0.0f, 3.0f, UniEngine::Curve(0.67f, 0.225f, {0, 0}, {1, 1})};

  m_leafBending.m_mean = {0.0f, 4.0f,
                          UniEngine::Curve(0.2f, 0.2f, {0, 0}, {1, 1})};
  m_leafBending.m_deviation = {0.0f, 0.0f,
                               UniEngine::Curve(0.5f, 0.5f, {0, 0}, {1, 1})};

  m_leafBendingAcceleration.m_mean = {
      -1.0f, 2.0f, UniEngine::Curve(0.72f, 0.28f, {0, 0}, {1, 1})};
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

  m_leafLength.m_mean = {0.0f, 2.5f,
                         UniEngine::Curve(0.165, 0.247, {0, 0}, {1, 1})};
  m_leafLength.m_deviation = {0.0f, 0.0f,
                              UniEngine::Curve(0.5f, 0.5f, {0, 0}, {1, 1})};

  m_leafWidth.m_mean = {0.0f, 0.075f,
                        UniEngine::Curve(0.5f, 0.5f, {0, 0}, {1, 1})};
  m_leafWidth.m_deviation = {0.0f, 0.0f,
                             UniEngine::Curve(0.5f, 0.5f, {0, 0}, {1, 1})};

  m_widthAlongStem = UniEngine::Curve(1.0f, 0.1f, {0, 0}, {1, 1});
  m_widthAlongLeaf = UniEngine::Curve(0.5f, 0.1f, {0, 0}, {1, 1});
  m_wavinessAlongLeaf = UniEngine::Curve(0.0f, 0.5f, {0, 0}, {1, 1});
}
