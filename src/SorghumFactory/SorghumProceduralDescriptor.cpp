#include <SorghumProceduralDescriptor.hpp>
#include <rapidxml.hpp>
#include <rapidxml_print.hpp>
#include <rapidxml_utils.hpp>
using namespace SorghumFactory;
void SorghumProceduralDescriptor::OnInspect() {
  ImGui::DragInt("Cascade level index", &m_cascadeIndex, 0, 1);
  if (ImGui::Button("Apply L1")) {
    L1ToBase();
  }
  if (ImGui::TreeNodeEx("Level 1")) {
    if (ImGui::DragInt("Leaf count", &m_l1LeafCount))
      m_saved = false;
    if(ImGui::TreeNode("Leaf waviness")){
      if (ImGui::DragFloat("Max waviness", &m_l1maxLeafWaviness, 0.01f))
        m_saved = false;
      if (m_l1LeafWavinessDistribution.CurveEditor("Waviness curve"))
        m_saved = false;
      if (ImGui::DragFloat("Max waviness period", &m_l1maxLeafWavinessPeriod, 0.01f))
        m_saved = false;
      if (m_l1LeafWavinessPeriodDistribution.CurveEditor("Waviness period curve"))
        m_saved = false;
      ImGui::TreePop();
    }
    if (ImGui::TreeNode("Leaf width")) {
      if (ImGui::DragFloat("Max stem width", &m_l1StemWidthMax, 0.01f))
        m_saved = false;
      if (m_l1StemWidthDistribution.CurveEditor("Stem width curve"))
        m_saved = false;

      if (ImGui::DragFloat("Max leaf width", &m_l1LeafWidthMax, 0.01f))
        m_saved = false;
      if (m_l1LeafWidthDistribution.CurveEditor("Leaf width curve"))
        m_saved = false;

      if (m_l1LeafLengthDecreaseStartingPointDistribution.CurveEditor(
              "Leaf width decrease starting point"))
        m_saved = false;

      ImGui::TreePop();
    }

    if (ImGui::TreeNode("Leaf length")) {
      if (ImGui::DragFloat("Stem length", &m_l1StemLength, 0.01f))
        m_saved = false;
      if (ImGui::DragFloat("First Leaf Starting Point",
                           &m_l1FirstLeafStartingPoint, 0.01f))
        m_saved = false;

      if (ImGui::DragFloat("Max leaf length", &m_l1LeafLengthMax, 0.01f))
        m_saved = false;
      if (m_l1LeafLengthDistribution.CurveEditor("Leaf length curve"))
        m_saved = false;

      if (ImGui::DragFloat("Max leaf length variance",
                           &m_l1LeafLengthVarianceMax, 0.01f))
        m_saved = false;
      if (m_l1LeafLengthVarianceDistribution.CurveEditor(
              "Leaf length variance curve"))
        m_saved = false;

      ImGui::TreePop();
    }
    if (ImGui::TreeNode("Leaf angle")) {
      if (ImGui::DragFloat("Max branching angle", &m_l1BranchingAngleMax,
                           0.01f))
        m_saved = false;
      if (m_l1BranchingAngleDistribution.CurveEditor("Branching angle curve"))
        m_saved = false;

      if (ImGui::DragFloat("Max branching angle variance",
                           &m_l1BranchingAngleVarianceMax, 0.01f))
        m_saved = false;
      if (m_l1BranchingAngleVarianceDistribution.CurveEditor(
              "Branching angle variance curve"))
        m_saved = false;

      if (ImGui::DragFloat("Max roll angle variance", &m_l1RollAngleVarianceMax,
                           0.01f))
        m_saved = false;
      if (m_l1RollAngleVarianceDistribution.CurveEditor(
              "Roll angle variance curve"))
        m_saved = false;

      ImGui::TreePop();
    }

    if (ImGui::TreeNode("Leaf bending")) {
      if (ImGui::DragFloat("Max bending", &m_l1GravitropismMax, 0.01f))
        m_saved = false;
      if (m_l1GravitropismDistribution.CurveEditor("Bending curve"))
        m_saved = false;

      if (ImGui::DragFloat("Max bending variance", &m_l1GravitropismVarianceMax,
                           0.01f))
        m_saved = false;
      if (m_l1GravitropismVarianceDistribution.CurveEditor(
              "Bending variance curve"))
        m_saved = false;

      if (ImGui::DragFloat("Max bending increase", &m_l1GravitropismFactorMax,
                           0.01f))
        m_saved = false;
      if (m_l1GravitropismFactorDistribution.CurveEditor(
              "Bending increase curve"))
        m_saved = false;

      if (ImGui::DragFloat("Max bending increase variance",
                           &m_l1GravitropismFactorVarianceMax, 0.01f))
        m_saved = false;
      if (m_l1GravitropismFactorVarianceDistribution.CurveEditor(
              "Bending increase variance curve"))
        m_saved = false;

      ImGui::TreePop();
    }
    ImGui::TreePop();
  }
  if (ImGui::TreeNodeEx("Base")) {
    if (ImGui::TreeNodeEx("Stem", ImGuiTreeNodeFlags_DefaultOpen)) {
      if (m_stemDescriptor.OnInspect())
        m_saved = false;
      ImGui::TreePop();
    }
    if (ImGui::TreeNodeEx("Leaves", ImGuiTreeNodeFlags_DefaultOpen)) {
      for (int i = 0; i < m_leafDescriptors.size(); i++) {
        if (ImGui::TreeNodeEx(("No. " + std::to_string(i)).c_str())) {
          if (m_leafDescriptors[i].OnInspect())
            m_saved = false;
          ImGui::TreePop();
        }
      }
      ImGui::TreePop();
    }
    ImGui::TreePop();
  }
}

void SorghumProceduralDescriptor::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_cascadeIndex" << YAML::Value << m_cascadeIndex;

  // L1
  out << YAML::Key << "m_l1maxLeafWaviness" << YAML::Value << m_l1maxLeafWaviness;
  out << YAML::Key << "m_l1LeafWavinessDistribution" << YAML::Value
      << YAML::BeginMap;
  m_l1LeafWavinessDistribution.Serialize(out);
  out << YAML::EndMap;

  out << YAML::Key << "m_l1maxLeafWavinessPeriod" << YAML::Value << m_l1maxLeafWavinessPeriod;
  out << YAML::Key << "m_l1LeafWavinessPeriodDistribution" << YAML::Value
      << YAML::BeginMap;
  m_l1LeafWavinessPeriodDistribution.Serialize(out);
  out << YAML::EndMap;

  out << YAML::Key << "m_l1StemWidthMax" << YAML::Value << m_l1StemWidthMax;
  out << YAML::Key << "m_l1StemWidthDistribution" << YAML::Value
      << YAML::BeginMap;
  m_l1StemWidthDistribution.Serialize(out);
  out << YAML::EndMap;

  out << YAML::Key << "m_l1LeafWidthMax" << YAML::Value << m_l1LeafWidthMax;
  out << YAML::Key << "m_l1LeafWidthDistribution" << YAML::Value
      << YAML::BeginMap;
  m_l1LeafWidthDistribution.Serialize(out);
  out << YAML::EndMap;

  out << YAML::Key << "m_l1LeafLengthDecreaseStartingPointDistribution"
      << YAML::Value << YAML::BeginMap;
  m_l1LeafLengthDecreaseStartingPointDistribution.Serialize(out);
  out << YAML::EndMap;

  out << YAML::Key << "m_l1LeafCount" << YAML::Value << m_l1LeafCount;
  out << YAML::Key << "m_l1StemLength" << YAML::Value << m_l1StemLength;
  out << YAML::Key << "m_l1FirstLeafStartingPoint" << YAML::Value
      << m_l1FirstLeafStartingPoint;

  out << YAML::Key << "m_l1LeafLengthMax" << YAML::Value << m_l1LeafLengthMax;
  out << YAML::Key << "m_l1LeafLengthDistribution" << YAML::Value
      << YAML::BeginMap;
  m_l1LeafLengthDistribution.Serialize(out);
  out << YAML::EndMap;

  out << YAML::Key << "m_l1LeafLengthVarianceMax" << YAML::Value
      << m_l1LeafLengthVarianceMax;
  out << YAML::Key << "m_l1LeafLengthVarianceDistribution" << YAML::Value
      << YAML::BeginMap;
  m_l1LeafLengthVarianceDistribution.Serialize(out);
  out << YAML::EndMap;

  out << YAML::Key << "m_l1BranchingAngleMax" << YAML::Value
      << m_l1BranchingAngleMax;
  out << YAML::Key << "m_l1BranchingAngleDistribution" << YAML::Value
      << YAML::BeginMap;
  m_l1BranchingAngleDistribution.Serialize(out);
  out << YAML::EndMap;

  out << YAML::Key << "m_l1BranchingAngleVarianceMax" << YAML::Value
      << m_l1BranchingAngleVarianceMax;
  out << YAML::Key << "m_l1BranchingAngleVarianceDistribution" << YAML::Value
      << YAML::BeginMap;
  m_l1BranchingAngleVarianceDistribution.Serialize(out);
  out << YAML::EndMap;

  out << YAML::Key << "m_l1RollAngleVarianceMax" << YAML::Value
      << m_l1RollAngleVarianceMax;
  out << YAML::Key << "m_l1RollAngleVarianceDistribution" << YAML::Value
      << YAML::BeginMap;
  m_l1RollAngleVarianceDistribution.Serialize(out);
  out << YAML::EndMap;

  out << YAML::Key << "m_l1GravitropismMax" << YAML::Value
      << m_l1GravitropismMax;
  out << YAML::Key << "m_l1GravitropismDistribution" << YAML::Value
      << YAML::BeginMap;
  m_l1GravitropismDistribution.Serialize(out);
  out << YAML::EndMap;

  out << YAML::Key << "m_l1GravitropismVarianceMax" << YAML::Value
      << m_l1GravitropismVarianceMax;
  out << YAML::Key << "m_l1GravitropismVarianceDistribution" << YAML::Value
      << YAML::BeginMap;
  m_l1GravitropismVarianceDistribution.Serialize(out);
  out << YAML::EndMap;

  out << YAML::Key << "m_l1GravitropismFactorMax" << YAML::Value
      << m_l1GravitropismFactorMax;
  out << YAML::Key << "m_l1GravitropismFactorDistribution" << YAML::Value
      << YAML::BeginMap;
  m_l1GravitropismFactorDistribution.Serialize(out);
  out << YAML::EndMap;

  out << YAML::Key << "m_l1GravitropismFactorVarianceMax" << YAML::Value
      << m_l1GravitropismFactorVarianceMax;
  out << YAML::Key << "m_l1GravitropismFactorVarianceDistribution"
      << YAML::Value << YAML::BeginMap;
  m_l1GravitropismFactorVarianceDistribution.Serialize(out);
  out << YAML::EndMap;

  // Base
  out << YAML::Key << "m_stemDescriptor" << YAML::Value << YAML::BeginMap;
  m_stemDescriptor.Serialize(out);
  out << YAML::EndMap;
  out << YAML::Key << "m_leafDescriptors" << YAML::Value << YAML::BeginSeq;
  for (auto &i : m_leafDescriptors) {
    out << YAML::BeginMap;
    i.Serialize(out);
    out << YAML::EndMap;
  }
  out << YAML::EndSeq;
}
void SorghumProceduralDescriptor::Deserialize(const YAML::Node &in) {
  m_cascadeIndex = in["m_cascadeIndex"].as<int>();
  // L1
  m_l1maxLeafWaviness = in["m_l1maxLeafWaviness"].as<float>();
  m_l1LeafWavinessDistribution.Deserialize(in["m_l1LeafWavinessDistribution"]);
  m_l1maxLeafWavinessPeriod = in["m_l1maxLeafWavinessPeriod"].as<float>();
  m_l1LeafWavinessPeriodDistribution.Deserialize(in["m_l1LeafWavinessPeriodDistribution"]);


  m_l1StemWidthMax = in["m_l1StemWidthMax"].as<float>();
  m_l1StemWidthDistribution.Deserialize(in["m_l1StemWidthDistribution"]);
  m_l1LeafWidthMax = in["m_l1LeafWidthMax"].as<float>();
  m_l1LeafWidthDistribution.Deserialize(in["m_l1LeafWidthDistribution"]);
  m_l1LeafLengthDecreaseStartingPointDistribution.Deserialize(
      in["m_l1LeafLengthDecreaseStartingPointDistribution"]);

  m_l1LeafCount = in["m_l1LeafCount"].as<int>();
  m_l1StemLength = in["m_l1StemLength"].as<float>();
  m_l1FirstLeafStartingPoint = in["m_l1FirstLeafStartingPoint"].as<float>();

  m_l1LeafLengthMax = in["m_l1LeafLengthMax"].as<float>();
  m_l1LeafLengthDistribution.Deserialize(in["m_l1LeafLengthDistribution"]);

  m_l1LeafLengthVarianceMax = in["m_l1LeafLengthVarianceMax"].as<float>();
  m_l1LeafLengthVarianceDistribution.Deserialize(
      in["m_l1LeafLengthVarianceDistribution"]);

  m_l1BranchingAngleMax = in["m_l1BranchingAngleMax"].as<float>();
  m_l1BranchingAngleDistribution.Deserialize(
      in["m_l1BranchingAngleDistribution"]);

  m_l1BranchingAngleVarianceMax =
      in["m_l1BranchingAngleVarianceMax"].as<float>();
  m_l1BranchingAngleVarianceDistribution.Deserialize(
      in["m_l1BranchingAngleVarianceDistribution"]);

  m_l1RollAngleVarianceMax = in["m_l1RollAngleVarianceMax"].as<float>();
  m_l1RollAngleVarianceDistribution.Deserialize(
      in["m_l1RollAngleVarianceDistribution"]);

  m_l1GravitropismMax = in["m_l1GravitropismMax"].as<float>();
  m_l1GravitropismDistribution.Deserialize(in["m_l1GravitropismDistribution"]);

  m_l1GravitropismVarianceMax = in["m_l1GravitropismVarianceMax"].as<float>();
  m_l1GravitropismVarianceDistribution.Deserialize(
      in["m_l1GravitropismVarianceDistribution"]);

  m_l1GravitropismFactorMax = in["m_l1GravitropismFactorMax"].as<float>();
  m_l1GravitropismFactorDistribution.Deserialize(
      in["m_l1GravitropismFactorDistribution"]);

  m_l1GravitropismFactorVarianceMax =
      in["m_l1GravitropismFactorVarianceMax"].as<float>();
  m_l1GravitropismFactorVarianceDistribution.Deserialize(
      in["m_l1GravitropismFactorVarianceDistribution"]);

  // Base
  m_stemDescriptor.Deserialize(in["m_stemDescriptor"]);
  m_leafDescriptors.clear();
  if (in["m_leafDescriptors"]) {
    for (const auto &i : in["m_leafDescriptors"]) {
      m_leafDescriptors.push_back(SorghumLeafDescriptor());
      m_leafDescriptors.back().Deserialize(i);
    }
  }
}
SorghumProceduralDescriptor::SorghumProceduralDescriptor() {
  m_l1LeafWavinessDistribution =
      UniEngine::Curve(0.5, 0.5, {0, 0}, {1, 1});

  m_l1LeafWavinessPeriodDistribution =
      UniEngine::Curve(0.5, 0.5, {0, 0}, {1, 1});

  m_l1LeafLengthDistribution = UniEngine::Curve(0.333, 0.247, {0, 0}, {1, 1});

  m_l1LeafLengthVarianceDistribution =
      UniEngine::Curve(0.5, 0.5, {0, 0}, {1, 1});

  m_l1BranchingAngleDistribution = UniEngine::Curve(0.5, 0.1, {0, 0}, {1, 1});

  m_l1BranchingAngleVarianceDistribution =
      UniEngine::Curve(0.5, 0.5, {0, 0}, {1, 1});

  m_l1RollAngleVarianceDistribution = UniEngine::Curve(0.1, 1, {0, 0}, {1, 1});

  m_l1GravitropismDistribution = UniEngine::Curve(0.5, 0.0, {0, 0}, {1, 1});

  m_l1GravitropismVarianceDistribution =
      UniEngine::Curve(0.5, 0.5, {0, 0}, {1, 1});

  m_l1GravitropismFactorDistribution =
      UniEngine::Curve(0.9, 0.0, {0, 0}, {1, 1});

  m_l1GravitropismFactorVarianceDistribution =
      UniEngine::Curve(0.5, 0.5, {0, 0}, {1, 1});

  m_l1StemWidthDistribution = UniEngine::Curve(1.0, 0.4, {0, 0}, {1, 1});

  m_l1LeafWidthDistribution = UniEngine::Curve(1.0, 0.8, {0, 0}, {1, 1});

  m_l1LeafLengthDecreaseStartingPointDistribution =
      UniEngine::Curve(0.5, 0.5, {0, 0}, {1, 1});
}
void SorghumProceduralDescriptor::L1ToBase() {
  m_stemDescriptor = SorghumStemDescriptor();
  m_stemDescriptor.m_direction = glm::vec3(0, 1, 0);
  m_stemDescriptor.m_length = m_l1StemLength;
  m_stemDescriptor.m_stemWidth =
      m_l1StemWidthMax * m_l1StemWidthDistribution.GetValue(1.0f);
  m_leafDescriptors.resize(m_l1LeafCount);
  for (int i = 0; i < m_l1LeafCount; i++) {
    float step =
        static_cast<float>(i) / (static_cast<float>(m_l1LeafCount) - 1.0f);
    auto &leafDescriptor = m_leafDescriptors[i];
    leafDescriptor.m_waviness =
        m_l1maxLeafWaviness * m_l1LeafWavinessDistribution.GetValue(step);
    leafDescriptor.m_wavinessPeriod =
        m_l1maxLeafWavinessPeriod * m_l1LeafWavinessPeriodDistribution.GetValue(step);

    leafDescriptor.m_stemWidth =
        m_l1StemWidthMax * m_l1StemWidthDistribution.GetValue(step);
    leafDescriptor.m_leafMaxWidth =
        m_l1LeafWidthMax * m_l1LeafWidthDistribution.GetValue(step);
    leafDescriptor.m_leafWidthDecreaseStart =
        m_l1LeafLengthDecreaseStartingPointDistribution.GetValue(step);

    leafDescriptor.m_leafIndex = i;
    leafDescriptor.m_leafStartingPoint =
        m_l1FirstLeafStartingPoint + step * (1.0f - m_l1FirstLeafStartingPoint);
    leafDescriptor.m_leafLength =
        m_l1LeafLengthMax * m_l1LeafLengthDistribution.GetValue(step) +
        glm::gaussRand(0.0f,
                       m_l1LeafLengthVarianceMax *
                           m_l1LeafLengthVarianceDistribution.GetValue(step));
    leafDescriptor.m_branchingAngle =
        m_l1BranchingAngleMax * m_l1BranchingAngleDistribution.GetValue(step) +
        glm::gaussRand(
            0.0f, m_l1BranchingAngleVarianceMax *
                      m_l1BranchingAngleVarianceDistribution.GetValue(step));
    leafDescriptor.m_rollAngle =
        i % 2 * 180.0f +
        glm::gaussRand(0.0f,
                       m_l1RollAngleVarianceMax *
                           m_l1RollAngleVarianceDistribution.GetValue(step));
    leafDescriptor.m_gravitropism =
        m_l1GravitropismMax * m_l1GravitropismDistribution.GetValue(step) +
        glm::gaussRand(0.0f,
                       m_l1GravitropismVarianceMax *
                           m_l1GravitropismVarianceDistribution.GetValue(step));
    leafDescriptor.m_gravitropismFactor =
        m_l1GravitropismFactorMax *
            m_l1GravitropismFactorDistribution.GetValue(step) +
        glm::gaussRand(
            0.0f,
            m_l1GravitropismFactorVarianceMax *
                m_l1GravitropismFactorVarianceDistribution.GetValue(step));
  }
}
void SorghumProceduralDescriptor::Ready() {
  if (m_cascadeIndex == 1) {
    L1ToBase();
  }
}
void SorghumStemDescriptor::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_direction" << YAML::Value << m_direction;
  out << YAML::Key << "m_length" << YAML::Value << m_length;
  out << YAML::Key << "m_stemWidth" << YAML::Value << m_stemWidth;
}

void SorghumStemDescriptor::Deserialize(const YAML::Node &in) {
  m_direction = in["m_direction"].as<glm::vec3>();
  m_length = in["m_length"].as<float>();
  m_stemWidth = in["m_stemWidth"].as<float>();
}

bool SorghumStemDescriptor::OnInspect() {
  bool changed = false;
  changed = changed || ImGui::DragFloat3("Direction", &m_direction.x, 0.01f);
  changed = changed || ImGui::DragFloat("Length", &m_length, 0.01f);
  changed = changed || ImGui::DragFloat("Stem Width", &m_stemWidth, 0.01f);
  return changed;
}

bool SorghumLeafDescriptor::OnInspect() {
  bool changed = false;
  changed = changed || ImGui::DragFloat("Starting point", &m_leafStartingPoint, 0.01f);
  changed = changed || ImGui::DragFloat("Leaf Length", &m_leafLength, 0.01f);
  changed = changed || ImGui::DragFloat("Branching angle", &m_branchingAngle, 0.01f);
  changed = changed || ImGui::DragFloat("Roll angle", &m_rollAngle, 0.01f);
  changed = changed || ImGui::DragFloat("Gravitropism", &m_gravitropism, 0.01f);
  changed = changed || ImGui::DragFloat("Gravitropism increase", &m_gravitropismFactor, 0.01f);

  changed = changed || ImGui::DragFloat("Stem width", &m_stemWidth, 0.01f);
  changed = changed || ImGui::DragFloat("Max width", &m_leafMaxWidth, 0.01f);
  changed = changed || ImGui::DragFloat("Width decrease starting point", &m_leafWidthDecreaseStart,
                   0.01f);

  changed = changed || ImGui::DragFloat("Waviness period", &m_wavinessPeriod, 0.01f);
  changed = changed || ImGui::DragFloat("Waviness", &m_waviness, 0.01f);

  return changed;
}
void SorghumLeafDescriptor::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_leafIndex" << YAML::Value << m_leafIndex;
  out << YAML::Key << "m_leafStartingPoint" << YAML::Value
      << m_leafStartingPoint;
  out << YAML::Key << "m_leafLength" << YAML::Value << m_leafLength;
  out << YAML::Key << "m_branchingAngle" << YAML::Value << m_branchingAngle;
  out << YAML::Key << "m_rollAngle" << YAML::Value << m_rollAngle;
  out << YAML::Key << "m_gravitropism" << YAML::Value << m_gravitropism;
  out << YAML::Key << "m_gravitropismFactor" << YAML::Value
      << m_gravitropismFactor;

  out << YAML::Key << "m_stemWidth" << YAML::Value << m_stemWidth;
  out << YAML::Key << "m_leafMaxWidth" << YAML::Value << m_leafMaxWidth;
  out << YAML::Key << "m_leafWidthDecreaseStart" << YAML::Value
      << m_leafWidthDecreaseStart;

  out << YAML::Key << "m_wavinessPeriod" << YAML::Value << m_wavinessPeriod;
  out << YAML::Key << "m_waviness" << YAML::Value << m_waviness;
}
void SorghumLeafDescriptor::Deserialize(const YAML::Node &in) {
  m_leafIndex = in["m_leafIndex"].as<int>();
  m_leafStartingPoint = in["m_leafStartingPoint"].as<float>();
  m_leafLength = in["m_leafLength"].as<float>();
  m_branchingAngle = in["m_branchingAngle"].as<float>();
  m_rollAngle = in["m_rollAngle"].as<float>();
  m_gravitropism = in["m_gravitropism"].as<float>();
  m_gravitropismFactor = in["m_gravitropismFactor"].as<float>();

  m_stemWidth = in["m_stemWidth"].as<float>();
  m_leafMaxWidth = in["m_leafMaxWidth"].as<float>();
  m_leafWidthDecreaseStart = in["m_leafWidthDecreaseStart"].as<float>();

  if(in["m_wavinessPeriod"]) m_wavinessPeriod = in["m_wavinessPeriod"].as<float>();
  if(in["m_waviness"]) m_waviness = in["m_waviness"].as<float>();
}
