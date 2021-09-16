#include <SorghumProceduralDescriptor.hpp>
#include <rapidxml.hpp>
#include <rapidxml_print.hpp>
#include <rapidxml_utils.hpp>
using namespace SorghumFactory;
void SorghumProceduralDescriptor::OnInspect() {
  if(ImGui::CollapsingHeader("L1")) {
    ImGui::Checkbox("Lock##L1", &m_l1Locked);
    if(ImGui::Button("Apply L1")){
      L1ToBase();
    }
    ImGui::DragInt("Leaf count", &m_l1LeafCount, 0.01f);
    ImGui::DragFloat("Stem length", &m_l1StemLength, 0.01f);
    ImGui::DragFloat("First Leaf Starting Point", &m_l1FirstLeafStartingPoint, 0.01f);

    ImGui::DragFloat("Max leaf length", &m_l1LeafLengthMax, 0.01f);
    m_l1LeafLengthDistribution.CurveEditor("Leaf length curve");

    ImGui::DragFloat("Max leaf length variance", &m_l1LeafLengthVarianceMax, 0.01f);
    m_l1LeafLengthVarianceDistribution.CurveEditor("Leaf length variance curve");

    ImGui::DragFloat("Max branching angle", &m_l1BranchingAngleMax, 0.01f);
    m_l1BranchingAngleDistribution.CurveEditor("Branching angle curve");

    ImGui::DragFloat("Max branching angle variance", &m_l1BranchingAngleVarianceMax, 0.01f);
    m_l1BranchingAngleVarianceDistribution.CurveEditor("Branching angle variance curve");

    ImGui::DragFloat("Max roll angle variance", &m_l1RollAngleVarianceMax, 0.01f);
    m_l1RollAngleVarianceDistribution.CurveEditor("Roll angle variance curve");

    ImGui::DragFloat("Max gravitropism", &m_l1GravitropismMax, 0.01f);
    m_l1GravitropismDistribution.CurveEditor("Gravitropism curve");

    ImGui::DragFloat("Max gravitropism variance", &m_l1GravitropismVarianceMax, 0.01f);
    m_l1GravitropismVarianceDistribution.CurveEditor("Gravitropism variance curve");

    ImGui::DragFloat("Max gravitropism increase", &m_l1GravitropismFactorMax, 0.01f);
    m_l1GravitropismFactorDistribution.CurveEditor("Gravitropism increase curve");

    ImGui::DragFloat("Max gravitropism increase variance", &m_l1GravitropismFactorVarianceMax, 0.01f);
    m_l1GravitropismFactorVarianceDistribution.CurveEditor("Gravitropism increase variance curve");
  }
  if(ImGui::CollapsingHeader("Base")){
    ImGui::Checkbox("Lock##Base", &m_baseLocked);
    if(ImGui::TreeNodeEx("Stem", ImGuiTreeNodeFlags_DefaultOpen)){
      m_stemDescriptor.OnInspect();
      ImGui::TreePop();
    }
    if(ImGui::TreeNodeEx("Leaves", ImGuiTreeNodeFlags_DefaultOpen)) {
      for (int i = 0; i < m_leafDescriptors.size(); i++) {
        if(ImGui::TreeNodeEx(("No. " + std::to_string(i)).c_str())){
          m_leafDescriptors[i].OnInspect();
          ImGui::TreePop();
        }
      }
      ImGui::TreePop();
    }
  }
}

void SorghumProceduralDescriptor::Serialize(YAML::Emitter &out) {

}
void SorghumProceduralDescriptor::Deserialize(const YAML::Node &in) {

}
SorghumProceduralDescriptor::SorghumProceduralDescriptor() {
  m_l1LeafLengthDistribution = UniEngine::Curve(0.5, 0.5, {0, 0}, {1, 1});

  m_l1LeafLengthVarianceDistribution = UniEngine::Curve(0.5, 0.5, {0, 0}, {1, 1});

  m_l1BranchingAngleDistribution = UniEngine::Curve(0.5, 0.5, {0, 0}, {1, 1});

  m_l1BranchingAngleVarianceDistribution = UniEngine::Curve(0.5, 0.5, {0, 0}, {1, 1});

  m_l1RollAngleVarianceDistribution = UniEngine::Curve(0.5, 0.5, {0, 0}, {1, 1});

  m_l1GravitropismDistribution = UniEngine::Curve(0.5, 0.5, {0, 0}, {1, 1});

  m_l1GravitropismVarianceDistribution = UniEngine::Curve(0.5, 0.5, {0, 0}, {1, 1});

  m_l1GravitropismFactorDistribution = UniEngine::Curve(0.5, 0.5, {0, 0}, {1, 1});

  m_l1GravitropismFactorVarianceDistribution = UniEngine::Curve(0.5, 0.5, {0, 0}, {1, 1});
}
void SorghumProceduralDescriptor::L1ToBase() {
  if(m_baseLocked){
    UNIENGINE_ERROR("Base parameter locked! Unlock and retry.");
    return;
  }

  m_stemDescriptor = SorghumStemDescriptor();
  m_stemDescriptor.m_direction = glm::vec3(0, 1, 0);
  m_stemDescriptor.m_length = m_l1StemLength;
  m_leafDescriptors.resize(m_l1LeafCount);
  for(int i = 0; i < m_l1LeafCount; i++){
    float step = static_cast<float>(i) / (static_cast<float>(m_l1LeafCount) - 1.0f);
    auto& leafDescriptor = m_leafDescriptors[i];
    leafDescriptor.m_leafIndex = i;
    leafDescriptor.m_leafStartingPoint = m_l1FirstLeafStartingPoint + step * (1.0f - m_l1FirstLeafStartingPoint);
    leafDescriptor.m_leafLength = m_l1LeafLengthMax * m_l1LeafLengthDistribution.GetValue(step) + glm::gaussRand(0.0f, m_l1LeafLengthVarianceMax * m_l1LeafLengthVarianceDistribution.GetValue(step));
    leafDescriptor.m_branchingAngle = m_l1BranchingAngleMax * m_l1BranchingAngleDistribution.GetValue(step) + glm::gaussRand(0.0f, m_l1BranchingAngleVarianceMax * m_l1BranchingAngleVarianceDistribution.GetValue(step));
    leafDescriptor.m_rollAngle = i % 2 * 180.0f + glm::gaussRand(0.0f, m_l1RollAngleVarianceMax * m_l1RollAngleVarianceDistribution.GetValue(step));
    leafDescriptor.m_gravitropism = m_l1GravitropismMax * m_l1GravitropismDistribution.GetValue(step) + glm::gaussRand(0.0f, m_l1GravitropismVarianceMax * m_l1GravitropismVarianceDistribution.GetValue(step));
    leafDescriptor.m_gravitropismFactor = m_l1GravitropismFactorMax * m_l1GravitropismFactorDistribution.GetValue(step) + glm::gaussRand(0.0f, m_l1GravitropismFactorVarianceMax * m_l1GravitropismFactorVarianceDistribution.GetValue(step));
  }
}
void SorghumStemDescriptor::Serialize(YAML::Emitter &out) {

}
void SorghumStemDescriptor::OnInspect() {
  ImGui::DragFloat3("Direction", &m_direction.x, 0.01f);
  ImGui::DragFloat("Length", &m_length, 0.01f);
}

void SorghumStemDescriptor::Deserialize(const YAML::Node &in) {

}
void SorghumLeafDescriptor::OnInspect() {
  ImGui::DragFloat("Starting point", &m_leafStartingPoint, 0.01f);
  ImGui::DragFloat("Leaf Length", &m_leafLength, 0.01f);
  ImGui::DragFloat("Branching angle", &m_branchingAngle, 0.01f);
  ImGui::DragFloat("Roll angle", &m_rollAngle, 0.01f);
  ImGui::DragFloat("Gravitropism", &m_gravitropism, 0.01f);
  ImGui::DragFloat("Gravitropism increase", &m_gravitropismFactor, 0.01f);
}
void SorghumLeafDescriptor::Serialize(YAML::Emitter &out) {

}
void SorghumLeafDescriptor::Deserialize(const YAML::Node &in) {

}
