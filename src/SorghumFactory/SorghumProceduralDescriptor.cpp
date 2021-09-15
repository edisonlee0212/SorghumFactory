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
    m_l1LeafLengthDistribution.DrawGraph("Leaf length curve");

    ImGui::DragFloat("Max leaf length variance", &m_l1LeafLengthVarianceMax, 0.01f);
    m_l1LeafLengthVarianceDistribution.DrawGraph("Leaf length variance curve");

    ImGui::DragFloat("Max branching angle", &m_l1BranchingAngleMax, 0.01f);
    m_l1BranchingAngleDistribution.DrawGraph("Branching angle curve");

    ImGui::DragFloat("Max branching angle variance", &m_l1BranchingAngleVarianceMax, 0.01f);
    m_l1BranchingAngleVarianceDistribution.DrawGraph("Branching angle variance curve");

    ImGui::DragFloat("Max roll angle variance", &m_l1RollAngleVarianceMax, 0.01f);
    m_l1RollAngleVarianceDistribution.DrawGraph("Roll angle variance curve");

    ImGui::DragFloat("Max gravitropism", &m_l1GravitropismMax, 0.01f);
    m_l1GravitropismDistribution.DrawGraph("Gravitropism curve");

    ImGui::DragFloat("Max gravitropism variance", &m_l1GravitropismVarianceMax, 0.01f);
    m_l1GravitropismVarianceDistribution.DrawGraph("Gravitropism variance curve");

    ImGui::DragFloat("Max gravitropism increase", &m_l1GravitropismFactorMax, 0.01f);
    m_l1GravitropismFactorDistribution.DrawGraph("Gravitropism increase curve");

    ImGui::DragFloat("Max gravitropism increase variance", &m_l1GravitropismFactorVarianceMax, 0.01f);
    m_l1GravitropismFactorVarianceDistribution.DrawGraph("Gravitropism increase variance curve");
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
  m_l1LeafLengthDistribution.m_fixed = false;
  m_l1LeafLengthVarianceDistribution.m_fixed = false;
  m_l1BranchingAngleDistribution.m_fixed = false;
  m_l1BranchingAngleVarianceDistribution.m_fixed = false;
  m_l1RollAngleVarianceDistribution.m_fixed = false;
  m_l1GravitropismDistribution.m_fixed = false;
  m_l1GravitropismVarianceDistribution.m_fixed = false;
  m_l1GravitropismFactorDistribution.m_fixed = false;
  m_l1GravitropismFactorVarianceDistribution.m_fixed = false;


  m_l1LeafLengthDistribution.m_controlPoints[0] = glm::vec2(0, 1);
  m_l1LeafLengthDistribution.m_controlPoints[1] = glm::vec2(0.3, 1);
  m_l1LeafLengthDistribution.m_controlPoints[2] = glm::vec2(0.7, 1);
  m_l1LeafLengthDistribution.m_controlPoints[3] = glm::vec2(1, 1);

  m_l1LeafLengthVarianceDistribution.m_controlPoints[0] = glm::vec2(0, 0.5);
  m_l1LeafLengthVarianceDistribution.m_controlPoints[1] = glm::vec2(0.3, 0.5);
  m_l1LeafLengthVarianceDistribution.m_controlPoints[2] = glm::vec2(0.7, 0.5);
  m_l1LeafLengthVarianceDistribution.m_controlPoints[3] = glm::vec2(1, 0.5);

  m_l1BranchingAngleDistribution.m_controlPoints[0] = glm::vec2(0, 1);
  m_l1BranchingAngleDistribution.m_controlPoints[1] = glm::vec2(0.3, 1);
  m_l1BranchingAngleDistribution.m_controlPoints[2] = glm::vec2(0.7, 1);
  m_l1BranchingAngleDistribution.m_controlPoints[3] = glm::vec2(1, 1);

  m_l1BranchingAngleVarianceDistribution.m_controlPoints[0] = glm::vec2(0, 0.5);
  m_l1BranchingAngleVarianceDistribution.m_controlPoints[1] = glm::vec2(0.3, 0.5);
  m_l1BranchingAngleVarianceDistribution.m_controlPoints[2] = glm::vec2(0.7, 0.5);
  m_l1BranchingAngleVarianceDistribution.m_controlPoints[3] = glm::vec2(1, 0.5);

  m_l1RollAngleVarianceDistribution.m_controlPoints[0] = glm::vec2(0, 0.5);
  m_l1RollAngleVarianceDistribution.m_controlPoints[1] = glm::vec2(0.3, 0.5);
  m_l1RollAngleVarianceDistribution.m_controlPoints[2] = glm::vec2(0.7, 0.5);
  m_l1RollAngleVarianceDistribution.m_controlPoints[3] = glm::vec2(1, 0.5);

  m_l1GravitropismDistribution.m_controlPoints[0] = glm::vec2(0, 1);
  m_l1GravitropismDistribution.m_controlPoints[1] = glm::vec2(0.3, 1);
  m_l1GravitropismDistribution.m_controlPoints[2] = glm::vec2(0.7, 1);
  m_l1GravitropismDistribution.m_controlPoints[3] = glm::vec2(1, 1);

  m_l1GravitropismVarianceDistribution.m_controlPoints[0] = glm::vec2(0, 0.5);
  m_l1GravitropismVarianceDistribution.m_controlPoints[1] = glm::vec2(0.3, 0.5);
  m_l1GravitropismVarianceDistribution.m_controlPoints[2] = glm::vec2(0.7, 0.5);
  m_l1GravitropismVarianceDistribution.m_controlPoints[3] = glm::vec2(1, 0.5);

  m_l1GravitropismFactorDistribution.m_controlPoints[0] = glm::vec2(0, 1);
  m_l1GravitropismFactorDistribution.m_controlPoints[1] = glm::vec2(0.3, 1);
  m_l1GravitropismFactorDistribution.m_controlPoints[2] = glm::vec2(0.7, 1);
  m_l1GravitropismFactorDistribution.m_controlPoints[3] = glm::vec2(1, 1);

  m_l1GravitropismFactorVarianceDistribution.m_controlPoints[0] = glm::vec2(0, 0.5);
  m_l1GravitropismFactorVarianceDistribution.m_controlPoints[1] = glm::vec2(0.3, 0.5);
  m_l1GravitropismFactorVarianceDistribution.m_controlPoints[2] = glm::vec2(0.7, 0.5);
  m_l1GravitropismFactorVarianceDistribution.m_controlPoints[3] = glm::vec2(1, 0.5);
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
    leafDescriptor.m_leafLength = m_l1LeafLengthMax * m_l1LeafLengthDistribution.GetPoint(step).y + glm::gaussRand(0.0f, m_l1LeafLengthVarianceMax * m_l1LeafLengthVarianceDistribution.GetPoint(step).y);
    leafDescriptor.m_branchingAngle = m_l1BranchingAngleMax * m_l1BranchingAngleDistribution.GetPoint(step).y + glm::gaussRand(0.0f, m_l1BranchingAngleVarianceMax * m_l1BranchingAngleVarianceDistribution.GetPoint(step).y);
    leafDescriptor.m_rollAngle = i % 2 * 180.0f + glm::gaussRand(0.0f, m_l1RollAngleVarianceMax * m_l1RollAngleVarianceDistribution.GetPoint(step).y);
    leafDescriptor.m_gravitropism = m_l1GravitropismMax * m_l1GravitropismDistribution.GetPoint(step).y + glm::gaussRand(0.0f, m_l1GravitropismVarianceMax * m_l1GravitropismVarianceDistribution.GetPoint(step).y);
    leafDescriptor.m_gravitropismFactor = m_l1GravitropismFactorMax * m_l1GravitropismFactorDistribution.GetPoint(step).y + glm::gaussRand(0.0f, m_l1GravitropismFactorVarianceMax * m_l1GravitropismFactorVarianceDistribution.GetPoint(step).y);
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
