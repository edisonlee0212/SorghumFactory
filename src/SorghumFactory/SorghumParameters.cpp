#include <SorghumParameters.hpp>
#include <rapidxml.hpp>
#include <rapidxml_print.hpp>
#include <rapidxml_utils.hpp>

void SorghumFactory::SorghumParameters::OnGui() {
  ImGui::DragInt("Leaf count", &m_leafCount, 0.01f);

  ImGui::DragFloat("Stem length", &m_stemLength, 0.01f);
  ImGui::DragFloat("First Leaf Starting Point", &m_firstLeafStartingPoint, 0.01f);
  ImGui::DragFloat("Max leaf length", &m_leafLengthBase, 0.01f);
  ImGui::DragFloat2("Branching angle Mean/Var", &m_branchingAngle, 0.01f);
  ImGui::DragFloat2("Gravitropism min/increase", &m_gravitropism, 0.01f);


  m_leafLength.Graph("Leaf length control");
}

void SorghumFactory::SorghumParameters::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_leafCount" << YAML::Value << m_leafCount;
}
void SorghumFactory::SorghumParameters::Deserialize(const YAML::Node &in) {
  m_leafCount = in["m_leafCount"].as<int>();
}
SorghumFactory::SorghumParameters::SorghumParameters() {
  m_leafLength.m_controlPoints[0] = glm::vec2(0, 1);
  m_leafLength.m_controlPoints[1] = glm::vec2(0.3, 1);
  m_leafLength.m_controlPoints[2] = glm::vec2(0.7, 1);
  m_leafLength.m_controlPoints[3] = glm::vec2(1, 1);
}
