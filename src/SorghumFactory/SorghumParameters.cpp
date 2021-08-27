#include <SorghumParameters.hpp>
#include <rapidxml.hpp>
#include <rapidxml_print.hpp>
#include <rapidxml_utils.hpp>

void SorghumFactory::SorghumParameters::OnGui() {
  ImGui::DragInt("Leaf count", &m_leafCount, 0.01f);

}

void SorghumFactory::SorghumParameters::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_leafCount" << YAML::Value << m_leafCount;


}
void SorghumFactory::SorghumParameters::Deserialize(const YAML::Node &in) {
  m_leafCount = in["m_leafCount"].as<int>();

}
