//
// Created by lllll on 10/17/2022.
//
#ifdef RAYTRACERFACILITY
#include "CompressedBTF.hpp"
using namespace RayTracerFacility;
#endif

#include "CBTFGroup.hpp"

void PlantArchitect::CBTFGroup::OnInspect() {
  AssetRef temp;
  if (Editor::DragAndDropButton<CompressedBTF>(temp, ("Drop to add..."))) {
    m_cBTFs.emplace_back(temp);
    temp.Clear();
  }
  if(ImGui::TreeNodeEx("List", ImGuiTreeNodeFlags_DefaultOpen))
  for(int i = 0; i < m_cBTFs.size(); i++){
    if(Editor::DragAndDropButton<CompressedBTF>(m_cBTFs[i], ("No." + std::to_string(i + 1))) && !m_cBTFs[i].Get<CompressedBTF>()){
      m_cBTFs.erase(m_cBTFs.begin() + i);
      i--;
    }
  }
  ImGui::TreePop();
}

void PlantArchitect::CBTFGroup::CollectAssetRef(std::vector<AssetRef> &list) {
  for (const auto &i : m_cBTFs)
    list.push_back(i);
}
void PlantArchitect::CBTFGroup::Serialize(YAML::Emitter &out) {
  if (!m_cBTFs.empty()) {
    out << YAML::Key << "m_cBTFs" << YAML::Value << YAML::BeginSeq;
    for (auto &cBTF : m_cBTFs) {
      out << YAML::BeginMap;
      cBTF.Serialize(out);
      out << YAML::EndMap;
    }
    out << YAML::EndSeq;
  }
}
void PlantArchitect::CBTFGroup::Deserialize(const YAML::Node &in) {
  auto inCBTFs = in["m_cBTFs"];
  if (inCBTFs) {
    for (const auto &i : inCBTFs) {
      AssetRef ref;
      ref.Deserialize(i);
      m_cBTFs.emplace_back(ref);
    }
  }
}
AssetRef PlantArchitect::CBTFGroup::GetRandom() const {
  if(!m_cBTFs.empty()) {
    return m_cBTFs[glm::linearRand(0, (int)m_cBTFs.size() - 1)];
  }
  return {};
}
