#include <SorghumData.hpp>
#include <SorghumSystem.hpp>

using namespace SorghumFactory;

void SorghumData::OnCreate() {}

void SorghumData::OnDestroy() {
  Entity rootInternode;
  GetOwner().ForEachChild([&](Entity child) {
    if (child.HasDataComponent<InternodeInfo>())
      rootInternode = child;
  });
  if (rootInternode.IsValid())
    EntityManager::DeleteEntity(rootInternode);
}

void SorghumData::OnGui() {
  if (ImGui::TreeNodeEx("I/O")) {
    if (m_meshGenerated) {
      FileUtils::SaveFile("Export OBJ", "3D Model", {".obj"},
                          [this](const std::filesystem::path &path) {
                            ExportModel(path.string());
                          });
    }
    ImGui::TreePop();
  }
  if (ImGui::TreeNodeEx("Parameters")) {
    m_parameters.OnGui();
    ImGui::TreePop();
  }
}

void SorghumData::ExportModel(const std::string &filename,
                              const bool &includeFoliage) const {
  std::ofstream of;
  of.open(filename, std::ofstream::out | std::ofstream::trunc);
  if (of.is_open()) {
    std::string start = "#Sorghum field, by Bosheng Li";
    start += "\n";
    of.write(start.c_str(), start.size());
    of.flush();
    unsigned startIndex = 1;
    SorghumSystem::ExportSorghum(GetOwner(), of, startIndex);
    of.close();
    Debug::Log("Sorghums saved as " + filename);
  } else {
    Debug::Error("Can't open file!");
  }
}
void SorghumData::Clone(const std::shared_ptr<IPrivateComponent> &target) {
  *this = *std::static_pointer_cast<SorghumData>(target);
}
void SorghumData::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_growthComplete" << YAML::Value << m_growthComplete;
  out << YAML::Key << "m_gravityDirection" << YAML::Value << m_gravityDirection;
  out << YAML::Key << "m_meshGenerated" << YAML::Value << m_meshGenerated;

  out << YAML::Key << "m_parameters" << YAML::BeginMap;
  m_parameters.Serialize(out);
  out << YAML::EndMap;
}
void SorghumData::Deserialize(const YAML::Node &in) {
  m_growthComplete = in["m_growthComplete"].as<float>();
  m_gravityDirection = in["m_gravityDirection"].as<glm::vec3>();
  m_meshGenerated = in["m_meshGenerated"].as<float>();

  m_parameters.Deserialize(in["m_parameters"]);
}
