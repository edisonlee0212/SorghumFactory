#include <SorghumData.hpp>
#include <SorghumSystem.hpp>

using namespace PlantFactory;

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
      FileUtils::SaveFile(
          "Export OBJ", ".obj",
          [this](const std::string &path) { ExportModel(path); });
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
