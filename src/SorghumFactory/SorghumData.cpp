#include <SorghumData.hpp>
#include <SorghumSystem.hpp>

using namespace SorghumFactory;

void SorghumData::OnCreate() {}

void SorghumData::OnDestroy() {
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
void SorghumData::ApplyParameters() {
  //1. Set owner's spline
  auto spline = GetOwner().GetOrSetPrivateComponent<Spline>().lock();
  spline->m_type = SplineType::Procedural;
  spline->m_order = -1;
  spline->m_startingPoint = -1;
  spline->m_left = glm::rotate(glm::vec3(1, 0, 0), glm::radians(glm::linearRand(0.0f, 360.0f)), glm::vec3(0, 1, 0));
  spline->m_initialDirection = glm::vec3(0, 1, 0);

  //2. Make sure children is enough.
  int childrenIndex = 0;
  GetOwner().ForEachChild([&](Entity child){
    auto spline = child.GetOrSetPrivateComponent<Spline>().lock();
    spline->m_type = SplineType::Procedural;
    spline->m_order = childrenIndex;
    spline->m_startingPoint = 0.1f + static_cast<float>(childrenIndex) / m_parameters.m_leafCount * 0.9f;
    spline->m_left = glm::rotate(glm::vec3(1, 0, 0), glm::radians(glm::linearRand(0.0f, 360.0f)), glm::vec3(0, 1, 0));
    spline->m_initialDirection = glm::rotate(glm::vec3(0, 1, 0), glm::radians(30.0f), spline->m_left);
    childrenIndex++;
  });
  for(int i = childrenIndex; i < m_parameters.m_leafCount; i++){
    Entity newLeaf = EntityManager::GetSystem<SorghumSystem>()->CreateSorghumLeaf(GetOwner());
    auto spline = newLeaf.GetOrSetPrivateComponent<Spline>().lock();
    spline->m_type = SplineType::Procedural;
    spline->m_order = i;
    spline->m_startingPoint = 0.1f + static_cast<float>(childrenIndex) / m_parameters.m_leafCount * 0.9f;
    spline->m_left = glm::rotate(glm::vec3(1, 0, 0), glm::radians(glm::linearRand(0.0f, 360.0f)), glm::vec3(0, 1, 0));
    spline->m_initialDirection = glm::rotate(glm::vec3(0, 1, 0), glm::radians(30.0f), spline->m_left);
  }
}
