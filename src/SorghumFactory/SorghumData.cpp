#include <SorghumData.hpp>
#include <SorghumLayer.hpp>

using namespace SorghumFactory;

void SorghumData::OnCreate() {}

void SorghumData::OnDestroy() {}

void SorghumData::OnInspect() {
  if (ImGui::TreeNodeEx("I/O")) {

      FileUtils::SaveFile("Export OBJ", "3D Model", {".obj"},
                          [this](const std::filesystem::path &path) {
                            ExportModel(path.string());
                          });

    ImGui::TreePop();
  }
  EditorManager::DragAndDropButton<SorghumProceduralDescriptor>(m_parameters, "Descriptor");
  if (ImGui::DragInt("Vertical subdivision", &m_segmentAmount)) {
    m_segmentAmount = glm::max(2, m_segmentAmount);
  }
  if (ImGui::DragInt("Horizontal subdivision", &m_step)) {
    m_step = glm::max(2, m_step);
  }
  if (ImGui::Button("Apply")) {
    ApplyParameters();
    GenerateGeometry();
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
    SorghumLayer::ExportSorghum(GetOwner(), of, startIndex);
    of.close();
    UNIENGINE_LOG("Sorghums saved as " + filename);
  } else {
    UNIENGINE_ERROR("Can't open file!");
  }
}

void SorghumData::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_gravityDirection" << YAML::Value << m_gravityDirection;
  out << YAML::Key << "m_meshGenerated" << YAML::Value << m_meshGenerated;

  out << YAML::Key << "m_parameters" << YAML::BeginMap;
  m_parameters.Serialize(out);
  out << YAML::EndMap;
}
void SorghumData::Deserialize(const YAML::Node &in) {
  m_gravityDirection = in["m_gravityDirection"].as<glm::vec3>();
  m_meshGenerated = in["m_meshGenerated"].as<bool>();

  m_parameters.Deserialize(in["m_parameters"]);
}
void SorghumData::ApplyParameters() {
  auto descriptor = m_parameters.Get<SorghumProceduralDescriptor>();
  if(!descriptor) return;
  descriptor->Ready();

  int unitAmount = 16;
  // 1. Set owner's spline
  auto stemSpline = GetOwner().GetOrSetPrivateComponent<Spline>().lock();
  stemSpline->m_unitAmount = unitAmount;
  stemSpline->m_unitLength = descriptor->m_stemDescriptor.m_length / unitAmount;
  stemSpline->m_gravitropism = 0.0f;
  stemSpline->m_gravitropismFactor = 0.0f;
  stemSpline->m_segmentAmount = m_segmentAmount;
  stemSpline->m_step = m_step;
  stemSpline->m_type = SplineType::Procedural;
  stemSpline->m_order = -1;
  stemSpline->m_startingPoint = -1;
  stemSpline->m_left =
      glm::rotate(glm::vec3(1, 0, 0), glm::radians(glm::linearRand(0.0f, 0.0f)),
                  glm::vec3(0, 1, 0));
  stemSpline->m_initialDirection = glm::vec3(0, 1, 0);
  stemSpline->m_stemWidth = descriptor->m_stemDescriptor.m_stemWidth;

  stemSpline->FormNodes(stemSpline);
  auto children = GetOwner().GetChildren();
  for (int i = 0; i < descriptor->m_leafDescriptors.size(); i++) {
    Entity child;
    if (i < children.size()) {
      child = children[i];
    } else {
      child = Application::GetLayer<SorghumLayer>()->CreateSorghumLeaf(GetOwner(), i);
    }
    auto leafDescriptor = descriptor->m_leafDescriptors[i];
    auto spline = child.GetOrSetPrivateComponent<Spline>().lock();
    spline->m_unitAmount = unitAmount;
    spline->m_unitLength =
        leafDescriptor.m_leafLength / unitAmount;
    spline->m_gravitropism = leafDescriptor.m_gravitropism;
    spline->m_gravitropismFactor = leafDescriptor.m_gravitropismFactor;
    spline->m_type = SplineType::Procedural;
    spline->m_order = i;
    spline->m_segmentAmount = m_segmentAmount;
    spline->m_step = m_step;
    spline->m_startingPoint = leafDescriptor.m_leafStartingPoint;
    spline->m_left = glm::rotate(
        glm::vec3(1, 0, 0), glm::radians(leafDescriptor.m_rollAngle),
        glm::vec3(0, 1, 0));
    spline->m_initialDirection = glm::rotate(
        glm::vec3(0, 1, 0),
        glm::radians(leafDescriptor.m_branchingAngle),
        spline->m_left);
    spline->m_stemWidth = leafDescriptor.m_stemWidth;
    spline->m_leafMaxWidth = leafDescriptor.m_leafMaxWidth;
    spline->m_leafWidthDecreaseStart = leafDescriptor.m_leafWidthDecreaseStart;

    spline->m_waviness = leafDescriptor.m_waviness;
    spline->m_wavinessPeriod = leafDescriptor.m_wavinessPeriod;
  }

  for (int i = descriptor->m_leafDescriptors.size(); i < children.size(); i++) {
    EntityManager::DeleteEntity(EntityManager::GetCurrentScene(), children[i]);
  }

  m_meshGenerated = false;
}
void SorghumData::GenerateGeometry() {
  auto stemSpline = GetOwner().GetOrSetPrivateComponent<Spline>().lock();
  stemSpline->FormNodes(stemSpline);
  stemSpline->GenerateGeometry(stemSpline);
  auto meshRenderer = GetOwner().GetOrSetPrivateComponent<MeshRenderer>().lock();
  meshRenderer->m_mesh.Get<Mesh>()->SetVertices(17, stemSpline->m_vertices,
                                                stemSpline->m_indices);
  GetOwner().ForEachChild([&](const std::shared_ptr<Scene>& scene, Entity child){
    auto spline = child.GetOrSetPrivateComponent<Spline>().lock();
    spline->GenerateGeometry(stemSpline);
    auto meshRenderer = child.GetOrSetPrivateComponent<MeshRenderer>().lock();
    meshRenderer->m_mesh.Get<Mesh>()->SetVertices(17, spline->m_vertices,
                                                  spline->m_indices);
  });
}
void SorghumData::CollectAssetRef(std::vector<AssetRef> &list) {
  list.push_back(m_parameters);
}
