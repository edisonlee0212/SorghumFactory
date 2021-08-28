#include <SorghumData.hpp>
#include <SorghumSystem.hpp>

using namespace SorghumFactory;

void SorghumData::OnCreate() {}

void SorghumData::OnDestroy() {}

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
  m_parameters.OnGui();

  if (ImGui::DragInt("Segment amount", &m_segmentAmount)) {
    m_segmentAmount = glm::max(2, m_segmentAmount);
  }
  if (ImGui::DragInt("Step amount", &m_step)) {
    m_step = glm::max(2, m_step);
  }
  ImGui::Checkbox("Force Same Rotation", &m_forceSameRotation);
  if (ImGui::Button("Apply parameters"))
    ApplyParameters();
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
  int unitAmount = 16;
  // 1. Set owner's spline
  auto stemSpline = GetOwner().GetOrSetPrivateComponent<Spline>().lock();
  stemSpline->m_unitAmount = unitAmount;
  stemSpline->m_unitLength = m_parameters.m_stemLength / unitAmount;
  stemSpline->m_gravitropism = m_parameters.m_gravitropism;
  stemSpline->m_gravitropismFactor = m_parameters.m_gravitropismFactor;
  stemSpline->m_segmentAmount = m_segmentAmount;
  stemSpline->m_step = m_step;
  stemSpline->m_type = SplineType::Procedural;
  stemSpline->m_order = -1;
  stemSpline->m_startingPoint = -1;
  stemSpline->m_left =
      glm::rotate(glm::vec3(1, 0, 0), glm::radians(glm::linearRand(0.0f, 0.0f)),
                  glm::vec3(0, 1, 0));
  stemSpline->m_initialDirection = glm::vec3(0, 1, 0);
  stemSpline->FormNodes(stemSpline);
  auto children = GetOwner().GetChildren();
  for (int i = 0; i < m_parameters.m_leafCount; i++) {
    Entity child;
    if (i < children.size()) {
      child = children[i];
    } else {
      child = EntityManager::GetSystem<SorghumSystem>()->CreateSorghumLeaf(
          GetOwner());
    }
    auto spline = child.GetOrSetPrivateComponent<Spline>().lock();
    spline->m_unitAmount = unitAmount;
    spline->m_unitLength =
        m_parameters.m_leafLengthBase *
        m_parameters.m_leafLength
            .GetPoint(static_cast<float>(i) / m_parameters.m_leafCount)
            .y /
        unitAmount;
    spline->m_gravitropism = m_parameters.m_gravitropism;
    spline->m_gravitropismFactor = m_parameters.m_gravitropismFactor;
    spline->m_type = SplineType::Procedural;
    spline->m_order = i;
    spline->m_segmentAmount = m_segmentAmount;
    spline->m_step = m_step;
    spline->m_startingPoint =
        m_parameters.m_firstLeafStartingPoint +
        static_cast<float>(i) / m_parameters.m_leafCount *
            (1.0f - m_parameters.m_firstLeafStartingPoint);
    spline->m_left = glm::rotate(
        glm::vec3(1, 0, 0),
        m_forceSameRotation ? 0.0f
                            : glm::radians(glm::linearRand(0.0f, 360.0f)),
        glm::vec3(0, 1, 0));
    spline->m_initialDirection = glm::rotate(
        glm::vec3(0, 1, 0),
        glm::radians(glm::gaussRand(m_parameters.m_branchingAngle,
                                    m_parameters.m_branchingAngleVariance)),
        spline->m_left);
    spline->GenerateGeometry(stemSpline);
    auto meshRenderer = child.GetOrSetPrivateComponent<MeshRenderer>().lock();
    meshRenderer->m_mesh.Get<Mesh>()->SetVertices(17, spline->m_vertices,
                                                  spline->m_indices);
  }

  for (int i = m_parameters.m_leafCount; i < children.size(); i++) {
    EntityManager::DeleteEntity(children[i]);
  }

  m_meshGenerated = true;
}
