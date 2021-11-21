#ifdef RAYTRACERFACILITY
#include "MLVQRenderer.hpp"
#endif
#include "IVolume.hpp"
#include <SorghumData.hpp>
#include <SorghumLayer.hpp>

#ifdef RAYTRACERFACILITY
using namespace RayTracerFacility;
#endif
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
  EditorManager::DragAndDropButton<SorghumProceduralDescriptor>(m_parameters,
                                                                "Descriptor");
  if (ImGui::DragInt("Vertical subdivision", &m_segmentAmount)) {
    m_segmentAmount = glm::max(2, m_segmentAmount);
  }
  if (ImGui::DragInt("Horizontal subdivision", &m_step)) {
    m_step = glm::max(2, m_step);
  }
  if (ImGui::Button("Apply")) {
    ApplyParameters();
    GenerateGeometry(false);
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
  if (!descriptor)
    return;
  descriptor->Ready();
  m_pinnacleDescriptor = descriptor->m_pinnacleDescriptor;
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
  stemSpline->m_stemWidthMax = descriptor->m_stemDescriptor.m_widthMax;
  stemSpline->m_stemWidthDistribution = descriptor->m_stemDescriptor.m_widthDistribution;

  stemSpline->FormNodes(stemSpline);
  auto children = GetOwner().GetChildren();
  for (int i = 0; i < descriptor->m_leafDescriptors.size(); i++) {
    Entity child;
    if (i < children.size()) {
      child = children[i];
    } else {
      child = Application::GetLayer<SorghumLayer>()->CreateSorghumLeaf(
          GetOwner(), i);
    }
    auto leafDescriptor = descriptor->m_leafDescriptors[i];
    auto spline = child.GetOrSetPrivateComponent<Spline>().lock();
    spline->m_unitAmount = unitAmount;
    spline->m_unitLength = leafDescriptor.m_leafLength / unitAmount;
    spline->m_gravitropism = leafDescriptor.m_gravitropism;
    spline->m_gravitropismFactor = leafDescriptor.m_gravitropismFactor;
    spline->m_type = SplineType::Procedural;
    spline->m_order = i;
    spline->m_segmentAmount = m_segmentAmount;
    spline->m_step = m_step;
    spline->m_startingPoint = leafDescriptor.m_leafStartingPoint;
    spline->m_left = glm::rotate(glm::vec3(1, 0, 0),
                                 glm::radians(leafDescriptor.m_rollAngle),
                                 glm::vec3(0, 1, 0));
    spline->m_initialDirection = glm::rotate(
        glm::vec3(0, 1, 0), glm::radians(leafDescriptor.m_branchingAngle),
        spline->m_left);
    spline->m_stemWidthMax = descriptor->m_stemDescriptor.m_widthMax;
    spline->m_stemWidthDistribution = descriptor->m_stemDescriptor.m_widthDistribution;
    spline->m_leafMaxWidth = leafDescriptor.m_leafMaxWidth;
    spline->m_leafWidthDecreaseStart = leafDescriptor.m_leafWidthDecreaseStart;

    spline->m_waviness = leafDescriptor.m_waviness;
    spline->m_wavinessPeriod = leafDescriptor.m_wavinessPeriod;
  }

  for (int i = descriptor->m_leafDescriptors.size(); i < children.size(); i++) {
    EntityManager::DeleteEntity(EntityManager::GetCurrentScene(), children[i]);
  }

  if (m_pinnacleDescriptor.m_hasPinnacle) {
    Application::GetLayer<SorghumLayer>()->CreateSorghumPinnacle(
        GetOwner());
  }

  m_meshGenerated = false;
}
void SorghumData::GenerateGeometry(bool segmentedMask, bool includeStem) {
  auto owner = GetOwner();
  auto stemSpline = owner.GetOrSetPrivateComponent<Spline>().lock();
  stemSpline->FormNodes(stemSpline);

  if (includeStem) {
    stemSpline->GenerateGeometry(stemSpline);
    auto meshRenderer = owner.GetOrSetPrivateComponent<MeshRenderer>().lock();
    meshRenderer->m_mesh.Get<Mesh>()->SetVertices(17, stemSpline->m_vertices,
                                                  stemSpline->m_indices);
    if (segmentedMask) {
      auto material = AssetManager::LoadMaterial(
          DefaultResources::GLPrograms::StandardProgram);
      meshRenderer->m_material = material;
      material->SetProgram(DefaultResources::GLPrograms::StandardProgram);
      material->m_cullingMode = MaterialCullingMode::Off;
      material->m_albedoColor = stemSpline->m_vertexColor;
      material->m_roughness = 1.0f;
      material->m_metallic = 0.0f;
    } else {
      meshRenderer->m_material =
          Application::GetLayer<SorghumLayer>()->m_leafMaterial;
    }
  }
#ifdef RAYTRACERFACILITY
  auto sorghumLayer = Application::GetLayer<SorghumLayer>();
  if (sorghumLayer->m_enableMLVQ) {
    auto rtt = owner.GetOrSetPrivateComponent<MLVQRenderer>().lock();
    rtt->Sync();
    rtt->m_materialIndex = sorghumLayer->m_MLVQMaterialIndex;
  }
#endif
  GetOwner().ForEachChild([&](const std::shared_ptr<Scene> &scene,
                              Entity child) {
    if (child.HasDataComponent<LeafTag>()) {
      auto leafSpline = child.GetOrSetPrivateComponent<Spline>().lock();
      leafSpline->GenerateGeometry(stemSpline);
      auto meshRenderer = child.GetOrSetPrivateComponent<MeshRenderer>().lock();
      meshRenderer->m_mesh.Get<Mesh>()->SetVertices(17, leafSpline->m_vertices,
                                                    leafSpline->m_indices);
      if (segmentedMask) {
        auto material = AssetManager::LoadMaterial(
            DefaultResources::GLPrograms::StandardProgram);
        meshRenderer->m_material = material;
        material->SetProgram(DefaultResources::GLPrograms::StandardProgram);
        material->m_cullingMode = MaterialCullingMode::Off;
        material->m_albedoColor = leafSpline->m_vertexColor;
        material->m_roughness = 1.0f;
        material->m_metallic = 0.0f;
      } else {
        meshRenderer->m_material =
            Application::GetLayer<SorghumLayer>()->m_leafMaterial;
      }
#ifdef RAYTRACERFACILITY
      if (sorghumLayer->m_enableMLVQ) {
        auto rtt = child.GetOrSetPrivateComponent<MLVQRenderer>().lock();
        rtt->Sync();
        rtt->m_materialIndex = sorghumLayer->m_MLVQMaterialIndex;
      }
#endif
    } else if (child.HasDataComponent<PinnacleTag>()) {
      auto center = stemSpline->EvaluatePoint(1.0f);
      auto meshRenderer = child.GetOrSetPrivateComponent<MeshRenderer>().lock();
      std::vector<UniEngine::Vertex> vertices;
      std::vector<glm::uvec3> triangles;
      PreparePinnacleMesh(center, vertices, triangles);
      meshRenderer->m_mesh.Get<Mesh>()->SetVertices(17, vertices, triangles);
      if (segmentedMask) {
        auto material = AssetManager::LoadMaterial(
            DefaultResources::GLPrograms::StandardProgram);
        meshRenderer->m_material = material;
        material->SetProgram(DefaultResources::GLPrograms::StandardProgram);
        material->m_cullingMode = MaterialCullingMode::Off;
        material->m_albedoColor = glm::vec3(165.0 / 256, 42.0 / 256, 42.0 / 256);
        material->m_roughness = 1.0f;
        material->m_metallic = 0.0f;
      } else {
        meshRenderer->m_material =
            Application::GetLayer<SorghumLayer>()->m_pinnacleMaterial;
      }
    }
  });
}
void SorghumData::CollectAssetRef(std::vector<AssetRef> &list) {
  list.push_back(m_parameters);
}
void SorghumData::PreparePinnacleMesh(const glm::vec3 &center, std::vector<UniEngine::Vertex> &vertices,
                                      std::vector<glm::uvec3> &triangles) {
  vertices.clear();
  triangles.clear();
  std::vector<glm::vec3> icosahedronVertices;
  std::vector<glm::uvec3> icosahedronTriangles;
  SphereMeshGenerator::Icosahedron(icosahedronVertices, icosahedronTriangles);
  int offset = 0;
  UniEngine::Vertex archetype = {};
  SphericalVolume volume;
  volume.m_radius = m_pinnacleDescriptor.m_pinnacleSize;
  for (int seedIndex = 0; seedIndex < m_pinnacleDescriptor.m_seedAmount; seedIndex++) {
    glm::vec3 positionOffset = volume.GetRandomPoint();
    for (const auto position : icosahedronVertices) {
      archetype.m_position =
          position * m_pinnacleDescriptor.m_seedRadius + positionOffset + center;
      vertices.push_back(archetype);
    }
    for (const auto triangle : icosahedronTriangles) {
      glm::uvec3 actualTriangle = triangle + glm::uvec3(offset);
      triangles.push_back(actualTriangle);
    }
    offset += icosahedronVertices.size();
  }
}
