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
  Editor::DragAndDropButton<ProceduralSorghumGrowthDescriptor>(m_parameters,
                                                               "Descriptor");
  auto descriptor = m_parameters.Get<ProceduralSorghumGrowthDescriptor>();
  if (descriptor) {
    static float time = 1.0f;
    static bool autoApply = true;
    ImGui::Checkbox("Auto apply", &autoApply);
    if(ImGui::DragFloat("Time", &time, 0.01f, 0.0f, descriptor->m_endTime) && autoApply){
      time = glm::clamp(time, 0.0f, descriptor->m_endTime);
      ApplyParameters(time);
      GenerateGeometry(false);
      ApplyGeometry(true, true, false);
    }
    time = glm::clamp(time, 0.0f, descriptor->m_endTime);
    if (ImGui::Button("Apply state only")) {
      ApplyParameters(time);
    }
    if (ImGui::Button("Apply state and build sorghum")) {
      ApplyParameters(time);
      GenerateGeometry(false);
      ApplyGeometry(true, true, false);
    }
    if (ImGui::Button("Generate geometry")) {
      GenerateGeometry(false);
      ApplyGeometry(true, true, false);
    }
    if(ImGui::Button("Apply +1/20")){
      time += descriptor->m_endTime / 20.0f;
      time = glm::clamp(time, 0.0f, descriptor->m_endTime);
      ApplyParameters(time);
      GenerateGeometry(false);
      ApplyGeometry(true, true, false);
    }
    ImGui::SameLine();
    if(ImGui::Button("Apply -1/20")){
      time -= descriptor->m_endTime / 20.0f;
      time = glm::clamp(time, 0.0f, descriptor->m_endTime);
      ApplyParameters(time);
      GenerateGeometry(false);
      ApplyGeometry(true, true, false);
    }
  }
  m_state.OnInspect();
  if (m_meshGenerated) {
    if (ImGui::TreeNodeEx("I/O")) {
      FileUtils::SaveFile("Export OBJ", "3D Model", {".obj"},
                          [this](const std::filesystem::path &path) {
                            ExportModel(path.string());
                          });

      ImGui::TreePop();
    }
    if (ImGui::Button("Scan point cloud")) {
      auto pointCloud =
          Application::GetLayer<SorghumLayer>()->ScanPointCloud(GetOwner());
      AssetManager::Share(pointCloud);
    }
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

  out << YAML::Key << "m_stemSubdivisionAmount" << YAML::Value
      << m_stemSubdivisionAmount;
  out << YAML::Key << "m_leafSubdivisionAmount" << YAML::Value
      << m_leafSubdivisionAmount;

  out << YAML::Key << "m_parameters" << YAML::BeginMap;
  m_parameters.Serialize(out);
  m_state.Serialize(out);
  out << YAML::EndMap;
}
void SorghumData::Deserialize(const YAML::Node &in) {
  m_gravityDirection = in["m_gravityDirection"].as<glm::vec3>();
  m_meshGenerated = in["m_meshGenerated"].as<bool>();
  if (in["m_stemSubdivisionAmount"])
    m_stemSubdivisionAmount = in["m_stemSubdivisionAmount"].as<float>();
  if (in["m_leafSubdivisionAmount"])
    m_leafSubdivisionAmount = in["m_leafSubdivisionAmount"].as<float>();
  m_parameters.Deserialize(in["m_parameters"]);
  if (in["m_state"])
    m_state.Deserialize(in["m_state"]);
}
void SorghumData::ApplyParameters(float time) {
  auto descriptor = m_parameters.Get<ProceduralSorghumGrowthDescriptor>();
  if (!descriptor)
    return;
  m_state = descriptor->Get(time);

  // 1. Set owner's spline
  auto stemSpline = GetOwner().GetOrSetPrivateComponent<Spline>().lock();
  auto children = GetOwner().GetChildren();
  for (int i = 0; i < children.size(); i++) {
    Entities::DeleteEntity(Entities::GetCurrentScene(), children[i]);
  }
  for (int i = 0; i < m_state.m_leaves.size(); i++) {
    Entity child;
    child =
        Application::GetLayer<SorghumLayer>()->CreateSorghumLeaf(GetOwner(), i);
    auto spline = child.GetOrSetPrivateComponent<Spline>().lock();
  }
  if (m_state.m_pinnacle.m_active) {
    Application::GetLayer<SorghumLayer>()->CreateSorghumPinnacle(GetOwner());
  }
  m_meshGenerated = false;
}

void SorghumData::CollectAssetRef(std::vector<AssetRef> &list) {
  list.push_back(m_parameters);
}
void SorghumData::PreparePinnacleMesh(const glm::vec3 &center,
                                      std::vector<UniEngine::Vertex> &vertices,
                                      std::vector<glm::uvec3> &triangles) {
  vertices.clear();
  triangles.clear();
  if (!m_state.m_pinnacle.m_active)
    return;
  std::vector<glm::vec3> icosahedronVertices;
  std::vector<glm::uvec3> icosahedronTriangles;
  SphereMeshGenerator::Icosahedron(icosahedronVertices, icosahedronTriangles);
  int offset = 0;
  UniEngine::Vertex archetype = {};
  SphericalVolume volume;
  volume.m_radius = m_state.m_pinnacle.m_pinnacleSize;
  for (int seedIndex = 0; seedIndex < m_state.m_pinnacle.m_seedAmount;
       seedIndex++) {
    glm::vec3 positionOffset = volume.GetRandomPoint();
    for (const auto position : icosahedronVertices) {
      archetype.m_position =
          position * m_state.m_pinnacle.m_seedRadius + positionOffset + center;
      vertices.push_back(archetype);
    }
    for (const auto triangle : icosahedronTriangles) {
      glm::uvec3 actualTriangle = triangle + glm::uvec3(offset);
      triangles.push_back(actualTriangle);
    }
    offset += icosahedronVertices.size();
  }
}

void SorghumData::GenerateGeometry(bool includeStem) {
  auto owner = GetOwner();
  auto stemSpline = owner.GetOrSetPrivateComponent<Spline>().lock();
  stemSpline->FormStem(m_state.m_stem, m_stemSubdivisionAmount);
  int i = 0;
  GetOwner().ForEachChild(
      [&](const std::shared_ptr<Scene> &scene, Entity child) {
        if (child.HasDataComponent<LeafTag>()) {
          auto leafSpline = child.GetOrSetPrivateComponent<Spline>().lock();
          leafSpline->FormLeaf(m_state.m_stem, m_state.m_leaves[i],
                               m_leafSubdivisionAmount);
        }
        i++;
      });
}
void SorghumData::ApplyGeometry(bool seperated, bool includeStem,
                                bool segmentedMask) {
  auto owner = GetOwner();
  auto stemSpline = owner.GetOrSetPrivateComponent<Spline>().lock();
  auto sorghumLayer = Application::GetLayer<SorghumLayer>();
  if (!seperated) {
    unsigned vertexCount = 0;
    std::vector<UniEngine::Vertex> vertices;
    std::vector<glm::uvec3> triangles;
    if (includeStem) {
      vertices.insert(vertices.end(), stemSpline->m_vertices.begin(),
                      stemSpline->m_vertices.end());
      for (const auto &triangle : stemSpline->m_triangles) {
        triangles.emplace_back(triangle.x + vertexCount,
                               triangle.y + vertexCount,
                               triangle.z + vertexCount);
      }
      vertexCount = vertices.size();
    }
    int i = 0;
    GetOwner().ForEachChild(
        [&](const std::shared_ptr<Scene> &scene, Entity child) {
          if (child.HasDataComponent<LeafTag>()) {
            auto leafSpline = child.GetOrSetPrivateComponent<Spline>().lock();
            vertices.insert(vertices.end(), leafSpline->m_vertices.begin(),
                            leafSpline->m_vertices.end());
            for (const auto &triangle : leafSpline->m_triangles) {
              triangles.emplace_back(triangle.x + vertexCount,
                                     triangle.y + vertexCount,
                                     triangle.z + vertexCount);
            }
            vertexCount = vertices.size();

          } else if (child.HasDataComponent<PinnacleTag>()) {
            auto center = m_state.m_stem.GetPoint(1.0f);
            auto meshRenderer =
                child.GetOrSetPrivateComponent<MeshRenderer>().lock();
            std::vector<UniEngine::Vertex> pvertices;
            std::vector<glm::uvec3> ptriangles;
            PreparePinnacleMesh(center, pvertices, ptriangles);
            meshRenderer->m_mesh.Get<Mesh>()->SetVertices(17, pvertices,
                                                          ptriangles);
            meshRenderer->m_material =
                Application::GetLayer<SorghumLayer>()->m_pinnacleMaterial;
          }
          i++;
#ifdef RAYTRACERFACILITY
          if (sorghumLayer->m_enableMLVQ) {
            auto rtt = child.GetOrSetPrivateComponent<MLVQRenderer>().lock();
            rtt->Sync();
            rtt->m_materialIndex = sorghumLayer->m_MLVQMaterialIndex;
          }
#endif
        });
    auto meshRenderer = owner.GetOrSetPrivateComponent<MeshRenderer>().lock();
    meshRenderer->m_material =
        Application::GetLayer<SorghumLayer>()->m_leafMaterial;
    meshRenderer->m_mesh.Get<Mesh>()->SetVertices(17, vertices, triangles);
#ifdef RAYTRACERFACILITY
    if (sorghumLayer->m_enableMLVQ) {
      auto rtt = owner.GetOrSetPrivateComponent<MLVQRenderer>().lock();
      rtt->Sync();
      rtt->m_materialIndex = sorghumLayer->m_MLVQMaterialIndex;
    }
#endif
  } else {
    if (includeStem) {
      auto meshRenderer = owner.GetOrSetPrivateComponent<MeshRenderer>().lock();
      meshRenderer->m_mesh.Get<Mesh>()->SetVertices(17, stemSpline->m_vertices,
                                                    stemSpline->m_triangles);
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
    if (sorghumLayer->m_enableMLVQ) {
      auto rtt = owner.GetOrSetPrivateComponent<MLVQRenderer>().lock();
      rtt->Sync();
      rtt->m_materialIndex = sorghumLayer->m_MLVQMaterialIndex;
    }
#endif
    int i = 0;
    GetOwner().ForEachChild([&](const std::shared_ptr<Scene> &scene,
                                Entity child) {
      if (child.HasDataComponent<LeafTag>()) {
        auto leafSpline = child.GetOrSetPrivateComponent<Spline>().lock();
        auto meshRenderer =
            child.GetOrSetPrivateComponent<MeshRenderer>().lock();
        meshRenderer->m_mesh.Get<Mesh>()->SetVertices(
            17, leafSpline->m_vertices, leafSpline->m_triangles);
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

      } else if (child.HasDataComponent<PinnacleTag>()) {
        auto center = m_state.m_stem.GetPoint(1.0f);
        auto meshRenderer =
            child.GetOrSetPrivateComponent<MeshRenderer>().lock();
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
          material->m_albedoColor =
              glm::vec3(165.0 / 256, 42.0 / 256, 42.0 / 256);
          material->m_roughness = 1.0f;
          material->m_metallic = 0.0f;
        } else {
          meshRenderer->m_material =
              Application::GetLayer<SorghumLayer>()->m_pinnacleMaterial;
        }
      }
#ifdef RAYTRACERFACILITY
      if (sorghumLayer->m_enableMLVQ) {
        auto rtt = child.GetOrSetPrivateComponent<MLVQRenderer>().lock();
        rtt->Sync();
        rtt->m_materialIndex = sorghumLayer->m_MLVQMaterialIndex;
      }
#endif
      i++;
    });
  }
  m_meshGenerated = true;
}
