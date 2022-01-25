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
  static const char *SorghumModes[]{"Procedural Growth", "Sorghum State"};
  ImGui::Combo("Mode", &m_mode, SorghumModes,
               IM_ARRAYSIZE(SorghumModes));
  switch ((SorghumMode)m_mode) {
  case SorghumMode::ProceduralSorghum: {
    Editor::DragAndDropButton<ProceduralSorghum>(m_descriptor,
                                                 "Procedural Sorghum");
    auto descriptor = m_descriptor.Get<ProceduralSorghum>();
    if (descriptor) {
      if (ImGui::SliderFloat("Time", &m_currentTime, 0.0f,
                             descriptor->m_endTime)) {
        Apply();
        GenerateGeometry();
        ApplyGeometry(true, true, false);
      }
      if (ImGui::Button("Apply")) {
        Apply();
        GenerateGeometry();
        ApplyGeometry(true, true, false);
      }
    }
  } break;
  case SorghumMode::SorghumStateGenerator: {
    Editor::DragAndDropButton<SorghumStateGenerator>(m_descriptor,
                                                     "Sorghum State Generator");
    auto descriptor = m_descriptor.Get<SorghumStateGenerator>();
    bool changed = ImGui::DragInt("Seed", &m_seed);
    if (descriptor) {
      if (ImGui::Button("Apply") || changed) {
        Apply();
        GenerateGeometry();
        ApplyGeometry(true, true, false);
      }
    }
  } break;
  }

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
  out << YAML::Key << "m_mode" << YAML::Value << m_mode;
  out << YAML::Key << "m_seed" << YAML::Value << m_seed;

  out << YAML::Key << "m_gravityDirection" << YAML::Value << m_gravityDirection;
  out << YAML::Key << "m_currentTime" << YAML::Value << m_currentTime;
  out << YAML::Key << "m_recordedVersion" << YAML::Value << m_recordedVersion;
  out << YAML::Key << "m_meshGenerated" << YAML::Value << m_meshGenerated;

  out << YAML::Key << "m_stemSubdivisionAmount" << YAML::Value
      << m_stemSubdivisionAmount;
  out << YAML::Key << "m_leafSubdivisionAmount" << YAML::Value
      << m_leafSubdivisionAmount;

  out << YAML::Key << "m_descriptor" << YAML::BeginMap;
  m_descriptor.Serialize(out);
  m_state.Serialize(out);
  out << YAML::EndMap;
}
void SorghumData::Deserialize(const YAML::Node &in) {
  if (in["m_mode"])
    m_mode = in["m_mode"].as<int>();
  if (in["m_seed"])
    m_seed = in["m_seed"].as<int>();
  if (in["m_gravityDirection"])
    m_gravityDirection = in["m_gravityDirection"].as<glm::vec3>();
  if (in["m_gravityDirection"])
    m_gravityDirection = in["m_gravityDirection"].as<glm::vec3>();
  if (in["m_meshGenerated"])
    m_meshGenerated = in["m_meshGenerated"].as<bool>();
  if (in["m_currentTime"])
    m_currentTime = in["m_currentTime"].as<float>();
  if (in["m_recordedVersion"])
    m_recordedVersion = in["m_recordedVersion"].as<unsigned>();
  if (in["m_stemSubdivisionAmount"])
    m_stemSubdivisionAmount = in["m_stemSubdivisionAmount"].as<float>();
  if (in["m_leafSubdivisionAmount"])
    m_leafSubdivisionAmount = in["m_leafSubdivisionAmount"].as<float>();
  if (in["m_descriptor"])
    m_descriptor.Deserialize(in["m_descriptor"]);
  if (in["m_state"])
    m_state.Deserialize(in["m_state"]);
}
void SorghumData::Apply() {
  switch ((SorghumMode)m_mode) {
  case SorghumMode::ProceduralSorghum: {
    auto descriptor = m_descriptor.Get<ProceduralSorghum>();
    if (!descriptor)
      return;
    m_currentTime = glm::clamp(m_currentTime, 0.0f, descriptor->m_endTime);
    m_state = descriptor->Get(m_currentTime);
    m_recordedVersion = descriptor->GetVersion();
  } break;
  case SorghumMode::SorghumStateGenerator: {
    auto descriptor = m_descriptor.Get<SorghumStateGenerator>();
    if (!descriptor)
      return;
    m_state = descriptor->Generate(m_seed);
    m_recordedVersion = descriptor->GetVersion();
  } break;
  }

  // 1. Set owner's spline
  auto stemSpline = GetOwner().GetOrSetPrivateComponent<Spline>().lock();
  auto children = GetOwner().GetChildren();
  for (int i = 0; i < children.size(); i++) {
    Entities::DeleteEntity(Entities::GetCurrentScene(), children[i]);
  }
  for (int i = 0; i < m_state.m_leaves.size(); i++) {
    if (!m_state.m_leaves[i].m_active)
      continue;
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
  list.push_back(m_descriptor);
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

void SorghumData::GenerateGeometry() {
  auto owner = GetOwner();
  auto stemSpline = owner.GetOrSetPrivateComponent<Spline>().lock();
  stemSpline->FormStem(m_state.m_stem, m_stemSubdivisionAmount);
  int i = 0;
  GetOwner().ForEachChild(
      [&](const std::shared_ptr<Scene> &scene, Entity child) {
        if (child.HasDataComponent<LeafTag>()) {
          auto leafSpline = child.GetOrSetPrivateComponent<Spline>().lock();
          while (!m_state.m_leaves[i].m_active) {
            i++;
          }
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

void SorghumData::SetTime(float time) {
  m_currentTime = time;
  Apply();
}
