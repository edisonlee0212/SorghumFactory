#ifdef RAYTRACERFACILITY
#include "MLVQRenderer.hpp"
#endif
#include "IVolume.hpp"
#include "LeafData.hpp"
#include "PinnacleData.hpp"
#include "StemData.hpp"
#include <SorghumData.hpp>
#include <SorghumLayer.hpp>
#ifdef RAYTRACERFACILITY
using namespace RayTracerFacility;
#endif
using namespace SorghumFactory;

void SorghumData::OnCreate() {}

void SorghumData::OnDestroy() {
  m_descriptor.Clear();
  m_meshGenerated = false;
}

void SorghumData::OnInspect() {
  static const char *SorghumModes[]{"Procedural Growth", "Sorghum State"};
  ImGui::Combo("Mode", &m_mode, SorghumModes, IM_ARRAYSIZE(SorghumModes));
  switch ((SorghumMode)m_mode) {
  case SorghumMode::ProceduralSorghum: {
    Editor::DragAndDropButton<ProceduralSorghum>(m_descriptor,
                                                 "Procedural Sorghum");
    auto descriptor = m_descriptor.Get<ProceduralSorghum>();
    if (descriptor) {
      if (ImGui::SliderFloat("Time", &m_currentTime, 0.0f,
                             descriptor->GetCurrentEndTime())) {
        GenerateGeometry();
        ApplyGeometry(true, true, false);
      }
      if (ImGui::Button("Apply")) {
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

  out << YAML::Key << "m_descriptor" << YAML::BeginMap;
  m_descriptor.Serialize(out);
  out << YAML::EndMap;
}
void SorghumData::Deserialize(const YAML::Node &in) {
  if (in["m_mode"])
    m_mode = in["m_mode"].as<int>();
  if (in["m_seed"])
    m_seed = in["m_seed"].as<int>();

  if (in["m_gravityDirection"])
    m_gravityDirection = in["m_gravityDirection"].as<glm::vec3>();
  if (in["m_meshGenerated"])
    m_meshGenerated = in["m_meshGenerated"].as<bool>();
  if (in["m_currentTime"])
    m_currentTime = in["m_currentTime"].as<float>();
  if (in["m_recordedVersion"])
    m_recordedVersion = in["m_recordedVersion"].as<unsigned>();
  if (in["m_descriptor"])
    m_descriptor.Deserialize(in["m_descriptor"]);
}

void SorghumData::CollectAssetRef(std::vector<AssetRef> &list) {
  list.push_back(m_descriptor);
}

void SorghumData::GenerateGeometry() {
  SorghumStatePair statePair;
  switch ((SorghumMode)m_mode) {
  case SorghumMode::ProceduralSorghum: {
    auto descriptor = m_descriptor.Get<ProceduralSorghum>();
    if (!descriptor)
      break;
    m_currentTime =
        glm::clamp(m_currentTime, 0.0f, descriptor->GetCurrentEndTime());
    statePair = descriptor->Get(m_currentTime);
    m_recordedVersion = descriptor->GetVersion();
  } break;
  case SorghumMode::SorghumStateGenerator: {
    auto descriptor = m_descriptor.Get<SorghumStateGenerator>();
    if (!descriptor)
      break;
    statePair.m_right = descriptor->Generate(m_seed);
    statePair.m_a = 1.0f;
    m_recordedVersion = descriptor->GetVersion();
  } break;
  }

  // 1. Set owner's spline

  auto children = GetOwner().GetChildren();
  for (int i = 0; i < children.size(); i++) {
    Entities::DeleteEntity(Entities::GetCurrentScene(), children[i]);
  }
  auto stem =
      Application::GetLayer<SorghumLayer>()->CreateSorghumStem(GetOwner());
  auto stemData = stem.GetOrSetPrivateComponent<StemData>().lock();
  stemData->FormStem(statePair);
  auto leafSize = statePair.SizeOfLeaf();
  for (int i = 0; i < leafSize; i++) {
    Entity leaf =
        Application::GetLayer<SorghumLayer>()->CreateSorghumLeaf(GetOwner(), i);
    auto leafData = leaf.GetOrSetPrivateComponent<LeafData>().lock();
    leafData->FormLeaf(statePair);
  }
  if (statePair.m_left.m_pinnacle.m_active ||
      (statePair.m_a == 1.0f && statePair.m_right.m_pinnacle.m_active)) {
    auto pinnacle =
        Application::GetLayer<SorghumLayer>()->CreateSorghumPinnacle(
            GetOwner());
    auto pinnacleData =
        pinnacle.GetOrSetPrivateComponent<PinnacleData>().lock();
    pinnacleData->FormPinnacle(statePair);
  }
}
void SorghumData::ApplyGeometry(bool seperated, bool includeStem,
                                bool segmentedMask) {
  auto owner = GetOwner();
  auto sorghumLayer = Application::GetLayer<SorghumLayer>();
  if (!seperated) {
    unsigned vertexCount = 0;
    std::vector<UniEngine::Vertex> vertices;
    std::vector<glm::uvec3> triangles;
    int i = 0;
    GetOwner().ForEachChild(
        [&](const std::shared_ptr<Scene> &scene, Entity child) {
          if (includeStem && child.HasDataComponent<StemTag>()) {
            auto stemData = child.GetOrSetPrivateComponent<StemData>().lock();
            vertices.insert(vertices.end(), stemData->m_vertices.begin(),
                            stemData->m_vertices.end());
            for (const auto &triangle : stemData->m_triangles) {
              triangles.emplace_back(triangle.x + vertexCount,
                                     triangle.y + vertexCount,
                                     triangle.z + vertexCount);
            }
            vertexCount = vertices.size();
          } else if (child.HasDataComponent<LeafTag>()) {
            auto leafData = child.GetOrSetPrivateComponent<LeafData>().lock();
            vertices.insert(vertices.end(), leafData->m_vertices.begin(),
                            leafData->m_vertices.end());
            for (const auto &triangle : leafData->m_triangles) {
              triangles.emplace_back(triangle.x + vertexCount,
                                     triangle.y + vertexCount,
                                     triangle.z + vertexCount);
            }
            vertexCount = vertices.size();

          } else if (child.HasDataComponent<PinnacleTag>()) {
            auto pinnacleData = child.GetOrSetPrivateComponent<PinnacleData>().lock();
            auto meshRenderer =
                child.GetOrSetPrivateComponent<MeshRenderer>().lock();
            meshRenderer->m_mesh.Get<Mesh>()->SetVertices(17, pinnacleData->m_vertices,
                                                          pinnacleData->m_triangles);
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


    int i = 0;
    GetOwner().ForEachChild([&](const std::shared_ptr<Scene> &scene,
                                Entity child) {
      if (includeStem && child.HasDataComponent<StemTag>()) {
        auto stemData = child.GetOrSetPrivateComponent<StemData>().lock();
        auto meshRenderer = child.GetOrSetPrivateComponent<MeshRenderer>().lock();
        meshRenderer->m_mesh.Get<Mesh>()->SetVertices(17, stemData->m_vertices,
                                                      stemData->m_triangles);
        if (segmentedMask) {
          auto material = AssetManager::LoadMaterial(
              DefaultResources::GLPrograms::StandardProgram);
          meshRenderer->m_material = material;
          material->SetProgram(DefaultResources::GLPrograms::StandardProgram);
          material->m_cullingMode = MaterialCullingMode::Off;
          material->m_albedoColor = stemData->m_vertexColor;
          material->m_roughness = 1.0f;
          material->m_metallic = 0.0f;
        } else {
          meshRenderer->m_material =
              Application::GetLayer<SorghumLayer>()->m_leafMaterial;
        }
#ifdef RAYTRACERFACILITY
        if (sorghumLayer->m_enableMLVQ) {
          auto rtt = owner.GetOrSetPrivateComponent<MLVQRenderer>().lock();
          rtt->Sync();
          rtt->m_materialIndex = sorghumLayer->m_MLVQMaterialIndex;
        }
#endif
      } else if (child.HasDataComponent<LeafTag>()) {
        auto leafData = child.GetOrSetPrivateComponent<LeafData>().lock();
        auto meshRenderer =
            child.GetOrSetPrivateComponent<MeshRenderer>().lock();
        meshRenderer->m_mesh.Get<Mesh>()->SetVertices(
            17, leafData->m_vertices, leafData->m_triangles);
        if (segmentedMask) {
          auto material = AssetManager::LoadMaterial(
              DefaultResources::GLPrograms::StandardProgram);
          meshRenderer->m_material = material;
          material->SetProgram(DefaultResources::GLPrograms::StandardProgram);
          material->m_cullingMode = MaterialCullingMode::Off;
          material->m_albedoColor = leafData->m_vertexColor;
          material->m_roughness = 1.0f;
          material->m_metallic = 0.0f;
        } else {
          meshRenderer->m_material =
              Application::GetLayer<SorghumLayer>()->m_leafMaterial;
        }

      } else if (child.HasDataComponent<PinnacleTag>()) {
        auto pinnacleData = child.GetOrSetPrivateComponent<PinnacleData>().lock();
        auto meshRenderer =
            child.GetOrSetPrivateComponent<MeshRenderer>().lock();
        meshRenderer->m_mesh.Get<Mesh>()->SetVertices(17, pinnacleData->m_vertices, pinnacleData->m_triangles);
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
  GenerateGeometry();
}
