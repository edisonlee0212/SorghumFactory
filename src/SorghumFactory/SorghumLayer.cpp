#include <Tinyply.hpp>
#ifdef RAYTRACERFACILITY
#include "RayTracerLayer.hpp"
#include <MLVQRenderer.hpp>
#include <TriangleIlluminationEstimator.hpp>
#endif
#include <SorghumData.hpp>
#include <SorghumLayer.hpp>

#include "DepthCamera.hpp"
#ifdef RAYTRACERFACILITY
using namespace RayTracerFacility;
#endif
using namespace SorghumFactory;
using namespace UniEngine;
using namespace tinyply;
void SorghumLayer::OnCreate() {
  ClassRegistry::RegisterDataComponent<PinnacleTag>("PinnacleTag");
  ClassRegistry::RegisterDataComponent<LeafTag>("LeafTag");
  ClassRegistry::RegisterDataComponent<SorghumTag>("SorghumTag");

  ClassRegistry::RegisterPrivateComponent<DepthCamera>("DepthCamera");

  ClassRegistry::RegisterPrivateComponent<Spline>("Spline");
  ClassRegistry::RegisterPrivateComponent<SorghumData>("SorghumData");
  ClassRegistry::RegisterAsset<ProceduralSorghum>("ProceduralSorghum",
                                                  ".proceduralsorghum");
  ClassRegistry::RegisterAsset<SorghumStateGenerator>("SorghumStateGenerator",
                                                      ".sorghumstategenerator");
  ClassRegistry::RegisterAsset<SorghumField>("SorghumField", ".sorghumfield");
  ClassRegistry::RegisterAsset<RectangularSorghumField>(
      "RectangularSorghumField", ".rectsorghumfield");
  ClassRegistry::RegisterAsset<PositionsField>("PositionsField",
                                               ".possorghumfield");

  auto &editorManager = Editor::GetInstance();
  auto texture2D = std::make_shared<Texture2D>();
  texture2D->Import(std::filesystem::absolute(
      std::filesystem::path("./SorghumFactoryResources/Textures") /
      "ProceduralSorghum.png"));
  editorManager.AssetIcons()["ProceduralSorghum"] = texture2D;
  texture2D = std::make_shared<Texture2D>();
  texture2D->Import(std::filesystem::absolute(
      std::filesystem::path("./SorghumFactoryResources/Textures") /
      "SorghumStateGenerator.png"));
  editorManager.AssetIcons()["SorghumStateGenerator"] = texture2D;
  texture2D = std::make_shared<Texture2D>();
  texture2D->Import(std::filesystem::absolute(
      std::filesystem::path("./SorghumFactoryResources/Textures") /
      "PositionsField.png"));
  editorManager.AssetIcons()["PositionsField"] = texture2D;

  m_leafArchetype = Entities::CreateEntityArchetype("Leaf", LeafTag());
  m_leafQuery = Entities::CreateEntityQuery();
  m_leafQuery.SetAllFilters(LeafTag());

  m_pinnacleArchetype =
      Entities::CreateEntityArchetype("Pinnacle", PinnacleTag());
  m_pinnacleQuery = Entities::CreateEntityQuery();
  m_pinnacleQuery.SetAllFilters(PinnacleTag());

  m_sorghumArchetype = Entities::CreateEntityArchetype("Sorghum", SorghumTag());
  m_sorghumQuery = Entities::CreateEntityQuery();
  m_sorghumQuery.SetAllFilters(SorghumTag());

  if (!m_leafAlbedoTexture.Get<Texture2D>()) {
    auto albedo = AssetManager::CreateAsset<Texture2D>("Leaf texture");
    albedo->Import(std::filesystem::absolute(
        std::filesystem::path("./SorghumFactoryResources/Textures") /
        "leafSurfaceDark.jpg"));
    m_leafAlbedoTexture.Set(albedo);
  }

  if (!m_leafMaterial.Get<Material>()) {
    auto material = AssetManager::LoadMaterial(
        DefaultResources::GLPrograms::StandardProgram);
    m_leafMaterial = material;
    material->m_albedoTexture = m_leafAlbedoTexture;
    material->SetProgram(DefaultResources::GLPrograms::StandardProgram);
    material->m_cullingMode = MaterialCullingMode::Off;
    material->m_albedoColor =
        glm::vec3(113.0f / 255, 169.0f / 255, 44.0f / 255);
    material->m_roughness = 1.0f;
    material->m_metallic = 0.0f;
  }

  if (!m_pinnacleMaterial.Get<Material>()) {
    auto material = AssetManager::LoadMaterial(
        DefaultResources::GLPrograms::StandardProgram);
    m_pinnacleMaterial = material;
    material->SetProgram(DefaultResources::GLPrograms::StandardProgram);
    material->m_cullingMode = MaterialCullingMode::Off;
    material->m_albedoColor = glm::vec3(165.0 / 256, 42.0 / 256, 42.0 / 256);
    material->m_roughness = 1.0f;
    material->m_metallic = 0.0f;
  }

  for (auto &i : m_segmentedLeafMaterials) {
    if (!i.Get<Material>()) {
      auto material = AssetManager::LoadMaterial(
          DefaultResources::GLPrograms::StandardProgram);
      i = material;
      material->SetProgram(DefaultResources::GLPrograms::StandardProgram);
      material->m_cullingMode = MaterialCullingMode::Off;
      material->m_albedoColor =
          glm::linearRand(glm::vec3(0.0f), glm::vec3(1.0f));
      material->m_roughness = 1.0f;
      material->m_metallic = 0.0f;
    }
  }
}

Entity SorghumLayer::CreateSorghum() {
  Transform transform;
  transform.SetScale(glm::vec3(1.0f));
  const Entity entity = Entities::CreateEntity(Entities::GetCurrentScene(),
                                               m_sorghumArchetype, "Sorghum");
  entity.GetOrSetPrivateComponent<Spline>();
  auto sorghumData = entity.GetOrSetPrivateComponent<SorghumData>().lock();
  entity.SetName("Sorghum");
#ifdef RAYTRACERFACILITY
  entity.GetOrSetPrivateComponent<TriangleIlluminationEstimator>();
#endif
  auto mmc = entity.GetOrSetPrivateComponent<MeshRenderer>().lock();
  // mmc->m_material = m_segmentedLeafMaterials[leafIndex];
  {
    auto material = AssetManager::LoadMaterial(
        DefaultResources::GLPrograms::StandardProgram);
    mmc->m_material = material;
    material->SetProgram(DefaultResources::GLPrograms::StandardProgram);
    material->m_cullingMode = MaterialCullingMode::Off;
    material->m_albedoColor = glm::vec3(0, 0, 0);
    material->m_roughness = 1.0f;
    material->m_metallic = 0.0f;
  }
  mmc->m_mesh = AssetManager::CreateAsset<Mesh>();
  return entity;
}

Entity SorghumLayer::CreateSorghumLeaf(const Entity &plantEntity,
                                       int leafIndex) {
  const Entity entity =
      Entities::CreateEntity(Entities::GetCurrentScene(), m_leafArchetype);
  entity.SetName("Leaf");
  entity.SetParent(plantEntity);
  Transform transform;
  transform.SetScale(glm::vec3(1.0f));
  auto spline = entity.GetOrSetPrivateComponent<Spline>().lock();
  LeafTag tag;
  tag.m_index = leafIndex;
  entity.SetDataComponent(tag);
  entity.SetDataComponent(transform);
  auto mmc = entity.GetOrSetPrivateComponent<MeshRenderer>().lock();
  mmc->m_mesh = AssetManager::CreateAsset<Mesh>();
  return entity;
}
Entity SorghumLayer::CreateSorghumPinnacle(const Entity &plantEntity) {
  const Entity entity =
      Entities::CreateEntity(Entities::GetCurrentScene(), m_pinnacleArchetype);
  entity.SetName("Pinnacle");
  entity.SetParent(plantEntity);
  Transform transform;
  transform.SetScale(glm::vec3(1.0f));
  auto spline = entity.GetOrSetPrivateComponent<Spline>().lock();
  PinnacleTag tag;
  entity.SetDataComponent(tag);
  entity.SetDataComponent(transform);
  auto mmc = entity.GetOrSetPrivateComponent<MeshRenderer>().lock();
  mmc->m_mesh = AssetManager::CreateAsset<Mesh>();
  return entity;
}

void SorghumLayer::GenerateMeshForAllSorghums(bool seperated, bool includeStem,
                                              bool segmentedMask,
                                              int segmentAmount, int step) {
  std::vector<Entity> plants;
  Entities::ForEach<GlobalTransform>(
      Entities::GetCurrentScene(), Jobs::Workers(), m_sorghumQuery,
      [=](int index, Entity entity, GlobalTransform &ltw) {
        if (entity.HasPrivateComponent<SorghumData>()) {
          auto sorghumData =
              entity.GetOrSetPrivateComponent<SorghumData>().lock();
          sorghumData->GenerateGeometry();
        }
      });

  m_sorghumQuery.ToEntityArray(Entities::GetCurrentScene(), plants);
  for (auto &plant : plants) {
    if (plant.HasPrivateComponent<SorghumData>())
      plant.GetOrSetPrivateComponent<SorghumData>().lock()->ApplyGeometry(
          seperated, includeStem, segmentedMask);
  }
}

Entity SorghumLayer::ImportPlant(const std::filesystem::path &path,
                                 const std::string &name) {
  /*
  std::ifstream file(path, std::fstream::in);
  if (!file.is_open()) {
    UNIENGINE_LOG("Failed to open file!");
    return Entity();
  }
  // Number of leaves in the file
  int leafCount;
  file >> leafCount;
  const auto sorghum = CreateSorghum();
  sorghum.SetName(name);
  auto truckSpline = sorghum.GetOrSetPrivateComponent<Spline>().lock();
  truckSpline->m_startingPoint = -1;
  truckSpline->m_startingPoint = -1;
  truckSpline->Import(file);

  // Recenter plant:
  glm::vec3 posSum = truckSpline->m_curves.front().m_p0;

  for (auto &curve : truckSpline->m_curves) {
    curve.m_p0 -= posSum;
    curve.m_p1 -= posSum;
    curve.m_p2 -= posSum;
    curve.m_p3 -= posSum;
  }
  truckSpline->m_left = glm::cross(glm::vec3(0.0f, 1.0f, 0.0f),
                                   truckSpline->m_curves.begin()->m_p0 -
                                       truckSpline->m_curves.back().m_p3);
  for (int i = 0; i < leafCount; i++) {
    Entity leaf = CreateSorghumLeaf(sorghum, i);
    auto leafSpline = leaf.GetOrSetPrivateComponent<Spline>().lock();
    float startingPoint;
    file >> startingPoint;

    leafSpline->m_startingPoint = startingPoint;
    leafSpline->Import(file);
    for (auto &curve : leafSpline->m_curves) {
      curve.m_p0 += truckSpline->EvaluatePoint(startingPoint);
      curve.m_p1 += truckSpline->EvaluatePoint(startingPoint);
      curve.m_p2 += truckSpline->EvaluatePoint(startingPoint);
      curve.m_p3 += truckSpline->EvaluatePoint(startingPoint);
    }

    leafSpline->m_left = glm::cross(glm::vec3(0.0f, 1.0f, 0.0f),
                                    leafSpline->m_curves.begin()->m_p0 -
                                        leafSpline->m_curves.back().m_p3);
  }
  return sorghum;
   */
  return Entity();
}

void SorghumLayer::OnInspect() {
  if (ImGui::Begin("Sorghum")) {
#ifdef RAYTRACERFACILITY
    if (ImGui::TreeNodeEx("Illumination Estimation")) {
      ImGui::Checkbox("Display light probes", &m_displayLightProbes);
      if (m_displayLightProbes) {
        ImGui::DragFloat("Size", &m_lightProbeSize, 0.0001f, 0.0001f, 0.2f,
                         "%.5f");
      }
      ImGui::DragInt("Seed", &m_seed);
      ImGui::DragFloat("Push distance along normal", &m_pushDistance, 0.0001f,
                       -1.0f, 1.0f, "%.5f");
      m_rayProperties.OnInspect();

      if (ImGui::Button("Calculate illumination")) {
        CalculateIlluminationFrameByFrame();
      }
      if (ImGui::Button("Calculate illumination instantly")) {
        CalculateIllumination();
      }
      ImGui::TreePop();
    }
#endif
    ImGui::Separator();
    if (ImGui::Button("Generate mesh for all sorghums")) {
      GenerateMeshForAllSorghums(false, true, false);
    }
    if (ImGui::DragInt("Segment amount", &m_segmentAmount)) {
      m_segmentAmount = glm::max(2, m_segmentAmount);
    }
    if (ImGui::DragInt("Step amount", &m_step)) {
      m_step = glm::max(2, m_step);
    }
    if (Editor::DragAndDropButton<Texture2D>(m_leafAlbedoTexture,
                                             "Replace Leaf Albedo Texture")) {
      auto tex = m_leafAlbedoTexture.Get<Texture2D>();
      if (tex) {
        m_leafMaterial.Get<Material>()->m_albedoTexture = m_leafAlbedoTexture;
        std::vector<Entity> sorghumEntities;
        m_sorghumQuery.ToEntityArray(Entities::GetCurrentScene(),
                                     sorghumEntities, false);
        for (const auto &i : sorghumEntities) {
          if (i.HasPrivateComponent<MeshRenderer>()) {
            i.GetOrSetPrivateComponent<MeshRenderer>()
                .lock()
                ->m_material.Get<Material>()
                ->m_albedoTexture = m_leafAlbedoTexture;
          }
        }
      }
    }

    if (Editor::DragAndDropButton<Texture2D>(m_leafNormalTexture,
                                             "Replace Leaf Normal Texture")) {
      auto tex = m_leafNormalTexture.Get<Texture2D>();
      if (tex) {
        m_leafMaterial.Get<Material>()->m_normalTexture = m_leafNormalTexture;
        std::vector<Entity> sorghumEntities;
        m_sorghumQuery.ToEntityArray(Entities::GetCurrentScene(),
                                     sorghumEntities, false);
        for (const auto &i : sorghumEntities) {
          if (i.HasPrivateComponent<MeshRenderer>()) {
            i.GetOrSetPrivateComponent<MeshRenderer>()
                .lock()
                ->m_material.Get<Material>()
                ->m_normalTexture = m_leafNormalTexture;
          }
        }
      }
    }

    ImGui::Separator();

    FileUtils::OpenFile("Import from Skeleton", "Skeleton", {".txt"},
                        [this](const std::filesystem::path &path) {
                          ImportPlant(path, "Sorghum");
                        });
    /*
    if (ImGui::Button("Create field...")) {
      ImGui::OpenPopup("Sorghum field wizard");
    }
    const ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if (ImGui::BeginPopupModal("Sorghum field wizard", nullptr,
                               ImGuiWindowFlags_AlwaysAutoResize)) {
      static RectangularSorghumFieldPattern field;
      ImGui::DragInt2("Size", &field.m_size.x, 1, 1, 10);
      ImGui::DragFloat2("Distance", &field.m_distances.x, 0.1f, 0.0f, 10.0f);
      ImGui::DragFloat3("Rotation variance", &field.m_rotationVariation.x, 0.1f,
                        0.0f, 10.0f);
      if (ImGui::Button("OK", ImVec2(120, 0))) {
        std::vector<Entity> candidates;
        candidates.push_back(
            ImportPlant(std::filesystem::path("../Resources") /
                            "Sorghum/skeleton_procedural_1.txt",
                        "Sorghum 1"));
        candidates.push_back(
            ImportPlant(std::filesystem::path("../Resources") /
                            "Sorghum/skeleton_procedural_2.txt",
                        "Sorghum 2"));
        candidates.push_back(
            ImportPlant(std::filesystem::path("../Resources") /
                            "Sorghum/skeleton_procedural_3.txt",
                        "Sorghum 3"));
        candidates.push_back(
            ImportPlant(std::filesystem::path("../Resources") /
                            "Sorghum/skeleton_procedural_4.txt",
                        "Sorghum 4"));
        GenerateMeshForAllSorghums();

        CreateGrid(field, candidates);
        for (auto &i : candidates)
          Entities::DeleteEntity(Entities::GetCurrentScene(), i);
        ImGui::CloseCurrentPopup();
      }
      ImGui::SetItemDefaultFocus();
      ImGui::SameLine();
      if (ImGui::Button("Cancel", ImVec2(120, 0))) {
        ImGui::CloseCurrentPopup();
      }
      ImGui::EndPopup();
    }
    */
    FileUtils::SaveFile("Export OBJ for all sorghums", "3D Model", {".obj"},
                        [this](const std::filesystem::path &path) {
                          ExportAllSorghumsModel(path.string());
                        });

    static bool opened = false;
#ifdef RAYTRACERFACILITY
    if (m_processing && !opened) {
      ImGui::OpenPopup("Illumination Estimation");
      opened = true;
    }
    if (ImGui::BeginPopupModal("Illumination Estimation", nullptr,
                               ImGuiWindowFlags_AlwaysAutoResize)) {
      ImGui::Text("Progress: ");
      float fraction = 1.0f - static_cast<float>(m_processingIndex) /
                                  m_processingEntities.size();
      std::string text =
          std::to_string(static_cast<int>(fraction * 100.0f)) + "% - " +
          std::to_string(m_processingEntities.size() - m_processingIndex) +
          "/" + std::to_string(m_processingEntities.size());
      ImGui::ProgressBar(fraction, ImVec2(240, 0), text.c_str());
      ImGui::SetItemDefaultFocus();
      ImGui::Text(("Estimation time for 1 plant: " +
                   std::to_string(m_perPlantCalculationTime) + " seconds")
                      .c_str());
      if (ImGui::Button("Cancel") || m_processing == false) {
        m_processing = false;
        opened = false;
        ImGui::CloseCurrentPopup();
      }
      ImGui::EndPopup();
    }
#endif
  }
  ImGui::End();
}

void SorghumLayer::CloneSorghums(const Entity &parent, const Entity &original,
                                 std::vector<glm::mat4> &matrices) {
  auto sorghumData = original.GetOrSetPrivateComponent<SorghumData>().lock();
  for (const auto &matrix : matrices) {
    Entity sorghum = CreateSorghum();
    Transform transform;
    transform.m_value = matrix;

    auto newSpline = sorghum.GetOrSetPrivateComponent<Spline>().lock();
    auto spline = original.GetOrSetPrivateComponent<Spline>().lock();
    newSpline->Copy(spline);

    original.ForEachChild([this, &sorghum,
                           &matrices](const std::shared_ptr<Scene> &scene,
                                      Entity child) {
      if (!child.HasDataComponent<LeafTag>())
        return;
      auto tag = child.GetDataComponent<LeafTag>();
      const auto newChild = CreateSorghumLeaf(sorghum, tag.m_index);
      newChild.SetDataComponent(tag);
      newChild.SetDataComponent(child.GetDataComponent<Transform>());
      auto newSpline = newChild.GetOrSetPrivateComponent<Spline>().lock();
      auto spline = child.GetOrSetPrivateComponent<Spline>().lock();
      newSpline->Copy(spline);
#ifdef RAYTRACERFACILITY
      if (m_enableMLVQ && child.HasPrivateComponent<MLVQRenderer>()) {
        auto newRayTracedRenderer =
            newChild.GetOrSetPrivateComponent<MLVQRenderer>().lock();
        newRayTracedRenderer->m_materialIndex = 1;
        newRayTracedRenderer->Sync();
      }
#endif
      auto newMeshRenderer =
          newChild.GetOrSetPrivateComponent<MeshRenderer>().lock();
      auto meshRenderer = child.GetOrSetPrivateComponent<MeshRenderer>().lock();
      newMeshRenderer->m_mesh = meshRenderer->m_mesh;
      newMeshRenderer->m_material = meshRenderer->m_material;
    });
    sorghum.SetParent(parent);
    sorghum.SetDataComponent(transform);
  }
}

void SorghumLayer::ExportSorghum(const Entity &sorghum, std::ofstream &of,
                                 unsigned &startIndex) {
  const std::string start = "#Sorghum\n";
  of.write(start.c_str(), start.size());
  of.flush();
  const auto position =
      sorghum.GetDataComponent<GlobalTransform>().GetPosition();

  const auto stemMesh = sorghum.GetOrSetPrivateComponent<MeshRenderer>()
                            .lock()
                            ->m_mesh.Get<Mesh>();
  ObjExportHelper(position, stemMesh, of, startIndex);

  sorghum.ForEachChild([&](const std::shared_ptr<Scene> &scene, Entity child) {
    if (!child.HasPrivateComponent<MeshRenderer>())
      return;
    const auto leafMesh = child.GetOrSetPrivateComponent<MeshRenderer>()
                              .lock()
                              ->m_mesh.Get<Mesh>();
    ObjExportHelper(position, leafMesh, of, startIndex);
  });
}

void SorghumLayer::ObjExportHelper(glm::vec3 position,
                                   std::shared_ptr<Mesh> mesh,
                                   std::ofstream &of, unsigned &startIndex) {
  if (!mesh->UnsafeGetTriangles().empty()) {
    std::string header =
        "#Vertices: " + std::to_string(mesh->GetVerticesAmount()) +
        ", tris: " + std::to_string(mesh->GetTriangleAmount());
    header += "\n";
    of.write(header.c_str(), header.size());
    of.flush();
    std::string o = "o ";
    o += "[" + std::to_string(position.x) + "," + std::to_string(position.z) +
         "]" + "\n";
    of.write(o.c_str(), o.size());
    of.flush();
    std::string data;
#pragma region Data collection

    for (auto i = 0; i < mesh->UnsafeGetVertices().size(); i++) {
      auto &vertexPosition = mesh->UnsafeGetVertices().at(i).m_position;
      auto &color = mesh->UnsafeGetVertices().at(i).m_color;
      data += "v " + std::to_string(vertexPosition.x + position.x) + " " +
              std::to_string(vertexPosition.y + position.y) + " " +
              std::to_string(vertexPosition.z + position.z) + " " +
              std::to_string(color.x) + " " + std::to_string(color.y) + " " +
              std::to_string(color.z) + "\n";
    }
    for (const auto &vertex : mesh->UnsafeGetVertices()) {
      data += "vn " + std::to_string(vertex.m_normal.x) + " " +
              std::to_string(vertex.m_normal.y) + " " +
              std::to_string(vertex.m_normal.z) + "\n";
    }

    for (const auto &vertex : mesh->UnsafeGetVertices()) {
      data += "vt " + std::to_string(vertex.m_texCoords.x) + " " +
              std::to_string(vertex.m_texCoords.y) + "\n";
    }
    // data += "s off\n";
    data += "# List of indices for faces vertices, with (x, y, z).\n";
    auto &triangles = mesh->UnsafeGetTriangles();
    for (auto i = 0; i < mesh->GetTriangleAmount(); i++) {
      const auto triangle = triangles[i];
      const auto f1 = triangle.x + startIndex;
      const auto f2 = triangle.y + startIndex;
      const auto f3 = triangle.z + startIndex;
      data += "f " + std::to_string(f1) + "/" + std::to_string(f1) + "/" +
              std::to_string(f1) + " " + std::to_string(f2) + "/" +
              std::to_string(f2) + "/" + std::to_string(f2) + " " +
              std::to_string(f3) + "/" + std::to_string(f3) + "/" +
              std::to_string(f3) + "\n";
    }
    startIndex += mesh->GetVerticesAmount();
#pragma endregion
    of.write(data.c_str(), data.size());
    of.flush();
  }
}

void SorghumLayer::ExportAllSorghumsModel(const std::string &filename) {
  std::ofstream of;
  of.open(filename, std::ofstream::out | std::ofstream::trunc);
  if (of.is_open()) {
    std::string start = "#Sorghum field, by Bosheng Li";
    start += "\n";
    of.write(start.c_str(), start.size());
    of.flush();

    unsigned startIndex = 1;
    std::vector<Entity> sorghums;
    m_sorghumQuery.ToEntityArray(Entities::GetCurrentScene(), sorghums);
    for (const auto &plant : sorghums) {
      ExportSorghum(plant, of, startIndex);
    }
    of.close();
    UNIENGINE_LOG("Sorghums saved as " + filename);
  } else {
    UNIENGINE_ERROR("Can't open file!");
  }
}
#ifdef RAYTRACERFACILITY
void SorghumLayer::RenderLightProbes() {
  if (m_probeTransforms.empty() || m_probeColors.empty() ||
      m_probeTransforms.size() != m_probeColors.size())
    return;
  Graphics::DrawGizmoMeshInstancedColored(DefaultResources::Primitives::Cube,
                                          m_probeColors, m_probeTransforms,
                                          glm::mat4(1.0f), m_lightProbeSize);
}
#endif
void SorghumLayer::CollectEntities(std::vector<Entity> &entities,
                                   const Entity &walker) {
  walker.ForEachChild([&](const std::shared_ptr<Scene> &scene, Entity child) {
    if (!child.HasPrivateComponent<MeshRenderer>())
      return;
    entities.push_back(child);
    CollectEntities(entities, child);
  });
}
#ifdef RAYTRACERFACILITY
void SorghumLayer::CalculateIlluminationFrameByFrame() {
  const auto *owners = Entities::UnsafeGetPrivateComponentOwnersList<
      TriangleIlluminationEstimator>(Entities::GetCurrentScene());
  if (!owners)
    return;
  m_processingEntities.clear();
  m_probeTransforms.clear();
  m_probeColors.clear();
  m_processingEntities.insert(m_processingEntities.begin(), owners->begin(),
                              owners->end());
  m_processingIndex = m_processingEntities.size();
  m_processing = true;
}
void SorghumLayer::CalculateIllumination() {
  const auto *owners = Entities::UnsafeGetPrivateComponentOwnersList<
      TriangleIlluminationEstimator>(Entities::GetCurrentScene());
  if (!owners)
    return;
  m_processingEntities.clear();
  m_probeTransforms.clear();
  m_probeColors.clear();
  m_processingEntities.insert(m_processingEntities.begin(), owners->begin(),
                              owners->end());
  m_processingIndex = m_processingEntities.size();
  while (m_processing) {
    m_processingIndex--;
    if (m_processingIndex == -1) {
      m_processing = false;
    } else {
      const float timer = Application::Time().CurrentTime();
      auto estimator =
          m_processingEntities[m_processingIndex]
              .GetOrSetPrivateComponent<TriangleIlluminationEstimator>()
              .lock();
      estimator->CalculateIlluminationForDescendents(m_rayProperties, m_seed,
                                                     m_pushDistance);
      m_probeTransforms.insert(m_probeTransforms.end(),
                               estimator->m_probeTransforms.begin(),
                               estimator->m_probeTransforms.end());
      m_probeColors.insert(m_probeColors.end(),
                           estimator->m_probeColors.begin(),
                           estimator->m_probeColors.end());
    }
  }
}
#endif
void SorghumLayer::Update() {
#ifdef RAYTRACERFACILITY
  if (m_displayLightProbes) {
    RenderLightProbes();
  }
  if (m_processing) {
    m_processingIndex--;
    if (m_processingIndex == -1) {
      m_processing = false;
    } else {
      const float timer = Application::Time().CurrentTime();
      auto estimator =
          m_processingEntities[m_processingIndex]
              .GetOrSetPrivateComponent<TriangleIlluminationEstimator>()
              .lock();
      estimator->CalculateIlluminationForDescendents(m_rayProperties, m_seed,
                                                     m_pushDistance);
      m_probeTransforms.insert(m_probeTransforms.end(),
                               estimator->m_probeTransforms.begin(),
                               estimator->m_probeTransforms.end());
      m_probeColors.insert(m_probeColors.end(),
                           estimator->m_probeColors.begin(),
                           estimator->m_probeColors.end());
      m_perPlantCalculationTime = Application::Time().CurrentTime() - timer;
    }
  }
#endif
}

void SorghumLayer::CreateGrid(RectangularSorghumFieldPattern &field,
                              const std::vector<Entity> &candidates) {
  const Entity entity =
      Entities::CreateEntity(Entities::GetCurrentScene(), "Field");
  std::vector<std::vector<glm::mat4>> matricesList;
  matricesList.resize(candidates.size());
  for (auto &i : matricesList) {
    i = std::vector<glm::mat4>();
  }
  field.GenerateField(matricesList);
  for (int i = 0; i < candidates.size(); i++) {
    CloneSorghums(entity, candidates[i], matricesList[i]);
  }
}

Entity SorghumLayer::CreateSorghum(
    const std::shared_ptr<ProceduralSorghum> &descriptor) {
  if (!descriptor) {
    UNIENGINE_ERROR("ProceduralSorghum empty!");
    return {};
  }
  Entity sorghum = CreateSorghum();
  auto sorghumData = sorghum.GetOrSetPrivateComponent<SorghumData>().lock();
  sorghumData->m_mode = (int)SorghumMode::ProceduralSorghum;
  sorghumData->m_descriptor = descriptor;
  sorghumData->SetTime(1.0f);
  sorghumData->GenerateGeometry();
  sorghumData->ApplyGeometry();
  return sorghum;
}
Entity SorghumLayer::CreateSorghum(
    const std::shared_ptr<SorghumStateGenerator> &descriptor) {
  if (!descriptor) {
    UNIENGINE_ERROR("SorghumStateGenerator empty!");
    return {};
  }
  Entity sorghum = CreateSorghum();
  auto sorghumData = sorghum.GetOrSetPrivateComponent<SorghumData>().lock();
  sorghumData->m_mode = (int)SorghumMode::SorghumStateGenerator;
  sorghumData->m_descriptor = descriptor;
  sorghumData->SetTime(1.0f);
  sorghumData->GenerateGeometry();
  sorghumData->ApplyGeometry();
  return sorghum;
}
std::shared_ptr<PointCloud>
SorghumLayer::ScanPointCloud(const Entity &sorghum, float boundingBoxRadius,
                             glm::vec2 boundingBoxHeightRange,
                             glm::vec2 pointDistance, float scannerAngle) {
  std::shared_ptr<PointCloud> pointCloud =
      AssetManager::CreateAsset<PointCloud>();
#ifdef RAYTRACERFACILITY
  auto boundingBoxHeight = boundingBoxHeightRange.y - boundingBoxHeightRange.x;
  auto planeSize =
      glm::vec2(boundingBoxRadius * 2.0f +
                    boundingBoxHeight / glm::cos(glm::radians(scannerAngle)),
                boundingBoxRadius * 2.0f);
  auto boundingBoxCenter =
      (boundingBoxHeightRange.y + boundingBoxHeightRange.x) / 2.0f;

  std::vector<uint64_t> entityHandles;
  std::vector<glm::vec3> points;
  std::vector<glm::vec3> colors;

  const auto column = unsigned(planeSize.x / pointDistance.x);
  const int columnStart = -(int)(column / 2);
  const auto row = unsigned(planeSize.y / pointDistance.y);
  const int rowStart = -(int)(row / 2);
  const auto size = column * row;
  auto gt = sorghum.GetDataComponent<GlobalTransform>();

  glm::vec3 front = glm::vec3(0, -1, 0);
  glm::vec3 up = glm::vec3(0, 0, -1);
  glm::vec3 left = glm::vec3(1, 0, 0);
  glm::vec3 actualVector =
      glm::normalize(glm::rotate(front, glm::radians(scannerAngle), up));
  glm::vec3 center = gt.GetPosition() + glm::vec3(0, boundingBoxCenter, 0) -
                     actualVector * (boundingBoxHeightRange.y / 2.0f /
                                     glm::cos(glm::radians(scannerAngle)));

  std::vector<PointCloudSample> pcSamples;
  pcSamples.resize(size * 2);
  std::vector<std::shared_future<void>> results;
  Jobs::ParallelFor(
      size,
      [&](unsigned i) {
        const int columnIndex = (int)i / row;
        const int rowIndex = (int)i % row;
        const auto position =
            center +
            left * (float)(columnStart + columnIndex) * pointDistance.x +
            up * (float)(rowStart + rowIndex) * pointDistance.y;
        pcSamples[i].m_start = position;
        pcSamples[i].m_direction = actualVector;
      },
      results);
  for (const auto &i : results)
    i.wait();
  auto plantPosition = gt.GetPosition();
  actualVector =
      glm::normalize(glm::rotate(front, glm::radians(-scannerAngle), up));
  center = gt.GetPosition() + glm::vec3(0, boundingBoxCenter, 0) -
           actualVector * (boundingBoxHeight / 2.0f /
                           glm::cos(glm::radians(scannerAngle)));

  std::vector<std::shared_future<void>> results2;
  Jobs::ParallelFor(
      size,
      [&](unsigned i) {
        const int columnIndex = (int)i / row;
        const int rowIndex = (int)i % row;
        const auto position =
            center +
            left * (float)(columnStart + columnIndex) * pointDistance.x +
            up * (float)(rowStart + rowIndex) * pointDistance.y;
        pcSamples[i + size].m_start = position;
        pcSamples[i + size].m_direction = actualVector;
      },
      results2);
  for (const auto &i : results2)
    i.wait();

  CudaModule::SamplePointCloud(
      Application::GetLayer<RayTracerLayer>()->m_environmentProperties,
      pcSamples);
  for (const auto &sample : pcSamples) {
    if (!sample.m_hit) {
      continue;
    }
    auto position = sample.m_end;
    if (glm::abs(position.x - plantPosition.x) > boundingBoxRadius ||
        position.y - plantPosition.y < boundingBoxHeightRange.x ||
        position.y - plantPosition.y > boundingBoxHeightRange.y) {
      continue;
    }
    points.push_back(sample.m_end);
    colors.push_back(sample.m_albedo);
    entityHandles.push_back(sample.m_handle);
  }
  for (int i = 0; i < points.size(); i++) {
    pointCloud->m_points.emplace_back(points[i]);
  }
#else
  UNIENGINE_ERROR("Ray tracer disabled!");
#endif
  return pointCloud;
}
void SorghumLayer::ScanPointCloudLabeled(
    const Entity &sorghum, const Entity &field, const std::filesystem::path &savePath,
    const PointCloudSampleSettings &settings) {
#ifdef RAYTRACERFACILITY
  auto boundingBoxHeight =
      settings.m_boundingBoxHeightRange.y - settings.m_boundingBoxHeightRange.x;
  auto planeSize = glm::vec2(
      settings.m_boundingBoxRadius * 2.0f +
          boundingBoxHeight / glm::cos(glm::radians(settings.m_scannerAngle)),
      settings.m_boundingBoxRadius * 2.0f);
  auto boundingBoxCenter = (settings.m_boundingBoxHeightRange.y +
                            settings.m_boundingBoxHeightRange.x) /
                           2.0f;

  std::vector<int> leafIndex;
  std::vector<int> leafPartIndex;
  std::vector<int> isMainPlant;
  std::vector<uint64_t> entityHandles;
  std::vector<glm::vec3> points;
  std::vector<glm::vec3> colors;

  const auto column = unsigned(planeSize.x / settings.m_pointDistance.x);
  const int columnStart = -(int)(column / 2);
  const auto row = unsigned(planeSize.y / settings.m_pointDistance.y);
  const int rowStart = -(int)(row / 2);
  const auto size = column * row;
  auto gt = sorghum.GetDataComponent<GlobalTransform>();

  glm::vec3 front = glm::vec3(0, -1, 0);
  glm::vec3 up = glm::vec3(0, 0, -1);
  glm::vec3 left = glm::vec3(1, 0, 0);
  glm::vec3 actualVector = glm::normalize(
      glm::rotate(front, glm::radians(settings.m_scannerAngle), up));
  glm::vec3 center =
      gt.GetPosition() + glm::vec3(0, boundingBoxCenter, 0) -
      actualVector * (settings.m_boundingBoxHeightRange.y / 2.0f /
                      glm::cos(glm::radians(settings.m_scannerAngle)));

  std::vector<PointCloudSample> pcSamples;
  pcSamples.resize(size * 2);
  std::vector<std::shared_future<void>> results;
  Jobs::ParallelFor(
      size,
      [&](unsigned i) {
        const int columnIndex = (int)i / row;
        const int rowIndex = (int)i % row;
        const auto position =
            center +
            left * (float)(columnStart + columnIndex) *
                settings.m_pointDistance.x +
            up * (float)(rowStart + rowIndex) * settings.m_pointDistance.y;
        pcSamples[i].m_start = position;
        pcSamples[i].m_direction = actualVector;
      },
      results);
  for (const auto &i : results)
    i.wait();
  auto plantPosition = gt.GetPosition();
  actualVector = glm::normalize(
      glm::rotate(front, glm::radians(-settings.m_scannerAngle), up));
  center = gt.GetPosition() + glm::vec3(0, boundingBoxCenter, 0) -
           actualVector * (boundingBoxHeight / 2.0f /
                           glm::cos(glm::radians(settings.m_scannerAngle)));

  std::vector<std::shared_future<void>> results2;
  Jobs::ParallelFor(
      size,
      [&](unsigned i) {
        const int columnIndex = (int)i / row;
        const int rowIndex = (int)i % row;
        const auto position =
            center +
            left * (float)(columnStart + columnIndex) *
                settings.m_pointDistance.x +
            up * (float)(rowStart + rowIndex) * settings.m_pointDistance.y;
        pcSamples[i + size].m_start = position;
        pcSamples[i + size].m_direction = actualVector;
      },
      results2);
  for (const auto &i : results2)
    i.wait();

  std::vector<std::pair<Handle, int>> mainPlantHandles = {};
  std::vector<std::pair<Handle, int>> plantHandles = {};
  mainPlantHandles.emplace_back(
      sorghum.GetOrSetPrivateComponent<MeshRenderer>().lock()->GetHandle(), 0);
  sorghum.ForEachChild([&](const std::shared_ptr<Scene> &scene, Entity child) {
    auto meshRenderer = child.GetOrSetPrivateComponent<MeshRenderer>();
    if (meshRenderer.expired() || !child.HasDataComponent<LeafTag>())
      return;
    auto handle = meshRenderer.lock()->GetHandle();
    auto index = child.GetDataComponent<LeafTag>().m_index + 1;
    mainPlantHandles.emplace_back(handle, index);
  });

  for(const auto& i : field.GetChildren()){
    if(i.GetIndex() == sorghum.GetIndex()) continue;
    plantHandles.emplace_back(
        i.GetOrSetPrivateComponent<MeshRenderer>().lock()->GetHandle(), 0);
    i.ForEachChild([&](const std::shared_ptr<Scene> &scene, Entity child) {
      auto meshRenderer = child.GetOrSetPrivateComponent<MeshRenderer>();
      if (meshRenderer.expired() || !child.HasDataComponent<LeafTag>())
        return;
      auto handle = meshRenderer.lock()->GetHandle();
      auto index = child.GetDataComponent<LeafTag>().m_index + 1;
      plantHandles.emplace_back(handle, index);
    });
  }

  CudaModule::SamplePointCloud(
      Application::GetLayer<RayTracerLayer>()->m_environmentProperties,
      pcSamples);
  if (settings.m_adjustBoundingBox) {
    std::mutex minX, minZ, maxX, maxZ;
    float minXVal, minZVal, maxXVal, maxZVal;
    minXVal = minZVal = 999999;
    maxXVal = maxZVal = -999999;
    std::vector<std::shared_future<void>> scanResults;
    Jobs::ParallelFor(
        pcSamples.size(),
        [&](unsigned i) {
          auto &sample = pcSamples[i];
          if (!sample.m_hit) {
            return;
          }
          for (const auto &pair : mainPlantHandles) {
            if (pair.first.GetValue() == sample.m_handle) {
              if (sample.m_end.x < minXVal) {
                std::lock_guard<std::mutex> lock(minX);
                minXVal = sample.m_end.x;
              } else if (sample.m_end.x > maxXVal) {
                std::lock_guard<std::mutex> lock(maxX);
                maxXVal = sample.m_end.x;
              }
              if (sample.m_end.z < minZVal) {
                std::lock_guard<std::mutex> lock(minZ);
                minZVal = sample.m_end.z;
              } else if (sample.m_end.z > maxZVal) {
                std::lock_guard<std::mutex> lock(maxZ);
                maxZVal = sample.m_end.z;
              }
              return;
            }
          }
        },
        scanResults);
    for (const auto &i : scanResults)
      i.wait();
    float xCenter = (minXVal + maxXVal) / 2.0f;
    float zCenter = (minZVal + maxZVal) / 2.0f;
    minXVal = xCenter + settings.m_adjustmentFactor * (minXVal - xCenter);
    maxXVal = xCenter + settings.m_adjustmentFactor * (maxXVal - xCenter);
    minZVal = zCenter + settings.m_adjustmentFactor * (minZVal - zCenter);
    maxZVal = zCenter + settings.m_adjustmentFactor * (maxZVal - zCenter);

    for (const auto &sample : pcSamples) {
      if (!sample.m_hit) {
        continue;
      }
      auto position = sample.m_end;
      if (position.x < minXVal || position.x > maxXVal ||
          position.z < minZVal || position.z > maxZVal ||
          position.y - plantPosition.y < settings.m_boundingBoxHeightRange.x ||
          position.y - plantPosition.y > settings.m_boundingBoxHeightRange.y) {
        continue;
      }
      points.push_back(sample.m_end);
      colors.push_back(sample.m_albedo);
      entityHandles.push_back(sample.m_handle);
    }
  } else {
    for (const auto &sample : pcSamples) {
      if (!sample.m_hit) {
        continue;
      }
      auto position = sample.m_end;
      if (glm::abs(position.x - plantPosition.x) >
              settings.m_boundingBoxRadius ||
          glm::abs(position.z - plantPosition.z) >
              settings.m_boundingBoxRadius ||
          position.y - plantPosition.y < settings.m_boundingBoxHeightRange.x ||
          position.y - plantPosition.y > settings.m_boundingBoxHeightRange.y) {
        continue;
      }
      points.push_back(sample.m_end);
      colors.push_back(sample.m_albedo);
      entityHandles.push_back(sample.m_handle);
    }
  }

  leafIndex.resize(points.size());
  leafPartIndex.resize(points.size());
  isMainPlant.resize(points.size());
  std::vector<std::shared_future<void>> results3;
  Jobs::ParallelFor(
      points.size(),
      [&](unsigned i) {
        if (colors[i].x != 0) {
          leafPartIndex[i] = 1;
        } else if (colors[i].y != 0) {
          leafPartIndex[i] = 2;
        } else {
          leafPartIndex[i] = 3;
        }
        isMainPlant[i] = 0;
        for (const auto &pair : mainPlantHandles) {
          if (pair.first.GetValue() == entityHandles[i]) {
            leafIndex[i] = pair.second;
            isMainPlant[i] = 1;
            return;
          }
        }
        for (const auto &pair : plantHandles) {
          if (pair.first.GetValue() == entityHandles[i]) {
            leafIndex[i] = pair.second;
            return;
          }
        }
      },
      results3);
  for (const auto &i : results3)
    i.wait();

  std::filebuf fb_binary;
  fb_binary.open(savePath.string(), std::ios::out | std::ios::binary);
  std::ostream outstream_binary(&fb_binary);
  if (outstream_binary.fail())
    throw std::runtime_error("failed to open " + savePath.string());
  /*
  std::filebuf fb_ascii;
  fb_ascii.open(filename + "-ascii.ply", std::ios::out);
  std::ostream outstream_ascii(&fb_ascii);
  if (outstream_ascii.fail()) throw std::runtime_error("failed to open " +
  filename);
  */
  PlyFile cube_file;

  cube_file.add_properties_to_element(
      "vertex", {"x", "z", "y"}, Type::FLOAT32, points.size(),
      reinterpret_cast<uint8_t *>(points.data()), Type::INVALID, 0);
  cube_file.add_properties_to_element(
      "color", {"red", "green", "blue"}, Type::FLOAT32, colors.size(),
      reinterpret_cast<uint8_t *>(colors.data()), Type::INVALID, 0);
  cube_file.add_properties_to_element(
      "leafIndex", {"value"}, Type::INT32, leafIndex.size(),
      reinterpret_cast<uint8_t *>(leafIndex.data()), Type::INVALID, 0);
  cube_file.add_properties_to_element(
      "leafPartIndex", {"value"}, Type::INT32, leafPartIndex.size(),
      reinterpret_cast<uint8_t *>(leafPartIndex.data()), Type::INVALID, 0);
  cube_file.add_properties_to_element(
      "isMainPlant", {"value"}, Type::INT32, isMainPlant.size(),
      reinterpret_cast<uint8_t *>(isMainPlant.data()), Type::INVALID, 0);
  // Write a binary file
  cube_file.write(outstream_binary, true);
#else
  UNIENGINE_ERROR("Ray tracer disabled!");
#endif
}
void SorghumLayer::LateUpdate() {
  std::vector<Entity> plants;
  m_sorghumQuery.ToEntityArray(Entities::GetCurrentScene(), plants);
  for (auto &plant : plants) {
    if (plant.HasPrivateComponent<SorghumData>()) {
      auto sorghumData = plant.GetOrSetPrivateComponent<SorghumData>().lock();
      auto proceduralSorghum =
          sorghumData->m_descriptor.Get<ProceduralSorghum>();
      if (proceduralSorghum &&
          proceduralSorghum->GetVersion() != sorghumData->m_recordedVersion) {
        sorghumData->Apply();
        sorghumData->GenerateGeometry();
        sorghumData->ApplyGeometry(true, true, false);
        continue;
      }
      auto sorghumStateGenerator =
          sorghumData->m_descriptor.Get<SorghumStateGenerator>();
      if (sorghumStateGenerator && sorghumStateGenerator->GetVersion() !=
                                       sorghumData->m_recordedVersion) {
        sorghumData->Apply();
        sorghumData->GenerateGeometry();
        sorghumData->ApplyGeometry(true, true, false);
      }
    }
  }
}

void PointCloudSampleSettings::OnInspect() {
  ImGui::DragFloat2("Point distance", &m_pointDistance.x, 0.0001f);
  ImGui::DragFloat("Scanner angle", &m_scannerAngle, 0.5f);

  ImGui::DragFloat2("Bounding box height range", &m_boundingBoxHeightRange.x,
                    0.01f);

  ImGui::Checkbox("Auto adjust bounding box", &m_adjustBoundingBox);
  if (m_adjustBoundingBox) {
    ImGui::DragFloat("Adjustment factor", &m_adjustmentFactor, 0.01f, 0.0f,
                     2.0f);
  } else {
    ImGui::DragFloat("Bounding box radius", &m_boundingBoxRadius, 0.01f);
  }
}
