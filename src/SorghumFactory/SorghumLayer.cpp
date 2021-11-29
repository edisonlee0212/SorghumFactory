#ifdef RAYTRACERFACILITY
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
void SorghumLayer::OnCreate() {
  ClassRegistry::RegisterDataComponent<PinnacleTag>("PinnacleTag");
  ClassRegistry::RegisterDataComponent<LeafTag>("LeafTag");
  ClassRegistry::RegisterDataComponent<SorghumTag>("SorghumTag");

  ClassRegistry::RegisterPrivateComponent<DepthCamera>("DepthCamera");

  ClassRegistry::RegisterPrivateComponent<Spline>("Spline");
  ClassRegistry::RegisterPrivateComponent<SorghumData>("SorghumData");

  ClassRegistry::RegisterAsset<SorghumProceduralDescriptor>(
      "SorghumProceduralDescriptor", ".spd");
  ClassRegistry::RegisterAsset<SorghumField>("SorghumField", ".sorghumfield");
  ClassRegistry::RegisterAsset<RectangularSorghumField>("RectangularSorghumField", ".rectsorghumfield");
  ClassRegistry::RegisterAsset<PositionsField>("PositionsField", ".possorghumfield");

  m_leafArchetype = EntityManager::CreateEntityArchetype("Leaf", LeafTag());
  m_leafQuery = EntityManager::CreateEntityQuery();
  m_leafQuery.SetAllFilters(LeafTag());

  m_pinnacleArchetype = EntityManager::CreateEntityArchetype("Pinnacle", PinnacleTag());
  m_pinnacleQuery = EntityManager::CreateEntityQuery();
  m_pinnacleQuery.SetAllFilters(PinnacleTag());

  m_sorghumArchetype =
      EntityManager::CreateEntityArchetype("Sorghum", SorghumTag());
  m_sorghumQuery = EntityManager::CreateEntityQuery();
  m_sorghumQuery.SetAllFilters(SorghumTag());

  if (!m_leafMaterial.Get<Material>()) {
    auto material = AssetManager::LoadMaterial(
        DefaultResources::GLPrograms::StandardProgram);
    m_leafMaterial = material;
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
    material->m_albedoColor =
        glm::vec3(165.0 / 256, 42.0 / 256, 42.0 / 256);
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
  const Entity entity = EntityManager::CreateEntity(
      EntityManager::GetCurrentScene(), m_sorghumArchetype, "Sorghum");
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
  const Entity entity = EntityManager::CreateEntity(
      EntityManager::GetCurrentScene(), m_leafArchetype);
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
  const Entity entity = EntityManager::CreateEntity(
      EntityManager::GetCurrentScene(), m_pinnacleArchetype);
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

void SorghumLayer::GenerateMeshForAllSorghums(int segmentAmount, int step) {
  std::vector<Entity> plants;
  m_sorghumQuery.ToEntityArray(EntityManager::GetCurrentScene(), plants);
  for (auto &plant : plants) {
    auto stemSpline = plant.GetOrSetPrivateComponent<Spline>().lock();
    // Form the stem spline. Feed with unused shared_ptr to itself.
    stemSpline->FormNodes(stemSpline);
  }
  EntityManager::ForEach<GlobalTransform>(
      EntityManager::GetCurrentScene(), JobManager::PrimaryWorkers(),
      m_leafQuery,
      [segmentAmount, step](int index, Entity entity, GlobalTransform &ltw) {
        auto spline = entity.GetOrSetPrivateComponent<Spline>().lock();
        auto stemSpline =
            entity.GetParent().GetOrSetPrivateComponent<Spline>().lock();
        if (stemSpline)
          spline->GenerateGeometry(stemSpline);
      });

  m_sorghumQuery.ToEntityArray(EntityManager::GetCurrentScene(), plants);
  for (auto &plant : plants) {
    plant.ForEachChild([](const std::shared_ptr<Scene> &scene, Entity child) {
      if (!child.HasPrivateComponent<Spline>())
        return;
      auto meshRenderer = child.GetOrSetPrivateComponent<MeshRenderer>().lock();
      auto spline = child.GetOrSetPrivateComponent<Spline>().lock();
      meshRenderer->m_mesh.Get<Mesh>()->SetVertices(17, spline->m_vertices,
                                                    spline->m_triangles);
    });
    if (plant.HasPrivateComponent<SorghumData>())
      plant.GetOrSetPrivateComponent<SorghumData>().lock()->m_meshGenerated =
          true;
  }
}

Entity SorghumLayer::ImportPlant(const std::filesystem::path &path,
                                 const std::string &name) {
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
}

void SorghumLayer::OnInspect() {
  if (ImGui::Begin("Sorghum")) {
#ifdef RAYTRACERFACILITY
    if(ImGui::TreeNodeEx("Illumination Estimation")) {
      ImGui::Checkbox("Display light probes", &m_displayLightProbes);
      if (m_displayLightProbes) {
        ImGui::DragFloat("Size", &m_lightProbeSize, 0.0001f, 0.0001f, 0.2f,
                         "%.5f");
      }
      ImGui::DragInt("Seed", &m_seed);
      ImGui::DragFloat("Push distance along normal", &m_pushDistance, 0.0001f, -1.0f, 1.0f, "%.5f");
      m_rayProperties.OnInspect();
      if (ImGui::Button("Generate mesh")) {
        GenerateMeshForAllSorghums();
      }

      if (ImGui::Button("Calculate illumination")) {
        CalculateIlluminationFrameByFrame();
      }
      if (ImGui::Button("Calculate illumination instantly")) {
        CalculateIllumination();
      }
      ImGui::TreePop();
    }
#endif

    if (ImGui::DragInt("Segment amount", &m_segmentAmount)) {
      m_segmentAmount = glm::max(2, m_segmentAmount);
    }
    if (ImGui::DragInt("Step amount", &m_step)) {
      m_step = glm::max(2, m_step);
    }

    ImGui::Separator();

    if(EditorManager::DragAndDropButton<Texture2D>(m_leafAlbedoTexture,
                                                   "Replace Leaf Albedo Texture")){
      auto tex = m_leafAlbedoTexture.Get<Texture2D>();
      if(tex){
        m_leafMaterial.Get<Material>()->m_albedoTexture = m_leafAlbedoTexture;
        std::vector<Entity> sorghumEntities;
        m_sorghumQuery.ToEntityArray(EntityManager::GetCurrentScene(), sorghumEntities, false);
        for(const auto& i : sorghumEntities){
          if(i.HasPrivateComponent<MeshRenderer>()){
            i.GetOrSetPrivateComponent<MeshRenderer>().lock()->m_material.Get<Material>()->m_albedoTexture = m_leafAlbedoTexture;
          }
        }
      }
    }

    if(EditorManager::DragAndDropButton<Texture2D>(m_leafNormalTexture,
                                                    "Replace Leaf Normal Texture")){
      auto tex = m_leafNormalTexture.Get<Texture2D>();
      if(tex){
        m_leafMaterial.Get<Material>()->m_normalTexture = m_leafNormalTexture;
        std::vector<Entity> sorghumEntities;
        m_sorghumQuery.ToEntityArray(EntityManager::GetCurrentScene(), sorghumEntities, false);
        for(const auto& i : sorghumEntities){
          if(i.HasPrivateComponent<MeshRenderer>()){
            i.GetOrSetPrivateComponent<MeshRenderer>().lock()->m_material.Get<Material>()->m_normalTexture = m_leafNormalTexture;
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
          EntityManager::DeleteEntity(EntityManager::GetCurrentScene(), i);
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
    m_sorghumQuery.ToEntityArray(EntityManager::GetCurrentScene(), sorghums);
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
  RenderManager::DrawGizmoMeshInstancedColored(
      DefaultResources::Primitives::Cube, m_probeColors, m_probeTransforms,
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
  const auto *owners = EntityManager::UnsafeGetPrivateComponentOwnersList<
      TriangleIlluminationEstimator>(EntityManager::GetCurrentScene());
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
  const auto *owners = EntityManager::UnsafeGetPrivateComponentOwnersList<
      TriangleIlluminationEstimator>(EntityManager::GetCurrentScene());
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
      estimator->CalculateIlluminationForDescendents(m_rayProperties, m_seed, m_pushDistance);
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
      estimator->CalculateIlluminationForDescendents(m_rayProperties, m_seed, m_pushDistance);
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
      EntityManager::CreateEntity(EntityManager::GetCurrentScene(), "Field");
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
    const std::shared_ptr<SorghumProceduralDescriptor> &descriptor) {
  if (!descriptor) {
    UNIENGINE_ERROR("Descriptor empty!");
    return Entity();
  }
  Entity sorghum = CreateSorghum();
  auto sorghumData = sorghum.GetOrSetPrivateComponent<SorghumData>().lock();
  sorghumData->m_parameters = descriptor;
  sorghumData->ApplyParameters();
  sorghumData->GenerateGeometrySeperated(false);
  return sorghum;
}
