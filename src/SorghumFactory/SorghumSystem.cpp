#include <MLVQRenderer.hpp>
#include <SorghumData.hpp>
#include <SorghumSystem.hpp>
#include <TriangleIlluminationEstimator.hpp>
using namespace RayTracerFacility;
using namespace SorghumFactory;
using namespace UniEngine;

void RectangularSorghumField::GenerateField(
    std::vector<std::vector<glm::mat4>> &matricesList) {
  const int size = matricesList.size();
  glm::vec2 center = glm::vec2(m_distances.x * (m_size.x - 1),
                               m_distances.y * (m_size.y - 1)) /
                     2.0f;
  for (int xi = 0; xi < m_size.x; xi++) {
    for (int yi = 0; yi < m_size.y; yi++) {
      const auto selectedIndex = glm::linearRand(0, size - 1);
      matricesList[selectedIndex].push_back(
          glm::translate(glm::vec3(xi * m_distances.x - center.x, 0.0f,
                                   yi * m_distances.y - center.y)) *
          glm::mat4_cast(glm::quat(glm::radians(
              glm::vec3(glm::gaussRand(0.0f, m_rotationVariation.x),
                        glm::gaussRand(0.0f, m_rotationVariation.y),
                        glm::gaussRand(0.0f, m_rotationVariation.z))))) *
              glm::scale(glm::vec3(1.0f)));
    }
  }
}
void SorghumSystem::OnCreate() { Enable(); }
void SorghumSystem::Start() {
  m_leafArchetype = EntityManager::CreateEntityArchetype("Leaf", LeafTag());
  m_leafQuery = EntityManager::CreateEntityQuery();
  m_leafQuery.SetAllFilters(LeafTag());

  m_sorghumArchetype =
      EntityManager::CreateEntityArchetype("Sorghum", SorghumTag());
  m_sorghumQuery = EntityManager::CreateEntityQuery();
  m_sorghumQuery.SetAllFilters(SorghumTag());

  if (!m_leafNodeMaterial.Get<Material>()) {
    m_leafNodeMaterial = AssetManager::LoadMaterial(
        DefaultResources::GLPrograms::StandardProgram);
    m_leafNodeMaterial.Get<Material>()->m_albedoColor = glm::vec3(0, 1, 0);
  }
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

  /*
  m_instancedLeafMaterial = AssetManager::LoadMaterial(
      DefaultResources::GLPrograms::StandardInstancedProgram);
  m_instancedLeafMaterial.Get<Material>()->m_cullingMode =
      MaterialCullingMode::Off;
  m_instancedLeafMaterial.Get<Material>()->m_roughness = 1.0f;
  m_instancedLeafMaterial.Get<Material>()->m_metallic = 0.0f;
  */
  m_ready = true;
}

Entity SorghumSystem::CreateSorghum(bool segmentedMask) {
  Transform transform;
  transform.SetScale(glm::vec3(1.0f));
  const Entity entity =
      EntityManager::CreateEntity(m_sorghumArchetype, "Sorghum");
  entity.GetOrSetPrivateComponent<Spline>();
  auto sorghumData = entity.GetOrSetPrivateComponent<SorghumData>().lock();
  sorghumData->m_segmentedMask = segmentedMask;
  entity.SetName("Sorghum");
  entity.GetOrSetPrivateComponent<TriangleIlluminationEstimator>();
  return entity;
}

Entity SorghumSystem::CreateSorghumLeaf(const Entity &plantEntity,
                                        int leafIndex) {
  const Entity entity = EntityManager::CreateEntity(m_leafArchetype);
  entity.SetName("Leaf");
  entity.SetParent(plantEntity);
  Transform transform;
  transform.SetScale(glm::vec3(1.0f));
  auto spline = entity.GetOrSetPrivateComponent<Spline>().lock();
  LeafTag tag;
  tag.m_index = leafIndex;
  entity.SetDataComponent(tag);
  entity.SetDataComponent(transform);
  if (plantEntity.GetOrSetPrivateComponent<SorghumData>()
          .lock()
          ->m_segmentedMask) {
    auto mmc = entity.GetOrSetPrivateComponent<MeshRenderer>().lock();
    // mmc->m_material = m_segmentedLeafMaterials[leafIndex];
    {
      auto material = AssetManager::LoadMaterial(
          DefaultResources::GLPrograms::StandardProgram);
      mmc->m_material = material;
      material->SetProgram(DefaultResources::GLPrograms::StandardProgram);
      material->m_cullingMode = MaterialCullingMode::Off;
      material->m_albedoColor =
          glm::linearRand(glm::vec3(0.0f), glm::vec3(1.0f));
      material->m_roughness = 1.0f;
      material->m_metallic = 0.0f;
    }
    mmc->m_mesh = AssetManager::CreateAsset<Mesh>();
  } else {
    auto mmc = entity.GetOrSetPrivateComponent<MeshRenderer>().lock();
    mmc->m_material = m_leafMaterial;
    mmc->m_mesh = AssetManager::CreateAsset<Mesh>();

    auto rtt = entity.GetOrSetPrivateComponent<MLVQRenderer>().lock();
    rtt->Sync();
    rtt->m_materialIndex = 1;
  }
  return entity;
}

void SorghumSystem::GenerateMeshForAllSorghums(int segmentAmount, int step) {
  std::vector<Entity> plants;
  m_sorghumQuery.ToEntityArray(plants);
  for (auto &plant : plants) {
    auto stemSpline = plant.GetOrSetPrivateComponent<Spline>().lock();
    // Form the stem spline. Feed with unused shared_ptr to itself.
    stemSpline->FormNodes(stemSpline);
  }
  EntityManager::ForEach<GlobalTransform>(
      JobManager::PrimaryWorkers(), m_leafQuery,
      [segmentAmount, step](int index, Entity entity, GlobalTransform &ltw) {
        auto spline = entity.GetOrSetPrivateComponent<Spline>().lock();
        auto stemSpline =
            entity.GetParent().GetOrSetPrivateComponent<Spline>().lock();
        if (stemSpline)
          spline->GenerateGeometry(stemSpline);
      });

  m_sorghumQuery.ToEntityArray(plants);
  for (auto &plant : plants) {
    plant.ForEachChild([](Entity child) {
      if (!child.HasPrivateComponent<Spline>())
        return;
      auto meshRenderer = child.GetOrSetPrivateComponent<MeshRenderer>().lock();
      auto spline = child.GetOrSetPrivateComponent<Spline>().lock();
      meshRenderer->m_mesh.Get<Mesh>()->SetVertices(17, spline->m_vertices,
                                                    spline->m_indices);
    });
    if (plant.HasPrivateComponent<SorghumData>())
      plant.GetOrSetPrivateComponent<SorghumData>().lock()->m_meshGenerated =
          true;
  }
}

Entity SorghumSystem::ImportPlant(const std::filesystem::path &path,
                                  const std::string &name, bool segmentedMask) {
  std::ifstream file(path, std::fstream::in);
  if (!file.is_open()) {
    UNIENGINE_LOG("Failed to open file!");
    return Entity();
  }
  // Number of leaves in the file
  int leafCount;
  file >> leafCount;
  const auto sorghum = CreateSorghum(segmentedMask);
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

void SorghumSystem::OnInspect() {
  if (!m_ready) {
    ImGui::Text("System not ready!");
    return;
  }
  ImGui::Checkbox("Display light probes", &m_displayLightProbes);
  ImGui::DragInt("Seed", &m_seed);
  if (ImGui::Button("Generate mesh")) {
    GenerateMeshForAllSorghums();
  }

  if (ImGui::Button("Calculate illumination")) {
    RayTracerFacility::IlluminationEstimationProperties properties;
    properties.m_skylightPower = 1.0f;
    properties.m_bounceLimit = 2;
    properties.m_seed = glm::abs(m_seed);
    properties.m_numPointSamples = 100;
    properties.m_numScatterSamples = 10;
    CalculateIllumination(properties);
  }
  if (ImGui::Button("Create...")) {
    ImGui::OpenPopup("New sorghum wizard");
  }
  if (ImGui::BeginPopupModal("New sorghum wizard", nullptr,
                             ImGuiWindowFlags_AlwaysAutoResize)) {
    static std::vector<SorghumParameters> newSorghumParameters;
    static std::vector<glm::vec3> newSorghumPositions;
    static std::vector<glm::vec3> newSorghumRotations;
    static int newSorghumAmount = 1;
    static int currentFocusedNewSorghumIndex = 0;
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
    ImGui::BeginChild("ChildL", ImVec2(300, 400), true,
                      ImGuiWindowFlags_None | ImGuiWindowFlags_MenuBar);
    if (ImGui::BeginMenuBar()) {
      if (ImGui::BeginMenu("Settings")) {
        static float distance = 10;
        static float variance = 4;
        static float yAxisVar = 180.0f;
        static float xzAxisVar = 0.0f;
        static int expand = 1;
        if (ImGui::BeginMenu("Create field...")) {
          ImGui::DragFloat("Avg. Y axis rotation", &yAxisVar, 0.01f, 0.0f,
                           180.0f);
          ImGui::DragFloat("Avg. XZ axis rotation", &xzAxisVar, 0.01f, 0.0f,
                           90.0f);
          ImGui::DragFloat("Avg. Distance", &distance, 0.01f);
          ImGui::DragFloat("Position variance", &variance, 0.01f);
          ImGui::DragInt("Expand", &expand, 1, 0, 3);
          if (ImGui::Button("Apply")) {
            newSorghumAmount = (2 * expand + 1) * (2 * expand + 1);
            newSorghumPositions.resize(newSorghumAmount);
            newSorghumRotations.resize(newSorghumAmount);
            const auto currentSize = newSorghumParameters.size();
            newSorghumParameters.resize(newSorghumAmount);
            for (auto i = currentSize; i < newSorghumAmount; i++) {
              newSorghumParameters[i] = newSorghumParameters[0];
            }
            int index = 0;
            for (int i = -expand; i <= expand; i++) {
              for (int j = -expand; j <= expand; j++) {
                glm::vec3 value = glm::vec3(i * distance, 0, j * distance);
                value.x += glm::linearRand(-variance, variance);
                value.z += glm::linearRand(-variance, variance);
                newSorghumPositions[index] = value;
                value = glm::vec3(glm::linearRand(-xzAxisVar, xzAxisVar),
                                  glm::linearRand(-yAxisVar, yAxisVar),
                                  glm::linearRand(-xzAxisVar, xzAxisVar));
                newSorghumRotations[index] = value;
                index++;
              }
            }
          }
          ImGui::EndMenu();
        }
        ImGui::InputInt("New sorghum amount", &newSorghumAmount);
        if (newSorghumAmount < 1)
          newSorghumAmount = 1;
        FileUtils::OpenFile(
            "Import parameters for all", "Sorghum Parameters",
            {".sorghumparam"}, [](const std::filesystem::path &path) {
              newSorghumParameters[0].Load(path);
              for (int i = 1; i < newSorghumParameters.size(); i++)
                newSorghumParameters[i] = newSorghumParameters[0];
            });
        ImGui::EndMenu();
      }
      ImGui::EndMenuBar();
    }
    ImGui::Columns(1);
    if (newSorghumPositions.size() < newSorghumAmount) {
      const auto currentSize = newSorghumPositions.size();
      newSorghumParameters.resize(newSorghumAmount);
      for (auto i = currentSize; i < newSorghumAmount; i++) {
        newSorghumParameters[i] = newSorghumParameters[0];
      }
      newSorghumPositions.resize(newSorghumAmount);
      newSorghumRotations.resize(newSorghumAmount);
    }
    for (auto i = 0; i < newSorghumAmount; i++) {
      std::string title = "New Sorghum No.";
      title += std::to_string(i);
      const bool opened = ImGui::TreeNodeEx(
          title.c_str(), ImGuiTreeNodeFlags_NoTreePushOnOpen |
                             ImGuiTreeNodeFlags_OpenOnArrow |
                             ImGuiTreeNodeFlags_NoAutoOpenOnLog |
                             (currentFocusedNewSorghumIndex == i
                                  ? ImGuiTreeNodeFlags_Framed
                                  : ImGuiTreeNodeFlags_FramePadding));
      if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
        currentFocusedNewSorghumIndex = i;
      }
      if (opened) {
        ImGui::TreePush();
        ImGui::InputFloat3(("Position##" + std::to_string(i)).c_str(),
                           &newSorghumPositions[i].x);
        ImGui::TreePop();
      }
    }

    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::SameLine();
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
    ImGui::BeginChild("ChildR", ImVec2(400, 400), true,
                      ImGuiWindowFlags_None | ImGuiWindowFlags_MenuBar);
    if (ImGui::BeginMenuBar()) {
      if (ImGui::BeginMenu("Parameters")) {
        FileUtils::OpenFile(
            "Import parameters", "Sorghum Params", {".sorghumparam"},
            [](const std::filesystem::path &path) {
              newSorghumParameters[currentFocusedNewSorghumIndex].Load(path);
            });

        FileUtils::SaveFile(
            "Export parameters", "Sorghum Params", {".sorghumparam"},
            [](const std::filesystem::path &path) {
              newSorghumParameters[currentFocusedNewSorghumIndex].Load(path);
            });
        ImGui::EndMenu();
      }
      ImGui::EndMenuBar();
    }
    ImGui::Columns(1);
    ImGui::PushItemWidth(200);
    newSorghumParameters[currentFocusedNewSorghumIndex].OnInspect();
    ImGui::PopItemWidth();
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::Separator();
    if (ImGui::Button("OK", ImVec2(120, 0))) {
      auto field = EntityManager::CreateEntity("Field");
      // Create sorghums here.
      for (auto i = 0; i < newSorghumAmount; i++) {
        Entity sorghum = CreateSorghum();
        auto sorghumTransform = sorghum.GetDataComponent<Transform>();
        sorghumTransform.SetPosition(newSorghumPositions[i]);
        sorghumTransform.SetEulerRotation(glm::radians(newSorghumRotations[i]));
        sorghum.SetDataComponent(sorghumTransform);
        auto sorghumData =
            sorghum.GetOrSetPrivateComponent<SorghumData>().lock();
        sorghumData->m_parameters = newSorghumParameters[i];
        sorghumData->ApplyParameters();
        sorghum.SetParent(field);
      }
      GenerateMeshForAllSorghums();
      ImGui::CloseCurrentPopup();
    }
    ImGui::SetItemDefaultFocus();
    ImGui::SameLine();
    if (ImGui::Button("Cancel", ImVec2(120, 0))) {
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
  if (ImGui::Button("Create field...")) {
    ImGui::OpenPopup("Sorghum field wizard");
  }
  const ImVec2 center = ImGui::GetMainViewport()->GetCenter();
  ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  if (ImGui::BeginPopupModal("Sorghum field wizard", nullptr,
                             ImGuiWindowFlags_AlwaysAutoResize)) {
    static RectangularSorghumField field;
    ImGui::DragInt2("Size", &field.m_size.x, 1, 1, 10);
    ImGui::DragFloat2("Distance", &field.m_distances.x, 0.1f, 0.0f, 10.0f);
    ImGui::DragFloat3("Rotation variance", &field.m_rotationVariation.x, 0.1f,
                      0.0f, 10.0f);
    if (ImGui::Button("OK", ImVec2(120, 0))) {
      std::vector<Entity> candidates;
      candidates.push_back(ImportPlant(std::filesystem::path("../Resources") /
                                           "Sorghum/skeleton_procedural_1.txt",
                                       "Sorghum 1", true));
      candidates.push_back(ImportPlant(std::filesystem::path("../Resources") /
                                           "Sorghum/skeleton_procedural_2.txt",
                                       "Sorghum 2", true));
      candidates.push_back(ImportPlant(std::filesystem::path("../Resources") /
                                           "Sorghum/skeleton_procedural_3.txt",
                                       "Sorghum 3", true));
      candidates.push_back(ImportPlant(std::filesystem::path("../Resources") /
                                           "Sorghum/skeleton_procedural_4.txt",
                                       "Sorghum 4", true));
      GenerateMeshForAllSorghums();

      CreateGrid(field, candidates);
      for (auto &i : candidates)
        EntityManager::DeleteEntity(i);
      ImGui::CloseCurrentPopup();
    }
    ImGui::SetItemDefaultFocus();
    ImGui::SameLine();
    if (ImGui::Button("Cancel", ImVec2(120, 0))) {
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
  FileUtils::SaveFile("Export OBJ for all sorghums", "3D Model", {".obj"},
                      [this](const std::filesystem::path &path) {
                        ExportAllSorghumsModel(path.string());
                      });

  static bool opened = false;
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
        std::to_string(m_processingEntities.size() - m_processingIndex) + "/" +
        std::to_string(m_processingEntities.size());
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
}

void SorghumSystem::CloneSorghums(const Entity &parent, const Entity &original,
                                  std::vector<glm::mat4> &matrices) {
  auto sorghumData = original.GetOrSetPrivateComponent<SorghumData>().lock();
  for (const auto &matrix : matrices) {
    Entity sorghum = CreateSorghum(sorghumData->m_segmentedMask);
    Transform transform;
    transform.m_value = matrix;

    auto newSpline = sorghum.GetOrSetPrivateComponent<Spline>().lock();
    auto spline = original.GetOrSetPrivateComponent<Spline>().lock();
    newSpline->Copy(spline);

    original.ForEachChild([this, &sorghum, &matrices](Entity child) {
      if (!child.HasDataComponent<LeafTag>())
        return;
      auto tag = child.GetDataComponent<LeafTag>();
      const auto newChild = CreateSorghumLeaf(sorghum, tag.m_index);
      newChild.SetDataComponent(tag);
      newChild.SetDataComponent(child.GetDataComponent<Transform>());
      auto newSpline = newChild.GetOrSetPrivateComponent<Spline>().lock();
      auto spline = child.GetOrSetPrivateComponent<Spline>().lock();
      newSpline->Copy(spline);

      if (child.HasPrivateComponent<MLVQRenderer>()) {
        auto newRayTracedRenderer =
            newChild.GetOrSetPrivateComponent<MLVQRenderer>().lock();
        newRayTracedRenderer->m_materialIndex = 1;
        newRayTracedRenderer->Sync();
      }
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

void SorghumSystem::ExportSorghum(const Entity &sorghum, std::ofstream &of,
                                  unsigned &startIndex) {
  const std::string start = "#Sorghum\n";
  of.write(start.c_str(), start.size());
  of.flush();
  const auto position =
      sorghum.GetDataComponent<GlobalTransform>().GetPosition();
  sorghum.ForEachChild([&](Entity child) {
    if (!child.HasPrivateComponent<MeshRenderer>())
      return;
    const auto leafMesh = child.GetOrSetPrivateComponent<MeshRenderer>()
                              .lock()
                              ->m_mesh.Get<Mesh>();
    ObjExportHelper(position, leafMesh, of, startIndex);
  });
}

void SorghumSystem::ObjExportHelper(glm::vec3 position,
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

void SorghumSystem::ExportAllSorghumsModel(const std::string &filename) {
  std::ofstream of;
  of.open(filename, std::ofstream::out | std::ofstream::trunc);
  if (of.is_open()) {
    std::string start = "#Sorghum field, by Bosheng Li";
    start += "\n";
    of.write(start.c_str(), start.size());
    of.flush();

    unsigned startIndex = 1;
    std::vector<Entity> sorghums;
    m_sorghumQuery.ToEntityArray(sorghums);
    for (const auto &plant : sorghums) {
      ExportSorghum(plant, of, startIndex);
    }
    of.close();
    UNIENGINE_LOG("Sorghums saved as " + filename);
  } else {
    UNIENGINE_ERROR("Can't open file!");
  }
}

void SorghumSystem::RenderLightProbes() {
  if (m_probeTransforms.empty() || m_probeColors.empty() ||
      m_probeTransforms.size() != m_probeColors.size())
    return;
  RenderManager::DrawGizmoMeshInstancedColored(
      DefaultResources::Primitives::Cube, m_probeColors, m_probeTransforms,
      glm::mat4(1.0f), 0.2f);
}

void SorghumSystem::CollectEntities(std::vector<Entity> &entities,
                                    const Entity &walker) {
  walker.ForEachChild([&](Entity child) {
    if (!child.HasPrivateComponent<MeshRenderer>())
      return;
    entities.push_back(child);
    CollectEntities(entities, child);
  });
}

void SorghumSystem::CalculateIllumination(
    const RayTracerFacility::IlluminationEstimationProperties &properties) {
  const auto *owners = EntityManager::UnsafeGetPrivateComponentOwnersList<
      TriangleIlluminationEstimator>();
  if (!owners)
    return;
  m_processingEntities.clear();
  m_probeTransforms.clear();
  m_probeColors.clear();
  m_properties = properties;
  m_properties.m_pushNormal = true;
  m_processingEntities.insert(m_processingEntities.begin(), owners->begin(),
                              owners->end());
  m_processingIndex = m_processingEntities.size();
  m_processing = true;
}

void SorghumSystem::Update() {
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
      estimator->CalculateIllumination(m_properties);
      m_probeTransforms.insert(m_probeTransforms.end(),
                               estimator->m_probeTransforms.begin(),
                               estimator->m_probeTransforms.end());
      m_probeColors.insert(m_probeColors.end(),
                           estimator->m_probeColors.begin(),
                           estimator->m_probeColors.end());
      m_perPlantCalculationTime = Application::Time().CurrentTime() - timer;
      const auto count = m_probeTransforms.size();
      m_lightProbeRenderingColorBuffer.SetData(
          static_cast<GLsizei>(count) * sizeof(glm::vec4), m_probeColors.data(),
          GL_DYNAMIC_DRAW);
      m_lightProbeRenderingTransformBuffer.SetData(
          static_cast<GLsizei>(count) * sizeof(glm::mat4),
          m_probeTransforms.data(), GL_DYNAMIC_DRAW);
    }
  }
}

void SorghumSystem::CreateGrid(SorghumField &field,
                               const std::vector<Entity> &candidates) {
  const Entity entity = EntityManager::CreateEntity("Field");
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

void SorghumSystem::Relink(const std::unordered_map<Handle, Handle> &map) {
  ISystem::Relink(map);
}
void SorghumSystem::Deserialize(const YAML::Node &in) {
  ISerializable::Deserialize(in);
}
void SorghumSystem::CollectAssetRef(std::vector<AssetRef> &list) {
  ISystem::CollectAssetRef(list);
}
void SorghumSystem::Serialize(YAML::Emitter &out) {
  ISerializable::Serialize(out);
}
