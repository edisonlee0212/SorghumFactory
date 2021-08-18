#include <PlantSystem.hpp>
#include <SorghumData.hpp>
#include <SorghumSystem.hpp>
#include <TriangleIlluminationEstimator.hpp>

using namespace PlantFactory;
using namespace UniEngine;

PlantNode::PlantNode(glm::vec3 position, float angle, float width,
                     glm::vec3 axis, bool isLeaf) {
  m_position = position;
  m_theta = angle;
  m_width = width;
  m_axis = axis;
  m_isLeaf = isLeaf;
}

void Spline::Import(std::ifstream &stream) {
  int curveAmount;
  stream >> curveAmount;
  m_curves.clear();
  for (int i = 0; i < curveAmount; i++) {
    glm::vec3 cp[4];
    float x, y, z;
    for (auto &j : cp) {
      stream >> x >> z >> y;
      j = glm::vec3(x, y, z) * 10.0f;
    }
    m_curves.emplace_back(cp[0], cp[1], cp[2], cp[3]);
  }
}

glm::vec3 Spline::EvaluatePointFromCurve(float point) {
  const float splineU = glm::clamp(point, 0.0f, 1.0f) * float(m_curves.size());

  // Decompose the global u coordinate on the spline
  float integerPart;
  const float fractionalPart = modff(splineU, &integerPart);

  auto curveIndex = int(integerPart);
  auto curveU = fractionalPart;

  // If evaluating the very last point on the spline
  if (curveIndex == m_curves.size() && curveU <= 0.0f) {
    // Flip to the end of the last patch
    curveIndex--;
    curveU = 1.0f;
  }
  return m_curves.at(curveIndex).GetPoint(curveU);
}

glm::vec3 Spline::EvaluateAxisFromCurve(float point) {
  const float splineU = glm::clamp(point, 0.0f, 1.0f) * float(m_curves.size());

  // Decompose the global u coordinate on the spline
  float integerPart;
  const float fractionalPart = modff(splineU, &integerPart);

  auto curveIndex = int(integerPart);
  auto curveU = fractionalPart;

  // If evaluating the very last point on the spline
  if (curveIndex == m_curves.size() && curveU <= 0.0f) {
    // Flip to the end of the last patch
    curveIndex--;
    curveU = 1.0f;
  }
  return m_curves.at(curveIndex).GetAxis(curveU);
}

void Spline::OnGui() {
  if (ImGui::TreeNodeEx("Curves", ImGuiTreeNodeFlags_DefaultOpen)) {
    for (int i = 0; i < m_curves.size(); i++) {
      ImGui::Text(("Curve" + std::to_string(i)).c_str());
      ImGui::InputFloat3("CP0", &m_curves[i].m_p0.x);
      ImGui::InputFloat3("CP1", &m_curves[i].m_p1.x);
      ImGui::InputFloat3("CP2", &m_curves[i].m_p2.x);
      ImGui::InputFloat3("CP3", &m_curves[i].m_p3.x);
    }
    ImGui::TreePop();
  }
}

void Spline::Clone(const std::shared_ptr<IPrivateComponent> &target) {
  *this = *std::static_pointer_cast<Spline>(target);
}

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
          glm::scale(glm::vec3(1.0f)));
    }
  }
}

void SorghumSystem::OnCreate() {

  m_plantSystem = EntityManager::GetSystem<PlantSystem>();

  m_leafNodeMaterial =
      AssetManager::LoadMaterial(DefaultResources::GLPrograms::StandardProgram);
  m_leafNodeMaterial->m_albedoColor = glm::vec3(0, 1, 0);

  m_leafArchetype = EntityManager::CreateEntityArchetype("Leaf", LeafInfo());
  m_leafQuery = EntityManager::CreateEntityQuery();
  m_leafQuery.SetAllFilters(LeafInfo());

  m_leafMaterial =
      AssetManager::LoadMaterial(DefaultResources::GLPrograms::StandardProgram);
  m_leafMaterial->SetProgram(DefaultResources::GLPrograms::StandardProgram);
  m_leafMaterial->m_cullingMode = MaterialCullingMode::Off;
  m_leafSurfaceTexture = AssetManager::Import<Texture2D>(
      std::filesystem::path(PLANT_FACTORY_RESOURCE_FOLDER) / "Textures/leafSurfaceBright.jpg");


  m_rayTracedLeafSurfaceTexture = AssetManager::Import<Texture2D>(
      std::filesystem::path(PLANT_FACTORY_RESOURCE_FOLDER) / "Textures/leafSurfaceBright.jpg");

  m_leafMaterial->SetTexture(TextureType::Albedo, m_leafSurfaceTexture);
  m_leafMaterial->m_roughness = 0.0f;
  m_leafMaterial->m_metallic = 0.0f;

  m_instancedLeafMaterial = AssetManager::LoadMaterial(
      DefaultResources::GLPrograms::StandardInstancedProgram);
  m_instancedLeafMaterial->m_cullingMode = MaterialCullingMode::Off;
  m_instancedLeafMaterial->SetTexture(TextureType::Albedo,
                                      m_leafSurfaceTexture);
  m_instancedLeafMaterial->m_roughness = 0.0f;
  m_instancedLeafMaterial->m_metallic = 0.0f;

  m_plantSystem->m_plantGrowthModels.insert_or_assign(
      PlantType::Sorghum, [this](std::vector<InternodeCandidate> &candidates) {
        FormCandidates(candidates);
      });

  m_plantSystem->m_plantMeshGenerators.insert_or_assign(
      PlantType::Sorghum, [this]() { GenerateLeavesForSorghum(); });

  m_plantSystem->m_plantMetaDataCalculators.insert_or_assign(
      PlantType::Sorghum, [this]() { FormLeafNodes(); });

  m_plantSystem->m_deleteAllPlants.insert_or_assign(
      PlantType::Sorghum, [this]() { DeleteAllPlantsHelper(); });

  Enable();
}

Entity SorghumSystem::CreateSorghum() {
  Transform transform;
  transform.SetScale(glm::vec3(1.0f));
  const Entity entity =
      m_plantSystem->CreatePlant(PlantType::Sorghum, transform);
  entity.ForEachChild([this](Entity child) {
    if (child.HasDataComponent<InternodeInfo>()) {
      auto internodeTransform = child.GetDataComponent<Transform>();
      internodeTransform.SetScale(glm::vec3(m_leafNodeSphereSize));
      child.SetDataComponent(internodeTransform);
    }
  });
  entity.GetOrSetPrivateComponent<Spline>();
  entity.GetOrSetPrivateComponent<SorghumData>();
  entity.SetName("Sorghum");
  entity.GetOrSetPrivateComponent<TriangleIlluminationEstimator>();
  return entity;
}

Entity SorghumSystem::CreateSorghumLeaf(const Entity &plantEntity) {
  const Entity entity = EntityManager::CreateEntity(m_leafArchetype);
  entity.SetName("Leaf");
  entity.SetParent(plantEntity);
  Transform transform;
  transform.SetScale(glm::vec3(1.0f));
  auto spline = entity.GetOrSetPrivateComponent<Spline>();
  entity.SetDataComponent(transform);

  auto mmc = entity.GetOrSetPrivateComponent<MeshRenderer>().lock();
  mmc->m_material = m_leafMaterial;
  mmc->m_mesh = AssetManager::CreateAsset<Mesh>();

  auto rtt =
      entity.GetOrSetPrivateComponent<RayTracerFacility::RayTracedRenderer>()
          .lock();
  rtt->m_mesh = mmc->m_mesh;
  rtt->m_albedoTexture = m_rayTracedLeafSurfaceTexture;
  if (m_rayTracedLeafNormalTexture)
    rtt->m_normalTexture = m_rayTracedLeafNormalTexture;

  return entity;
}

void SorghumSystem::GenerateMeshForAllSorghums(int segmentAmount, int step) {
  std::mutex meshMutex;
  EntityManager::ForEach<GlobalTransform>(
      JobManager::PrimaryWorkers(), m_leafQuery,
      [&meshMutex, segmentAmount, step](int index, Entity entity,
                                        GlobalTransform &ltw) {
        auto spline = entity.GetOrSetPrivateComponent<Spline>().lock();
        spline->m_nodes.clear();
        int stemNodeCount = 0;
        if (spline->m_startingPoint != -1) {
          auto truckSpline =
              entity.GetParent().GetOrSetPrivateComponent<Spline>().lock();
          float width = 0.1f - spline->m_startingPoint * 0.05f;
          for (float i = 0.0f; i < spline->m_startingPoint - 0.05f;
               i += 0.05f) {
            spline->m_nodes.emplace_back(
                truckSpline->EvaluatePointFromCurve(i), 180.0f, width,
                truckSpline->EvaluateAxisFromCurve(i), false);
          }
          stemNodeCount = spline->m_nodes.size();
          for (float i = 0.05f; i <= 1.0f; i += 0.05f) {
            float w = 0.2f;
            if (i > 0.75f)
              w -= (i - 0.75f) * 0.75f;
            spline->m_nodes.emplace_back(
                spline->EvaluatePointFromCurve(i), i == 0.05f ? 60.0f : 10.0f,
                w, spline->EvaluateAxisFromCurve(i), true);
          }
        } else {
          for (float i = 0.0f; i <= 1.0f; i += 0.05f) {
            spline->m_nodes.emplace_back(
                spline->EvaluatePointFromCurve(i), 180.0f, 0.04f,
                spline->EvaluateAxisFromCurve(i), false);
          }
          auto endPoint = spline->EvaluatePointFromCurve(1.0f);
          auto endAxis = spline->EvaluateAxisFromCurve(1.0f);
          spline->m_nodes.emplace_back(endPoint + endAxis * 0.05f, 10.0f,
                                       0.001f, endAxis, false);
          stemNodeCount = spline->m_nodes.size();
        }
        spline->m_vertices.clear();
        spline->m_indices.clear();
        spline->m_segments.clear();

        float temp = 0.0f;

        float leftPeriod = 0.0f;
        float rightPeriod = 0.0f;
        float leftFlatness =
            glm::gaussRand(1.75f,
                           0.5f); // glm::linearRand(0.5f, 2.0f);
        float rightFlatness =
            glm::gaussRand(1.75f,
                           0.5f); // glm::linearRand(0.5f, 2.0f);
        float leftFlatnessFactor =
            glm::gaussRand(1.25f,
                           0.2f); // glm::linearRand(1.0f, 2.5f);
        float rightFlatnessFactor =
            glm::gaussRand(1.25f,
                           0.2f); // glm::linearRand(1.0f, 2.5f);

        int stemSegmentCount = 0;
        for (int i = 1; i < spline->m_nodes.size(); i++) {
          auto &prev = spline->m_nodes.at(i - 1);
          auto &curr = spline->m_nodes.at(i);
          if (i == stemNodeCount) {
            stemSegmentCount = spline->m_segments.size();
          }
          float distance = glm::distance(prev.m_position, curr.m_position);
          BezierCurve curve = BezierCurve(
              prev.m_position, prev.m_position + distance / 5.0f * prev.m_axis,
              curr.m_position - distance / 5.0f * curr.m_axis, curr.m_position);
          for (float div = 1.0f / segmentAmount; div <= 1.0f;
               div += 1.0f / segmentAmount) {
            auto front = prev.m_axis * (1.0f - div) + curr.m_axis * div;

            auto up = glm::normalize(glm::cross(spline->m_left, front));
            if (prev.m_isLeaf) {
              leftPeriod += glm::gaussRand(1.25f, 0.5f) / segmentAmount;
              rightPeriod += glm::gaussRand(1.25f, 0.5f) / segmentAmount;
              spline->m_segments.emplace_back(
                  curve.GetPoint(div), up, front,
                  prev.m_width * (1.0f - div) + curr.m_width * div,
                  prev.m_theta * (1.0f - div) + curr.m_theta * div,
                  curr.m_isLeaf, glm::sin(leftPeriod) * leftFlatness,
                  glm::sin(rightPeriod) * rightFlatness, leftFlatnessFactor,
                  rightFlatnessFactor);
            } else {
              spline->m_segments.emplace_back(
                  curve.GetPoint(div), up, front,
                  prev.m_width * (1.0f - div) + curr.m_width * div,
                  prev.m_theta * (1.0f - div) + curr.m_theta * div,
                  curr.m_isLeaf);
            }
          }
        }

        const int vertexIndex = spline->m_vertices.size();
        Vertex archetype;
        const float xStep = 1.0f / step / 2.0f;
        const float yStemStep = 0.5f / static_cast<float>(stemSegmentCount);
        const float yLeafStep =
            0.5f / (spline->m_segments.size() -
                    static_cast<float>(stemSegmentCount) + 1);
        for (int i = 0; i < spline->m_segments.size(); i++) {
          auto &segment = spline->m_segments.at(i);
          const float angleStep = segment.m_theta / step;
          const int vertsCount = step * 2 + 1;
          for (int j = 0; j < vertsCount; j++) {
            const auto position = segment.GetPoint((j - step) * angleStep);
            archetype.m_position =
                glm::vec3(position.x, position.y, position.z);
            float yPos = (i < stemSegmentCount)
                             ? yStemStep * i
                             : 0.5f + yLeafStep * (i - stemSegmentCount + 1);
            archetype.m_texCoords = glm::vec2(j * xStep, yPos);
            spline->m_vertices.push_back(archetype);
          }
          if (i != 0) {
            for (int j = 0; j < vertsCount - 1; j++) {
              // Down triangle
              spline->m_indices.push_back(vertexIndex +
                                          ((i - 1) + 1) * vertsCount + j);
              spline->m_indices.push_back(vertexIndex + (i - 1) * vertsCount +
                                          j + 1);
              spline->m_indices.push_back(vertexIndex + (i - 1) * vertsCount +
                                          j);
              // Up triangle
              spline->m_indices.push_back(vertexIndex + (i - 1) * vertsCount +
                                          j + 1);
              spline->m_indices.push_back(vertexIndex +
                                          ((i - 1) + 1) * vertsCount + j);
              spline->m_indices.push_back(vertexIndex +
                                          ((i - 1) + 1) * vertsCount + j + 1);
            }
          }
        }
      });
  std::vector<Entity> plants;
  m_plantSystem->m_plantQuery.ToEntityArray(plants);
  for (auto &plant : plants) {
    if (plant.GetDataComponent<PlantInfo>().m_plantType != PlantType::Sorghum)
      continue;
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
                                  const std::string &name) {
  std::ifstream file(path, std::fstream::in);
  if (!file.is_open()) {
    Debug::Log("Failed to open file!");
    return Entity();
  }
  // Number of leaves in the file
  int leafCount;
  file >> leafCount;
  const auto sorghum = CreateSorghum();
  sorghum.RemovePrivateComponent<SorghumData>();
  auto children = sorghum.GetChildren();
  for (const auto &child : children) {
    EntityManager::DeleteEntity(child);
  }
  sorghum.SetName(name);
  auto truckSpline = sorghum.GetOrSetPrivateComponent<Spline>().lock();
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
    Entity leaf = CreateSorghumLeaf(sorghum);
    auto leafSpline = leaf.GetOrSetPrivateComponent<Spline>().lock();
    float startingPoint;
    file >> startingPoint;

    leafSpline->m_startingPoint = startingPoint;
    leafSpline->Import(file);
    for (auto &curve : leafSpline->m_curves) {
      curve.m_p0 += truckSpline->EvaluatePointFromCurve(startingPoint);
      curve.m_p1 += truckSpline->EvaluatePointFromCurve(startingPoint);
      curve.m_p2 += truckSpline->EvaluatePointFromCurve(startingPoint);
      curve.m_p3 += truckSpline->EvaluatePointFromCurve(startingPoint);
    }

    leafSpline->m_left = glm::cross(glm::vec3(0.0f, 1.0f, 0.0f),
                                    leafSpline->m_curves.begin()->m_p0 -
                                        leafSpline->m_curves.back().m_p3);
  }
  return sorghum;
}

void SorghumSystem::OnGui() {
  if (ImGui::BeginMainMenuBar()) {
    if (ImGui::BeginMenu("Sorghum Manager")) {
      ImGui::Checkbox("Display light probes", &m_displayLightProbes);
      ImGui::DragInt("Seed", &m_seed);
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
            if (ImGui::BeginMenu("Create forest...")) {
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
                "Import parameters for all", ".sorghumparam",
                [](const std::string &path) {
                  newSorghumParameters[0].Deserialize(path);
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
                "Import parameters", ".treeparam", [](const std::string &path) {
                  newSorghumParameters[currentFocusedNewSorghumIndex]
                      .Deserialize(path);
                });

            FileUtils::SaveFile(
                "Export parameters", ".treeparam", [](const std::string &path) {
                  newSorghumParameters[currentFocusedNewSorghumIndex].Serialize(
                      path);
                });
            ImGui::EndMenu();
          }
          ImGui::EndMenuBar();
        }
        ImGui::Columns(1);
        ImGui::PushItemWidth(200);
        newSorghumParameters[currentFocusedNewSorghumIndex].OnGui();
        ImGui::PopItemWidth();
        ImGui::EndChild();
        ImGui::PopStyleVar();
        ImGui::Separator();
        if (ImGui::Button("OK", ImVec2(120, 0))) {
          // Create tree here.
          for (auto i = 0; i < newSorghumAmount; i++) {
            Entity sorghum = CreateSorghum();
            auto sorghumTransform = sorghum.GetDataComponent<Transform>();
            sorghumTransform.SetPosition(newSorghumPositions[i]);
            sorghumTransform.SetEulerRotation(
                glm::radians(newSorghumRotations[i]));
            sorghum.SetDataComponent(sorghumTransform);
            sorghum.GetOrSetPrivateComponent<SorghumData>()
                .lock()
                ->m_parameters = newSorghumParameters[i];
          }
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
        if (ImGui::Button("OK", ImVec2(120, 0))) {
          std::vector<Entity> candidates;
          candidates.push_back(
              ImportPlant(std::filesystem::path(PLANT_FACTORY_RESOURCE_FOLDER) /
                              "Sorghum/skeleton_procedural_1.txt",
                          "Sorghum 1"));
          candidates.push_back(
              ImportPlant(std::filesystem::path(PLANT_FACTORY_RESOURCE_FOLDER) /
                              "Sorghum/skeleton_procedural_2.txt",
                          "Sorghum 2"));
          candidates.push_back(
              ImportPlant(std::filesystem::path(PLANT_FACTORY_RESOURCE_FOLDER) /
                              "Sorghum/skeleton_procedural_3.txt",
                          "Sorghum 3"));
          candidates.push_back(
              ImportPlant(std::filesystem::path(PLANT_FACTORY_RESOURCE_FOLDER) /
                              "Sorghum/skeleton_procedural_4.txt",
                          "Sorghum 4"));
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
      if (ImGui::Button("Lock structure")) {
        std::vector<Entity> sorghums;
        m_plantSystem->m_plantQuery.ToEntityArray<PlantInfo>(
            sorghums, [](const Entity &plant, const PlantInfo &plantInfo) {
              return plantInfo.m_plantType == PlantType::Sorghum;
            });
        for (const auto &sorghum : sorghums) {
          if (!sorghum.HasPrivateComponent<SorghumData>())
            continue;
          sorghum.RemovePrivateComponent<SorghumData>();
          Entity rootInternode;
          sorghum.ForEachChild([&](Entity child) {
            if (child.HasDataComponent<InternodeInfo>())
              rootInternode = child;
          });
          if (rootInternode.IsValid())
            EntityManager::DeleteEntity(rootInternode);
        }
      }
      FileUtils::SaveFile(
          "Export OBJ for all sorghums", ".obj",
          [this](const std::string &path) { ExportAllSorghumsModel(path); });
      ImGui::EndMenu();
    }
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

    ImGui::EndMainMenuBar();
  }
}

void SorghumSystem::CloneSorghums(const Entity &parent, const Entity &original,
                                  std::vector<glm::mat4> &matrices) {
  for (const auto &matrix : matrices) {
    Entity sorghum = CreateSorghum();
    sorghum.RemovePrivateComponent<SorghumData>();
    auto children = sorghum.GetChildren();
    for (const auto &child : children) {
      EntityManager::DeleteEntity(child);
    }
    Transform transform;
    transform.m_value = matrix;

    auto newSpline = sorghum.GetOrSetPrivateComponent<Spline>().lock();
    auto spline = original.GetOrSetPrivateComponent<Spline>().lock();
    newSpline->Clone(std::static_pointer_cast<IPrivateComponent>(spline));

    original.ForEachChild([this, &sorghum, &matrices](Entity child) {
      if (!child.HasDataComponent<LeafInfo>())
        return;
      const auto newChild = CreateSorghumLeaf(sorghum);
      newChild.SetDataComponent(child.GetDataComponent<LeafInfo>());
      newChild.SetDataComponent(child.GetDataComponent<Transform>());
      auto newSpline = newChild.GetOrSetPrivateComponent<Spline>().lock();
      auto spline = child.GetOrSetPrivateComponent<Spline>().lock();
      newSpline->Clone(std::static_pointer_cast<IPrivateComponent>(spline));
      auto newMeshRenderer =
          newChild.GetOrSetPrivateComponent<MeshRenderer>().lock();
      auto meshRenderer = child.GetOrSetPrivateComponent<MeshRenderer>().lock();
      newMeshRenderer->m_mesh = meshRenderer->m_mesh;
      auto newRayTracedRenderer =
          newChild
              .GetOrSetPrivateComponent<RayTracerFacility::RayTracedRenderer>()
              .lock();
      newRayTracedRenderer->m_mesh = meshRenderer->m_mesh;
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
    m_plantSystem->m_plantQuery.ToEntityArray<PlantInfo>(
        sorghums, [](const Entity &plant, const PlantInfo &plantInfo) {
          return plantInfo.m_plantType == PlantType::Sorghum;
        });

    for (const auto &plant : sorghums) {
      ExportSorghum(plant, of, startIndex);
    }
    of.close();
    Debug::Log("Sorghums saved as " + filename);
  } else {
    Debug::Error("Can't open file!");
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

void SorghumSystem::GenerateLeavesForSorghum() {
  // Remove previous leaf internodes.
  std::vector<Entity> sorghums;
  m_plantSystem->m_plantQuery.ToEntityArray<PlantInfo>(
      sorghums, [](const Entity &plant, const PlantInfo &plantInfo) {
        return plantInfo.m_plantType == PlantType::Sorghum;
      });
  for (const auto &sorghum : sorghums) {
    if (!sorghum.HasPrivateComponent<SorghumData>())
      continue;
#pragma region Clear all child with spline->
    auto children = sorghum.GetChildren();
    for (const auto &child : children) {
      if (child.HasPrivateComponent<Spline>()) {
        EntityManager::DeleteEntity(child);
      }
    }
#pragma endregion
    auto truckSpline = sorghum.GetOrSetPrivateComponent<Spline>().lock();
    Entity walker = sorghum;
    std::vector<Entity> centerNode;
    while (!walker.IsNull()) {
      Entity temp;
      walker.ForEachChild([&](Entity child) {
        if (child.HasDataComponent<InternodeInfo>() &&
            child.GetDataComponent<InternodeInfo>().m_order == 1) {
          temp = child;
          centerNode.push_back(child);
        }
      });
      walker = temp;
    }
    glm::vec3 startPosition =
        sorghum.GetDataComponent<GlobalTransform>().GetPosition();
    glm::vec3 endPosition =
        centerNode.back().GetDataComponent<GlobalTransform>().GetPosition();
    auto truckCurve = BezierCurve(
        glm::vec3(0.0f), (-startPosition + endPosition) / 2.0f,
        (-startPosition + endPosition) / 2.0f, endPosition - startPosition);
    truckSpline->m_curves.push_back(truckCurve);

    int leafAmount = 0;
    for (int i = 0; i < centerNode.size(); i++) {
      glm::vec3 centerNodePosition =
          centerNode[i].GetDataComponent<GlobalTransform>().GetPosition();
      centerNode[i].ForEachChild([&](Entity child) {
        if (child.HasDataComponent<InternodeInfo>() &&
            child.GetDataComponent<InternodeInfo>().m_order == 2) {
          const auto leafEntity = CreateSorghumLeaf(sorghum);
          auto leafSpline =
              leafEntity.GetOrSetPrivateComponent<Spline>().lock();
          auto cp0 = centerNodePosition;
          auto cp1 = child.GetDataComponent<GlobalTransform>().GetPosition();
          auto child2 = child.GetChildren()[0];
          auto cp2 = child2.GetDataComponent<GlobalTransform>().GetPosition();
          auto child3 = child2.GetChildren()[0];
          auto cp3 = child3.GetDataComponent<GlobalTransform>().GetPosition();
          auto leafCurve =
              BezierCurve(cp0 - startPosition, cp1 - startPosition,
                          cp2 - startPosition, cp3 - startPosition);
          leafSpline->m_curves.push_back(leafCurve);

          auto child0 = child3.GetChildren()[0];
          cp0 = child0.GetDataComponent<GlobalTransform>().GetPosition();
          auto child1 = child0.GetChildren()[0];
          cp1 = child1.GetDataComponent<GlobalTransform>().GetPosition();
          child2 = child1.GetChildren()[0];
          cp2 = child2.GetDataComponent<GlobalTransform>().GetPosition();
          child3 = child2.GetChildren()[0];
          cp3 = child3.GetDataComponent<GlobalTransform>().GetPosition();
          leafCurve = BezierCurve(cp0 - startPosition, cp1 - startPosition,
                                  cp2 - startPosition, cp3 - startPosition);
          leafSpline->m_curves.push_back(leafCurve);

          leafSpline->m_startingPoint =
              static_cast<float>(i + 1) / (centerNode.size() + 1);
          leafSpline->m_left = glm::cross(glm::vec3(0.0f, 1.0f, 0.0f),
                                          leafSpline->m_curves.begin()->m_p0 -
                                              leafSpline->m_curves.back().m_p3);

          leafAmount++;
        }
      });
    }

    if (leafAmount == sorghum.GetOrSetPrivateComponent<SorghumData>()
                          .lock()
                          ->m_parameters.m_leafCount) {
      sorghum.GetOrSetPrivateComponent<SorghumData>().lock()->m_growthComplete =
          true;
    }
  }

  GenerateMeshForAllSorghums();
}

void SorghumSystem::FormCandidates(
    std::vector<InternodeCandidate> &candidates) {
  const float globalTime = m_plantSystem->m_globalTime;
  const float sphereSize = m_leafNodeSphereSize;
  std::mutex mutex;
  EntityManager::ForEach<GlobalTransform, Transform, InternodeInfo,
                         InternodeGrowth, InternodeStatistics, Illumination>(
      JobManager::PrimaryWorkers(), m_plantSystem->m_internodeQuery,
      [&, globalTime, sphereSize](
          int index, Entity internode, GlobalTransform &globalTransform,
          Transform &transform, InternodeInfo &internodeInfo,
          InternodeGrowth &internodeGrowth,
          InternodeStatistics &internodeStatistics,
          Illumination &internodeIllumination) {
        if (internodeInfo.m_plantType != PlantType::Sorghum)
          return;
        if (internode.GetChildrenAmount() != 0)
          return;
        auto internodeData =
            internode.GetOrSetPrivateComponent<InternodeData>().lock();
        auto plant = internodeData->m_plant.Get();
        if (!plant.IsEnabled())
          return;
        auto sorghumData = plant.GetOrSetPrivateComponent<SorghumData>().lock();
        if (sorghumData->m_growthComplete)
          return;
        auto plantInfo = plant.GetDataComponent<PlantInfo>();
        auto parameters = sorghumData->m_parameters;
        if (internodeInfo.m_order == 1) {
          auto stemCandidate = InternodeCandidate();
          stemCandidate.m_info.m_plantType = PlantType::Sorghum;
          stemCandidate.m_parent = internode;
          stemCandidate.m_info.m_startGlobalTime = globalTime;
          stemCandidate.m_plant = plant;
          stemCandidate.m_info.m_startAge = plantInfo.m_age;
          stemCandidate.m_info.m_order = internodeInfo.m_order;
          stemCandidate.m_info.m_level = internodeInfo.m_level + 1;
          stemCandidate.m_growth.m_internodeLength =
              glm::gaussRand(parameters.m_internodeLength,
                             parameters.m_internodeLengthVariance);
          stemCandidate.m_growth.m_distanceToRoot =
              internodeGrowth.m_distanceToRoot + 1;
          float rotateAngle = glm::gaussRand(
              180.0f, parameters.m_rollAngleVariance +
                          parameters.m_rollAngleVarianceDistanceFactor *
                              internodeGrowth.m_distanceToRoot);
          stemCandidate.m_growth.m_desiredLocalRotation =
              glm::radians(glm::vec3(0.0f, 0.0f, rotateAngle));
          stemCandidate.m_statistics.m_isEndNode = true;
          stemCandidate.m_buds = std::vector<Bud>();
#pragma region Calculate transform
          glm::quat globalRotation =
              globalTransform.GetRotation() *
              stemCandidate.m_growth.m_desiredLocalRotation;
          glm::vec3 front = globalRotation * glm::vec3(0, 0, -1);
          glm::vec3 positionDelta =
              front * stemCandidate.m_growth.m_internodeLength;
          glm::vec3 newInternodePosition =
              globalTransform.GetPosition() + positionDelta;
          stemCandidate.m_globalTransform.m_value =
              glm::translate(newInternodePosition) *
              glm::mat4_cast(globalRotation) *
              glm::scale(glm::vec3(sphereSize));
#pragma endregion
          candidates.push_back(std::move(stemCandidate));
          if (internodeGrowth.m_distanceToRoot > 0) {
            auto leafCandidate = InternodeCandidate();
            leafCandidate.m_info.m_plantType = PlantType::Sorghum;
            leafCandidate.m_parent = internode;
            leafCandidate.m_info.m_startGlobalTime = globalTime;
            leafCandidate.m_plant = plant;
            leafCandidate.m_info.m_startAge = plantInfo.m_age;
            leafCandidate.m_info.m_order = internodeInfo.m_order + 1;
            leafCandidate.m_info.m_level = internodeInfo.m_level + 1;
            leafCandidate.m_growth.m_distanceToRoot =
                internodeGrowth.m_distanceToRoot + 1;
            leafCandidate.m_statistics.m_isEndNode = true;
            leafCandidate.m_buds = std::vector<Bud>();
            leafCandidate.m_growth.m_desiredLocalRotation =
                glm::radians(glm::vec3(0.0f, 30.0f, 0.0f));
            leafCandidate.m_growth.m_internodeLength =
                sorghumData->m_parameters.m_leafLengthBase;
#pragma region Calculate transform
            glm::quat globalRotation =
                globalTransform.GetRotation() *
                leafCandidate.m_growth.m_desiredLocalRotation;
            glm::vec3 front = globalRotation * glm::vec3(0, 0, -1);
            glm::vec3 positionDelta =
                front * (sorghumData->m_parameters.m_leafLengthBase / 8.0f);
            glm::vec3 newInternodePosition =
                globalTransform.GetPosition() + positionDelta;
            leafCandidate.m_globalTransform.m_value =
                glm::translate(newInternodePosition) *
                glm::mat4_cast(globalRotation) *
                glm::scale(glm::vec3(sphereSize));
#pragma endregion
            std::lock_guard lock(mutex);
            candidates.push_back(std::move(leafCandidate));
          }
        }
      });
}

void SorghumSystem::FormLeafNodes() {
  std::vector<std::pair<Entity, SorghumParameters>> candidates;
  std::mutex mutex;
  const float globalTime = m_plantSystem->m_globalTime;
  EntityManager::ForEach<GlobalTransform, Transform, InternodeInfo,
                         InternodeGrowth, InternodeStatistics, Illumination>(
      JobManager::PrimaryWorkers(), m_plantSystem->m_internodeQuery,
      [&, globalTime](int index, Entity internode,
                      GlobalTransform &globalTransform, Transform &transform,
                      InternodeInfo &internodeInfo,
                      InternodeGrowth &internodeGrowth,
                      InternodeStatistics &internodeStatistics,
                      Illumination &internodeIllumination) {
        if (internodeInfo.m_plantType != PlantType::Sorghum)
          return;
        auto internodeData =
            internode.GetOrSetPrivateComponent<InternodeData>().lock();
        auto plant = internodeData->m_plant.Get();
        if (!plant.IsEnabled())
          return;
        if (internodeInfo.m_order != 2)
          return;
        auto sorghumData = plant.GetOrSetPrivateComponent<SorghumData>().lock();
        auto parameters = sorghumData->m_parameters;
        std::lock_guard lock(mutex);
        candidates.push_back(std::make_pair(internode, parameters));
      });

  for (auto &i : candidates) {
    if (i.first.GetChildrenAmount() != 0)
      EntityManager::DeleteEntity(i.first.GetChildren()[0]);
    const int level = i.first.GetDataComponent<InternodeInfo>().m_level;
    const float leafNodeDistance =
        i.second.m_leafLengthBase / 8.0f / m_leafNodeSphereSize;
    const float leafGravityBending = i.second.m_leafGravityBending;
    const float leafGravityBendingIncrease =
        i.second.m_leafGravityBendingIncreaseFactor;

    auto meshRenderer = i.first.GetOrSetPrivateComponent<MeshRenderer>().lock();
    meshRenderer->m_mesh = DefaultResources::Primitives::Sphere;
    meshRenderer->m_material = m_leafNodeMaterial;
    meshRenderer->m_material.Get<Material>()->m_albedoColor =
        glm::vec3(0, 1, 0);

    InternodeGrowth internodeGrowth;
    Entity leafNode1 =
        m_plantSystem->CreateInternode(PlantType::Sorghum, i.first);
    auto internodeInfo = leafNode1.GetDataComponent<InternodeInfo>();
    internodeInfo.m_order = 3;
    Transform transform;
    transform.SetPosition(glm::vec3(0, 0, -leafNodeDistance));
    transform.SetEulerRotation(
        glm::radians(glm::vec3(0, leafGravityBending, 0)));
    leafNode1.SetDataComponent(transform);
    leafNode1.SetDataComponent(internodeInfo);
    leafNode1.SetDataComponent(internodeGrowth);

    auto meshRenderer1 =
        leafNode1.GetOrSetPrivateComponent<MeshRenderer>().lock();
    meshRenderer1->m_mesh = DefaultResources::Primitives::Sphere;
    meshRenderer1->m_material = m_leafNodeMaterial;
    meshRenderer1->m_material.Get<Material>()->m_albedoColor =
        glm::vec3(0, 1, 0);

    Entity leafNode2 =
        m_plantSystem->CreateInternode(PlantType::Sorghum, leafNode1);
    internodeInfo = leafNode2.GetDataComponent<InternodeInfo>();
    internodeInfo.m_order = 3;
    transform.SetPosition(glm::vec3(0, 0, -leafNodeDistance));
    transform.SetEulerRotation(glm::radians(
        glm::vec3(0, leafGravityBending + leafGravityBendingIncrease, 0)));
    leafNode2.SetDataComponent(transform);
    leafNode2.SetDataComponent(internodeInfo);
    leafNode2.SetDataComponent(internodeGrowth);
    auto meshRenderer2 =
        leafNode2.GetOrSetPrivateComponent<MeshRenderer>().lock();
    meshRenderer2->m_mesh = DefaultResources::Primitives::Sphere;
    meshRenderer2->m_material = m_leafNodeMaterial;
    meshRenderer2->m_material.Get<Material>()->m_albedoColor =
        glm::vec3(0, 1, 0);

    Entity leafNode3 =
        m_plantSystem->CreateInternode(PlantType::Sorghum, leafNode2);
    internodeInfo = leafNode3.GetDataComponent<InternodeInfo>();
    internodeInfo.m_order = 3;
    transform.SetPosition(glm::vec3(0, 0, -leafNodeDistance));
    transform.SetEulerRotation(glm::radians(glm::vec3(
        0, leafGravityBending + 2.0 * leafGravityBendingIncrease, 0)));
    leafNode3.SetDataComponent(transform);
    leafNode3.SetDataComponent(internodeInfo);
    leafNode3.SetDataComponent(internodeGrowth);
    auto meshRenderer3 =
        leafNode3.GetOrSetPrivateComponent<MeshRenderer>().lock();
    meshRenderer3->m_mesh = DefaultResources::Primitives::Sphere;
    meshRenderer3->m_material = m_leafNodeMaterial;
    meshRenderer3->m_material.Get<Material>()->m_albedoColor =
        glm::vec3(0, 1, 0);

    Entity leafNode4 =
        m_plantSystem->CreateInternode(PlantType::Sorghum, leafNode3);
    internodeInfo = leafNode4.GetDataComponent<InternodeInfo>();
    internodeInfo.m_order = 3;
    transform.SetPosition(glm::vec3(0, 0, -leafNodeDistance));
    transform.SetEulerRotation(glm::radians(glm::vec3(
        0, leafGravityBending + 3.0 * leafGravityBendingIncrease, 0)));
    leafNode4.SetDataComponent(transform);
    leafNode4.SetDataComponent(internodeInfo);
    leafNode4.SetDataComponent(internodeGrowth);

    auto meshRenderer4 =
        leafNode4.GetOrSetPrivateComponent<MeshRenderer>().lock();
    meshRenderer4->m_mesh = DefaultResources::Primitives::Sphere;
    meshRenderer4->m_material = m_leafNodeMaterial;
    meshRenderer4->m_material.Get<Material>()->m_albedoColor =
        glm::vec3(0, 1, 0);

    Entity leafNode5 =
        m_plantSystem->CreateInternode(PlantType::Sorghum, leafNode4);
    internodeInfo = leafNode5.GetDataComponent<InternodeInfo>();
    internodeInfo.m_order = 3;
    transform.SetPosition(glm::vec3(0, 0, -leafNodeDistance));
    transform.SetEulerRotation(glm::radians(glm::vec3(
        0, leafGravityBending + 4.0 * leafGravityBendingIncrease, 0)));
    leafNode5.SetDataComponent(transform);
    leafNode5.SetDataComponent(internodeInfo);
    leafNode5.SetDataComponent(internodeGrowth);

    auto meshRenderer5 =
        leafNode5.GetOrSetPrivateComponent<MeshRenderer>().lock();
    meshRenderer5->m_mesh = DefaultResources::Primitives::Sphere;
    meshRenderer5->m_material = m_leafNodeMaterial;
    meshRenderer5->m_material.Get<Material>()->m_albedoColor =
        glm::vec3(0, 1, 0);

    Entity leafNode6 =
        m_plantSystem->CreateInternode(PlantType::Sorghum, leafNode5);
    internodeInfo = leafNode6.GetDataComponent<InternodeInfo>();
    internodeInfo.m_order = 3;
    transform.SetPosition(glm::vec3(0, 0, -leafNodeDistance));
    transform.SetEulerRotation(glm::radians(glm::vec3(
        0, leafGravityBending + 5.0 * leafGravityBendingIncrease, 0)));
    leafNode6.SetDataComponent(transform);
    leafNode6.SetDataComponent(internodeInfo);
    leafNode6.SetDataComponent(internodeGrowth);

    auto meshRenderer6 =
        leafNode6.GetOrSetPrivateComponent<MeshRenderer>().lock();
    meshRenderer6->m_mesh = DefaultResources::Primitives::Sphere;
    meshRenderer6->m_material = m_leafNodeMaterial;
    meshRenderer6->m_material.Get<Material>()->m_albedoColor =
        glm::vec3(0, 1, 0);
  }
}

void SorghumSystem::RemoveInternodes(const Entity &sorghum) {
  if (!sorghum.HasPrivateComponent<SorghumData>())
    return;
  sorghum.RemovePrivateComponent<SorghumData>();
  Entity rootInternode;
  sorghum.ForEachChild([&](Entity child) {
    if (child.HasDataComponent<InternodeInfo>())
      rootInternode = child;
  });
  if (rootInternode.IsValid())
    EntityManager::DeleteEntity(rootInternode);
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
void SorghumSystem::DeleteAllPlantsHelper() {
  m_probeColors.clear();
  m_probeTransforms.clear();
}
