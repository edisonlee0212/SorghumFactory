#include <CUDAModule.hpp>
#include <CubeVolume.hpp>
#include <Joint.hpp>
#include <PhysicsManager.hpp>
#include <PlantSystem.hpp>
#include <RayTracedRenderer.hpp>
#include <RayTracerManager.hpp>
#include <RigidBody.hpp>
#include <SorghumSystem.hpp>
#include <TreeSystem.hpp>
#include <Utilities.hpp>
#include <Volume.hpp>
using namespace PlantFactory;

#pragma region GUI Related
void ResourceParcel::OnGui() const {
  ImGui::Text("%s", ("Nutrient: " + std::to_string(m_nutrient)).c_str());
  ImGui::Text("%s", ("Carbon: " + std::to_string(m_carbon)).c_str());
}

void InternodeData::OnGui() {
  if (ImGui::TreeNode("Display buds")) {
    for (int i = 0; i < m_buds.size(); i++) {
      ImGui::Text("%s", ("Bud: " + std::to_string(i)).c_str());
      if (ImGui::TreeNode("Info")) {
        ImGui::Text(m_buds[i].m_isApical ? "Type: Apical" : "Type: Lateral");
        ImGui::Text(m_buds[i].m_active ? "Status: Active"
                                       : "Status: Not active");
        ImGui::Text(m_buds[i].m_enoughForGrowth ? "Has enough resource: True"
                                                : "Has enough resource: False");
        ImGui::Text("%s", ("ResourceWeight: " +
                           std::to_string(m_buds[i].m_resourceWeight))
                              .c_str());
        if (ImGui::TreeNode("Current Resource")) {
          m_buds[i].m_currentResource.OnGui();
          ImGui::TreePop();
        }
        ImGui::TreePop();
      }
    }
    ImGui::TreePop();
  }
}

void PlantSystem::OnGui() {
  if (ImGui::Begin("Plant Manager")) {
    if (ImGui::Button("Delete all plants")) {
      ImGui::OpenPopup("Delete Warning");
    }
    if (ImGui::BeginPopupModal("Delete Warning", nullptr,
                               ImGuiWindowFlags_AlwaysAutoResize)) {
      ImGui::Text("Are you sure? All plants will be removed!");
      if (ImGui::Button("Yes, delete all!", ImVec2(120, 0))) {
        DeleteAllPlants();
        ImGui::CloseCurrentPopup();
      }
      ImGui::SetItemDefaultFocus();
      ImGui::SameLine();
      if (ImGui::Button("Cancel", ImVec2(120, 0))) {
        ImGui::CloseCurrentPopup();
      }
      ImGui::EndPopup();
    }
    ImGui::Text(
        "%s",
        ("Internode amount: " + std::to_string(m_internodes.size())).c_str());
    if (ImGui::CollapsingHeader("Growth", ImGuiTreeNodeFlags_DefaultOpen)) {
      if (ImGui::Button("Recalculate illumination"))
        CalculateIlluminationForInternodes();
      static int pushAmount = 20;
      ImGui::DragInt("Amount", &pushAmount, 1, 0, 60.0f / m_deltaTime);
      if (ImGui::Button("Push and start (grow by iteration)")) {

        m_iterationsToGrow = pushAmount;
        Application::SetPlaying(true);
      }
      if (Application::IsPlaying() &&
          ImGui::Button("Push time (grow instantly)")) {
        const float time = Application::Time().CurrentTime();
        GrowAllPlants(pushAmount);
        const std::string spendTime =
            std::to_string(Application::Time().CurrentTime() - time);
        Debug::Log("Growth finished in " + spendTime + " sec.");
      }

      ImGui::SliderFloat("Time speed", &m_deltaTime, 0.1f, 1.0f);

      if (ImGui::TreeNode("Timers")) {
        ImGui::Text("Resource allocation: %.3fs",

                    m_resourceAllocationTimer);
        ImGui::Text("Form internodes: %.3fs", m_internodeFormTimer);
        ImGui::Text("Create internodes: %.3fs", m_internodeCreateTimer);
        ImGui::Text("Illumination: %.3fs",

                    m_illuminationCalculationTimer);
        ImGui::Text("Pruning & Metadata: %.3fs", m_metaDataTimer);
        ImGui::TreePop();
      }
    }
  }
  ImGui::End();
}
#pragma endregion
#pragma region Growth related
bool PlantSystem::GrowAllPlants() {
  Refresh();

  m_globalTime += m_deltaTime;
  float time = Application::Time().CurrentTime();
  std::vector<ResourceParcel> totalResources;
  totalResources.resize(m_plants.size());
  std::vector<ResourceParcel> resourceAvailable;
  resourceAvailable.resize(m_plants.size());
  for (auto &i : resourceAvailable)
    i.m_nutrient = 5000000.0f;
  CollectNutrient(m_plants, totalResources, resourceAvailable);
  for (auto &i : m_plantResourceAllocators) {
    i.second(totalResources);
  }

  m_resourceAllocationTimer = Application::Time().CurrentTime() - time;

  for (const auto &plant : m_plants) {
    auto plantInfo = plant.GetDataComponent<PlantInfo>();
    plantInfo.m_age += m_deltaTime;
    plant.SetDataComponent(plantInfo);
  }

  time = Application::Time().CurrentTime();
  std::vector<InternodeCandidate> candidates;
  for (auto &i : m_plantGrowthModels) {
    i.second(candidates);
  }

  m_internodeFormTimer = Application::Time().CurrentTime() - time;

  if (GrowCandidates(candidates)) {
    time = Application::Time().CurrentTime();
    std::vector<std::pair<GlobalTransform, Volume *>> obstacles;
    const auto *entities =
        EntityManager::UnsafeGetPrivateComponentOwnersList<CubeVolume>();
    if (entities) {
      for (const auto &entity : *entities) {
        if (!entity.IsEnabled())
          continue;
        auto &volume = entity.GetPrivateComponent<CubeVolume>();
        if (volume.IsEnabled() && volume.m_asObstacle)
          obstacles.emplace_back(
              volume.GetOwner().GetDataComponent<GlobalTransform>(), &volume);
      }
    }
    for (auto &i : m_plantInternodePruners) {
      i.second(obstacles);
    }

    m_pruningTimer = Application::Time().CurrentTime() - time;
    time = Application::Time().CurrentTime();
    for (auto &i : m_plantMetaDataCalculators) {
      i.second();
    }

    m_metaDataTimer = Application::Time().CurrentTime() - time;
    return true;
  }
  return false;
}
bool PlantSystem::GrowAllPlants(const unsigned &iterations) {
  bool grew = false;
  for (unsigned i = 0; i < iterations; i++) {
    const bool grewInThisIteration = GrowAllPlants();
    grew = grew || grewInThisIteration;
  }
  if (grew)
    Refresh();
  return grew;
}

bool PlantSystem::GrowCandidates(std::vector<InternodeCandidate> &candidates) {
  const float time = Application::Time().CurrentTime();
  if (candidates.empty())
    return false;
  auto entities = EntityManager::CreateEntities(m_internodeArchetype,
                                                candidates.size(), "Internode");
  int i = 0;
  for (auto &candidate : candidates) {
    auto newInternode = entities[i];
    newInternode.SetDataComponent(candidate.m_info);
    newInternode.SetDataComponent(candidate.m_growth);
    newInternode.SetDataComponent(candidate.m_statistics);
    newInternode.SetDataComponent(candidate.m_globalTransform);
    newInternode.SetDataComponent(candidate.m_transform);
    auto &newInternodeData = newInternode.SetPrivateComponent<InternodeData>();
    newInternodeData.m_buds.swap(candidate.m_buds);
    newInternodeData.m_owner = candidate.m_owner;
    newInternode.SetParent(candidate.m_parent);
    const auto search =
        m_plantInternodePostProcessors.find(candidate.m_info.m_plantType);
    if (search != m_plantInternodePostProcessors.end()) {
      search->second(newInternode, candidate);
    }
    i++;
  }
  m_internodeCreateTimer = Application::Time().CurrentTime() - time;
  return true;
}

void PlantSystem::CalculateIlluminationForInternodes() {
  if (m_internodeTransforms.empty())
    return;
  const float time = Application::Time().CurrentTime();
  // Upload geometries to OptiX.
  RayTracerFacility::RayTracerManager::GetInstance().UpdateScene();
  RayTracerFacility::IlluminationEstimationProperties properties;
  properties.m_bounceLimit = 1;
  properties.m_numPointSamples = 1000;
  properties.m_numScatterSamples = 1;
  properties.m_seed = glm::linearRand(16384, 32768);
  properties.m_skylightPower = 1.0f;
  properties.m_pushNormal = true;
  std::vector<RayTracerFacility::LightSensor<float>> lightProbes;
  lightProbes.resize(m_internodeQuery.GetEntityAmount());
  EntityManager::ForEach<GlobalTransform>(
      JobManager::PrimaryWorkers(), m_internodeQuery,
      [&](int i, Entity leafEntity, GlobalTransform &globalTransform) {
        lightProbes[i].m_position = globalTransform.GetPosition();
        lightProbes[i].m_surfaceNormal =
            globalTransform.GetRotation() * glm::vec3(0.0f, 0.0f, -1.0f);
      },
      false);
  if (lightProbes.empty())
    return;
  RayTracerFacility::CudaModule::EstimateIlluminationRayTracing(properties,
                                                                lightProbes);

  EntityManager::ForEach<Illumination>(
      JobManager::PrimaryWorkers(), m_internodeQuery,
      [&](int i, Entity leafEntity, Illumination &illumination) {
        const auto &lightProbe = lightProbes[i];
        illumination.m_accumulatedDirection +=
            lightProbe.m_direction * m_deltaTime;
        illumination.m_currentIntensity = lightProbe.m_energy;
      },
      false);

  m_illuminationCalculationTimer = Application::Time().CurrentTime() - time;
}
void PlantSystem::CollectNutrient(
    std::vector<Entity> &trees, std::vector<ResourceParcel> &totalNutrients,
    std::vector<ResourceParcel> &nutrientsAvailable) {
  for (int i = 0; i < trees.size(); i++) {
    totalNutrients[i].m_nutrient = nutrientsAvailable[i].m_nutrient;
  }
}

void PlantSystem::ApplyTropism(const glm::vec3 &targetDir, float tropism,
                               glm::vec3 &front, glm::vec3 &up) {
  const glm::vec3 dir = glm::normalize(targetDir);
  const float dotP = glm::abs(glm::dot(front, dir));
  if (dotP < 0.99f && dotP > -0.99f) {
    const glm::vec3 left = glm::cross(front, dir);
    const float maxAngle = glm::acos(dotP);
    const float rotateAngle = maxAngle * tropism;
    front = glm::normalize(
        glm::rotate(front, glm::min(maxAngle, rotateAngle), left));
    up = glm::normalize(glm::cross(glm::cross(front, up), front));
    // up = glm::normalize(glm::rotate(up, glm::min(maxAngle, rotateAngle),
    // left));
  }
}
#pragma endregion
#pragma region ResourceParcel
ResourceParcel::ResourceParcel() {
  m_nutrient = 0;
  m_carbon = 0;
}

ResourceParcel::ResourceParcel(const float &water, const float &carbon) {
  m_nutrient = water;
  m_carbon = carbon;
}

ResourceParcel &ResourceParcel::operator+=(const ResourceParcel &value) {
  m_nutrient += value.m_nutrient;
  m_carbon += value.m_carbon;
  return *this;
}

bool ResourceParcel::IsEnough() const {
  return m_nutrient > 1.0f && m_carbon > 1.0f;
}

#pragma endregion
#pragma region Helpers

Entity PlantSystem::CreateCubeObstacle() {
  const auto volumeEntity = EntityManager::CreateEntity("Volume");
  volumeEntity.SetEnabled(false);
  Transform transform;
  transform.SetPosition(glm::vec3(0, 10, 0));
  transform.SetScale(glm::vec3(4, 2, 4));
  GlobalTransform globalTransform;
  globalTransform.m_value = transform.m_value;
  volumeEntity.SetDataComponent(transform);
  volumeEntity.SetDataComponent(globalTransform);
  volumeEntity.SetStatic(true);

  auto &meshRenderer = volumeEntity.SetPrivateComponent<MeshRenderer>();
  meshRenderer.m_mesh = DefaultResources::Primitives::Cube;
  meshRenderer.m_material = DefaultResources::Materials::StandardMaterial;

  auto &volume = volumeEntity.SetPrivateComponent<CubeVolume>();
  volume.ApplyMeshRendererBounds();
  return volumeEntity;
}

void PlantSystem::DeleteAllPlants() {
  m_globalTime = 0;
  std::vector<Entity> trees;
  m_plantQuery.ToEntityArray(trees);
  for (const auto &tree : trees)
    EntityManager::DeleteEntity(tree);
  Refresh();
  for(auto& i : m_deleteAllPlants) i.second();
}

Entity PlantSystem::CreatePlant(const PlantType &type,
                                const Transform &transform) {
  const auto entity = EntityManager::CreateEntity(m_plantArchetype);

  GlobalTransform globalTransform;
  globalTransform.m_value = transform.m_value;
  entity.SetDataComponent(globalTransform);
  entity.SetDataComponent(transform);
  entity.SetName("Tree");
  PlantInfo treeInfo{};
  treeInfo.m_plantType = type;
  treeInfo.m_age = 0;
  treeInfo.m_startTime = m_globalTime;
  entity.SetDataComponent(treeInfo);
#pragma region Set root internode
  const auto rootInternode = EntityManager::CreateEntity(m_internodeArchetype);
  rootInternode.SetName("Internode");
  InternodeInfo internodeInfo;
  internodeInfo.m_plantType = type;
  internodeInfo.m_plant = entity;
  internodeInfo.m_startAge = 0;
  internodeInfo.m_startGlobalTime = treeInfo.m_startTime;
  InternodeGrowth internodeGrowth;
  internodeGrowth.m_desiredLocalRotation =
      glm::quat(glm::vec3(glm::radians(90.0f), 0.0f, 0.0f));

  GlobalTransform internodeGlobalTransform;
  internodeGlobalTransform.m_value =
      entity.GetDataComponent<GlobalTransform>().m_value *
      glm::mat4_cast(internodeGrowth.m_desiredLocalRotation);
  InternodeStatistics internodeStatistics;
  rootInternode.SetDataComponent(internodeInfo);
  rootInternode.SetDataComponent(internodeGrowth);
  rootInternode.SetDataComponent(internodeStatistics);
  rootInternode.SetDataComponent(internodeGlobalTransform);

  auto &rootInternodeData = rootInternode.SetPrivateComponent<InternodeData>();
  Bud bud;
  bud.m_isApical = true;
  rootInternodeData.m_buds.push_back(bud);
  rootInternodeData.m_owner = entity;
  rootInternode.SetParent(entity);

#pragma endregion
  return entity;
}

Entity PlantSystem::CreateInternode(const PlantType &type,
                                    const Entity &parentEntity) {
  const auto entity = EntityManager::CreateEntity(m_internodeArchetype);
  entity.SetName("Internode");
  InternodeInfo internodeInfo;
  internodeInfo.m_plantType = type;
  internodeInfo.m_plant =
      parentEntity.GetDataComponent<InternodeInfo>().m_plant;
  internodeInfo.m_startAge =
      internodeInfo.m_plant.GetDataComponent<PlantInfo>().m_age;
  entity.SetDataComponent(internodeInfo);
  entity.SetPrivateComponent<InternodeData>();
  entity.SetParent(parentEntity);
  return entity;
}

#pragma endregion
#pragma region Runtime
void PlantSystem::OnCreate() {


#pragma region Ground

  m_ground = EntityManager::CreateEntity("Ground");

  auto &meshRenderer = m_ground.SetPrivateComponent<MeshRenderer>();
  meshRenderer.m_mesh = DefaultResources::Primitives::Quad;
  meshRenderer.m_material = AssetManager::LoadMaterial(
      false, DefaultResources::GLPrograms::StandardProgram);
  meshRenderer.m_material->m_name = "Ground mat";
  meshRenderer.m_material->m_roughness = 1.0f;
  meshRenderer.m_material->m_metallic = 0.5f;
  meshRenderer.m_material->m_albedoColor = glm::vec3(1.0f);

  Transform groundTransform;
  GlobalTransform groundGlobalTransform;
  groundTransform.SetScale(glm::vec3(500.0f, 1.0f, 500.0f));
  groundTransform.SetPosition(glm::vec3(0.0f, 0.0f, 0.0f));
  groundGlobalTransform.m_value = groundTransform.m_value;

  m_ground.SetDataComponent(groundTransform);

  m_ground.SetDataComponent(groundGlobalTransform);

  auto &rayTracedRenderer =
      m_ground.SetPrivateComponent<RayTracerFacility::RayTracedRenderer>();
  rayTracedRenderer.SyncWithMeshRenderer();
  rayTracedRenderer.m_enableMLVQ = true;

  auto &cubeVolume = m_ground.SetPrivateComponent<CubeVolume>();
  cubeVolume.m_asObstacle = true;
  cubeVolume.m_minMaxBound.m_max = glm::vec3(500, -0.1f, 500);
  cubeVolume.m_minMaxBound.m_min = glm::vec3(-500, -10.0f, -500);

#pragma endregion
#pragma region Mask material
  std::string vertShaderCode =
      std::string("#version 460 core\n") +
      *DefaultResources::ShaderIncludes::Uniform + +"\n" +
      FileIO::LoadFileAsString(
          FileIO::GetResourcePath("Shaders/Vertex/Standard.vert"));
  std::string fragShaderCode =
      std::string("#version 460 core\n") +
      *DefaultResources::ShaderIncludes::Uniform + "\n" +
      FileIO::LoadFileAsString(FileIO::GetAssetFolderPath() +
                               "Shaders/Fragment/SemanticBranch.frag");

  auto standardVert =
      std::make_shared<OpenGLUtils::GLShader>(OpenGLUtils::ShaderType::Vertex);
  standardVert->Compile(vertShaderCode);
  auto standardFrag = std::make_shared<OpenGLUtils::GLShader>(
      OpenGLUtils::ShaderType::Fragment);
  standardFrag->Compile(fragShaderCode);
  auto branchProgram =
      AssetManager::CreateResource<OpenGLUtils::GLProgram>();
  branchProgram->Link(standardVert, standardFrag);

  vertShaderCode = std::string("#version 460 core\n") +
                   *DefaultResources::ShaderIncludes::Uniform + +"\n" +
                   FileIO::LoadFileAsString(FileIO::GetResourcePath(
                       "Shaders/Vertex/StandardInstanced.vert"));
  fragShaderCode =
      std::string("#version 460 core\n") +
      *DefaultResources::ShaderIncludes::Uniform + "\n" +
      FileIO::LoadFileAsString(FileIO::GetAssetFolderPath() +
                               "Shaders/Fragment/SemanticLeaf.frag");
  standardVert =
      std::make_shared<OpenGLUtils::GLShader>(OpenGLUtils::ShaderType::Vertex);
  standardVert->Compile(vertShaderCode);
  standardFrag = std::make_shared<OpenGLUtils::GLShader>(
      OpenGLUtils::ShaderType::Fragment);
  standardFrag->Compile(fragShaderCode);
  auto leafProgram = AssetManager::CreateResource<OpenGLUtils::GLProgram>();
  leafProgram->Link(standardVert, standardFrag);
#pragma endregion
#pragma region Entity

  m_internodeArchetype = EntityManager::CreateEntityArchetype(
      "Internode", BranchCylinder(), BranchCylinderWidth(), BranchPointer(),
      BranchColor(), Ray(), Illumination(), InternodeInfo(), InternodeGrowth(),
      InternodeStatistics());

  m_plantArchetype = EntityManager::CreateEntityArchetype("Tree", PlantInfo());

  m_internodeQuery = EntityManager::CreateEntityQuery();

  m_internodeQuery.SetAllFilters(InternodeInfo());

  m_plantQuery = EntityManager::CreateEntityQuery();

  m_plantQuery.SetAllFilters(PlantInfo());
#pragma endregion
#pragma region GUI
  EditorManager::RegisterComponentDataInspector<InternodeStatistics>(
      [](Entity entity, IDataComponent *data, bool isRoot) {
        auto *internodeStatistics = static_cast<InternodeStatistics *>(data);
        ImGui::Text(("MaxChildOrder: " +
                     std::to_string(internodeStatistics->m_maxChildOrder))
                        .c_str());
        ImGui::Text(("MaxChildLevel: " +
                     std::to_string(internodeStatistics->m_maxChildLevel))
                        .c_str());
        ImGui::Text(
            ("ChildrenEndNodeAmount: " +
             std::to_string(internodeStatistics->m_childrenEndNodeAmount))
                .c_str());
        ImGui::Text(("DistanceToBranchEnd: " +
                     std::to_string(internodeStatistics->m_distanceToBranchEnd))
                        .c_str());
        ImGui::Text(
            ("LongestDistanceToAnyEndNode: " +
             std::to_string(internodeStatistics->m_longestDistanceToAnyEndNode))
                .c_str());
        ImGui::Text(("TotalLength: " +
                     std::to_string(internodeStatistics->m_totalLength))
                        .c_str());
        ImGui::Text(
            ("DistanceToBranchStart: " +
             std::to_string(internodeStatistics->m_distanceToBranchStart))
                .c_str());
        ImGui::Checkbox("IsMaxChild: ", &internodeStatistics->m_isMaxChild);
        ImGui::Checkbox("IsEndNode: ", &internodeStatistics->m_isEndNode);
      });

  EditorManager::RegisterComponentDataInspector<
      InternodeGrowth>([](Entity entity, IDataComponent *data, bool isRoot) {
    auto *internodeGrowth = static_cast<InternodeGrowth *>(data);
    ImGui::Text(
        ("Inhibitor: " + std::to_string(internodeGrowth->m_inhibitor)).c_str());
    ImGui::Text(("InhibitorTransmitFactor: " +
                 std::to_string(internodeGrowth->m_inhibitorTransmitFactor))
                    .c_str());
    ImGui::Text(
        ("DistanceToRoot: " + std::to_string(internodeGrowth->m_distanceToRoot))
            .c_str());
    ImGui::Text(
        ("Thickness: " + std::to_string(internodeGrowth->m_thickness)).c_str());
    ImGui::InputFloat("Gravity sagging", &internodeGrowth->m_sagging,
                      ImGuiInputTextFlags_ReadOnly);
    ImGui::InputFloat("Mass of Children", &internodeGrowth->m_MassOfChildren,
                      ImGuiInputTextFlags_ReadOnly);
    ImGui::InputFloat2("Torque",
                       static_cast<float *>(static_cast<void *>(
                           &internodeGrowth->m_childrenTotalTorque)),
                       "%.3f", ImGuiInputTextFlags_ReadOnly);
    ImGui::InputFloat2("Mean position",
                       static_cast<float *>(static_cast<void *>(
                           &internodeGrowth->m_childMeanPosition)),
                       "%.3f", ImGuiInputTextFlags_ReadOnly);
    glm::vec3 desiredAngles =
        glm::degrees(glm::eulerAngles(internodeGrowth->m_desiredLocalRotation));
    ImGui::InputFloat3("Desired Rotation##Internode", &desiredAngles.x, "%.3f",
                       ImGuiInputTextFlags_ReadOnly);
    ImGui::InputFloat3("Desired Position##Internode",
                       &internodeGrowth->m_desiredGlobalPosition.x, "%.3f",
                       ImGuiInputTextFlags_ReadOnly);
  });

  EditorManager::RegisterComponentDataInspector<InternodeInfo>(
      [](Entity entity, IDataComponent *data, bool isRoot) {
        auto *internodeInfo = static_cast<InternodeInfo *>(data);
        ImGui::Checkbox("Activated", &internodeInfo->m_activated);
        ImGui::Text(
            ("StartAge: " + std::to_string(internodeInfo->m_startAge)).c_str());
        ImGui::Text(("StartGlobalTime: " +
                     std::to_string(internodeInfo->m_startGlobalTime))
                        .c_str());
        ImGui::Text(
            ("Order: " + std::to_string(internodeInfo->m_order)).c_str());
        ImGui::Text(
            ("Level: " + std::to_string(internodeInfo->m_level)).c_str());
      });

  EditorManager::RegisterComponentDataInspector<Illumination>(
      [](Entity entity, IDataComponent *data, bool isRoot) {
        auto *illumination = static_cast<Illumination *>(data);
        ImGui::Text(("CurrentIntensity: " +
                     std::to_string(illumination->m_currentIntensity))
                        .c_str());
        ImGui::InputFloat3("Direction", &illumination->m_accumulatedDirection.x,
                           "%.3f", ImGuiInputTextFlags_ReadOnly);
      });

  EditorManager::RegisterComponentDataInspector<PlantInfo>(
      [](Entity entity, IDataComponent *data, bool isRoot) {
        auto *info = static_cast<PlantInfo *>(data);
        ImGui::Text(
            ("Start time: " + std::to_string(info->m_startTime)).c_str());
        ImGui::Text(("Age: " + std::to_string(info->m_age)).c_str());
      });
#pragma endregion

  m_ready = true;

  m_globalTime = 0;
  Enable();
}

void PlantSystem::Update() {
  if (Application::IsPlaying()) {
    if (m_iterationsToGrow > 0) {
      m_iterationsToGrow--;
      if (GrowAllPlants()) {
        m_endUpdate = true;
      }
    } else if (m_endUpdate) {
      Refresh();
      m_endUpdate = false;
    }
  }
}

void PlantSystem::Refresh() {

  m_plants.resize(0);

  m_plantQuery.ToEntityArray(m_plants);

  m_internodes.resize(0);

  m_internodeTransforms.resize(0);

  m_internodeQuery.ToComponentDataArray(m_internodeTransforms);

  m_internodeQuery.ToEntityArray(m_internodes);

  float time = Application::Time().CurrentTime();
  for (auto &i : m_plantMeshGenerators) {
    i.second();
  }

  m_meshGenerationTimer = Application::Time().CurrentTime() - time;
  CalculateIlluminationForInternodes();
}

void PlantSystem::End() {}

#pragma endregion