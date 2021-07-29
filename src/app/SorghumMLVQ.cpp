// PlantFactory.cpp : This file contains the 'main' function. Program execution
// begins and ends there.
//
#include <Application.hpp>
#include <CUDAModule.hpp>
#include <CameraControlSystem.hpp>
#include <EditorManager.hpp>
#include <FileSystem.hpp>
#include <PhysicsManager.hpp>
#include <PlantSystem.hpp>
#include <PostProcessing.hpp>
#include <RayTracerManager.hpp>
#include <SorghumSystem.hpp>
#include <TreeSystem.hpp>
#include <TreeLeaves.hpp>
#include <CubeVolume.hpp>
#include <RadialBoundingVolume.hpp>
#include <SorghumData.hpp>
#include <TriangleIlluminationEstimator.hpp>
using namespace PlantFactory;
using namespace RayTracerFacility;

void EngineSetup();

int main() {


  SerializableFactory::RegisterDataComponent<LeafInfo>("LeafInfo");
  SerializableFactory::RegisterSerializable<Spline>("Spline");
  SerializableFactory::RegisterSerializable<SorghumData>("SorghumData");
  SerializableFactory::RegisterSerializable<TriangleIlluminationEstimator>("TriangleIlluminationEstimator");
  SerializableFactory::RegisterDataComponent<TreeLeavesTag>("TreeLeavesTag");
  SerializableFactory::RegisterDataComponent<RbvTag>("RbvTag");
  SerializableFactory::RegisterSerializable<TreeData>("TreeData");
  SerializableFactory::RegisterSerializable<TreeLeaves>("TreeLeaves");
  SerializableFactory::RegisterSerializable<RadialBoundingVolume>("RadialBoundingVolume");
  SerializableFactory::RegisterSerializable<CubeVolume>("CubeVolume");
  SerializableFactory::RegisterDataComponent<PlantInfo>("PlantInfo");
  SerializableFactory::RegisterDataComponent<BranchCylinder>("BranchCylinder");
  SerializableFactory::RegisterDataComponent<BranchCylinderWidth>("BranchCylinderWidth");
  SerializableFactory::RegisterDataComponent<BranchPointer>("BranchPointer");
  SerializableFactory::RegisterDataComponent<Illumination>("Illumination");
  SerializableFactory::RegisterDataComponent<BranchColor>("BranchColor");

  SerializableFactory::RegisterDataComponent<InternodeInfo>("InternodeInfo");
  SerializableFactory::RegisterDataComponent<InternodeGrowth>("InternodeGrowth");
  SerializableFactory::RegisterDataComponent<InternodeStatistics>("InternodeStatistics");

  SerializableFactory::RegisterSerializable<InternodeData>("InternodeData");

  SerializableFactory::RegisterSerializable<PlantSystem>("PlantSystem");
  SerializableFactory::RegisterSerializable<SorghumSystem>("SorghumSystem");
  SerializableFactory::RegisterSerializable<TreeSystem>("TreeSystem");
  SerializableFactory::RegisterSerializable<CameraControlSystem>("CameraControlSystem");

  EngineSetup();

  const bool enableRayTracing = true;
  if (enableRayTracing)
    RayTracerManager::Init();

  auto plantSystem =
      EntityManager::GetOrCreateSystem<PlantSystem>(
          EntityManager::GetCurrentScene(), SystemGroup::SimulationSystemGroup);
  auto treeSystem = EntityManager::GetOrCreateSystem<TreeSystem>(
      EntityManager::GetCurrentScene(), SystemGroup::SimulationSystemGroup + 0.1f);
  auto sorghumSystem =
      EntityManager::GetOrCreateSystem<SorghumSystem>(
          EntityManager::GetCurrentScene(), SystemGroup::SimulationSystemGroup + 0.1f);

  EntityManager::GetSystem<PhysicsSystem>()->Disable();
#pragma region Engine Loop
  Application::Run();
#pragma endregion
  if (enableRayTracing)
    RayTracerManager::End();
  Application::End();
}

void EngineSetup() {
  AssetManager::SetProjectPath(PLANT_FACTORY_RESOURCE_FOLDER);
  Application::Init();
#pragma region Engine Setup
#pragma region Global light settings
  RenderManager::GetInstance().m_lightSettings.m_ambientLight = 0.2f;
  RenderManager::GetInstance().m_stableFit = false;
  RenderManager::GetInstance().m_maxShadowDistance = 100;
  RenderManager::SetSplitRatio(0.15f, 0.3f, 0.5f, 1.0f);

#pragma endregion

  Transform transform;
  transform.SetEulerRotation(glm::radians(glm::vec3(150, 30, 0)));

#pragma region Preparations
  Application::SetTimeStep(0.016f);

  auto ccs = EntityManager::GetOrCreateSystem<CameraControlSystem>(
      EntityManager::GetCurrentScene(), SystemGroup::SimulationSystemGroup);
  ccs->SetVelocity(15.0f);

  transform = Transform();
  transform.SetPosition(glm::vec3(0, 2, 35));
  transform.SetEulerRotation(glm::radians(glm::vec3(15, 0, 0)));
  auto mainCamera = RenderManager::GetMainCamera();
  if (mainCamera) {
    auto &postProcessing =
        mainCamera->GetOwner().SetPrivateComponent<PostProcessing>();
    auto *ssao = postProcessing.GetLayer<SSAO>();
    ssao->m_kernelRadius = 0.1;
    mainCamera->GetOwner().SetDataComponent(transform);
    mainCamera->m_useClearColor = true;
    mainCamera->m_clearColor = glm::vec3(0.5f);
  }
#pragma endregion
#pragma endregion

  const Entity lightEntity = EntityManager::CreateEntity("Light source");
  auto &pointLight = lightEntity.SetPrivateComponent<PointLight>();
  pointLight.m_diffuseBrightness = 6;
  pointLight.m_lightSize = 0.25f;
  pointLight.m_quadratic = 0.0001f;
  pointLight.m_linear = 0.01f;
  pointLight.m_lightSize = 0.08f;
  transform.SetPosition(glm::vec3(0, 30, 0));
  transform.SetEulerRotation(glm::radians(glm::vec3(0, 0, 0)));
  lightEntity.SetDataComponent(transform);
}
