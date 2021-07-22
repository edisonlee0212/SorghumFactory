// PlantFactory.cpp : This file contains the 'main' function. Program execution
// begins and ends there.
//
#include <Application.hpp>
#include <CUDAModule.hpp>
#include <CameraControlSystem.hpp>
#include <EditorManager.hpp>
#include <FileIO.hpp>
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
  EngineSetup();

  ComponentFactory::RegisterDataComponent<LeafInfo>("LeafInfo");
  ComponentFactory::RegisterSerializable<Spline>("Spline");
  ComponentFactory::RegisterSerializable<SorghumData>("SorghumData");
  ComponentFactory::RegisterSerializable<TriangleIlluminationEstimator>("TriangleIlluminationEstimator");
  ComponentFactory::RegisterDataComponent<TreeLeavesTag>("TreeLeavesTag");
  ComponentFactory::RegisterDataComponent<RbvTag>("RbvTag");
  ComponentFactory::RegisterSerializable<TreeData>("TreeData");
  ComponentFactory::RegisterSerializable<TreeLeaves>("TreeLeaves");
  ComponentFactory::RegisterSerializable<RadialBoundingVolume>("RadialBoundingVolume");
  ComponentFactory::RegisterSerializable<CubeVolume>("CubeVolume");
  ComponentFactory::RegisterDataComponent<PlantInfo>("PlantInfo");
  ComponentFactory::RegisterDataComponent<BranchCylinder>("BranchCylinder");
  ComponentFactory::RegisterDataComponent<BranchCylinderWidth>("BranchCylinderWidth");
  ComponentFactory::RegisterDataComponent<BranchPointer>("BranchPointer");
  ComponentFactory::RegisterDataComponent<Illumination>("Illumination");
  ComponentFactory::RegisterDataComponent<BranchColor>("BranchColor");

  ComponentFactory::RegisterDataComponent<InternodeInfo>("InternodeInfo");
  ComponentFactory::RegisterDataComponent<InternodeGrowth>("InternodeGrowth");
  ComponentFactory::RegisterDataComponent<InternodeStatistics>("InternodeStatistics");

  ComponentFactory::RegisterSerializable<InternodeData>("InternodeData");

  const bool enableRayTracing = true;
  if (enableRayTracing)
    RayTracerManager::Init();

  auto plantSystem =
      EntityManager::GetOrCreateSystem<PlantSystem>(
          "PlantSystem", SystemGroup::SimulationSystemGroup);
  auto treeSystem = EntityManager::GetOrCreateSystem<TreeSystem>(
      "TreeSystem", SystemGroup::SimulationSystemGroup + 0.1f);
  auto sorghumSystem =
      EntityManager::GetOrCreateSystem<SorghumSystem>(
          "SorghumSystem", SystemGroup::SimulationSystemGroup + 0.1f);


#pragma region Engine Loop
  Application::Run();
#pragma endregion
  if (enableRayTracing)
    RayTracerManager::End();
  Application::End();
}

void EngineSetup() {
  FileIO::SetProjectPath(PLANT_FACTORY_RESOURCE_FOLDER);
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
      "CameraControlSystem", SystemGroup::SimulationSystemGroup);
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
