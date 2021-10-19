// PlantFactory.cpp : This file contains the 'main' function. Program execution
// begins and ends there.
//
#include <Application.hpp>
#ifdef RAYTRACERFACILITY
#include <CUDAModule.hpp>
#include <MLVQRenderer.hpp>
#include <RayTracerManager.hpp>
#endif
#include <EditorManager.hpp>
#include <PhysicsManager.hpp>
#include <PostProcessing.hpp>
#include <ProjectManager.hpp>
#include <Utilities.hpp>

#include <ClassRegistry.hpp>
#include <ObjectRotator.hpp>
#include <SorghumData.hpp>
#include <SorghumSystem.hpp>
#include <TriangleIlluminationEstimator.hpp>

#include <AutoSorghumGenerationPipeline.hpp>
#include <DepthCamera.hpp>
#include <ProceduralSorghumSegmentationMask.hpp>
#include <SorghumField.hpp>
#include <SorghumProceduralDescriptor.hpp>
using namespace Scripts;
using namespace SorghumFactory;
#ifdef RAYTRACERFACILITY
using namespace RayTracerFacility;
#endif
void EngineSetup();

int main() {
  ClassRegistry::RegisterDataComponent<LeafTag>("LeafTag");
  ClassRegistry::RegisterDataComponent<SorghumTag>("SorghumTag");

  ClassRegistry::RegisterPrivateComponent<DepthCamera>("DepthCamera");
  ClassRegistry::RegisterPrivateComponent<ObjectRotator>("ObjectRotator");
  ClassRegistry::RegisterPrivateComponent<Spline>("Spline");
  ClassRegistry::RegisterPrivateComponent<SorghumData>("SorghumData");
  ClassRegistry::RegisterPrivateComponent<TriangleIlluminationEstimator>(
      "TriangleIlluminationEstimator");

  ClassRegistry::RegisterSystem<SorghumSystem>("SorghumSystem");
#ifdef RAYTRACERFACILITY
  ClassRegistry::RegisterPrivateComponent<MLVQRenderer>("MLVQRenderer");
#endif
  ClassRegistry::RegisterAsset<SorghumProceduralDescriptor>(
      "SorghumProceduralDescriptor", ".spd");
  ClassRegistry::RegisterAsset<SorghumField>("SorghumField", ".sorghumfield");
  const bool enableRayTracing = true;
  EngineSetup();
  ApplicationConfigs applicationConfigs;
  Application::Init(applicationConfigs);
#ifdef RAYTRACERFACILITY
  if (enableRayTracing)
    RayTracerManager::Init();
#endif
#pragma region Engine Loop
  Application::Run();
#pragma endregion
#ifdef RAYTRACERFACILITY
  if (enableRayTracing)
    RayTracerManager::End();
#endif
  Application::End();
}

void EngineSetup() {
  ProjectManager::SetScenePostLoadActions([=]() {
#pragma region Engine Setup
#pragma region Global light settings
    RenderManager::GetInstance().m_maxShadowDistance = 100;
    RenderManager::SetSplitRatio(0.15f, 0.3f, 0.5f, 1.0f);
#pragma endregion

    auto sorghumSystem =
        EntityManager::GetCurrentScene()->GetOrCreateSystem<SorghumSystem>(
            SystemGroup::SimulationSystemGroup + 0.1f);
  });
}
