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
#include <Utilities.hpp>
#include <ProjectManager.hpp>
#include <PhysicsManager.hpp>
#include <PostProcessing.hpp>

#include <SorghumSystem.hpp>
#include <SorghumData.hpp>
#include <TriangleIlluminationEstimator.hpp>
#include <ClassRegistry.hpp>
#include <ObjectRotator.hpp>


#include <AutoSorghumGenerationPipeline.hpp>
#include <ProceduralSorghumSegmentationMask.hpp>
#include <SorghumProceduralDescriptor.hpp>
#include <SorghumField.hpp>
#include <DepthCamera.hpp>
using namespace Scripts;
using namespace SorghumFactory;
#ifdef RAYTRACERFACILITY
using namespace RayTracerFacility;
#endif
void EngineSetup(bool enableRayTracing);

int main() {
  ClassRegistry::RegisterDataComponent<LeafTag>("LeafTag");
  ClassRegistry::RegisterDataComponent<SorghumTag>("SorghumTag");

  ClassRegistry::RegisterPrivateComponent<DepthCamera>("DepthCamera");
  ClassRegistry::RegisterPrivateComponent<ObjectRotator>("ObjectRotator");
  ClassRegistry::RegisterPrivateComponent<Spline>("Spline");
  ClassRegistry::RegisterPrivateComponent<SorghumData>("SorghumData");
  ClassRegistry::RegisterPrivateComponent<TriangleIlluminationEstimator>("TriangleIlluminationEstimator");

  ClassRegistry::RegisterSystem<SorghumSystem>("SorghumSystem");
#ifdef RAYTRACERFACILITY
  ClassRegistry::RegisterPrivateComponent<MLVQRenderer>(
      "MLVQRenderer");
#endif
  ClassRegistry::RegisterAsset<SorghumProceduralDescriptor>("SorghumProceduralDescriptor", ".spd");
  ClassRegistry::RegisterAsset<SorghumField>("SorghumField", ".sorghumfield");
  const bool enableRayTracing = true;
  EngineSetup(enableRayTracing);

  ApplicationConfigs applicationConfigs;
  Application::Init(applicationConfigs);

#pragma region Engine Loop
  Application::Run();
#pragma endregion
#ifdef RAYTRACERFACILITY
  if (enableRayTracing)
    RayTracerManager::End();
#endif
  Application::End();
}

void EngineSetup(bool enableRayTracing) {
  ProjectManager::SetScenePostLoadActions([=](){
    #pragma region Engine Setup
#pragma region Global light settings
    RenderManager::GetInstance().m_maxShadowDistance = 100;
    RenderManager::SetSplitRatio(0.15f, 0.3f, 0.5f, 1.0f);
#pragma endregion

#ifdef RAYTRACERFACILITY
    if (enableRayTracing)
      RayTracerManager::Init();
#endif
    auto sorghumSystem =
        EntityManager::GetOrCreateSystem<SorghumSystem>(
            EntityManager::GetCurrentScene(), SystemGroup::SimulationSystemGroup + 0.1f);
  });
}
