// PlantFactory.cpp : This file contains the 'main' function. Program execution
// begins and ends there.
//
#include <Application.hpp>
#ifdef RAYTRACERFACILITY
#include <CUDAModule.hpp>
#include <MLVQRenderer.hpp>
#include <RayTracerManager.hpp>
#endif
#include <ClassRegistry.hpp>
#include <EditorManager.hpp>
#include <ObjectRotator.hpp>
#include <PhysicsManager.hpp>
#include <PostProcessing.hpp>
#include <ProjectManager.hpp>
#include <SorghumData.hpp>
#include <SorghumSystem.hpp>
#include <TriangleIlluminationEstimator.hpp>
#include <Utilities.hpp>

#include <AutoSorghumGenerationPipeline.hpp>
#include <DepthCamera.hpp>
#include <ProceduralSorghumSegmentationMask.hpp>
#include <SDFDataCapture.hpp>
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

  ClassRegistry::RegisterPrivateComponent<AutoSorghumGenerationPipeline>(
      "AutoSorghumGenerationPipeline");
  ClassRegistry::RegisterAsset<SDFDataCapture>("SDFDataCapture",
                                               ".sdfdatacapture");
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
  applicationConfigs.m_projectPath = "temp/SDF/SDF.ueproj";
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

    Transform transform;
    transform.SetEulerRotation(glm::radians(glm::vec3(150, 30, 0)));

#pragma region Preparations
    Application::Time().SetTimeStep(0.016f);
    transform = Transform();
    transform.SetPosition(glm::vec3(0, 2, 35));
    transform.SetEulerRotation(glm::radians(glm::vec3(15, 0, 0)));
    auto mainCamera =
        EntityManager::GetCurrentScene()->m_mainCamera.Get<UniEngine::Camera>();
    if (mainCamera) {
      auto postProcessing = mainCamera->GetOwner()
                                .GetOrSetPrivateComponent<PostProcessing>()
                                .lock();
      auto ssao = postProcessing->GetLayer<SSAO>().lock();
      ssao->m_kernelRadius = 0.1;
      mainCamera->GetOwner().SetDataComponent(transform);
      mainCamera->m_useClearColor = true;
      mainCamera->m_clearColor = glm::vec3(0.5f);
    }
#pragma endregion
#pragma endregion
    /*
    const Entity lightEntity = EntityManager::CreateEntity("Light source");
    auto pointLight = lightEntity.GetOrSetPrivateComponent<PointLight>().lock();
    pointLight->m_diffuseBrightness = 6;
    pointLight->m_lightSize = 0.25f;
    pointLight->m_quadratic = 0.0001f;
    pointLight->m_linear = 0.01f;
    pointLight->m_lightSize = 0.08f;
    transform.SetPosition(glm::vec3(0, 30, 0));
    transform.SetEulerRotation(glm::radians(glm::vec3(0, 0, 0)));
    lightEntity.SetDataComponent(transform);
    */

    auto sorghumSystem =
        EntityManager::GetCurrentScene()->GetOrCreateSystem<SorghumSystem>(
            SystemGroup::SimulationSystemGroup + 0.1f);

    auto sdfEntity = EntityManager::CreateEntity(
        EntityManager::GetCurrentScene(), "SDFPipeline");
    auto pipeline =
        sdfEntity.GetOrSetPrivateComponent<AutoSorghumGenerationPipeline>()
            .lock();
    auto capture = AssetManager::CreateAsset<SDFDataCapture>();
    pipeline->m_pipelineBehaviour = capture;
    capture->m_cameraEntity = mainCamera->GetOwner();
  });
}
