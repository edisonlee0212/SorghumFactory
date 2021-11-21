// PlantFactory.cpp : This file contains the 'main' function. Program execution
// begins and ends there.
//
#include <Application.hpp>
#ifdef RAYTRACERFACILITY
#include <RayTracerManager.hpp>
#include "RayTracerCamera.hpp"
#endif
#include <ClassRegistry.hpp>
#include <EditorManager.hpp>
#include <ObjectRotator.hpp>
#include <PhysicsLayer.hpp>
#include <PostProcessing.hpp>
#include <ProjectManager.hpp>
#include <SorghumLayer.hpp>

#include <AutoSorghumGenerationPipeline.hpp>
#include <DepthCamera.hpp>
#include <SDFDataCapture.hpp>
using namespace Scripts;
using namespace SorghumFactory;
#ifdef RAYTRACERFACILITY
using namespace RayTracerFacility;
#endif

void EngineSetup();

int main() {
  ClassRegistry::RegisterPrivateComponent<AutoSorghumGenerationPipeline>(
      "AutoSorghumGenerationPipeline");
  ClassRegistry::RegisterAsset<SDFDataCapture>("SDFDataCapture",
                                               ".sdfdatacapture");
  ClassRegistry::RegisterPrivateComponent<ObjectRotator>("ObjectRotator");

  const bool enableRayTracing = true;
  EngineSetup();

  ApplicationConfigs applicationConfigs;
  applicationConfigs.m_projectPath = "temp/SDF/SDF.ueproj";
  Application::Create(applicationConfigs);
#ifdef RAYTRACERFACILITY
  if (enableRayTracing)
    Application::PushLayer<RayTracerManager>();
#endif
  Application::PushLayer<SorghumLayer>();
#pragma region Engine Loop
  Application::Start();
#pragma endregion
  Application::End();
}

void EngineSetup() {
  ProjectManager::SetScenePostLoadActions([=]() {
#pragma region Engine Setup

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

    auto sdfEntity = EntityManager::CreateEntity(
        EntityManager::GetCurrentScene(), "SDFPipeline");
    auto pipeline =
        sdfEntity.GetOrSetPrivateComponent<AutoSorghumGenerationPipeline>()
            .lock();
    auto capture = AssetManager::CreateAsset<SDFDataCapture>();
    pipeline->m_pipelineBehaviour = capture;

#ifdef RAYTRACERFACILITY
    auto rayTracerCamera = mainCamera->GetOwner().GetOrSetPrivateComponent<RayTracerCamera>().lock();
    capture->m_rayTracerCamera = rayTracerCamera;
    Application::GetLayer<RayTracerManager>()->m_rayTracerCamera = rayTracerCamera;
#else
    capture->m_rayTracerCamera = mainCamera;
#endif
    auto depthCamera = mainCamera->GetOwner().GetOrSetPrivateComponent<DepthCamera>().lock();
    capture->m_depthCamera = depthCamera;
  });
}
