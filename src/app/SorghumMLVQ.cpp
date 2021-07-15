// PlantFactory.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <FileIO.hpp>
#include <Application.hpp>
#include <CameraControlSystem.hpp>
#include <PlantManager.hpp>
#include <PostProcessing.hpp>
#include <CUDAModule.hpp>
#include <TreeManager.hpp>
#include <SorghumManager.hpp>
#include <EditorManager.hpp>
#include <RayTracerManager.hpp>
#include <PhysicsManager.hpp>
using namespace PlantFactory;
using namespace RayTracerFacility;

void EngineSetup();

int main() {
    EngineSetup();
    PlantManager::Init();
    SorghumManager::Init();
    TreeManager::Init();
    const bool enableRayTracing = true;
    if (enableRayTracing) RayTracerManager::Init();
    EntityManager::GetCurrentWorld()->GetSystem<PhysicsSystem>()->Disable();
    Application::SetTimeStep(1.0 / 100.0f);
#pragma region Engine Loop
    Application::RegisterUpdateFunction([&]() {
                                            PlantManager::Update();
                                            SorghumManager::Update();
                                            TreeManager::Update();
                                            if (enableRayTracing) RayTracerManager::Update();
                                        }
    );
    Application::RegisterLateUpdateFunction([&]() {
                                                PlantManager::OnGui();
                                                TreeManager::OnGui();
                                                SorghumManager::OnGui();
                                                if (enableRayTracing) RayTracerManager::OnGui();
                                            }
    );
    Application::RegisterFixedUpdateFunction([&]() {
                                                 if(Application::IsPlaying()) EntityManager::GetCurrentWorld()->GetSystem<PhysicsSystem>()->Simulate(0.016);
                                             }
    );
    Application::Run();
#pragma endregion
    if (enableRayTracing) RayTracerManager::End();
    Application::End();
}

void EngineSetup() {
    FileIO::SetProjectPath(PLANT_FACTORY_RESOURCE_FOLDER);
    Application::Init();
#pragma region Engine Setup
#pragma region Global light settings
    RenderManager::GetInstance().m_lightSettings.m_blockerSearchAmount = 6;
    RenderManager::GetInstance().m_lightSettings.m_pcfSampleAmount = 16;
    RenderManager::GetInstance().m_lightSettings.m_scaleFactor = 1.0f;
    RenderManager::GetInstance().m_lightSettings.m_ambientLight = 0.2f;
    RenderManager::SetShadowMapResolution(8192);
    RenderManager::GetInstance().m_stableFit = false;
    RenderManager::GetInstance().m_lightSettings.m_seamFixRatio = 0.05f;
    RenderManager::GetInstance().m_maxShadowDistance = 100;
    RenderManager::SetSplitRatio(0.15f, 0.3f, 0.5f, 1.0f);
#pragma endregion

    Transform transform;
    transform.SetEulerRotation(glm::radians(glm::vec3(150, 30, 0)));

#pragma region Preparations
    Application::SetTimeStep(0.016f);
    auto &world = EntityManager::GetCurrentWorld();

    const bool enableCameraControl = true;
    if (enableCameraControl) {
        auto *ccs = world->CreateSystem<CameraControlSystem>(SystemGroup::SimulationSystemGroup);
        ccs->Enable();
        ccs->SetVelocity(15.0f);
    }
    transform = Transform();
    transform.SetPosition(glm::vec3(0, 2, 35));
    transform.SetEulerRotation(glm::radians(glm::vec3(15, 0, 0)));
    auto mainCamera = RenderManager::GetMainCamera();
    if (mainCamera) {
        mainCamera->GetOwner().SetDataComponent(transform);
        mainCamera->m_useClearColor = true;
        mainCamera->m_clearColor = glm::vec3(0.5f);
    }
#pragma endregion
#pragma endregion

    const Entity lightEntity = EntityManager::CreateEntity("Light source");
    auto& pointLight = lightEntity.SetPrivateComponent<PointLight>();
    pointLight.m_diffuseBrightness = 15;
    pointLight.m_lightSize = 0.25f;
    transform.SetPosition(glm::vec3(0, 30, 0));
    transform.SetEulerRotation(glm::radians(glm::vec3(0, 0, 0)));
    lightEntity.SetDataComponent(transform);
}
