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
using namespace PlantFactory;
void EngineSetup();
int main()
{
	EngineSetup();
	PlantManager::Init();
	SorghumManager::Init();
	TreeManager::Init();
#pragma region Engine Loop
	Application::RegisterUpdateFunction([]()
		{
			PlantManager::Update();
			SorghumManager::Update();
			TreeManager::Update();
			PlantManager::RenderRayTracerDebugOutput();
		}
	);
	Application::RegisterLateUpdateFunction([]()
		{
			PlantManager::OnGui();
			TreeManager::OnGui();
			SorghumManager::OnGui();
		}
	);
	Application::Run();
#pragma endregion
	Application::End();
}
void EngineSetup()
{
	FileIO::SetProjectPath(SORGHUMMLVQ_RESOURCE_FOLDER);
	Application::Init();
#pragma region Engine Setup
#pragma region Global light settings
	RenderManager::GetInstance().m_lightSettings.m_blockerSearchAmount = 6;
	RenderManager::GetInstance().m_lightSettings.m_pcfSampleAmount = 16;
	RenderManager::GetInstance().m_lightSettings.m_scaleFactor = 1.0f;
	RenderManager::GetInstance().m_lightSettings.m_ambientLight = 0.2f;
	RenderManager::SetShadowMapResolution(4096);
	RenderManager::GetInstance().m_stableFit = false;
	RenderManager::GetInstance().m_lightSettings.m_seamFixRatio = 0.05f;
	RenderManager::GetInstance().m_maxShadowDistance = 100;
	RenderManager::SetSplitRatio(0.15f, 0.3f, 0.5f, 1.0f);
#pragma endregion

	Transform transform;
	transform.SetEulerRotation(glm::radians(glm::vec3(150, 30, 0)));

#pragma region Preparations
	Application::SetTimeStep(0.016f);
	auto& world = Application::GetCurrentWorld();

	const bool enableCameraControl = true;
	if (enableCameraControl) {
		auto* ccs = world->CreateSystem<CameraControlSystem>(SystemGroup::SimulationSystemGroup);
		ccs->Enable();
		ccs->SetVelocity(15.0f);
	}
	transform = Transform();
	transform.SetPosition(glm::vec3(0, 2, 35));
	transform.SetEulerRotation(glm::radians(glm::vec3(15, 0, 0)));
	auto mainCamera = RenderManager::GetMainCamera();
	if (mainCamera) {
		mainCamera->GetOwner().SetComponentData(transform);
		mainCamera->m_drawSkyBox = false;
		mainCamera->m_clearColor = glm::vec3(0.2f);
		auto postProcessing = std::make_unique<PostProcessing>();
		postProcessing->PushLayer(std::make_unique<Bloom>());
		//postProcessing->PushLayer(std::make_unique<SSAO>());
		mainCamera->GetOwner().SetPrivateComponent(std::move(postProcessing));
	}
	EditorManager::GetSceneCamera()->m_clearColor = glm::vec3(0.2f);
#pragma endregion
	JobManager::ResizeSecondaryWorkers(0);
	JobManager::ResizePrimaryWorkers(18);
#pragma endregion

	const Entity lightEntity = EntityManager::CreateEntity("Light source");
	auto pointLight = std::make_unique<PointLight>();
	pointLight->m_diffuseBrightness = 15;
	pointLight->m_lightSize = 0.25f;
	transform.SetPosition(glm::vec3(0, 30, 0));
	transform.SetEulerRotation(glm::radians(glm::vec3(0, 0, 0)));
	lightEntity.SetComponentData(transform);
	lightEntity.SetPrivateComponent(std::move(pointLight));
}
