#include <RayTracerManager.hpp>
#include <RayTracedRenderer.hpp>

using namespace RayMLVQ;

void RayTracerManager::UpdateScene() const
{
	bool rebuildAccelerationStructure = false;
	bool updateShaderBindingTable = false;
	auto& meshesStorage = CudaModule::GetRayTracer()->m_instances;
	for (auto& i : meshesStorage)
	{
		i.m_removeTag = true;
	}
	if (const auto* rayTracerEntities = EntityManager::GetPrivateComponentOwnersList<RayTracedRenderer>(); rayTracerEntities)
	{
		for (auto entity : *rayTracerEntities) {
			if (!entity.IsEnabled()) continue;
			auto& rayTracerMaterial = entity.GetPrivateComponent<RayTracedRenderer>();
			if (!rayTracerMaterial->IsEnabled()) continue;
			if (!rayTracerMaterial->m_mesh || rayTracerMaterial->m_mesh->UnsafeGetVertexPositions().empty()) continue;
			auto globalTransform = entity.GetComponentData<GlobalTransform>().m_value;
			RayTracerInstance newCudaTriangleMesh;
			RayTracerInstance* cudaTriangleMesh = &newCudaTriangleMesh;
			bool needVerticesUpdate = false;
			bool needTransformUpdate = false;
			bool fromNew = true;
			bool needMaterialUpdate = false;
			for (auto& triangleMesh : meshesStorage)
			{
				if (triangleMesh.m_entityId == entity.m_index && triangleMesh.m_entityVersion == entity.m_version)
				{
					fromNew = false;
					cudaTriangleMesh = &triangleMesh;
					triangleMesh.m_removeTag = false;
					if (globalTransform != triangleMesh.m_globalTransform) {
						needTransformUpdate = true;
					}
					if (cudaTriangleMesh->m_version != rayTracerMaterial->m_mesh->GetVersion())
						needVerticesUpdate = true;
					if (cudaTriangleMesh->m_surfaceColor != rayTracerMaterial->m_surfaceColor
						|| cudaTriangleMesh->m_metallic != (rayTracerMaterial->m_metallic == 1.0f ? -1.0f : 1.0f / glm::pow(1.0f - rayTracerMaterial->m_metallic, 3.0f))
						|| cudaTriangleMesh->m_roughness != rayTracerMaterial->m_roughness
						)
					{
						needMaterialUpdate = true;
					}
				}
			}
			cudaTriangleMesh->m_version = rayTracerMaterial->m_mesh->GetVersion();
			if (fromNew || needVerticesUpdate || needTransformUpdate || needMaterialUpdate) {
				updateShaderBindingTable = true;
				cudaTriangleMesh->m_surfaceColor = rayTracerMaterial->m_surfaceColor;
				cudaTriangleMesh->m_metallic = rayTracerMaterial->m_metallic == 1.0f ? -1.0f : 1.0f / glm::pow(1.0f - rayTracerMaterial->m_metallic, 3.0f);
				cudaTriangleMesh->m_roughness = rayTracerMaterial->m_roughness;
				cudaTriangleMesh->m_normalTexture = 0;
				cudaTriangleMesh->m_albedoTexture = 0;
				cudaTriangleMesh->m_entityId = entity.m_index;
				cudaTriangleMesh->m_entityVersion = entity.m_version;
			}
			if (rayTracerMaterial->m_albedoTexture && rayTracerMaterial->m_albedoTexture->Texture()->Id() != cudaTriangleMesh->m_albedoTexture)
			{
				updateShaderBindingTable = true;
				cudaTriangleMesh->m_albedoTexture = rayTracerMaterial->m_albedoTexture->Texture()->Id();
			}
			else if (!rayTracerMaterial->m_albedoTexture && cudaTriangleMesh->m_albedoTexture != 0)
			{
				updateShaderBindingTable = true;
				cudaTriangleMesh->m_albedoTexture = 0;
			}
			if (rayTracerMaterial->m_normalTexture && rayTracerMaterial->m_normalTexture->Texture()->Id() != cudaTriangleMesh->m_normalTexture)
			{
				updateShaderBindingTable = true;
				cudaTriangleMesh->m_normalTexture = rayTracerMaterial->m_normalTexture->Texture()->Id();
			}
			else if (!rayTracerMaterial->m_normalTexture && cudaTriangleMesh->m_normalTexture != 0)
			{
				updateShaderBindingTable = true;
				cudaTriangleMesh->m_normalTexture = 0;
			}
			if (cudaTriangleMesh->m_diffuseIntensity != rayTracerMaterial->m_diffuseIntensity)
			{
				updateShaderBindingTable = true;
				cudaTriangleMesh->m_diffuseIntensity = rayTracerMaterial->m_diffuseIntensity;
			}
			if (fromNew || needVerticesUpdate) {
				rebuildAccelerationStructure = true;
				cudaTriangleMesh->m_verticesUpdateFlag = true;
				if (fromNew) {
					cudaTriangleMesh->m_transformUpdateFlag = true;
					cudaTriangleMesh->m_globalTransform = globalTransform;
				}
				cudaTriangleMesh->m_positions = &rayTracerMaterial->m_mesh->UnsafeGetVertexPositions();
				cudaTriangleMesh->m_normals = &rayTracerMaterial->m_mesh->UnsafeGetVertexNormals();
				cudaTriangleMesh->m_tangents = &rayTracerMaterial->m_mesh->UnsafeGetVertexTangents();
				cudaTriangleMesh->m_colors = &rayTracerMaterial->m_mesh->UnsafeGetVertexColors();
				cudaTriangleMesh->m_triangles = &rayTracerMaterial->m_mesh->UnsafeGetTriangles();
				cudaTriangleMesh->m_texCoords = &rayTracerMaterial->m_mesh->UnsafeGetVertexTexCoords();
			}
			else if (needTransformUpdate)
			{
				rebuildAccelerationStructure = true;
				cudaTriangleMesh->m_globalTransform = globalTransform;
				cudaTriangleMesh->m_transformUpdateFlag = true;
			}
			if (fromNew) meshesStorage.push_back(newCudaTriangleMesh);
		}
	}
	else
	{
		for (auto& i : meshesStorage)
		{
			i.m_removeTag = true;
		}
	}
	for (int i = 0; i < meshesStorage.size(); i++)
	{
		if (meshesStorage[i].m_removeTag)
		{
			meshesStorage.erase(meshesStorage.begin() + i);
			i--;
			rebuildAccelerationStructure = true;
		}
	}
	if (rebuildAccelerationStructure && !meshesStorage.empty()) {
		CudaModule::GetRayTracer()->BuildAccelerationStructure();
		CudaModule::GetRayTracer()->ClearAccumulate();
	}
	else if (updateShaderBindingTable)
	{
		CudaModule::GetRayTracer()->ClearAccumulate();
	}
}

RayTracerManager& RayTracerManager::GetInstance()
{
	static RayTracerManager instance;
	return instance;
}

void RayTracerManager::Init()
{
	auto& manager = GetInstance();
#pragma region Environmental map
	{
		const std::vector facesPath
		{
			FileIO::GetResourcePath("Textures/Skyboxes/Default/posx.jpg"),
		FileIO::GetResourcePath("Textures/Skyboxes/Default/negx.jpg"),
		FileIO::GetResourcePath("Textures/Skyboxes/Default/posy.jpg"),
		FileIO::GetResourcePath("Textures/Skyboxes/Default/negy.jpg"),
		FileIO::GetResourcePath("Textures/Skyboxes/Default/posz.jpg"),
		FileIO::GetResourcePath("Textures/Skyboxes/Default/negz.jpg"),
		};
		manager.m_environmentalMap = ResourceManager::LoadCubemap(false, facesPath, true);
	}
#pragma endregion
	CudaModule::Init();
	EditorManager::RegisterPrivateComponentMenu<RayTracedRenderer>([](Entity owner)
		{
			if (owner.HasPrivateComponent<RayTracedRenderer>()) return;
			if (ImGui::SmallButton("RayTracerMaterial"))
			{
				owner.SetPrivateComponent(std::make_unique<RayTracedRenderer>());
			}
		}
	);

	manager.m_defaultWindow.Init("Ray:Default");
	manager.m_rayMLVQWindow.Init("Ray:MLVQ");
}

void RayTracerRenderWindow::Init(const std::string& name)
{
	m_output = std::make_unique<OpenGLUtils::GLTexture2D>(0, GL_RGBA32F, 1, 1, false);
	m_output->SetData(0, GL_RGBA32F, GL_RGBA, GL_FLOAT, 0);
	m_output->SetInt(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	m_output->SetInt(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	m_output->SetInt(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	m_output->SetInt(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	m_name = name;
}

void RayTracerManager::Update()
{
	auto& manager = GetInstance();
	manager.UpdateScene();
	auto& size = manager.m_defaultWindow.m_outputSize;
	manager.m_defaultWindow.m_output->ReSize(0, GL_RGBA32F, GL_RGBA, GL_FLOAT, 0, size.x, size.y);
	manager.m_defaultRenderingProperties.m_camera.Set(EditorManager::GetInstance().m_sceneCameraRotation, EditorManager::GetInstance().m_sceneCameraPosition, EditorManager::GetInstance().m_sceneCamera->m_fov, size);
	manager.m_defaultRenderingProperties.m_environmentalMapId = manager.m_environmentalMap->Texture()->Id();
	manager.m_defaultRenderingProperties.m_frameSize = size;
	manager.m_defaultRenderingProperties.m_outputTextureId = manager.m_defaultWindow.m_output->Id();
	if (!CudaModule::GetRayTracer()->m_instances.empty()) {
		manager.m_defaultWindow.m_rendered = CudaModule::GetRayTracer()->RenderDefault(manager.m_defaultRenderingProperties);
	}

	size = manager.m_rayMLVQWindow.m_outputSize;
	manager.m_rayMLVQWindow.m_output->ReSize(0, GL_RGBA32F, GL_RGBA, GL_FLOAT, 0, size.x, size.y);
	manager.m_rayMLVQRenderingProperties.m_camera.Set(EditorManager::GetInstance().m_sceneCameraRotation, EditorManager::GetInstance().m_sceneCameraPosition, EditorManager::GetInstance().m_sceneCamera->m_fov, size);
	manager.m_rayMLVQRenderingProperties.m_environmentalMapId = manager.m_environmentalMap->Texture()->Id();
	manager.m_rayMLVQRenderingProperties.m_frameSize = size;
	manager.m_rayMLVQRenderingProperties.m_outputTextureId = manager.m_rayMLVQWindow.m_output->Id();
	if (!CudaModule::GetRayTracer()->m_instances.empty()) {
		manager.m_rayMLVQWindow.m_rendered = CudaModule::GetRayTracer()->RenderRayMLVQ(manager.m_rayMLVQRenderingProperties);
	}
}

void RayTracerManager::OnGui()
{
	auto& manager = GetInstance();
	manager.m_defaultWindow.OnGui();
	manager.m_rayMLVQWindow.OnGui();
}

void RayTracerManager::End()
{
	CudaModule::Terminate();
}

void RayTracerRenderWindow::Resize()
{
	auto& size = m_outputSize;
	m_output->ReSize(0, GL_RGBA32F, GL_RGBA, GL_FLOAT, 0, size.x, size.y);
}

void RayTracerRenderWindow::OnGui()
{
	if (m_rightMouseButtonHold && !InputManager::GetMouseInternal(GLFW_MOUSE_BUTTON_RIGHT, WindowManager::GetWindow()))
	{
		m_rightMouseButtonHold = false;
		m_startMouse = false;
	}
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{ 0, 0 });
	ImGui::Begin(m_name.c_str());
	{
		if (ImGui::BeginChild("CameraRenderer", ImVec2(0, 0), false, ImGuiWindowFlags_None | ImGuiWindowFlags_MenuBar)) {
			if (ImGui::BeginMenuBar())
			{
				if (ImGui::BeginMenu("Settings"))
				{
					/*
					ImGui::DragFloat("FOV", &m_cameraFov, 1, 1, 120);
					ImGui::Checkbox("Use Geometry normal", &renderingProperties.m_useGeometryNormal);
					ImGui::Checkbox("Accumulate", &renderingProperties.m_accumulate);
					ImGui::DragInt("bounce limit", &renderingProperties.m_bounceLimit, 1, 1, 8);
					ImGui::DragInt("pixel samples", &renderingProperties.m_samplesPerPixel, 1, 1, 32);
					ImGui::Checkbox("Use environmental map", &renderingProperties.m_useEnvironmentalMap);
					ImGui::DragFloat("Skylight intensity", &renderingProperties.m_skylightIntensity, 0.01f, 0.0f, 5.0f);
					*/
					ImGui::EndMenu();
				}
				ImGui::EndMenuBar();
			}
			ImVec2 viewPortSize = ImGui::GetWindowSize();
			viewPortSize.y -= 20;
			if (viewPortSize.y < 0) viewPortSize.y = 0;
			m_outputSize = glm::ivec2(viewPortSize.x, viewPortSize.y);
			if (m_rendered) ImGui::Image(reinterpret_cast<ImTextureID>(m_output->Id()), viewPortSize, ImVec2(0, 1), ImVec2(1, 0));
			else ImGui::Text("No mesh in the scene!");
			if (ImGui::IsWindowFocused())
			{
				const bool valid = true;
				const glm::vec2 mousePosition = InputManager::GetMouseAbsolutePositionInternal(WindowManager::GetWindow());
				if (valid) {
					if (!m_startMouse) {
						m_lastX = mousePosition.x;
						m_lastY = mousePosition.y;
						m_startMouse = true;
					}
					const float xOffset = mousePosition.x - m_lastX;
					const float yOffset = -mousePosition.y + m_lastY;
					m_lastX = mousePosition.x;
					m_lastY = mousePosition.y;
#pragma region Scene Camera Controller
					if (!m_rightMouseButtonHold && InputManager::GetMouseInternal(GLFW_MOUSE_BUTTON_RIGHT, WindowManager::GetWindow())) {
						m_rightMouseButtonHold = true;
					}
					if (m_rightMouseButtonHold && !EditorManager::GetInstance().m_lockCamera)
					{
						const glm::vec3 front = EditorManager::GetInstance().m_sceneCameraRotation * glm::vec3(0, 0, -1);
						glm::vec3 right;
						right = EditorManager::GetInstance().m_sceneCameraRotation * glm::vec3(1, 0, 0);
						if (InputManager::GetKeyInternal(GLFW_KEY_W, WindowManager::GetWindow())) {
							EditorManager::GetInstance().m_sceneCameraPosition += front * static_cast<float>(Application::GetCurrentWorld()->Time()->DeltaTime()) * EditorManager::GetInstance().m_velocity;
						}
						if (InputManager::GetKeyInternal(GLFW_KEY_S, WindowManager::GetWindow())) {
							EditorManager::GetInstance().m_sceneCameraPosition -= front * static_cast<float>(Application::GetCurrentWorld()->Time()->DeltaTime()) * EditorManager::GetInstance().m_velocity;
						}
						if (InputManager::GetKeyInternal(GLFW_KEY_A, WindowManager::GetWindow())) {
							EditorManager::GetInstance().m_sceneCameraPosition -= right * static_cast<float>(Application::GetCurrentWorld()->Time()->DeltaTime()) * EditorManager::GetInstance().m_velocity;
						}
						if (InputManager::GetKeyInternal(GLFW_KEY_D, WindowManager::GetWindow())) {
							EditorManager::GetInstance().m_sceneCameraPosition += right * static_cast<float>(Application::GetCurrentWorld()->Time()->DeltaTime()) * EditorManager::GetInstance().m_velocity;
						}
						if (InputManager::GetKeyInternal(GLFW_KEY_LEFT_SHIFT, WindowManager::GetWindow())) {
							EditorManager::GetInstance().m_sceneCameraPosition.y += EditorManager::GetInstance().m_velocity * static_cast<float>(Application::GetCurrentWorld()->Time()->DeltaTime());
						}
						if (InputManager::GetKeyInternal(GLFW_KEY_LEFT_CONTROL, WindowManager::GetWindow())) {
							EditorManager::GetInstance().m_sceneCameraPosition.y -= EditorManager::GetInstance().m_velocity * static_cast<float>(Application::GetCurrentWorld()->Time()->DeltaTime());
						}
						if (xOffset != 0.0f || yOffset != 0.0f) {
							EditorManager::GetInstance().m_sceneCameraYawAngle += xOffset * EditorManager::GetInstance().m_sensitivity;
							EditorManager::GetInstance().m_sceneCameraPitchAngle += yOffset * EditorManager::GetInstance().m_sensitivity;
							if (EditorManager::GetInstance().m_sceneCameraPitchAngle > 89.0f)
								EditorManager::GetInstance().m_sceneCameraPitchAngle = 89.0f;
							if (EditorManager::GetInstance().m_sceneCameraPitchAngle < -89.0f)
								EditorManager::GetInstance().m_sceneCameraPitchAngle = -89.0f;

							EditorManager::GetInstance().m_sceneCameraRotation = CameraComponent::ProcessMouseMovement(EditorManager::GetInstance().m_sceneCameraYawAngle, EditorManager::GetInstance().m_sceneCameraPitchAngle, false);
						}
					}
#pragma endregion
				}
			}
		}
		ImGui::EndChild();
		auto* window = ImGui::FindWindowByName(m_name.c_str());
		m_renderingEnabled = !(window->Hidden && !window->Collapsed);
	}
	ImGui::End();
	ImGui::PopStyleVar();
}