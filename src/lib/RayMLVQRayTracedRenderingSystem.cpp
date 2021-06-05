#include <RayMLVQRayTracedRenderingSystem.hpp>
#include <RayTracedRenderer.hpp>
#include <InputManager.hpp>
#include <WindowManager.hpp>
#include <EditorManager.hpp>
#include <ResourceManager.hpp>
#include <MeshRenderer.hpp>
#include <Cubemap.hpp>
#include <RayTracer.hpp>
using namespace RayMLVQ;

void RayMLVQRayTracedRenderingSystem::OnGui()
{
	if (m_rightMouseButtonHold && !InputManager::GetMouseInternal(GLFW_MOUSE_BUTTON_RIGHT, WindowManager::GetWindow()))
	{
		m_rightMouseButtonHold = false;
		m_startMouse = false;
	}
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{ 0, 0 });
	ImGui::Begin("MLVQ");
	{
		if (ImGui::BeginChild("CameraRenderer", ImVec2(0, 0), false, ImGuiWindowFlags_None | ImGuiWindowFlags_MenuBar)) {
			if (ImGui::BeginMenuBar())
			{
				if (ImGui::BeginMenu("Settings"))
				{
					static float lightSize = 1.0f;
					static glm::vec3 lightDir = glm::vec3(0, -1, 0);
					ImGui::DragFloat("FOV", &m_cameraFov, 1, 1, 120);
					ImGui::Checkbox("Use Geometry normal", &m_properties.m_useGeometryNormal);
					ImGui::Checkbox("Accumulate", &m_properties.m_accumulate);
					ImGui::DragInt("bounce limit", &m_properties.m_bounceLimit, 1, 1, 8);
					ImGui::DragInt("pixel samples", &m_properties.m_samplesPerPixel, 1, 1, 32);
					ImGui::Checkbox("Use environmental map", &m_properties.m_useEnvironmentalMap);
					ImGui::DragFloat("Skylight intensity", &m_properties.m_skylightIntensity, 0.01f, 0.0f, 5.0f);
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
		auto* window = ImGui::FindWindowByName("Ray Tracer");
		m_renderingEnabled = !(window->Hidden && !window->Collapsed);
	}
	ImGui::End();
	ImGui::PopStyleVar();
}

void RayMLVQRayTracedRenderingSystem::OnCreate()
{
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
		m_environmentalMap = ResourceManager::LoadCubemap(false, facesPath, true);
	}
#pragma endregion
	m_output = std::make_unique<OpenGLUtils::GLTexture2D>(0, GL_RGBA32F, 1, 1, false);
	m_output->SetData(0, GL_RGBA32F, GL_RGBA, GL_FLOAT, 0);
	m_output->SetInt(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	m_output->SetInt(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	m_output->SetInt(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	m_output->SetInt(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

void RayMLVQRayTracedRenderingSystem::OnDestroy()
{
}

void RayMLVQRayTracedRenderingSystem::PreUpdate()
{
}

void RayMLVQRayTracedRenderingSystem::Update()
{
	auto& size = m_outputSize;
	m_output->ReSize(0, GL_RGBA32F, GL_RGBA, GL_FLOAT, 0, size.x, size.y);
	m_properties.m_camera.Set(EditorManager::GetInstance().m_sceneCameraRotation, EditorManager::GetInstance().m_sceneCameraPosition, EditorManager::GetInstance().m_sceneCamera->m_fov, size);
	m_properties.m_environmentalMapId = m_environmentalMap->Texture()->Id();
	m_properties.m_frameSize = size;
	m_properties.m_outputTextureId = m_output->Id();
	if (!CudaModule::GetInstance().m_meshes.empty()) {
		m_rendered = CudaModule::RenderRayMLVQ(m_properties);
	}
}

void RayMLVQRayTracedRenderingSystem::FixedUpdate()
{
}

void RayMLVQRayTracedRenderingSystem::LateUpdate()
{
	OnGui();
}
