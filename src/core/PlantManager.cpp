#include <PlantManager.hpp>
#include <CubeVolume.hpp>
#include <CUDAModule.hpp>
#include <Utilities.hpp>
#include <Volume.hpp>
#include <concurrent_vector.h>
#include <Curve.hpp>
#include <RayTracerMaterial.hpp>
#include <SorghumManager.hpp>
#include <TreeManager.hpp>

using namespace PlantFactory;

#pragma region GUI Related
void ResourceParcel::OnGui() const
{
	ImGui::Text(("Nutrient: " + std::to_string(m_nutrient)).c_str());
	ImGui::Text(("Carbon: " + std::to_string(m_carbon)).c_str());
}

void InternodeData::OnGui()
{
	if (ImGui::TreeNode("Display buds")) {
		for (int i = 0; i < m_buds.size(); i++)
		{
			ImGui::Text(("Bud: " + std::to_string(i)).c_str());
			if (ImGui::TreeNode("Info")) {
				ImGui::Text(m_buds[i].m_isApical ? "Type: Apical" : "Type: Lateral");
				ImGui::Text(m_buds[i].m_active ? "Status: Active" : "Status: Not active");
				ImGui::Text(m_buds[i].m_enoughForGrowth ? "Has enough resource: True" : "Has enough resource: False");
				ImGui::Text(("ResourceWeight: " + std::to_string(m_buds[i].m_resourceWeight)).c_str());
				if (ImGui::TreeNode("Current Resource")) {
					m_buds[i].m_currentResource.OnGui();
					ImGui::TreePop();
				}
				ImGui::TreePop();
			}
		}
		ImGui::TreePop();
	}
}
const char* DebugOutputRenderTypes[]{
	"Shadow",
	"Glass",
	"BRDF"
};

void PlantManager::OnGui()
{
	auto& manager = GetInstance();

	if (ImGui::Begin("Plant Manager"))
	{
		if (ImGui::Button("Delete all plants")) {
			ImGui::OpenPopup("Delete Warning");
		}
		if (ImGui::BeginPopupModal("Delete Warning", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
		{
			ImGui::Text("Are you sure? All plants will be removed!");
			if (ImGui::Button("Yes, delete all!", ImVec2(120, 0))) {
				DeleteAllPlants();
				TreeManager::GenerateLeavesForTree(manager);
				ImGui::CloseCurrentPopup();
			}
			ImGui::SetItemDefaultFocus();
			ImGui::SameLine();
			if (ImGui::Button("Cancel", ImVec2(120, 0))) { ImGui::CloseCurrentPopup(); }
			ImGui::EndPopup();
		}
		ImGui::Text(("Internode amount: " + std::to_string(manager.m_internodes.size())).c_str());
		if (ImGui::CollapsingHeader("Growth", ImGuiTreeNodeFlags_DefaultOpen))
		{
			if (ImGui::Button("Recalculate illumination")) CalculateIlluminationForInternodes(manager);
			static int pushAmount = 20;
			ImGui::DragInt("Amount", &pushAmount, 1, 0, 60.0f / manager.m_deltaTime);
			if (ImGui::Button("Push time (grow by iteration)")) manager.m_iterationsToGrow = pushAmount;
			if (Application::IsPlaying() && ImGui::Button("Push time (grow instantly)")) {
				const float time = Application::EngineTime();
				GrowAllPlants(pushAmount);
				const std::string spendTime = std::to_string(Application::EngineTime() - time);
				Debug::Log("Growth finished in " + spendTime + " sec.");
			}

			ImGui::SliderFloat("Time speed", &manager.m_deltaTime, 0.1f, 1.0f);

			if (ImGui::TreeNode("Timers")) {
				ImGui::Text("Resource allocation: %.3fs", manager.m_resourceAllocationTimer);
				ImGui::Text("Form internodes: %.3fs", manager.m_internodeFormTimer);
				ImGui::Text("Create internodes: %.3fs", manager.m_internodeCreateTimer);
				ImGui::Text("Illumination: %.3fs", manager.m_illuminationCalculationTimer);
				ImGui::Text("Pruning & Metadata: %.3fs", manager.m_metaDataTimer);
				ImGui::TreePop();
			}
		}

	}
	ImGui::End();
	if (manager.m_rightMouseButtonHold && !InputManager::GetMouseInternal(GLFW_MOUSE_BUTTON_RIGHT, WindowManager::GetWindow()))
	{
		manager.m_rightMouseButtonHold = false;
		manager.m_startMouse = false;
	}
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{ 0, 0 });
	ImGui::Begin("Ray Tracer");
	{
		if (ImGui::BeginChild("CameraRenderer", ImVec2(0, 0), false, ImGuiWindowFlags_None | ImGuiWindowFlags_MenuBar)) {
			if (ImGui::BeginMenuBar())
			{
				if (ImGui::BeginMenu("Settings"))
				{
					static float lightSize = 1.0f;
					static glm::vec3 lightDir = glm::vec3(0, -1, 0);
					ImGui::DragFloat("FOV", &manager.m_cameraFov, 1, 1, 120);
					ImGui::Checkbox("Use Geometry normal", &manager.m_properties.m_useGeometryNormal);
					ImGui::Checkbox("Accumulate", &manager.m_properties.m_accumulate);
					ImGui::DragInt("bounce limit", &manager.m_properties.m_bounceLimit, 1, 1, 8);
					ImGui::DragInt("pixel samples", &manager.m_properties.m_samplesPerPixel, 1, 1, 32);
					ImGui::Checkbox("Use environmental map", &manager.m_properties.m_useEnvironmentalMap);
					ImGui::DragFloat("Skylight intensity", &manager.m_properties.m_skylightIntensity, 0.01f, 0.0f, 5.0f);
					ImGui::Combo("Render type", (int*)&manager.m_properties.m_debugRenderingType, DebugOutputRenderTypes, IM_ARRAYSIZE(DebugOutputRenderTypes));
					switch (manager.m_properties.m_debugRenderingType)
					{
					case RayMLVQ::DebugOutputRenderType::SoftShadow:
					{
						ImGui::Text("Shadow softness");
						if (ImGui::DragFloat("Shadow softness", &lightSize, 0.002f, 0.0f, 2.0f))
						{
							RayMLVQ::CudaModule::SetSkylightSize(lightSize);
						}
						ImGui::Text("Shadow direction");
						if (ImGui::DragFloat3("Shadow dir", &lightDir.x, 0.002f, -7.0f, 7.0f))
						{
							RayMLVQ::CudaModule::SetSkylightDir(glm::normalize(lightDir));
						}
					}
					break;
					case RayMLVQ::DebugOutputRenderType::Glass:
					{

					}
					break;
					case RayMLVQ::DebugOutputRenderType::Brdf:
					{

					}
					break;
					default: break;
					}
					ImGui::EndMenu();
				}
				ImGui::EndMenuBar();
			}
			ImVec2 viewPortSize = ImGui::GetWindowSize();
			viewPortSize.y -= 20;
			if (viewPortSize.y < 0) viewPortSize.y = 0;
			manager.m_rayTracerTestOutputSize = glm::ivec2(viewPortSize.x, viewPortSize.y);
			if (manager.m_rendered) ImGui::Image(reinterpret_cast<ImTextureID>(manager.m_rayTracerTestOutput->Id()), viewPortSize, ImVec2(0, 1), ImVec2(1, 0));
			else ImGui::Text("No mesh in the scene!");
			if (ImGui::IsWindowFocused())
			{
				const bool valid = true;
				const glm::vec2 mousePosition = InputManager::GetMouseAbsolutePositionInternal(WindowManager::GetWindow());
				if (valid) {
					if (!manager.m_startMouse) {
						manager.m_lastX = mousePosition.x;
						manager.m_lastY = mousePosition.y;
						manager.m_startMouse = true;
					}
					const float xOffset = mousePosition.x - manager.m_lastX;
					const float yOffset = -mousePosition.y + manager.m_lastY;
					manager.m_lastX = mousePosition.x;
					manager.m_lastY = mousePosition.y;
#pragma region Scene Camera Controller
					if (!manager.m_rightMouseButtonHold && InputManager::GetMouseInternal(GLFW_MOUSE_BUTTON_RIGHT, WindowManager::GetWindow())) {
						manager.m_rightMouseButtonHold = true;
					}
					if (manager.m_rightMouseButtonHold && !EditorManager::GetInstance().m_lockCamera)
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
		manager.m_rayTracerDebugRenderingEnabled = !(window->Hidden && !window->Collapsed);
	}
	ImGui::End();
	ImGui::PopStyleVar();

}
#pragma endregion
#pragma region Growth related
bool PlantManager::GrowAllPlants()
{
	auto& manager = GetInstance();
	Refresh();
	manager.m_globalTime += manager.m_deltaTime;
	float time = Application::EngineTime();
	std::vector<ResourceParcel> totalResources;
	totalResources.resize(manager.m_plants.size());
	std::vector<ResourceParcel> resourceAvailable;
	resourceAvailable.resize(manager.m_plants.size());
	for (auto& i : resourceAvailable) i.m_nutrient = 5000000.0f;
	CollectNutrient(manager.m_plants, totalResources, resourceAvailable);
	for (auto& i : manager.m_plantResourceAllocators)
	{
		i.second(manager, totalResources);
	}
	manager.m_resourceAllocationTimer = Application::EngineTime() - time;

	for (const auto& plant : manager.m_plants)
	{
		auto plantInfo = plant.GetComponentData<PlantInfo>();
		plantInfo.m_age += manager.m_deltaTime;
		plant.SetComponentData(plantInfo);
	}

	time = Application::EngineTime();
	Concurrency::concurrent_vector<InternodeCandidate> candidates;
	for (auto& i : manager.m_plantGrowthModels)
	{
		i.second(manager, candidates);
	}
	manager.m_internodeFormTimer = Application::EngineTime() - time;

	if (GrowCandidates(candidates))
	{
		time = Application::EngineTime();
		std::vector<Volume*> obstacles;
		const auto* entities = EntityManager::GetPrivateComponentOwnersList<CubeVolume>();
		if (entities) {
			for (const auto& entity : *entities)
			{
				if (!entity.IsEnabled()) continue;
				auto& volume = entity.GetPrivateComponent<CubeVolume>();
				if (volume->IsEnabled() && volume->m_asObstacle) obstacles.push_back(volume.get());
			}
		}
		for (auto& i : manager.m_plantInternodePruners)
		{
			i.second(manager, obstacles);
		}
		manager.m_pruningTimer = Application::EngineTime() - time;
		time = Application::EngineTime();
		for (auto& i : manager.m_plantMetaDataCalculators)
		{
			i.second(manager);
		}
		manager.m_metaDataTimer = Application::EngineTime() - time;
		return true;
	}
	return false;
}
bool PlantManager::GrowAllPlants(const unsigned& iterations)
{
	bool grew = false;
	for (unsigned i = 0; i < iterations; i++)
	{
		const bool grewInThisIteration = GrowAllPlants();
		grew = grew || grewInThisIteration;
	}
	if (grew) Refresh();
	return grew;
}

bool PlantManager::GrowCandidates(Concurrency::concurrent_vector<InternodeCandidate>& candidates)
{
	const float time = Application::EngineTime();
	if (candidates.empty()) return false;
	auto entities = EntityManager::CreateEntities(GetInstance().m_internodeArchetype, candidates.size(), "Internode");
	int i = 0;
	for (auto& candidate : candidates) {
		auto newInternode = entities[i];
		newInternode.SetComponentData(candidate.m_info);
		newInternode.SetComponentData(candidate.m_growth);
		newInternode.SetComponentData(candidate.m_statistics);
		newInternode.SetComponentData(candidate.m_globalTransform);
		newInternode.SetComponentData(candidate.m_transform);
		auto newInternodeData = std::make_unique<InternodeData>();
		newInternodeData->m_buds.swap(candidate.m_buds);
		newInternodeData->m_owner = candidate.m_owner;
		EntityManager::SetPrivateComponent<InternodeData>(newInternode, std::move(newInternodeData));
		EntityManager::SetParent(newInternode, candidate.m_parent);
		i++;
	}
	GetInstance().m_internodeCreateTimer = Application::EngineTime() - time;
	return true;
}

void PlantManager::CalculateIlluminationForInternodes(PlantManager& manager)
{
	if (manager.m_internodeTransforms.empty()) return;
	const float time = Application::EngineTime();
	//Upload geometries to OptiX.
	UpdateDebugRenderOutputScene();
	RayMLVQ::IlluminationEstimationProperties properties;
	properties.m_bounceLimit = 1;
	properties.m_numPointSamples = 1000;
	properties.m_numScatterSamples = 1;
	properties.m_seed = glm::linearRand(16384, 32768);
	properties.m_skylightPower = 1.0f;
	properties.m_pushNormal = true;
	std::vector<RayMLVQ::LightProbe<float>> lightProbes;
	lightProbes.resize(manager.m_internodeQuery.GetEntityAmount());
	EntityManager::ForEach<GlobalTransform>(JobManager::PrimaryWorkers(), manager.m_internodeQuery, [&](int i, Entity leafEntity, GlobalTransform& globalTransform)
		{
			lightProbes[i].m_position = globalTransform.GetPosition();
			lightProbes[i].m_surfaceNormal = globalTransform.GetRotation() * glm::vec3(0.0f, 0.0f, -1.0f);
		}, false
		);
	if (lightProbes.empty()) return;
	RayMLVQ::CudaModule::EstimateIlluminationRayTracing(properties, lightProbes);

	EntityManager::ForEach<Illumination>(JobManager::PrimaryWorkers(), manager.m_internodeQuery, [&](int i, Entity leafEntity, Illumination& illumination)
		{
			const auto& lightProbe = lightProbes[i];
			illumination.m_accumulatedDirection += lightProbe.m_direction * manager.m_deltaTime;
			illumination.m_currentIntensity = lightProbe.m_energy;
		}, false
		);

	manager.m_illuminationCalculationTimer = Application::EngineTime() - time;
}
void PlantManager::CollectNutrient(std::vector<Entity>& trees, std::vector<ResourceParcel>& totalNutrients,
	std::vector<ResourceParcel>& nutrientsAvailable)
{
	for (int i = 0; i < trees.size(); i++)
	{
		totalNutrients[i].m_nutrient = nutrientsAvailable[i].m_nutrient;
	}
}

void PlantManager::ApplyTropism(const glm::vec3& targetDir, float tropism, glm::vec3& front, glm::vec3& up)
{
	const glm::vec3 dir = glm::normalize(targetDir);
	const float dotP = glm::abs(glm::dot(front, dir));
	if (dotP < 0.99f && dotP > -0.99f)
	{
		const glm::vec3 left = glm::cross(front, dir);
		const float maxAngle = glm::acos(dotP);
		const float rotateAngle = maxAngle * tropism;
		front = glm::normalize(glm::rotate(front, glm::min(maxAngle, rotateAngle), left));
		up = glm::normalize(glm::cross(glm::cross(front, up), front));
		//up = glm::normalize(glm::rotate(up, glm::min(maxAngle, rotateAngle), left));
	}
}
#pragma endregion
#pragma region ResourceParcel
ResourceParcel::ResourceParcel()
{
	m_nutrient = 0;
	m_carbon = 0;
}

ResourceParcel::ResourceParcel(const float& water, const float& carbon)
{
	m_nutrient = water;
	m_carbon = carbon;
}

ResourceParcel& ResourceParcel::operator+=(const ResourceParcel& value)
{
	m_nutrient += value.m_nutrient;
	m_carbon += value.m_carbon;
	return *this;
}

bool ResourceParcel::IsEnough() const
{
	return m_nutrient > 1.0f && m_carbon > 1.0f;
}


#pragma endregion
#pragma region Helpers

Entity PlantManager::CreateCubeObstacle()
{
	const auto volumeEntity = EntityManager::CreateEntity("Volume");
	volumeEntity.SetEnabled(false);
	Transform transform;
	transform.SetPosition(glm::vec3(0, 10, 0));
	transform.SetScale(glm::vec3(4, 2, 4));
	GlobalTransform globalTransform;
	globalTransform.m_value = transform.m_value;
	volumeEntity.SetComponentData(transform);
	volumeEntity.SetComponentData(globalTransform);
	volumeEntity.SetStatic(true);
	volumeEntity.SetPrivateComponent(std::make_unique<CubeVolume>());
	auto meshRenderer = std::make_unique<MeshRenderer>();
	meshRenderer->m_mesh = Default::Primitives::Cube;
	meshRenderer->m_material = Default::Materials::StandardMaterial;
	volumeEntity.SetPrivateComponent(std::move(meshRenderer));
	auto& volume = volumeEntity.GetPrivateComponent<CubeVolume>();
	volume->ApplyMeshRendererBounds();
	return volumeEntity;
}

void PlantManager::UpdateDebugRenderOutputScene()
{
	auto& manager = GetInstance();
	if(!manager.m_rayTracerDebugRenderingEnabled) return;
	bool needGeometryUpdate = false;
	bool needSbtUpdate = false;
	auto& meshesStorage = RayMLVQ::CudaModule::GetInstance().m_meshes;
	for (auto& i : meshesStorage)
	{
		i.m_removeTag = true;
	}
	const auto* rayTracerEntities = EntityManager::GetPrivateComponentOwnersList<RayTracerMaterial>();
	if (rayTracerEntities)
	{
		for (auto entity : *rayTracerEntities) {
			if (!entity.IsEnabled()) continue;
			if (!entity.HasPrivateComponent<MeshRenderer>()) continue;
			auto& rayTracerMaterial = entity.GetPrivateComponent<RayTracerMaterial>();
			if (!rayTracerMaterial->IsEnabled()) continue;
			auto& meshRenderer = entity.GetPrivateComponent<MeshRenderer>();
			if (!meshRenderer->IsEnabled()) continue;
			if (!meshRenderer->m_mesh || meshRenderer->m_mesh->UnsafeGetVertices().empty()) continue;
			auto globalTransform = entity.GetComponentData<GlobalTransform>().m_value;
			RayMLVQ::TriangleMesh newCudaTriangleMesh;
			RayMLVQ::TriangleMesh* cudaTriangleMesh = &newCudaTriangleMesh;
			bool needUpdate = false;
			bool fromNew = true;
			auto vertices = meshRenderer->m_mesh->UnsafeGetVertices();
			auto triangles = meshRenderer->m_mesh->UnsafeGetTriangles();
			bool needMaterialUpdate = false;
			for (auto& i : meshesStorage)
			{
				if (entity.m_index == i.m_id && entity.m_version == i.m_version)
				{
					fromNew = false;
					cudaTriangleMesh = &i;
					i.m_removeTag = false;
					if (globalTransform != i.m_globalTransform) needUpdate = true;
					if (cudaTriangleMesh->m_vertices.size() != vertices.size())
						needUpdate = true;
					if (cudaTriangleMesh->m_color != meshRenderer->m_material->m_albedoColor
						|| cudaTriangleMesh->m_metallic != (meshRenderer->m_material->m_metallic == 1.0f ? -1.0f : 1.0f / glm::pow(1.0f - meshRenderer->m_material->m_metallic, 3.0f))
						|| cudaTriangleMesh->m_roughness != meshRenderer->m_material->m_roughness
						)
					{
						needMaterialUpdate = true;
					}

				}
			}
			if (fromNew || needUpdate || needMaterialUpdate) {
				needSbtUpdate = true;
				cudaTriangleMesh->m_color = meshRenderer->m_material->m_albedoColor;
				cudaTriangleMesh->m_metallic = meshRenderer->m_material->m_metallic == 1.0f ? -1.0f : 1.0f / glm::pow(1.0f - meshRenderer->m_material->m_metallic, 3.0f);
				cudaTriangleMesh->m_roughness = meshRenderer->m_material->m_roughness;
				cudaTriangleMesh->m_id = entity.m_index;
				cudaTriangleMesh->m_version = entity.m_version;
				cudaTriangleMesh->m_normalTexture = 0;
				cudaTriangleMesh->m_albedoTexture = 0;
			}


			if (rayTracerMaterial->m_albedoTexture && rayTracerMaterial->m_albedoTexture->Texture()->Id() != cudaTriangleMesh->m_albedoTexture)
			{
				needSbtUpdate = true;
				cudaTriangleMesh->m_albedoTexture = rayTracerMaterial->m_albedoTexture->Texture()->Id();
			}
			else if (!rayTracerMaterial->m_albedoTexture && cudaTriangleMesh->m_albedoTexture != 0)
			{
				needSbtUpdate = true;
				cudaTriangleMesh->m_albedoTexture = 0;
			}
			if (rayTracerMaterial->m_normalTexture && rayTracerMaterial->m_normalTexture->Texture()->Id() != cudaTriangleMesh->m_normalTexture)
			{
				needSbtUpdate = true;
				cudaTriangleMesh->m_normalTexture = rayTracerMaterial->m_normalTexture->Texture()->Id();
			}
			else if (!rayTracerMaterial->m_normalTexture && cudaTriangleMesh->m_normalTexture != 0)
			{
				needSbtUpdate = true;
				cudaTriangleMesh->m_normalTexture = 0;
			}
			if (cudaTriangleMesh->m_diffuseIntensity != rayTracerMaterial->m_diffuseIntensity)
			{
				needSbtUpdate = true;
				cudaTriangleMesh->m_diffuseIntensity = rayTracerMaterial->m_diffuseIntensity;
			}


			if (fromNew || needUpdate) {
				needGeometryUpdate = true;
				cudaTriangleMesh->m_globalTransform = globalTransform;
				cudaTriangleMesh->m_vertices.resize(vertices.size());
				cudaTriangleMesh->m_vertexInfos.resize(vertices.size());
				for (int index = 0; index < vertices.size(); index++)
				{
					cudaTriangleMesh->m_vertices[index] = globalTransform * glm::vec4(vertices[index].m_position, 1.0f);
					cudaTriangleMesh->m_vertexInfos[index].m_normal = glm::normalize(glm::vec3(globalTransform * glm::vec4(vertices[index].m_normal, 0.0f)));
					cudaTriangleMesh->m_vertexInfos[index].m_tangent = glm::normalize(glm::vec3(globalTransform * glm::vec4(vertices[index].m_tangent, 0.0f)));
					cudaTriangleMesh->m_vertexInfos[index].m_texCoords = vertices[index].m_texCoords0;
				}
				cudaTriangleMesh->m_indices.clear();
				cudaTriangleMesh->m_indices.insert(cudaTriangleMesh->m_indices.begin(), triangles.begin(), triangles.end());
			}
			if (fromNew && !cudaTriangleMesh->m_vertices.empty()) meshesStorage.push_back(std::move(newCudaTriangleMesh));
		}
	}else
	{
		for (int i = 0; i < meshesStorage.size(); i++)
		{
			meshesStorage[i].m_removeTag = true;
		}
	}
	for (int i = 0; i < meshesStorage.size(); i++)
	{
		if (meshesStorage[i].m_removeTag && meshesStorage[i].m_id != 0)
		{
			meshesStorage.erase(meshesStorage.begin() + i);
			i--;
			needGeometryUpdate = true;
		}
	}
	if (needGeometryUpdate && !meshesStorage.empty()) {
		RayMLVQ::CudaModule::PrepareScene();
		RayMLVQ::CudaModule::SetStatusChanged();
	}
	else if (needSbtUpdate)
	{
		RayMLVQ::CudaModule::SetStatusChanged();
	}
}

void PlantManager::RenderRayTracerDebugOutput()
{
	UpdateDebugRenderOutputScene();
	auto& manager = GetInstance();
	auto& size = manager.m_rayTracerTestOutputSize;
	manager.m_rayTracerTestOutput->ReSize(0, GL_RGBA32F, GL_RGBA, GL_FLOAT, 0, size.x, size.y);
	manager.m_properties.m_camera.Set(EditorManager::GetInstance().m_sceneCameraRotation, EditorManager::GetInstance().m_sceneCameraPosition, EditorManager::GetInstance().m_sceneCamera->m_fov, size);
	manager.m_properties.m_environmentalMapId = GetInstance().m_environmentalMap->Texture()->Id();
	manager.m_properties.m_frameSize = size;
	manager.m_properties.m_outputTextureId = manager.m_rayTracerTestOutput->Id();
	manager.m_rendered = RayMLVQ::CudaModule::RenderRayTracerDebugOutput(
		manager.m_properties
	);
}

void PlantManager::DeleteAllPlants()
{
	GetInstance().m_globalTime = 0;
	std::vector<Entity> trees;
	GetInstance().m_plantQuery.ToEntityArray(trees);
	for (const auto& tree : trees) EntityManager::DeleteEntity(tree);
	Refresh();
	SorghumManager::GetInstance().m_probeColors.clear();
	SorghumManager::GetInstance().m_probeTransforms.clear();
}

Entity PlantManager::CreatePlant(const PlantType& type, const Transform& transform)
{
	const auto entity = EntityManager::CreateEntity(GetInstance().m_plantArchetype);

	GlobalTransform globalTransform;
	globalTransform.m_value = transform.m_value;
	entity.SetComponentData(globalTransform);
	entity.SetComponentData(transform);
	entity.SetName("Tree");
	PlantInfo treeInfo;
	treeInfo.m_plantType = type;
	treeInfo.m_age = 0;
	treeInfo.m_startTime = GetInstance().m_globalTime;
	entity.SetComponentData(treeInfo);

#pragma region Set root internode
	const auto rootInternode = EntityManager::CreateEntity(GetInstance().m_internodeArchetype);
	rootInternode.SetName("Internode");
	InternodeInfo internodeInfo;
	internodeInfo.m_plantType = type;
	internodeInfo.m_plant = entity;
	internodeInfo.m_startAge = 0;
	internodeInfo.m_startGlobalTime = treeInfo.m_startTime;
	InternodeGrowth internodeGrowth;
	internodeGrowth.m_desiredLocalRotation = glm::quat(glm::vec3(glm::radians(90.0f), 0.0f, 0.0f));

	GlobalTransform internodeGlobalTransform;
	internodeGlobalTransform.m_value = entity.GetComponentData<GlobalTransform>().m_value * glm::mat4_cast(internodeGrowth.m_desiredLocalRotation);
	InternodeStatistics internodeStatistics;
	rootInternode.SetComponentData(internodeInfo);
	rootInternode.SetComponentData(internodeGrowth);
	rootInternode.SetComponentData(internodeStatistics);
	rootInternode.SetComponentData(internodeGlobalTransform);

	rootInternode.SetPrivateComponent(std::make_unique<InternodeData>());
	auto& rootInternodeData = rootInternode.GetPrivateComponent<InternodeData>();
	Bud bud;
	bud.m_isApical = true;
	rootInternodeData->m_buds.push_back(bud);
	rootInternodeData->m_owner = entity;
	EntityManager::SetParent(rootInternode, entity);
#pragma endregion
	return entity;
}

Entity PlantManager::CreateInternode(const PlantType& type, const Entity& parentEntity)
{
	const auto entity = EntityManager::CreateEntity(GetInstance().m_internodeArchetype);
	entity.SetName("Internode");
	InternodeInfo internodeInfo;
	internodeInfo.m_plantType = type;
	internodeInfo.m_plant = parentEntity.GetComponentData<InternodeInfo>().m_plant;
	internodeInfo.m_startAge = internodeInfo.m_plant.GetComponentData<PlantInfo>().m_age;
	entity.SetComponentData(internodeInfo);
	entity.SetPrivateComponent(std::make_unique<InternodeData>());
	EntityManager::SetParent(entity, parentEntity);
	return entity;
}




#pragma endregion
#pragma region Runtime
PlantManager& PlantManager::GetInstance()
{
	static PlantManager instance;
	return instance;
}

void PlantManager::Init()
{
	auto& manager = GetInstance();


#pragma region Ground
	manager.m_ground = EntityManager::CreateEntity("Ground");
	auto meshRenderer = std::make_unique<MeshRenderer>();
	meshRenderer->m_mesh = Default::Primitives::Quad;
	meshRenderer->m_material = ResourceManager::LoadMaterial(true, Default::GLPrograms::StandardProgram);
	meshRenderer->m_material->m_name = "Ground mat";
	meshRenderer->m_material->m_roughness = 0.0f;
	meshRenderer->m_material->m_metallic = 0.7f;
	meshRenderer->m_material->m_albedoColor = glm::vec3(1.0f);
	manager.m_ground.SetPrivateComponent(std::move(meshRenderer));
	Transform groundTransform;
	GlobalTransform groundGlobalTransform;
	groundTransform.SetScale(glm::vec3(1000.0f, 1.0f, 1000.0f));
	groundTransform.SetPosition(glm::vec3(0.0f, 0.0f, 0.0f));
	groundGlobalTransform.m_value = groundTransform.m_value;
	manager.m_ground.SetComponentData(groundTransform);
	manager.m_ground.SetComponentData(groundGlobalTransform);
	manager.m_ground.SetStatic(true);
	manager.m_ground.SetPrivateComponent(std::make_unique<RayTracerMaterial>());

	auto cubeVolume = std::make_unique<CubeVolume>();
	cubeVolume->m_asObstacle = true;
	cubeVolume->m_minMaxBound.m_max = glm::vec3(1000, -0.1f, 1000);
	cubeVolume->m_minMaxBound.m_min = glm::vec3(-1000, -10.0f, -1000);
	manager.m_ground.SetPrivateComponent(std::move(cubeVolume));
#pragma endregion
#pragma region Environmental map
	{
		const std::vector<std::string> facesPath
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
#pragma region Mask material
	std::string vertShaderCode = std::string("#version 460 core\n")
		+ *Default::ShaderIncludes::Uniform +
		+"\n"
		+ FileIO::LoadFileAsString(FileIO::GetResourcePath("Shaders/Vertex/Standard.vert"));
	std::string fragShaderCode = std::string("#version 460 core\n")
		+ *Default::ShaderIncludes::Uniform
		+ "\n"
		+ FileIO::LoadFileAsString(FileIO::GetAssetFolderPath() + "Shaders/Fragment/SemanticBranch.frag");

	auto standardVert = std::make_shared<OpenGLUtils::GLShader>(OpenGLUtils::ShaderType::Vertex);
	standardVert->Compile(vertShaderCode);
	auto standardFrag = std::make_shared<OpenGLUtils::GLShader>(OpenGLUtils::ShaderType::Fragment);
	standardFrag->Compile(fragShaderCode);
	auto branchProgram = std::make_shared<OpenGLUtils::GLProgram>(standardVert, standardFrag);


	vertShaderCode = std::string("#version 460 core\n")
		+ *Default::ShaderIncludes::Uniform +
		+"\n"
		+ FileIO::LoadFileAsString(FileIO::GetResourcePath("Shaders/Vertex/StandardInstanced.vert"));
	fragShaderCode = std::string("#version 460 core\n")
		+ *Default::ShaderIncludes::Uniform
		+ "\n"
		+ FileIO::LoadFileAsString(FileIO::GetAssetFolderPath() + "Shaders/Fragment/SemanticLeaf.frag");
	standardVert = std::make_shared<OpenGLUtils::GLShader>(OpenGLUtils::ShaderType::Vertex);
	standardVert->Compile(vertShaderCode);
	standardFrag = std::make_shared<OpenGLUtils::GLShader>(OpenGLUtils::ShaderType::Fragment);
	standardFrag->Compile(fragShaderCode);
	auto leafProgram = std::make_shared<OpenGLUtils::GLProgram>(standardVert, standardFrag);
#pragma endregion
#pragma region Entity
	manager.m_internodeArchetype = EntityManager::CreateEntityArchetype(
		"Internode",
		BranchCylinder(), BranchCylinderWidth(), BranchPointer(), BranchColor(), Ray(),
		Illumination(),
		InternodeInfo(),
		InternodeGrowth(),
		InternodeStatistics()
	);
	manager.m_plantArchetype = EntityManager::CreateEntityArchetype(
		"Tree",
		PlantInfo()
	);

	manager.m_internodeQuery = EntityManager::CreateEntityQuery();
	manager.m_internodeQuery.SetAllFilters(InternodeInfo());

	manager.m_plantQuery = EntityManager::CreateEntityQuery();
	manager.m_plantQuery.SetAllFilters(PlantInfo());
#pragma endregion
#pragma region GUI
	EditorManager::RegisterComponentDataInspector<InternodeStatistics>(
		[](Entity entity, ComponentDataBase* data, bool isRoot)
		{
			auto* internodeStatistics = static_cast<InternodeStatistics*>(data);
			ImGui::Text(("MaxChildOrder: " + std::to_string(internodeStatistics->m_maxChildOrder)).c_str());
			ImGui::Text(("MaxChildLevel: " + std::to_string(internodeStatistics->m_maxChildLevel)).c_str());
			ImGui::Text(("ChildrenEndNodeAmount: " + std::to_string(internodeStatistics->m_childrenEndNodeAmount)).c_str());
			ImGui::Text(("DistanceToBranchEnd: " + std::to_string(internodeStatistics->m_distanceToBranchEnd)).c_str());
			ImGui::Text(("LongestDistanceToAnyEndNode: " + std::to_string(internodeStatistics->m_longestDistanceToAnyEndNode)).c_str());
			ImGui::Text(("TotalLength: " + std::to_string(internodeStatistics->m_totalLength)).c_str());
			ImGui::Text(("DistanceToBranchStart: " + std::to_string(internodeStatistics->m_distanceToBranchStart)).c_str());
			ImGui::Checkbox("IsMaxChild: ", &internodeStatistics->m_isMaxChild);
			ImGui::Checkbox("IsEndNode: ", &internodeStatistics->m_isEndNode);
		}
	);

	EditorManager::RegisterComponentDataInspector<InternodeGrowth>(
		[](Entity entity, ComponentDataBase* data, bool isRoot)
		{
			auto* internodeGrowth = static_cast<InternodeGrowth*>(data);
			ImGui::Text(("Inhibitor: " + std::to_string(internodeGrowth->m_inhibitor)).c_str());
			ImGui::Text(("InhibitorTransmitFactor: " + std::to_string(internodeGrowth->m_inhibitorTransmitFactor)).c_str());
			ImGui::Text(("DistanceToRoot: " + std::to_string(internodeGrowth->m_distanceToRoot)).c_str());
			ImGui::Text(("Thickness: " + std::to_string(internodeGrowth->m_thickness)).c_str());
			ImGui::InputFloat("Gravity sagging", &internodeGrowth->m_sagging, ImGuiInputTextFlags_ReadOnly);
			ImGui::InputFloat("Mass of Children", &internodeGrowth->m_MassOfChildren, ImGuiInputTextFlags_ReadOnly);
			ImGui::InputFloat2("Torque", static_cast<float*>(static_cast<void*>(&internodeGrowth->m_childrenTotalTorque)), "%.3f", ImGuiInputTextFlags_ReadOnly);
			ImGui::InputFloat2("Mean position", static_cast<float*>(static_cast<void*>(&internodeGrowth->m_childMeanPosition)), "%.3f", ImGuiInputTextFlags_ReadOnly);
			glm::vec3 desiredAngles = glm::degrees(glm::eulerAngles(internodeGrowth->m_desiredLocalRotation));
			ImGui::InputFloat3("Desired Rotation##Internode", &desiredAngles.x, "%.3f", ImGuiInputTextFlags_ReadOnly);
			ImGui::InputFloat3("Desired Position##Internode", &internodeGrowth->m_desiredGlobalPosition.x, "%.3f", ImGuiInputTextFlags_ReadOnly);
		}
	);

	EditorManager::RegisterComponentDataInspector<InternodeInfo>(
		[](Entity entity, ComponentDataBase* data, bool isRoot)
		{
			auto* internodeInfo = static_cast<InternodeInfo*>(data);
			ImGui::Checkbox("Activated", &internodeInfo->m_activated);
			ImGui::Text(("StartAge: " + std::to_string(internodeInfo->m_startAge)).c_str());
			ImGui::Text(("StartGlobalTime: " + std::to_string(internodeInfo->m_startGlobalTime)).c_str());
			ImGui::Text(("Order: " + std::to_string(internodeInfo->m_order)).c_str());
			ImGui::Text(("Level: " + std::to_string(internodeInfo->m_level)).c_str());
		}
	);

	EditorManager::RegisterComponentDataInspector<Illumination>(
		[](Entity entity, ComponentDataBase* data, bool isRoot)
		{
			auto* illumination = static_cast<Illumination*>(data);
			ImGui::Text(("CurrentIntensity: " + std::to_string(illumination->m_currentIntensity)).c_str());
			ImGui::InputFloat3("Direction", &illumination->m_accumulatedDirection.x, "%.3f", ImGuiInputTextFlags_ReadOnly);
		}
	);

	EditorManager::RegisterComponentDataInspector<PlantInfo>(
		[](Entity entity, ComponentDataBase* data, bool isRoot)
		{
			auto* info = static_cast<PlantInfo*>(data);
			ImGui::Text(("Start time: " + std::to_string(info->m_startTime)).c_str());
			ImGui::Text(("Age: " + std::to_string(info->m_age)).c_str());
		}
	);
#pragma endregion

#pragma region Cuda and OptiX

	RayMLVQ::CudaModule::Init();
	EditorManager::RegisterPrivateComponentMenu<RayTracerMaterial>([](Entity owner)
		{
			if (owner.HasPrivateComponent<RayTracerMaterial>()) return;
			if (ImGui::SmallButton("RayTracerMaterial"))
			{
				owner.SetPrivateComponent(std::make_unique<RayTracerMaterial>());
			}
		}
	);
	manager.m_rayTracerTestOutput = std::make_unique<OpenGLUtils::GLTexture2D>(0, GL_RGBA32F, 1, 1, false);
	manager.m_rayTracerTestOutput->SetData(0, GL_RGBA32F, GL_RGBA, GL_FLOAT, 0);
	manager.m_rayTracerTestOutput->SetInt(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	manager.m_rayTracerTestOutput->SetInt(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	manager.m_rayTracerTestOutput->SetInt(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	manager.m_rayTracerTestOutput->SetInt(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
#pragma endregion

	manager.m_ready = true;
	manager.m_globalTime = 0;

}

void PlantManager::Update()
{
	auto& manager = GetInstance();
	if (Application::IsPlaying()) {
		if (manager.m_iterationsToGrow > 0) {
			manager.m_iterationsToGrow--;
			if (GrowAllPlants())
			{
				manager.m_endUpdate = true;
			}
		}
		else if (manager.m_endUpdate) {
			Refresh();
			manager.m_endUpdate = false;
		}
	}
}

void PlantManager::Refresh()
{
	auto& manager = GetInstance();
	manager.m_plants.resize(0);
	manager.m_plantQuery.ToEntityArray(manager.m_plants);
	manager.m_internodes.resize(0);
	manager.m_internodeTransforms.resize(0);
	manager.m_internodeQuery.ToComponentDataArray(manager.m_internodeTransforms);
	manager.m_internodeQuery.ToEntityArray(manager.m_internodes);
	float time = Application::EngineTime();
	for (auto& i : manager.m_plantFoliageGenerators)
	{
		i.second(manager);
	}
	manager.m_foliageGenerationTimer = Application::EngineTime() - time;

	time = Application::EngineTime();
	for (auto& i : manager.m_plantMeshGenerators)
	{
		i.second(manager);
	}
	manager.m_meshGenerationTimer = Application::EngineTime() - time;
	CalculateIlluminationForInternodes(manager);
}

void PlantManager::End()
{
	RayMLVQ::CudaModule::Terminate();
}

#pragma endregion
