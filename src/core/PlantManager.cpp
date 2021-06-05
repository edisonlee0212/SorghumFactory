#include <PlantManager.hpp>
#include <CubeVolume.hpp>
#include <CUDAModule.hpp>
#include <Utilities.hpp>
#include <Volume.hpp>
#include <SorghumManager.hpp>
#include <TreeManager.hpp>
#include <RayTracerManager.hpp>
#include <RayTracedRenderer.hpp>
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
	std::vector<InternodeCandidate> candidates;
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

bool PlantManager::GrowCandidates(std::vector<InternodeCandidate>& candidates)
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
	RayMLVQ::RayTracerManager::GetInstance().UpdateScene();
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
	manager.m_ground.SetPrivateComponent(std::make_unique<RayMLVQ::RayTracedRenderer>());
	manager.m_ground.GetPrivateComponent<RayMLVQ::RayTracedRenderer>()->SyncWithMeshRenderer();
	
	auto cubeVolume = std::make_unique<CubeVolume>();
	cubeVolume->m_asObstacle = true;
	cubeVolume->m_minMaxBound.m_max = glm::vec3(1000, -0.1f, 1000);
	cubeVolume->m_minMaxBound.m_min = glm::vec3(-1000, -10.0f, -1000);
	manager.m_ground.SetPrivateComponent(std::move(cubeVolume));
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
}

#pragma endregion
