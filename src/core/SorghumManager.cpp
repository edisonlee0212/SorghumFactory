#include <SorghumManager.hpp>
#include <PlantManager.hpp>
#include <SorghumData.hpp>
#include <TriangleIlluminationEstimator.hpp>

using namespace PlantFactory;
using namespace RayMLVQ;
PlantNode::PlantNode(glm::vec3 position, float angle, float width, glm::vec3 axis, bool isLeaf)
{
	m_position = position;
	m_theta = angle;
	m_width = width;
	m_axis = axis;
	m_isLeaf = isLeaf;
}

void Spline::Import(std::ifstream& stream)
{
	int curveAmount;
	stream >> curveAmount;
	m_curves.clear();
	for (int i = 0; i < curveAmount; i++) {
		glm::vec3 cp[4];
		float x, y, z;
		for (auto& j : cp)
		{
			stream >> x >> z >> y;
			j = glm::vec3(x, y, z) * 10.0f;
		}
		m_curves.emplace_back(cp[0], cp[1], cp[2], cp[3]);
	}
}

glm::vec3 Spline::EvaluatePointFromCurve(float point)
{
	const float splineU = glm::clamp(point, 0.0f, 1.0f) * float(m_curves.size());

	// Decompose the global u coordinate on the spline
	float integerPart;
	const float fractionalPart = modff(splineU, &integerPart);

	auto curveIndex = int(integerPart);
	auto curveU = fractionalPart;

	// If evaluating the very last point on the spline
	if (curveIndex == m_curves.size() && curveU <= 0.0f)
	{
		// Flip to the end of the last patch
		curveIndex--;
		curveU = 1.0f;
	}
	return m_curves.at(curveIndex).GetPoint(curveU);
}

glm::vec3 Spline::EvaluateAxisFromCurve(float point)
{
	const float splineU = glm::clamp(point, 0.0f, 1.0f) * float(m_curves.size());

	// Decompose the global u coordinate on the spline
	float integerPart;
	const float fractionalPart = modff(splineU, &integerPart);

	auto curveIndex = int(integerPart);
	auto curveU = fractionalPart;

	// If evaluating the very last point on the spline
	if (curveIndex == m_curves.size() && curveU <= 0.0f)
	{
		// Flip to the end of the last patch
		curveIndex--;
		curveU = 1.0f;
	}
	return m_curves.at(curveIndex).GetAxis(curveU);
}

void Spline::OnGui()
{
	if (ImGui::TreeNodeEx("Curves", ImGuiTreeNodeFlags_DefaultOpen))
	{
		for (int i = 0; i < m_curves.size(); i++)
		{
			ImGui::Text(("Curve" + std::to_string(i)).c_str());
			ImGui::InputFloat3("CP0", &m_curves[i].m_p0.x);
			ImGui::InputFloat3("CP1", &m_curves[i].m_p1.x);
			ImGui::InputFloat3("CP2", &m_curves[i].m_p2.x);
			ImGui::InputFloat3("CP3", &m_curves[i].m_p3.x);
		}
		ImGui::TreePop();
	}
}

void RectangularSorghumField::GenerateField(std::vector<std::vector<glm::mat4>>& matricesList)
{
	const int size = matricesList.size();
	glm::vec2 center = glm::vec2(m_distances.x * (m_size.x - 1), m_distances.y * (m_size.y - 1)) / 2.0f;
	for (int xi = 0; xi < m_size.x; xi++)
	{
		for (int yi = 0; yi < m_size.y; yi++)
		{
			const auto selectedIndex = glm::linearRand(0, size - 1);
			matricesList[selectedIndex].push_back(glm::translate(glm::vec3(xi * m_distances.x - center.x, 0.0f, yi * m_distances.y - center.y)) * glm::scale(glm::vec3(1.0f)));
		}
	}
}

SorghumManager& SorghumManager::GetInstance()
{
	static SorghumManager instance;
	return instance;
}

void SorghumManager::Init()
{
	auto& sorghumManager = GetInstance();
	sorghumManager.m_leafNodeMaterial = std::make_shared<Material>();
	sorghumManager.m_leafNodeMaterial->SetProgram(Default::GLPrograms::StandardProgram);
	sorghumManager.m_leafNodeMaterial->m_albedoColor = glm::vec3(0, 1, 0);

	sorghumManager.m_leafArchetype = EntityManager::CreateEntityArchetype("Leaf",
		LeafInfo()
	);
	sorghumManager.m_leafQuery = EntityManager::CreateEntityQuery();
	EntityManager::SetEntityQueryAllFilters(sorghumManager.m_leafQuery, LeafInfo());

	sorghumManager.m_leafMaterial = std::make_shared<Material>();
	sorghumManager.m_leafMaterial->SetProgram(Default::GLPrograms::StandardProgram);
	sorghumManager.m_leafMaterial->m_cullingMode = MaterialCullingMode::Off;
	const auto textureLeaf = ResourceManager::LoadTexture(false, FileIO::GetAssetFolderPath() + "Textures/leafSurfaceBright.jpg");
	sorghumManager.m_leafSurfaceTexture = ResourceManager::LoadTexture(false, FileIO::GetAssetFolderPath() + "Textures/leafSurfaceBright.jpg");
	sorghumManager.m_leafMaterial->SetTexture(textureLeaf);
	sorghumManager.m_leafMaterial->m_roughness = 0.0f;
	sorghumManager.m_leafMaterial->m_metallic = 0.0f;

	sorghumManager.m_instancedLeafMaterial = std::make_shared<Material>();
	sorghumManager.m_instancedLeafMaterial->m_cullingMode = MaterialCullingMode::Off;
	sorghumManager.m_instancedLeafMaterial->SetProgram(Default::GLPrograms::StandardInstancedProgram);
	sorghumManager.m_instancedLeafMaterial->SetTexture(textureLeaf);
	sorghumManager.m_instancedLeafMaterial->m_roughness = 0.0f;
	sorghumManager.m_instancedLeafMaterial->m_metallic = 0.0f;

	PlantManager::GetInstance().m_plantGrowthModels.insert_or_assign(PlantType::Sorghum, [](PlantManager& manager, std::vector<InternodeCandidate>& candidates)
		{
			FormCandidates(manager, candidates);
		}
	);

	PlantManager::GetInstance().m_plantFoliageGenerators.insert_or_assign(PlantType::Sorghum, [](PlantManager& manager)
		{
			GenerateLeavesForSorghum(manager);
		}
	);

	PlantManager::GetInstance().m_plantMetaDataCalculators.insert_or_assign(PlantType::Sorghum, [](PlantManager& manager)
		{
			FormLeafNodes(manager);
		}
	);
}

Entity SorghumManager::CreateSorghum()
{
	Transform transform;
	transform.SetScale(glm::vec3(1.0f));
	const Entity entity = PlantManager::CreatePlant(PlantType::Sorghum, transform);
	EntityManager::ForEachChild(entity, [](Entity child)
		{
			if (child.HasComponentData<InternodeInfo>())
			{
				auto internodeTransform = child.GetComponentData<Transform>();
				internodeTransform.SetScale(glm::vec3(GetInstance().m_leafNodeSphereSize));
				child.SetComponentData(internodeTransform);
			}
		}
	);
	EntityManager::SetPrivateComponent(entity, std::make_unique<Spline>());
	EntityManager::SetPrivateComponent(entity, std::make_unique<SorghumData>());
	entity.SetName("Sorghum");
	EntityManager::SetPrivateComponent(entity, std::make_unique<TriangleIlluminationEstimator>());
	return entity;
}

Entity SorghumManager::CreateSorghumLeaf(const Entity& plantEntity)
{
	const Entity entity = EntityManager::CreateEntity(GetInstance().m_leafArchetype);
	entity.SetName("Leaf");
	EntityManager::SetParent(entity, plantEntity);
	Transform transform;
	transform.SetScale(glm::vec3(1.0f));
	auto spline = std::make_unique<Spline>();
	EntityManager::SetPrivateComponent(entity, std::move(spline));
	EntityManager::SetComponentData(entity, transform);

	auto mmc = std::make_unique<MeshRenderer>();
	mmc->m_material = GetInstance().m_leafMaterial;
	mmc->m_mesh = std::make_shared<Mesh>();
	EntityManager::SetPrivateComponent(entity, std::move(mmc));

	auto rtt = std::make_unique<RayTracerMaterial>();
	rtt->m_albedoTexture = GetInstance().m_leafSurfaceTexture;
	if (GetInstance().m_leafNormalTexture) rtt->m_normalTexture = GetInstance().m_leafNormalTexture;
	entity.SetPrivateComponent(std::move(rtt));
	return entity;
}

void SorghumManager::GenerateMeshForAllSorghums(int segmentAmount, int step)
{
	std::mutex meshMutex;
	EntityManager::ForEach<GlobalTransform>(JobManager::PrimaryWorkers(), GetInstance().m_leafQuery, [&meshMutex, segmentAmount, step]
	(int index, Entity entity, GlobalTransform& ltw)
		{
			auto& spline = EntityManager::GetPrivateComponent<Spline>(entity);
			spline->m_nodes.clear();
			int stemNodeCount = 0;
			if (spline->m_startingPoint != -1) {
				auto& truckSpline = EntityManager::GetPrivateComponent<Spline>(EntityManager::GetParent(entity));
				float width = 0.1f - spline->m_startingPoint * 0.05f;
				for (float i = 0.0f; i < spline->m_startingPoint - 0.05f; i += 0.05f)
				{
					spline->m_nodes.emplace_back(truckSpline->EvaluatePointFromCurve(i), 180.0f, width, truckSpline->EvaluateAxisFromCurve(i), false);

				}
				stemNodeCount = spline->m_nodes.size();
				for (float i = 0.05f; i <= 1.0f; i += 0.05f)
				{
					float w = 0.2f;
					if (i > 0.75f) w -= (i - 0.75f) * 0.75f;
					spline->m_nodes.emplace_back(spline->EvaluatePointFromCurve(i), i == 0.05f ? 60.0f : 10.0f, w, spline->EvaluateAxisFromCurve(i), true);
				}
			}
			else
			{
				for (float i = 0.0f; i <= 1.0f; i += 0.05f)
				{
					spline->m_nodes.emplace_back(spline->EvaluatePointFromCurve(i), 180.0f, 0.04f, spline->EvaluateAxisFromCurve(i), false);
				}
				auto endPoint = spline->EvaluatePointFromCurve(1.0f);
				auto endAxis = spline->EvaluateAxisFromCurve(1.0f);
				spline->m_nodes.emplace_back(endPoint + endAxis * 0.05f, 10.0f, 0.001f, endAxis, false);
				stemNodeCount = spline->m_nodes.size();
			}
			spline->m_vertices.clear();
			spline->m_indices.clear();
			spline->m_segments.clear();


			float temp = 0.0f;

			float leftPeriod = 0.0f;
			float rightPeriod = 0.0f;
			float leftFlatness = glm::gaussRand(1.75f, 0.5f);//glm::linearRand(0.5f, 2.0f);
			float rightFlatness = glm::gaussRand(1.75f, 0.5f);//glm::linearRand(0.5f, 2.0f);
			float leftFlatnessFactor = glm::gaussRand(1.25f, 0.2f);//glm::linearRand(1.0f, 2.5f);
			float rightFlatnessFactor = glm::gaussRand(1.25f, 0.2f);//glm::linearRand(1.0f, 2.5f);

			int stemSegmentCount = 0;
			for (int i = 1; i < spline->m_nodes.size(); i++)
			{
				auto& prev = spline->m_nodes.at(i - 1);
				auto& curr = spline->m_nodes.at(i);
				if (i == stemNodeCount)
				{
					stemSegmentCount = spline->m_segments.size();
				}
				float distance = glm::distance(prev.m_position, curr.m_position);
				BezierCurve curve = BezierCurve(prev.m_position, prev.m_position + distance / 5.0f * prev.m_axis, curr.m_position - distance / 5.0f * curr.m_axis, curr.m_position);
				for (float div = 1.0f / segmentAmount; div <= 1.0f; div += 1.0f / segmentAmount)
				{
					auto front = prev.m_axis * (1.0f - div) + curr.m_axis * div;

					auto up = glm::normalize(glm::cross(spline->m_left, front));
					if (prev.m_isLeaf) {
						leftPeriod += glm::gaussRand(1.25f, 0.5f) / segmentAmount;
						rightPeriod += glm::gaussRand(1.25f, 0.5f) / segmentAmount;
						spline->m_segments.emplace_back(
							curve.GetPoint(div),
							up,
							front,
							prev.m_width * (1.0f - div) + curr.m_width * div,
							prev.m_theta * (1.0f - div) + curr.m_theta * div,
							curr.m_isLeaf,
							glm::sin(leftPeriod) * leftFlatness,
							glm::sin(rightPeriod) * rightFlatness,
							leftFlatnessFactor, rightFlatnessFactor);
					}
					else
					{
						spline->m_segments.emplace_back(
							curve.GetPoint(div),
							up,
							front,
							prev.m_width * (1.0f - div) + curr.m_width * div,
							prev.m_theta * (1.0f - div) + curr.m_theta * div,
							curr.m_isLeaf);
					}
				}
			}

			const int vertexIndex = spline->m_vertices.size();
			Vertex archetype;
			const float xStep = 1.0f / step / 2.0f;
			const float yStemStep = 0.5f / static_cast<float>(stemSegmentCount);
			const float yLeafStep = 0.5f / (spline->m_segments.size() - static_cast<float>(stemSegmentCount) + 1);
			for (int i = 0; i < spline->m_segments.size(); i++)
			{
				auto& segment = spline->m_segments.at(i);
				const float angleStep = segment.m_theta / step;
				const int vertsCount = step * 2 + 1;
				for (int j = 0; j < vertsCount; j++)
				{
					const auto position = segment.GetPoint((j - step) * angleStep);
					archetype.m_position = glm::vec3(position.x, position.y, position.z);
					float yPos = (i < stemSegmentCount) ? yStemStep * i : 0.5f + yLeafStep * (i - stemSegmentCount + 1);
					archetype.m_texCoords0 = glm::vec2(j * xStep, yPos);
					spline->m_vertices.push_back(archetype);
				}
				if (i != 0) {
					for (int j = 0; j < vertsCount - 1; j++) {
						//Down triangle
						spline->m_indices.push_back(vertexIndex + ((i - 1) + 1) * vertsCount + j);
						spline->m_indices.push_back(vertexIndex + (i - 1) * vertsCount + j + 1);
						spline->m_indices.push_back(vertexIndex + (i - 1) * vertsCount + j);
						//Up triangle
						spline->m_indices.push_back(vertexIndex + (i - 1) * vertsCount + j + 1);
						spline->m_indices.push_back(vertexIndex + ((i - 1) + 1) * vertsCount + j);
						spline->m_indices.push_back(vertexIndex + ((i - 1) + 1) * vertsCount + j + 1);
					}
				}
			}

		}
	);
	std::vector<Entity> plants;
	PlantManager::GetInstance().m_plantQuery.ToEntityArray(plants);
	for (auto& plant : plants) {
		if (plant.GetComponentData<PlantInfo>().m_plantType != PlantType::Sorghum) continue;
		EntityManager::ForEachChild(plant, [](Entity child)
			{
				if (!child.HasPrivateComponent<Spline>()) return;
				auto& meshRenderer = EntityManager::GetPrivateComponent<MeshRenderer>(child);
				auto& spline = EntityManager::GetPrivateComponent<Spline>(child);
				meshRenderer->m_mesh->SetVertices(17, spline->m_vertices, spline->m_indices, true);
			}
		);
		if (plant.HasPrivateComponent<SorghumData>())plant.GetPrivateComponent<SorghumData>()->m_meshGenerated = true;
	}
}

Entity SorghumManager::ImportPlant(const std::string& path, const std::string& name)
{
	std::ifstream file(path, std::fstream::in);
	if (!file.is_open())
	{
		Debug::Log("Failed to open file!");
		return Entity();
	}
	// Number of leaves in the file
	int leafCount;
	file >> leafCount;
	const auto sorghum = CreateSorghum();
	sorghum.RemovePrivateComponent<SorghumData>();
	auto children = EntityManager::GetChildren(sorghum);
	for (const auto& child : children)
	{
		EntityManager::DeleteEntity(child);
	}
	sorghum.SetName(name);
	auto& truckSpline = EntityManager::GetPrivateComponent<Spline>(sorghum);
	truckSpline->m_startingPoint = -1;
	truckSpline->Import(file);

	//Recenter plant:
	glm::vec3 posSum = truckSpline->m_curves.front().m_p0;

	for (auto& curve : truckSpline->m_curves) {
		curve.m_p0 -= posSum;
		curve.m_p1 -= posSum;
		curve.m_p2 -= posSum;
		curve.m_p3 -= posSum;
	}
	truckSpline->m_left = glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), truckSpline->m_curves.begin()->m_p0 - truckSpline->m_curves.back().m_p3);
	for (int i = 0; i < leafCount; i++) {
		Entity leaf = CreateSorghumLeaf(sorghum);
		auto& leafSpline = EntityManager::GetPrivateComponent<Spline>(leaf);
		float startingPoint;
		file >> startingPoint;

		leafSpline->m_startingPoint = startingPoint;
		leafSpline->Import(file);
		for (auto& curve : leafSpline->m_curves) {
			curve.m_p0 += truckSpline->EvaluatePointFromCurve(startingPoint);
			curve.m_p1 += truckSpline->EvaluatePointFromCurve(startingPoint);
			curve.m_p2 += truckSpline->EvaluatePointFromCurve(startingPoint);
			curve.m_p3 += truckSpline->EvaluatePointFromCurve(startingPoint);
		}

		leafSpline->m_left = glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), leafSpline->m_curves.begin()->m_p0 - leafSpline->m_curves.back().m_p3);
	}
	return sorghum;
}


void SorghumManager::OnGui()
{
	auto& manager = GetInstance();
	if (ImGui::BeginMainMenuBar())
	{
		if (ImGui::BeginMenu("Sorghum Manager"))
		{
			ImGui::Checkbox("Display light probes", &GetInstance().m_displayLightProbes);
			ImGui::DragInt("Seed", &manager.m_seed);
			if (ImGui::Button("Calculate illumination"))
			{
				RayMLVQ::IlluminationEstimationProperties properties;
				properties.m_skylightPower = 1.0f;
				properties.m_bounceLimit = 2;
				properties.m_seed = glm::abs(manager.m_seed);
				properties.m_numPointSamples = 100;
				properties.m_numScatterSamples = 10;
				CalculateIllumination(properties);
			}
			if (ImGui::Button("Create...")) {
				ImGui::OpenPopup("New sorghum wizard");
			}
			if (ImGui::BeginPopupModal("New sorghum wizard", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
			{
				static std::vector<SorghumParameters> newSorghumParameters;
				static std::vector<glm::vec3> newSorghumPositions;
				static std::vector<glm::vec3> newSorghumRotations;
				static int newSorghumAmount = 1;
				static int currentFocusedNewSorghumIndex = 0;
				ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
				ImGui::BeginChild("ChildL", ImVec2(300, 400), true, ImGuiWindowFlags_None | ImGuiWindowFlags_MenuBar);
				if (ImGui::BeginMenuBar())
				{
					if (ImGui::BeginMenu("Settings"))
					{
						static float distance = 10;
						static float variance = 4;
						static float yAxisVar = 180.0f;
						static float xzAxisVar = 0.0f;
						static int expand = 1;
						if (ImGui::BeginMenu("Create forest...")) {
							ImGui::DragFloat("Avg. Y axis rotation", &yAxisVar, 0.01f, 0.0f, 180.0f);
							ImGui::DragFloat("Avg. XZ axis rotation", &xzAxisVar, 0.01f, 0.0f, 90.0f);
							ImGui::DragFloat("Avg. Distance", &distance, 0.01f);
							ImGui::DragFloat("Position variance", &variance, 0.01f);
							ImGui::DragInt("Expand", &expand, 1, 0, 3);
							if (ImGui::Button("Apply"))
							{
								newSorghumAmount = (2 * expand + 1) * (2 * expand + 1);
								newSorghumPositions.resize(newSorghumAmount);
								newSorghumRotations.resize(newSorghumAmount);
								const auto currentSize = newSorghumParameters.size();
								newSorghumParameters.resize(newSorghumAmount);
								for (auto i = currentSize; i < newSorghumAmount; i++)
								{
									newSorghumParameters[i] = newSorghumParameters[0];
								}
								int index = 0;
								for (int i = -expand; i <= expand; i++)
								{
									for (int j = -expand; j <= expand; j++)
									{
										glm::vec3 value = glm::vec3(i * distance, 0, j * distance);
										value.x += glm::linearRand(-variance, variance);
										value.z += glm::linearRand(-variance, variance);
										newSorghumPositions[index] = value;
										value = glm::vec3(glm::linearRand(-xzAxisVar, xzAxisVar), glm::linearRand(-yAxisVar, yAxisVar), glm::linearRand(-xzAxisVar, xzAxisVar));
										newSorghumRotations[index] = value;
										index++;
									}
								}
							}
							ImGui::EndMenu();
						}
						ImGui::InputInt("New sorghum amount", &newSorghumAmount);
						if (newSorghumAmount < 1) newSorghumAmount = 1;
						FileIO::OpenFile("Import parameters for all", ".sorghumparam", [](const std::string& path)
							{
								newSorghumParameters[0].Deserialize(path);
								for (int i = 1; i < newSorghumParameters.size(); i++) newSorghumParameters[i] = newSorghumParameters[0];
							}
						);
						ImGui::EndMenu();
					}
					ImGui::EndMenuBar();
				}
				ImGui::Columns(1);
				if (newSorghumPositions.size() < newSorghumAmount) {
					const auto currentSize = newSorghumPositions.size();
					newSorghumParameters.resize(newSorghumAmount);
					for (auto i = currentSize; i < newSorghumAmount; i++)
					{
						newSorghumParameters[i] = newSorghumParameters[0];
					}
					newSorghumPositions.resize(newSorghumAmount);
					newSorghumRotations.resize(newSorghumAmount);
				}
				for (auto i = 0; i < newSorghumAmount; i++)
				{
					std::string title = "New Sorghum No.";
					title += std::to_string(i);
					const bool opened = ImGui::TreeNodeEx(title.c_str(), ImGuiTreeNodeFlags_NoTreePushOnOpen | ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_NoAutoOpenOnLog | (currentFocusedNewSorghumIndex == i ? ImGuiTreeNodeFlags_Framed : ImGuiTreeNodeFlags_FramePadding));
					if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
						currentFocusedNewSorghumIndex = i;
					}
					if (opened) {
						ImGui::TreePush();
						ImGui::InputFloat3(("Position##" + std::to_string(i)).c_str(), &newSorghumPositions[i].x);
						ImGui::TreePop();
					}
				}

				ImGui::EndChild();
				ImGui::PopStyleVar();
				ImGui::SameLine();
				ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
				ImGui::BeginChild("ChildR", ImVec2(400, 400), true, ImGuiWindowFlags_None | ImGuiWindowFlags_MenuBar);
				if (ImGui::BeginMenuBar())
				{
					if (ImGui::BeginMenu("Parameters")) {
						FileIO::OpenFile("Import parameters", ".treeparam", [](const std::string& path)
							{
								newSorghumParameters[currentFocusedNewSorghumIndex].Deserialize(path);
							}
						);

						FileIO::SaveFile("Export parameters", ".treeparam", [](const std::string& path)
							{
								newSorghumParameters[currentFocusedNewSorghumIndex].Serialize(path);
							}
						);
						ImGui::EndMenu();
					}
					ImGui::EndMenuBar();
				}
				ImGui::Columns(1);
				ImGui::PushItemWidth(200);
				newSorghumParameters[currentFocusedNewSorghumIndex].OnGui();
				ImGui::PopItemWidth();
				ImGui::EndChild();
				ImGui::PopStyleVar();
				ImGui::Separator();
				if (ImGui::Button("OK", ImVec2(120, 0))) {
					//Create tree here.
					for (auto i = 0; i < newSorghumAmount; i++) {
						Entity sorghum = CreateSorghum();
						auto sorghumTransform = sorghum.GetComponentData<Transform>();
						sorghumTransform.SetPosition(newSorghumPositions[i]);
						sorghumTransform.SetEulerRotation(glm::radians(newSorghumRotations[i]));
						sorghum.SetComponentData(sorghumTransform);
						sorghum.GetPrivateComponent<SorghumData>()->m_parameters = newSorghumParameters[i];
					}
					ImGui::CloseCurrentPopup();
				}
				ImGui::SetItemDefaultFocus();
				ImGui::SameLine();
				if (ImGui::Button("Cancel", ImVec2(120, 0))) { ImGui::CloseCurrentPopup(); }
				ImGui::EndPopup();
			}
			if (ImGui::Button("Create field...")) {
				ImGui::OpenPopup("Sorghum field wizard");
			}
			const ImVec2 center = ImGui::GetMainViewport()->GetCenter();
			ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
			if (ImGui::BeginPopupModal("Sorghum field wizard", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
			{
				static RectangularSorghumField field;
				ImGui::DragInt2("Size", &field.m_size.x, 1, 1, 10);
				ImGui::DragFloat2("Distance", &field.m_distances.x, 0.1f, 0.0f, 10.0f);
				if (ImGui::Button("OK", ImVec2(120, 0))) {
					std::vector<Entity> candidates;
					candidates.push_back(ImportPlant(FileIO::GetAssetFolderPath() + "Sorghum/skeleton_procedural_1.txt", "Sorghum 1"));
					candidates.push_back(ImportPlant(FileIO::GetAssetFolderPath() + "Sorghum/skeleton_procedural_2.txt", "Sorghum 2"));
					candidates.push_back(ImportPlant(FileIO::GetAssetFolderPath() + "Sorghum/skeleton_procedural_3.txt", "Sorghum 3"));
					candidates.push_back(ImportPlant(FileIO::GetAssetFolderPath() + "Sorghum/skeleton_procedural_4.txt", "Sorghum 4"));
					GenerateMeshForAllSorghums();

					CreateGrid(field, candidates);
					for (auto& i : candidates) EntityManager::DeleteEntity(i);
					ImGui::CloseCurrentPopup();
				}
				ImGui::SetItemDefaultFocus();
				ImGui::SameLine();
				if (ImGui::Button("Cancel", ImVec2(120, 0))) { ImGui::CloseCurrentPopup(); }
				ImGui::EndPopup();
			}
			if (ImGui::Button("Lock structure"))
			{
				std::vector<Entity> sorghums;
				PlantManager::GetInstance().m_plantQuery.ToEntityArray<PlantInfo>(sorghums, [](const Entity& plant, const PlantInfo& plantInfo)
					{
						return plantInfo.m_plantType == PlantType::Sorghum;
					}
				);
				for (const auto& sorghum : sorghums)
				{
					if (!sorghum.HasPrivateComponent<SorghumData>()) continue;
					sorghum.RemovePrivateComponent<SorghumData>();
					Entity rootInternode;
					EntityManager::ForEachChild(sorghum, [&](Entity child)
						{
							if (child.HasComponentData<InternodeInfo>()) rootInternode = child;
						}
					);
					if (rootInternode.IsValid()) EntityManager::DeleteEntity(rootInternode);
				}
			}
			FileIO::SaveFile("Export OBJ for all sorghums", ".obj", [](const std::string& path)
				{
					ExportAllSorghumsModel(path);
				}
			);
			ImGui::EndMenu();
		}
		static bool opened = false;
		if (manager.m_processing && !opened)
		{
			ImGui::OpenPopup("Illumination Estimation");
			opened = true;
		}
		if (ImGui::BeginPopupModal("Illumination Estimation", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
		{
			ImGui::Text("Progress: ");
			float fraction = 1.0f - static_cast<float>(manager.m_processingIndex) / manager.m_processingEntities.size();
			std::string text = std::to_string(static_cast<int>(fraction * 100.0f)) + "% - " + std::to_string(manager.m_processingEntities.size() - manager.m_processingIndex) + "/" + std::to_string(manager.m_processingEntities.size());
			ImGui::ProgressBar(fraction, ImVec2(240, 0), text.c_str());
			ImGui::SetItemDefaultFocus();
			ImGui::Text(("Estimation time for 1 plant: " + std::to_string(manager.m_perPlantCalculationTime) + " seconds").c_str());
			if (ImGui::Button("Cancel") || manager.m_processing == false)
			{
				manager.m_processing = false;
				opened = false;
				ImGui::CloseCurrentPopup();
			}
			ImGui::EndPopup();
		}

		ImGui::EndMainMenuBar();
	}
}

void SorghumManager::CloneSorghums(const Entity& parent, const Entity& original, std::vector<glm::mat4>& matrices)
{
	for (const auto& matrix : matrices)
	{
		Entity sorghum = CreateSorghum();
		sorghum.RemovePrivateComponent<SorghumData>();
		auto children = EntityManager::GetChildren(sorghum);
		for (const auto& child : children)
		{
			EntityManager::DeleteEntity(child);
		}
		Transform transform;
		transform.m_value = matrix;
		sorghum.SetComponentData(transform);
		auto newSpline = std::make_unique<Spline>();
		auto& spline = EntityManager::GetPrivateComponent<Spline>(original);
		*newSpline = *spline;
		sorghum.SetPrivateComponent(std::move(newSpline));
		EntityManager::ForEachChild(original, [&sorghum, &matrices](Entity child)
			{
				if (!child.HasComponentData<LeafInfo>()) return;
				const auto newChild = CreateSorghumLeaf(sorghum);
				newChild.SetComponentData(EntityManager::GetComponentData<LeafInfo>(child));
				newChild.SetComponentData(EntityManager::GetComponentData<Transform>(child));
				auto newSpline = std::make_unique<Spline>();
				auto& spline = EntityManager::GetPrivateComponent<Spline>(child);
				*newSpline = *spline;
				newChild.SetPrivateComponent(std::move(newSpline));
				auto& newMeshRenderer = newChild.GetPrivateComponent<MeshRenderer>();
				auto& meshRenderer = EntityManager::GetPrivateComponent<MeshRenderer>(child);
				newMeshRenderer->m_mesh = meshRenderer->m_mesh;
			});
		EntityManager::SetParent(sorghum, parent, false);
	}
}

void SorghumManager::ExportSorghum(const Entity& sorghum, std::ofstream& of, unsigned& startIndex)
{
	const std::string start = "#Sorghum\n";
	of.write(start.c_str(), start.size());
	of.flush();
	const auto position = EntityManager::GetComponentData<GlobalTransform>(sorghum).GetPosition();
	EntityManager::ForEachChild(sorghum, [&](Entity child)
		{
			if (!EntityManager::HasPrivateComponent<MeshRenderer>(child)) return;
			const auto& leafMesh = EntityManager::GetPrivateComponent<MeshRenderer>(child)->m_mesh;
			ObjExportHelper(position, leafMesh, of, startIndex);
		}
	);
}

void SorghumManager::ObjExportHelper(glm::vec3 position, std::shared_ptr<Mesh> mesh,
	std::ofstream& of, unsigned& startIndex)
{
	if (!mesh->UnsafeGetVertices().empty() && !mesh->UnsafeGetTriangles().empty())
	{
		std::string header = "#Vertices: " + std::to_string(mesh->UnsafeGetVertices().size()) + ", tris: " + std::to_string(mesh->UnsafeGetTriangles().size() / 3);
		header += "\n";
		of.write(header.c_str(), header.size());
		of.flush();
		std::string o = "o ";
		o +=
			"["
			+ std::to_string(position.x) + ","
			+ std::to_string(position.z)
			+ "]" + "\n";
		of.write(o.c_str(), o.size());
		of.flush();
		std::string data;
#pragma region Data collection

		for (const auto& vertex : mesh->UnsafeGetVertices()) {
			data += "v " + std::to_string(vertex.m_position.x + position.x)
				+ " " + std::to_string(vertex.m_position.y + position.y)
				+ " " + std::to_string(vertex.m_position.z + position.z)
				+ " " + std::to_string(vertex.m_color.x)
				+ " " + std::to_string(vertex.m_color.y)
				+ " " + std::to_string(vertex.m_color.z)
				+ "\n";
		}
		for (const auto& vertex : mesh->UnsafeGetVertices()) {
			data += "vn " + std::to_string(vertex.m_normal.x)
				+ " " + std::to_string(vertex.m_normal.y)
				+ " " + std::to_string(vertex.m_normal.z)
				+ "\n";
		}

		for (const auto& vertex : mesh->UnsafeGetVertices()) {
			data += "vt " + std::to_string(vertex.m_texCoords0.x)
				+ " " + std::to_string(vertex.m_texCoords0.y)
				+ "\n";
		}
		//data += "s off\n";
		data += "# List of indices for faces vertices, with (x, y, z).\n";
		auto& triangles = mesh->UnsafeGetTriangles();
		for (auto i = 0; i < mesh->GetTriangleAmount(); i++) {
			const auto triangle = triangles[i];
			const auto f1 = triangle.x + startIndex;
			const auto f2 = triangle.y + startIndex;
			const auto f3 = triangle.z + startIndex;
			data += "f " + std::to_string(f1) + "/" + std::to_string(f1) + "/" + std::to_string(f1)
				+ " " + std::to_string(f2) + "/" + std::to_string(f2) + "/" + std::to_string(f2)
				+ " " + std::to_string(f3) + "/" + std::to_string(f3) + "/" + std::to_string(f3)
				+ "\n";
		}
		startIndex += mesh->UnsafeGetVertices().size();
#pragma endregion
		of.write(data.c_str(), data.size());
		of.flush();
	}
}

void SorghumManager::ExportAllSorghumsModel(const std::string& filename)
{
	std::ofstream of;
	of.open(filename, std::ofstream::out | std::ofstream::trunc);
	if (of.is_open())
	{
		std::string start = "#Sorghum field, by Bosheng Li";
		start += "\n";
		of.write(start.c_str(), start.size());
		of.flush();

		unsigned startIndex = 1;
		std::vector<Entity> sorghums;
		PlantManager::GetInstance().m_plantQuery.ToEntityArray<PlantInfo>(sorghums, [](const Entity& plant, const PlantInfo& plantInfo)
			{
				return plantInfo.m_plantType == PlantType::Sorghum;
			}
		);
		
		for (const auto& plant : sorghums) {
			ExportSorghum(plant, of, startIndex);
		}
		of.close();
		Debug::Log("Sorghums saved as " + filename);
	}
	else
	{
		Debug::Error("Can't open file!");
	}
}

void SorghumManager::RenderLightProbes()
{
	auto& manager = GetInstance();
	if (manager.m_probeTransforms.empty() || manager.m_probeColors.empty() || manager.m_probeTransforms.size() != manager.m_probeColors.size()) return;
	RenderManager::DrawGizmoMeshInstancedColored(Default::Primitives::Cube.get(), EditorManager::GetSceneCamera().get(), manager.m_probeColors.data(), manager.m_probeTransforms.data(), manager.m_probeTransforms.size(), glm::mat4(1.0f), 0.2f);
	/*
	if (!EditorManager::GetSceneCamera()->IsEnabled()) return;
#pragma region Render
	CameraComponent::m_cameraInfoBlock.UpdateMatrices(EditorManager::GetSceneCamera().get(),
		EditorManager::GetInstance().m_sceneCameraPosition,
		EditorManager::GetInstance().m_sceneCameraRotation
	);
	CameraComponent::m_cameraInfoBlock.UploadMatrices(EditorManager::GetSceneCamera().get());
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glDrawBuffer(GL_COLOR_ATTACHMENT0);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	EditorManager::GetSceneCamera()->Bind();
	auto mesh = Default::Primitives::Cube;
	mesh->Enable();
	const size_t count = manager.m_probeColors.size();
	const auto vao = mesh->Vao();
	manager.m_lightProbeRenderingColorBuffer.Bind();
	vao->EnableAttributeArray(11);
	vao->SetAttributePointer(11, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);
	vao->SetAttributeDivisor(11, 1);

	manager.m_lightProbeRenderingTransformBuffer.Bind();
	vao->EnableAttributeArray(12);
	vao->SetAttributePointer(12, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)0);
	vao->EnableAttributeArray(13);
	vao->SetAttributePointer(13, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(sizeof(glm::vec4)));
	vao->EnableAttributeArray(14);
	vao->SetAttributePointer(14, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(2 * sizeof(glm::vec4)));
	vao->EnableAttributeArray(15);
	vao->SetAttributePointer(15, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(3 * sizeof(glm::vec4)));
	vao->SetAttributeDivisor(12, 1);
	vao->SetAttributeDivisor(13, 1);
	vao->SetAttributeDivisor(14, 1);
	vao->SetAttributeDivisor(15, 1);

	PlantManager::GetInstance().m_internodeRenderProgram->Bind();
	const glm::mat4 translation = glm::translate(glm::identity<glm::mat4>(), glm::vec3(0.0f));
	const glm::mat4 scale = glm::scale(glm::identity<glm::mat4>(), glm::vec3(0.2f));
	const glm::mat4 model = translation * scale;
	PlantManager::GetInstance().m_internodeRenderProgram->SetFloat4x4("model", model);

	glDrawElementsInstanced(GL_TRIANGLES, static_cast<GLsizei>(mesh->GetTriangleAmount() * 3), GL_UNSIGNED_INT, 0, static_cast<GLsizei>(count));
	vao->DisableAttributeArray(11);
	vao->DisableAttributeArray(12);
	vao->DisableAttributeArray(13);
	vao->DisableAttributeArray(14);
	vao->DisableAttributeArray(15);
#pragma endregion
	*/
}

void SorghumManager::CollectEntities(std::vector<Entity>& entities, const Entity& walker)
{
	EntityManager::ForEachChild(walker, [&](Entity child)
		{
			if (!child.HasPrivateComponent<MeshRenderer>()) return;
			entities.push_back(child);
			CollectEntities(entities, child);
		});
}

void SorghumManager::CalculateIllumination(const RayMLVQ::IlluminationEstimationProperties& properties)
{
	auto& manager = GetInstance();
	const auto* owners = EntityManager::GetPrivateComponentOwnersList<TriangleIlluminationEstimator>();
	if (!owners) return;
	manager.m_processingEntities.clear();
	manager.m_probeTransforms.clear();
	manager.m_probeColors.clear();
	manager.m_properties = properties;
	manager.m_properties.m_pushNormal = true;
	manager.m_processingEntities.insert(manager.m_processingEntities.begin(), owners->begin(), owners->end());
	manager.m_processingIndex = manager.m_processingEntities.size();
	manager.m_processing = true;
}

void SorghumManager::GenerateLeavesForSorghum(PlantManager& manager)
{
	//Remove previous leaf internodes.
	std::vector<Entity> sorghums;
	PlantManager::GetInstance().m_plantQuery.ToEntityArray<PlantInfo>(sorghums, [](const Entity& plant, const PlantInfo& plantInfo)
		{
			return plantInfo.m_plantType == PlantType::Sorghum;
		});
	for (const auto& sorghum : sorghums)
	{
		if (!sorghum.HasPrivateComponent<SorghumData>()) continue;
#pragma region Clear all child with spline.
		auto children = EntityManager::GetChildren(sorghum);
		for (const auto& child : children)
		{
			if (child.HasPrivateComponent<Spline>())
			{
				EntityManager::DeleteEntity(child);
			}
		}
#pragma endregion
		auto truckSpline = std::make_unique<Spline>();
		Entity walker = sorghum;
		std::vector<Entity> centerNode;
		while (!walker.IsNull())
		{
			Entity temp;
			EntityManager::ForEachChild(walker, [&](Entity child)
				{
					if (child.HasComponentData<InternodeInfo>() && child.GetComponentData<InternodeInfo>().m_order == 1)
					{
						temp = child;
						centerNode.push_back(child);
					}
				}
			);
			walker = temp;
		}
		glm::vec3 startPosition = sorghum.GetComponentData<GlobalTransform>().GetPosition();
		glm::vec3 endPosition = centerNode.back().GetComponentData<GlobalTransform>().GetPosition();
		auto truckCurve = BezierCurve(glm::vec3(0.0f), (-startPosition + endPosition) / 2.0f, (-startPosition + endPosition) / 2.0f, endPosition - startPosition);
		truckSpline->m_curves.push_back(truckCurve);
		sorghum.SetPrivateComponent(std::move(truckSpline));
		int leafAmount = 0;
		for (int i = 0; i < centerNode.size(); i++)
		{
			glm::vec3 centerNodePosition = centerNode[i].GetComponentData<GlobalTransform>().GetPosition();
			EntityManager::ForEachChild(centerNode[i], [&](Entity child)
				{
					if (child.HasComponentData<InternodeInfo>() && child.GetComponentData<InternodeInfo>().m_order == 2)
					{
						const auto leafEntity = CreateSorghumLeaf(sorghum);
						auto leafSpline = std::make_unique<Spline>();
						auto cp0 = centerNodePosition;
						auto cp1 = child.GetComponentData<GlobalTransform>().GetPosition();
						auto child2 = EntityManager::GetChildren(child)[0];
						auto cp2 = child2.GetComponentData<GlobalTransform>().GetPosition();
						auto child3 = EntityManager::GetChildren(child2)[0];
						auto cp3 = child3.GetComponentData<GlobalTransform>().GetPosition();
						auto leafCurve = BezierCurve(cp0 - startPosition, cp1 - startPosition, cp2 - startPosition, cp3 - startPosition);
						leafSpline->m_curves.push_back(leafCurve);

						auto child0 = EntityManager::GetChildren(child3)[0];
						cp0 = child0.GetComponentData<GlobalTransform>().GetPosition();
						auto child1 = EntityManager::GetChildren(child0)[0];
						cp1 = child1.GetComponentData<GlobalTransform>().GetPosition();
						child2 = EntityManager::GetChildren(child1)[0];
						cp2 = child2.GetComponentData<GlobalTransform>().GetPosition();
						child3 = EntityManager::GetChildren(child2)[0];
						cp3 = child3.GetComponentData<GlobalTransform>().GetPosition();
						leafCurve = BezierCurve(cp0 - startPosition, cp1 - startPosition, cp2 - startPosition, cp3 - startPosition);
						leafSpline->m_curves.push_back(leafCurve);

						leafSpline->m_startingPoint = static_cast<float>(i + 1) / (centerNode.size() + 1);
						leafSpline->m_left = glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), leafSpline->m_curves.begin()->m_p0 - leafSpline->m_curves.back().m_p3);
						leafEntity.SetPrivateComponent(std::move(leafSpline));
						leafAmount++;
					}
				}
			);
		}

		if (leafAmount == sorghum.GetPrivateComponent<SorghumData>()->m_parameters.m_leafCount)
		{
			sorghum.GetPrivateComponent<SorghumData>()->m_growthComplete = true;
		}

	}

	GenerateMeshForAllSorghums();
}

void SorghumManager::FormCandidates(PlantManager& manager,
	std::vector<InternodeCandidate>& candidates)
{
	const float globalTime = manager.m_globalTime;
	const float sphereSize = GetInstance().m_leafNodeSphereSize;
	std::mutex mutex;
	EntityManager::ForEach<GlobalTransform, Transform, InternodeInfo, InternodeGrowth, InternodeStatistics, Illumination>(JobManager::PrimaryWorkers(), PlantManager::GetInstance().m_internodeQuery,
		[&, globalTime, sphereSize](int index, Entity internode, GlobalTransform& globalTransform, Transform& transform, InternodeInfo& internodeInfo, InternodeGrowth& internodeGrowth, InternodeStatistics& internodeStatistics, Illumination& internodeIllumination)
		{
			if (internodeInfo.m_plantType != PlantType::Sorghum) return;
			if (EntityManager::GetChildrenAmount(internode) != 0) return;
			if (!internodeInfo.m_plant.IsEnabled()) return;
			auto& internodeData = internode.GetPrivateComponent<InternodeData>();
			auto& sorghumData = internodeInfo.m_plant.GetPrivateComponent<SorghumData>();
			if (sorghumData->m_growthComplete) return;
			auto plantInfo = internodeInfo.m_plant.GetComponentData<PlantInfo>();
			auto parameters = sorghumData->m_parameters;
			if (internodeInfo.m_order == 1)
			{
				auto stemCandidate = InternodeCandidate();
				stemCandidate.m_info.m_plantType = PlantType::Sorghum;
				stemCandidate.m_owner = internodeData->m_owner;
				stemCandidate.m_parent = internode;
				stemCandidate.m_info.m_startGlobalTime = globalTime;
				stemCandidate.m_info.m_plant = internodeInfo.m_plant;
				stemCandidate.m_info.m_startAge = plantInfo.m_age;
				stemCandidate.m_info.m_order = internodeInfo.m_order;
				stemCandidate.m_info.m_level = internodeInfo.m_level + 1;
				stemCandidate.m_growth.m_internodeLength = glm::gaussRand(parameters.m_internodeLength, parameters.m_internodeLengthVariance);
				stemCandidate.m_growth.m_distanceToRoot = internodeGrowth.m_distanceToRoot + 1;
				float rotateAngle = glm::gaussRand(180.0f, parameters.m_rollAngleVariance + parameters.m_rollAngleVarianceDistanceFactor * internodeGrowth.m_distanceToRoot);
				stemCandidate.m_growth.m_desiredLocalRotation = glm::radians(glm::vec3(0.0f, 0.0f, rotateAngle));
				stemCandidate.m_statistics.m_isEndNode = true;
				stemCandidate.m_buds = std::vector<Bud>();
#pragma region Calculate transform
				glm::quat globalRotation = globalTransform.GetRotation() * stemCandidate.m_growth.m_desiredLocalRotation;
				glm::vec3 front = globalRotation * glm::vec3(0, 0, -1);
				glm::vec3 positionDelta = front * stemCandidate.m_growth.m_internodeLength;
				glm::vec3 newInternodePosition = globalTransform.GetPosition() + positionDelta;
				stemCandidate.m_globalTransform.m_value = glm::translate(newInternodePosition)
					* glm::mat4_cast(globalRotation) * glm::scale(glm::vec3(sphereSize));
				stemCandidate.m_transform.m_value = glm::inverse(globalTransform.m_value) * stemCandidate.m_globalTransform.m_value;
#pragma endregion
				candidates.push_back(std::move(stemCandidate));
				if (internodeGrowth.m_distanceToRoot > 0)
				{
					auto leafCandidate = InternodeCandidate();
					leafCandidate.m_info.m_plantType = PlantType::Sorghum;
					leafCandidate.m_owner = internodeData->m_owner;
					leafCandidate.m_parent = internode;
					leafCandidate.m_info.m_startGlobalTime = globalTime;
					leafCandidate.m_info.m_plant = internodeInfo.m_plant;
					leafCandidate.m_info.m_startAge = plantInfo.m_age;
					leafCandidate.m_info.m_order = internodeInfo.m_order + 1;
					leafCandidate.m_info.m_level = internodeInfo.m_level + 1;
					leafCandidate.m_growth.m_distanceToRoot = internodeGrowth.m_distanceToRoot + 1;
					leafCandidate.m_statistics.m_isEndNode = true;
					leafCandidate.m_buds = std::vector<Bud>();
					leafCandidate.m_growth.m_desiredLocalRotation = glm::radians(glm::vec3(0.0f, 30.0f, 0.0f));
					leafCandidate.m_growth.m_internodeLength = sorghumData->m_parameters.m_leafLengthBase;
#pragma region Calculate transform
					glm::quat globalRotation = globalTransform.GetRotation() * leafCandidate.m_growth.m_desiredLocalRotation;
					glm::vec3 front = globalRotation * glm::vec3(0, 0, -1);
					glm::vec3 positionDelta = front * (sorghumData->m_parameters.m_leafLengthBase / 8.0f);
					glm::vec3 newInternodePosition = globalTransform.GetPosition() + positionDelta;
					leafCandidate.m_globalTransform.m_value = glm::translate(newInternodePosition)
						* glm::mat4_cast(globalRotation) * glm::scale(glm::vec3(sphereSize));
					leafCandidate.m_transform.m_value = glm::inverse(globalTransform.m_value) * leafCandidate.m_globalTransform.m_value;
#pragma endregion
					std::lock_guard lock(mutex);
					candidates.push_back(std::move(leafCandidate));
				}
			}
		}
	);
}

void SorghumManager::FormLeafNodes(PlantManager& plantManager)
{
	auto& manager = GetInstance();
	std::vector<std::pair<Entity, SorghumParameters>> candidates;
	std::mutex mutex;
	const float globalTime = plantManager.m_globalTime;
	EntityManager::ForEach<GlobalTransform, Transform, InternodeInfo, InternodeGrowth, InternodeStatistics, Illumination>(JobManager::PrimaryWorkers(), PlantManager::GetInstance().m_internodeQuery,
		[&, globalTime](int index, Entity internode, GlobalTransform& globalTransform, Transform& transform, InternodeInfo& internodeInfo, InternodeGrowth& internodeGrowth, InternodeStatistics& internodeStatistics, Illumination& internodeIllumination)
		{
			if (internodeInfo.m_plantType != PlantType::Sorghum) return;
			if (!internodeInfo.m_plant.IsEnabled()) return;
			if (internodeInfo.m_order != 2) return;
			auto& sorghumData = internodeInfo.m_plant.GetPrivateComponent<SorghumData>();
			auto parameters = sorghumData->m_parameters;
			std::lock_guard lock(mutex);
			candidates.push_back(std::make_pair(internode, parameters));
		}
	);
	
	for (auto& i : candidates)
	{
		if(EntityManager::GetChildrenAmount(i.first) != 0)EntityManager::DeleteEntity(EntityManager::GetChildren(i.first)[0]);
		const int level = i.first.GetComponentData<InternodeInfo>().m_level;
		const float leafNodeDistance = i.second.m_leafLengthBase / 8.0f / manager.m_leafNodeSphereSize;
		const float leafGravityBending = i.second.m_leafGravityBending;
		const float leafGravityBendingIncrease = i.second.m_leafGravityBendingIncreaseFactor;
		auto meshRenderer = std::make_unique<MeshRenderer>();
		meshRenderer->m_mesh = Default::Primitives::Sphere;
		meshRenderer->m_material = GetInstance().m_leafNodeMaterial;
		meshRenderer->m_material->m_albedoColor = glm::vec3(0, 1, 0);
		i.first.SetPrivateComponent(std::move(meshRenderer));
		InternodeGrowth internodeGrowth;
		Entity leafNode = PlantManager::CreateInternode(PlantType::Sorghum, i.first);
		auto internodeInfo = leafNode.GetComponentData<InternodeInfo>();
		internodeInfo.m_order = 3;
		Transform transform;
		transform.SetPosition(glm::vec3(0, 0, -leafNodeDistance));
		transform.SetEulerRotation(glm::radians(glm::vec3(0, leafGravityBending, 0)));
		leafNode.SetComponentData(transform);
		leafNode.SetComponentData(internodeInfo);
		leafNode.SetComponentData(internodeGrowth);
		meshRenderer = std::make_unique<MeshRenderer>();
		meshRenderer->m_mesh = Default::Primitives::Sphere;
		meshRenderer->m_material = GetInstance().m_leafNodeMaterial;
		meshRenderer->m_material->m_albedoColor = glm::vec3(0, 1, 0);
		leafNode.SetPrivateComponent(std::move(meshRenderer));

		Entity leafNode2 = PlantManager::CreateInternode(PlantType::Sorghum, leafNode);
		internodeInfo = leafNode2.GetComponentData<InternodeInfo>();
		internodeInfo.m_order = 3;
		transform.SetPosition(glm::vec3(0, 0, -leafNodeDistance));
		transform.SetEulerRotation(glm::radians(glm::vec3(0, leafGravityBending + leafGravityBendingIncrease, 0)));
		leafNode2.SetComponentData(transform);
		leafNode2.SetComponentData(internodeInfo);
		leafNode2.SetComponentData(internodeGrowth);
		meshRenderer = std::make_unique<MeshRenderer>();
		meshRenderer->m_mesh = Default::Primitives::Sphere;
		meshRenderer->m_material = GetInstance().m_leafNodeMaterial;
		meshRenderer->m_material->m_albedoColor = glm::vec3(0, 1, 0);
		leafNode2.SetPrivateComponent(std::move(meshRenderer));

		Entity leafNode3 = PlantManager::CreateInternode(PlantType::Sorghum, leafNode2);
		internodeInfo = leafNode3.GetComponentData<InternodeInfo>();
		internodeInfo.m_order = 3;
		transform.SetPosition(glm::vec3(0, 0, -leafNodeDistance));
		transform.SetEulerRotation(glm::radians(glm::vec3(0, leafGravityBending + 2.0 * leafGravityBendingIncrease, 0)));
		leafNode3.SetComponentData(transform);
		leafNode3.SetComponentData(internodeInfo);
		leafNode3.SetComponentData(internodeGrowth);
		meshRenderer = std::make_unique<MeshRenderer>();
		meshRenderer->m_mesh = Default::Primitives::Sphere;
		meshRenderer->m_material = GetInstance().m_leafNodeMaterial;
		meshRenderer->m_material->m_albedoColor = glm::vec3(0, 1, 0);
		leafNode3.SetPrivateComponent(std::move(meshRenderer));

		Entity leafNode4 = PlantManager::CreateInternode(PlantType::Sorghum, leafNode3);
		internodeInfo = leafNode4.GetComponentData<InternodeInfo>();
		internodeInfo.m_order = 3;
		transform.SetPosition(glm::vec3(0, 0, -leafNodeDistance));
		transform.SetEulerRotation(glm::radians(glm::vec3(0, leafGravityBending + 3.0 * leafGravityBendingIncrease, 0)));
		leafNode4.SetComponentData(transform);
		leafNode4.SetComponentData(internodeInfo);
		leafNode4.SetComponentData(internodeGrowth);
		meshRenderer = std::make_unique<MeshRenderer>();
		meshRenderer->m_mesh = Default::Primitives::Sphere;
		meshRenderer->m_material = GetInstance().m_leafNodeMaterial;
		meshRenderer->m_material->m_albedoColor = glm::vec3(0, 1, 0);
		leafNode4.SetPrivateComponent(std::move(meshRenderer));

		Entity leafNode5 = PlantManager::CreateInternode(PlantType::Sorghum, leafNode4);
		internodeInfo = leafNode5.GetComponentData<InternodeInfo>();
		internodeInfo.m_order = 3;
		transform.SetPosition(glm::vec3(0, 0, -leafNodeDistance));
		transform.SetEulerRotation(glm::radians(glm::vec3(0, leafGravityBending + 4.0 * leafGravityBendingIncrease, 0)));
		leafNode5.SetComponentData(transform);
		leafNode5.SetComponentData(internodeInfo);
		leafNode5.SetComponentData(internodeGrowth);
		meshRenderer = std::make_unique<MeshRenderer>();
		meshRenderer->m_mesh = Default::Primitives::Sphere;
		meshRenderer->m_material = GetInstance().m_leafNodeMaterial;
		meshRenderer->m_material->m_albedoColor = glm::vec3(0, 1, 0);
		leafNode5.SetPrivateComponent(std::move(meshRenderer));

		Entity leafNode7 = PlantManager::CreateInternode(PlantType::Sorghum, leafNode5);
		internodeInfo = leafNode7.GetComponentData<InternodeInfo>();
		internodeInfo.m_order = 3;
		transform.SetPosition(glm::vec3(0, 0, -leafNodeDistance));
		transform.SetEulerRotation(glm::radians(glm::vec3(0, leafGravityBending + 5.0 * leafGravityBendingIncrease, 0)));
		leafNode7.SetComponentData(transform);
		leafNode7.SetComponentData(internodeInfo);
		leafNode7.SetComponentData(internodeGrowth);
		meshRenderer = std::make_unique<MeshRenderer>();
		meshRenderer->m_mesh = Default::Primitives::Sphere;
		meshRenderer->m_material = GetInstance().m_leafNodeMaterial;
		meshRenderer->m_material->m_albedoColor = glm::vec3(0, 1, 0);
		leafNode7.SetPrivateComponent(std::move(meshRenderer));
	}
}

void SorghumManager::RemoveInternodes(const Entity& sorghum)
{
	if (!sorghum.HasPrivateComponent<SorghumData>()) return;
	sorghum.RemovePrivateComponent<SorghumData>();
	Entity rootInternode;
	EntityManager::ForEachChild(sorghum, [&](Entity child)
		{
			if (child.HasComponentData<InternodeInfo>()) rootInternode = child;
		}
	);
	if (rootInternode.IsValid()) EntityManager::DeleteEntity(rootInternode);
}

void SorghumManager::Update()
{
	auto& manager = GetInstance();
	if (manager.m_displayLightProbes)
	{
		RenderLightProbes();
	}

	if (manager.m_processing)
	{
		manager.m_processingIndex--;
		if (manager.m_processingIndex == -1)
		{
			manager.m_processing = false;
		}
		else
		{
			const float timer = Application::EngineTime();
			auto& estimator = manager.m_processingEntities[manager.m_processingIndex].GetPrivateComponent<TriangleIlluminationEstimator>();
			estimator->CalculateIllumination(manager.m_properties);
			manager.m_probeTransforms.insert(manager.m_probeTransforms.end(), estimator->m_probeTransforms.begin(), estimator->m_probeTransforms.end());
			manager.m_probeColors.insert(manager.m_probeColors.end(), estimator->m_probeColors.begin(), estimator->m_probeColors.end());
			manager.m_perPlantCalculationTime = Application::EngineTime() - timer;
			const auto count = manager.m_probeTransforms.size();
			manager.m_lightProbeRenderingColorBuffer.SetData(static_cast<GLsizei>(count) * sizeof(glm::vec4), manager.m_probeColors.data(), GL_DYNAMIC_DRAW);
			manager.m_lightProbeRenderingTransformBuffer.SetData(static_cast<GLsizei>(count) * sizeof(glm::mat4), manager.m_probeTransforms.data(), GL_DYNAMIC_DRAW);
		}
	}
}

void SorghumManager::CreateGrid(SorghumField& field, const std::vector<Entity>& candidates)
{
	const Entity entity = EntityManager::CreateEntity("Field");
	std::vector<std::vector<glm::mat4>> matricesList;
	matricesList.resize(candidates.size());
	for (auto& i : matricesList)
	{
		i = std::vector<glm::mat4>();
	}
	field.GenerateField(matricesList);
	for (int i = 0; i < candidates.size(); i++)
	{
		CloneSorghums(entity, candidates[i], matricesList[i]);
	}
}
