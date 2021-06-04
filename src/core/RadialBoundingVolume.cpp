#include <RadialBoundingVolume.hpp>
#include <PlantManager.hpp>
#include <RayTracedRenderer.hpp>
#include <OptixRayTracer.hpp>

using namespace PlantFactory;
using namespace RayMLVQ;
glm::vec3 RadialBoundingVolume::GetRandomPoint()
{
	if(!m_meshGenerated) return glm::vec3(0);
	return glm::vec3(0);
}

glm::ivec2 RadialBoundingVolume::SelectSlice(glm::vec3 position) const
{
	glm::ivec2 retVal;
	const float heightLevel = m_maxHeight / m_layerAmount;
	const float sliceAngle = 360.0f / m_sectorAmount;
	auto x = static_cast<int>(position.y / heightLevel);
	if (x < 0) x = 0;
	retVal.x = x;
	if (retVal.x >= m_layerAmount) retVal.x = m_layerAmount - 1;
	if (position.x == 0 && position.z == 0) retVal.y = 0;
	else retVal.y = static_cast<int>((glm::degrees(glm::atan(position.x, position.z)) + 180.0f) / sliceAngle);
	if (retVal.y >= m_sectorAmount) retVal.y = m_sectorAmount - 1;
	return retVal;
}

void RadialBoundingVolume::GenerateMesh()
{
	m_boundMeshes.clear();
	if (m_cakeTiers.empty()) return;
	for (int tierIndex = 0; tierIndex < m_layerAmount; tierIndex++)
	{
		auto mesh = std::make_shared<Mesh>();
		std::vector<Vertex> vertices;
		std::vector<unsigned> indices;

		const float sliceAngle = 360.0f / m_sectorAmount;
		const int totalAngleStep = 360.0f / m_sectorAmount;
		const int totalLevelStep = 2;
		const float stepAngle = sliceAngle / (totalAngleStep - 1);
		const float heightLevel = m_maxHeight / m_layerAmount;
		const float stepLevel = heightLevel / (totalLevelStep - 1);
		vertices.resize(totalLevelStep * m_sectorAmount * totalAngleStep * 2 + totalLevelStep);
		indices.resize((12 * (totalLevelStep - 1) * totalAngleStep) * m_sectorAmount);
		for (int levelStep = 0; levelStep < totalLevelStep; levelStep++) {
			const float currentHeight = heightLevel * tierIndex + stepLevel * levelStep;
			for (int sliceIndex = 0; sliceIndex < m_sectorAmount; sliceIndex++)
			{
				for (int angleStep = 0; angleStep < totalAngleStep; angleStep++) {
					const int actualAngleStep = sliceIndex * totalAngleStep + angleStep;

					float currentAngle = sliceAngle * sliceIndex + stepAngle * angleStep;
					if (currentAngle >= 360) currentAngle = 0;
					float x = glm::abs(glm::tan(glm::radians(currentAngle)));
					float z = 1.0f;
					if (currentAngle >= 0 && currentAngle <= 90)
					{
						z *= -1;
						x *= -1;
					}
					else if (currentAngle > 90 && currentAngle <= 180)
					{
						x *= -1;
					}
					else if (currentAngle > 270 && currentAngle <= 360)
					{
						z *= -1;
					}
					glm::vec3 position = glm::normalize(glm::vec3(x, 0.0f, z)) * m_cakeTiers[tierIndex][sliceIndex].m_maxDistance;
					position.y = currentHeight;
					vertices[levelStep * totalAngleStep * m_sectorAmount + actualAngleStep].m_position = position;
					vertices[levelStep * totalAngleStep * m_sectorAmount + actualAngleStep].m_texCoords = glm::vec2((float)levelStep / (totalLevelStep - 1), (float)angleStep / (totalAngleStep - 1));
					vertices[levelStep * totalAngleStep * m_sectorAmount + actualAngleStep].m_normal = glm::normalize(position);
					vertices[totalLevelStep * m_sectorAmount * totalAngleStep + levelStep * totalAngleStep * m_sectorAmount + actualAngleStep].m_position = position;
					vertices[totalLevelStep * m_sectorAmount * totalAngleStep + levelStep * totalAngleStep * m_sectorAmount + actualAngleStep].m_texCoords = glm::vec2((float)levelStep / (totalLevelStep - 1), (float)angleStep / (totalAngleStep - 1));
					vertices[totalLevelStep * m_sectorAmount * totalAngleStep + levelStep * totalAngleStep * m_sectorAmount + actualAngleStep].m_normal = glm::vec3(0, levelStep == 0 ? -1 : 1, 0);
				}
			}
			vertices[vertices.size() - totalLevelStep + levelStep].m_position = glm::vec3(0, currentHeight, 0);
			vertices[vertices.size() - totalLevelStep + levelStep].m_normal = glm::vec3(0, levelStep == 0 ? -1 : 1, 0);
			vertices[vertices.size() - totalLevelStep + levelStep].m_texCoords = glm::vec2(0.0f);
		}
		for (int levelStep = 0; levelStep < totalLevelStep - 1; levelStep++)
		{
			for (int sliceIndex = 0; sliceIndex < m_sectorAmount; sliceIndex++)
			{
				for (int angleStep = 0; angleStep < totalAngleStep; angleStep++) {
					const int actualAngleStep = sliceIndex * totalAngleStep + angleStep; //0-5
					//Fill a quad here.
					if (actualAngleStep < m_sectorAmount * totalAngleStep - 1) {
						indices[12 * (levelStep * totalAngleStep * m_sectorAmount + actualAngleStep)] = levelStep * totalAngleStep * m_sectorAmount + actualAngleStep;
						indices[12 * (levelStep * totalAngleStep * m_sectorAmount + actualAngleStep) + 1] = levelStep * totalAngleStep * m_sectorAmount + actualAngleStep + 1;
						indices[12 * (levelStep * totalAngleStep * m_sectorAmount + actualAngleStep) + 2] = (levelStep + 1) * totalAngleStep * m_sectorAmount + actualAngleStep;

						indices[12 * (levelStep * totalAngleStep * m_sectorAmount + actualAngleStep) + 3] = (levelStep + 1) * totalAngleStep * m_sectorAmount + actualAngleStep + 1;
						indices[12 * (levelStep * totalAngleStep * m_sectorAmount + actualAngleStep) + 4] = (levelStep + 1) * totalAngleStep * m_sectorAmount + actualAngleStep;
						indices[12 * (levelStep * totalAngleStep * m_sectorAmount + actualAngleStep) + 5] = levelStep * totalAngleStep * m_sectorAmount + actualAngleStep + 1;
						//Connect with center here.
						indices[12 * (levelStep * totalAngleStep * m_sectorAmount + actualAngleStep) + 6] = totalLevelStep * m_sectorAmount * totalAngleStep + levelStep * totalAngleStep * m_sectorAmount + actualAngleStep;
						indices[12 * (levelStep * totalAngleStep * m_sectorAmount + actualAngleStep) + 7] = vertices.size() - totalLevelStep + levelStep;
						indices[12 * (levelStep * totalAngleStep * m_sectorAmount + actualAngleStep) + 8] = totalLevelStep * m_sectorAmount * totalAngleStep + levelStep * totalAngleStep * m_sectorAmount + actualAngleStep + 1;

						indices[12 * (levelStep * totalAngleStep * m_sectorAmount + actualAngleStep) + 9] = totalLevelStep * m_sectorAmount * totalAngleStep + (levelStep + 1) * totalAngleStep * m_sectorAmount + actualAngleStep + 1;
						indices[12 * (levelStep * totalAngleStep * m_sectorAmount + actualAngleStep) + 10] = vertices.size() - totalLevelStep + (levelStep + 1);
						indices[12 * (levelStep * totalAngleStep * m_sectorAmount + actualAngleStep) + 11] = totalLevelStep * m_sectorAmount * totalAngleStep + (levelStep + 1) * totalAngleStep * m_sectorAmount + actualAngleStep;
					}
					else
					{
						indices[12 * (levelStep * totalAngleStep * m_sectorAmount + actualAngleStep)] = levelStep * totalAngleStep * m_sectorAmount + actualAngleStep;
						indices[12 * (levelStep * totalAngleStep * m_sectorAmount + actualAngleStep) + 1] = levelStep * totalAngleStep * m_sectorAmount;
						indices[12 * (levelStep * totalAngleStep * m_sectorAmount + actualAngleStep) + 2] = (levelStep + 1) * totalAngleStep * m_sectorAmount + actualAngleStep;

						indices[12 * (levelStep * totalAngleStep * m_sectorAmount + actualAngleStep) + 3] = (levelStep + 1) * totalAngleStep * m_sectorAmount;
						indices[12 * (levelStep * totalAngleStep * m_sectorAmount + actualAngleStep) + 4] = (levelStep + 1) * totalAngleStep * m_sectorAmount + actualAngleStep;
						indices[12 * (levelStep * totalAngleStep * m_sectorAmount + actualAngleStep) + 5] = levelStep * totalAngleStep * m_sectorAmount;
						//Connect with center here.
						indices[12 * (levelStep * totalAngleStep * m_sectorAmount + actualAngleStep) + 6] = totalLevelStep * m_sectorAmount * totalAngleStep + levelStep * totalAngleStep * m_sectorAmount + actualAngleStep;
						indices[12 * (levelStep * totalAngleStep * m_sectorAmount + actualAngleStep) + 7] = vertices.size() - totalLevelStep + levelStep;
						indices[12 * (levelStep * totalAngleStep * m_sectorAmount + actualAngleStep) + 8] = totalLevelStep * m_sectorAmount * totalAngleStep + levelStep * totalAngleStep * m_sectorAmount;

						indices[12 * (levelStep * totalAngleStep * m_sectorAmount + actualAngleStep) + 9] = totalLevelStep * m_sectorAmount * totalAngleStep + (levelStep + 1) * totalAngleStep * m_sectorAmount;
						indices[12 * (levelStep * totalAngleStep * m_sectorAmount + actualAngleStep) + 10] = vertices.size() - totalLevelStep + (levelStep + 1);
						indices[12 * (levelStep * totalAngleStep * m_sectorAmount + actualAngleStep) + 11] = totalLevelStep * m_sectorAmount * totalAngleStep + (levelStep + 1) * totalAngleStep * m_sectorAmount + actualAngleStep;
					}

				}
			}
		}
		mesh->SetVertices(19, vertices, indices);
		m_boundMeshes.push_back(std::move(mesh));
	}
	m_meshGenerated = true;
}

void RadialBoundingVolume::FormEntity()
{
	if (!m_meshGenerated) CalculateVolume();
	if (!m_meshGenerated) return;
	auto children = EntityManager::GetChildren(GetOwner());
	for (auto& child : children)
	{
		EntityManager::DeleteEntity(child);
	}
	children.clear();
	for (auto i = 0; i < m_boundMeshes.size(); i++)
	{
		auto slice = EntityManager::CreateEntity( "RBV_" + std::to_string(i));
		auto mmc = std::make_unique<MeshRenderer>();
		mmc->m_material = ResourceManager::LoadMaterial(false, Default::GLPrograms::StandardProgram);
		mmc->m_forwardRendering = false;
		mmc->m_mesh = m_boundMeshes[i];
		slice.SetPrivateComponent(std::move(mmc));
		slice.SetPrivateComponent(std::make_unique<RayTracedRenderer>());
		EntityManager::SetParent(slice, GetOwner(), false);
		slice.GetPrivateComponent<RayTracedRenderer>()->SyncWithMeshRenderer();
	}
}

std::string RadialBoundingVolume::Save()
{
	if (!m_meshGenerated) CalculateVolume();
	std::string output;
	output += std::to_string(m_layerAmount) + "\n";
	output += std::to_string(m_sectorAmount) + "\n";
	output += std::to_string(m_maxHeight) + "\n";
	output += std::to_string(m_maxRadius) + "\n";
	int tierIndex = 0;
	for (const auto& tier : m_cakeTiers)
	{
		int sliceIndex = 0;
		for (const auto& slice : tier)
		{
			output += std::to_string(slice.m_maxDistance);
			output += "\n";
			sliceIndex++;
		}
		tierIndex++;
	}
	output += "\n";
	for (const auto& tier : m_cakeTiers)
	{
		int sliceIndex = 0;
		for (const auto& slice : tier)
		{
			output += std::to_string(slice.m_maxDistance);
			output += ",";
			sliceIndex++;
		}
		tierIndex++;
	}
	output += "\n";
	return output;
}

void RadialBoundingVolume::ExportAsObj(const std::string& filename)
{
	if (!m_meshGenerated) CalculateVolume();
	auto& meshes = m_boundMeshes;

	std::ofstream of;
	of.open((filename + ".obj").c_str(), std::ofstream::out | std::ofstream::trunc);
	if (of.is_open())
	{
		std::string o = "o ";
		o += "RBV\n";
		of.write(o.c_str(), o.size());
		of.flush();
		std::string data;
		int offset = 1;
#pragma region Data collection
		for (auto& mesh : meshes) {
			for (const auto& position : mesh->UnsafeGetVertexPositions()) {
				data += "v " + std::to_string(position.x)
					+ " " + std::to_string(-position.y)
					+ " " + std::to_string(position.z)
					+ "\n";
			}
		}
		for (auto& mesh : meshes)
		{
			data += "# List of indices for faces vertices, with (x, y, z).\n";
			auto& triangles = mesh->UnsafeGetTriangles();
			for (auto i = 0; i < triangles.size(); i++) {
				auto f1 = triangles.at(i).x + offset;
				auto f2 = triangles.at(i).y + offset;
				auto f3 = triangles.at(i).z + offset;
				data += "f " + std::to_string(f1)
					+ " " + std::to_string(f2)
					+ " " + std::to_string(f3)
					+ "\n";
			}
			offset += mesh->GetVerticesAmount();
		}
#pragma endregion
		of.write(data.c_str(), data.size());
		of.flush();
	}
}

void RadialBoundingVolume::Load(const std::string& path)
{
	std::ifstream ifs;
	ifs.open(path.c_str());
	Debug::Log("Loading from " + path);
	if (ifs.is_open())
	{
		ifs >> m_layerAmount;
		ifs >> m_sectorAmount;
		ifs >> m_maxHeight;
		ifs >> m_maxRadius;
		m_cakeTiers.resize(m_layerAmount);
		for (auto& tier : m_cakeTiers)
		{
			tier.resize(m_sectorAmount);
			for (auto& slice : tier)
			{
				ifs >> slice.m_maxDistance;
			}
		}
		GenerateMesh();
	}
}

void RadialBoundingVolume::CalculateVolume()
{
	const auto tree = EntityManager::GetParent(GetOwner());
	EntityQuery internodeDataQuery = EntityManager::CreateEntityQuery();
	EntityManager::SetEntityQueryAllFilters(internodeDataQuery, InternodeInfo());
	std::vector<InternodeInfo> internodeInfos;
	internodeDataQuery.ToComponentDataArray<InternodeInfo, InternodeInfo>(internodeInfos, [=](const InternodeInfo& info)
		{
			return info.m_plant == tree;
		}
	);
	std::vector<Entity> internodes;
	internodeDataQuery.ToEntityArray<InternodeInfo>(internodes, [=](const Entity& entity, const InternodeInfo& info)
		{
			return info.m_plant == tree;
		}
	);
	m_maxHeight = 0;
	m_maxRadius = 0;
	const auto treeGlobalTransform = tree.GetComponentData<GlobalTransform>().m_value;
	std::vector<glm::vec3> positions;
	for (auto& i : internodes)
	{
		auto globalTransform = i.GetComponentData<GlobalTransform>().m_value;
		const glm::vec3 position = (glm::inverse(treeGlobalTransform) * globalTransform)[3];
		positions.push_back(position);
		if (position.y > m_maxHeight) m_maxHeight = position.y;
		const float radius = glm::length(glm::vec2(position.x, position.z));
		if (radius > m_maxRadius) m_maxRadius = radius;
	}

	m_cakeTiers.resize(m_layerAmount);
	for (auto& tier : m_cakeTiers)
	{
		tier.resize(m_sectorAmount);
		for (auto& slice : tier)
		{
			slice.m_maxDistance = 0.0f;
		}
	}
	auto positionIndex = 0;
	for (auto& internode : internodes)
	{
		const auto internodeGrowth = internode.GetComponentData<InternodeGrowth>();
		auto parentGlobalTransform = EntityManager::GetParent(internode).GetComponentData<GlobalTransform>().m_value;
		const glm::vec3 parentNodePosition = (glm::inverse(treeGlobalTransform) * parentGlobalTransform)[3];
		const int segments = 3;
		for (int i = 0; i < segments; i++) {
			const glm::vec3 position = positions[positionIndex] + (parentNodePosition - positions[positionIndex]) * (float)i / (float)segments;
			const auto sliceIndex = SelectSlice(position);
			const float currentDistance = glm::length(glm::vec2(position.x, position.z));
			if (currentDistance <= internodeGrowth.m_thickness)
			{
				for (auto& slice : m_cakeTiers[sliceIndex.x])
				{
					if (slice.m_maxDistance < currentDistance + internodeGrowth.m_thickness) slice.m_maxDistance = currentDistance + internodeGrowth.m_thickness;
				}
			}
			else if (m_cakeTiers[sliceIndex.x][sliceIndex.y].m_maxDistance < currentDistance) m_cakeTiers[sliceIndex.x][sliceIndex.y].m_maxDistance = currentDistance;
		}
		positionIndex++;
	}
	GenerateMesh();
}

void RadialBoundingVolume::CalculateVolume(float maxHeight)
{
	const auto tree = EntityManager::GetParent(GetOwner());
	EntityQuery internodeDataQuery = EntityManager::CreateEntityQuery();
	EntityManager::SetEntityQueryAllFilters(internodeDataQuery, InternodeInfo());
	std::vector<InternodeInfo> internodeInfos;
	internodeDataQuery.ToComponentDataArray<InternodeInfo, InternodeInfo>(internodeInfos, [=](const InternodeInfo& info)
		{
			return info.m_plant == tree;
		}
	);
	std::vector<Entity> internodes;
	internodeDataQuery.ToEntityArray<InternodeInfo>(internodes, [=](const Entity& entity, const InternodeInfo& info)
		{
			return info.m_plant == tree;
		}
	);
	m_maxHeight = maxHeight;
	m_maxRadius = 0;
	const auto treeGlobalTransform = tree.GetComponentData<GlobalTransform>().m_value;
	std::vector<glm::vec3> positions;
	for (auto& i : internodes)
	{
		auto globalTransform = i.GetComponentData<GlobalTransform>().m_value;
		const glm::vec3 position = (glm::inverse(treeGlobalTransform) * globalTransform)[3];
		positions.push_back(position);
		const float radius = glm::length(glm::vec2(position.x, position.z));
		if (radius > m_maxRadius) m_maxRadius = radius;
	}

	m_cakeTiers.resize(m_layerAmount);
	for (auto& tier : m_cakeTiers)
	{
		tier.resize(m_sectorAmount);
		for (auto& slice : tier)
		{
			slice.m_maxDistance = 0.0f;
		}
	}
	const auto threadsAmount = JobManager::PrimaryWorkers().Size();
	std::vector<std::vector<std::vector<RadialBoundingVolumeSlice>>> tempCakeTowers;
	tempCakeTowers.resize(threadsAmount);
	for (int i = 0; i < threadsAmount; i++)
	{
		tempCakeTowers[i].resize(m_layerAmount);
		for (auto& tier : tempCakeTowers[i])
		{
			tier.resize(m_sectorAmount);
			for (auto& slice : tier)
			{
				slice.m_maxDistance = 0.0f;
			}
		}
	}
	auto positionIndex = 0;
	for (auto& internode : internodes)
	{
		const auto internodeGrowth = internode.GetComponentData<InternodeGrowth>();
		auto parentGlobalTransform = EntityManager::GetParent(internode).GetComponentData<GlobalTransform>().m_value;
		const glm::vec3 parentNodePosition = (glm::inverse(treeGlobalTransform) * parentGlobalTransform)[3];
		const int segments = 3;
		for (int i = 0; i < segments; i++) {
			const glm::vec3 position = positions[positionIndex] + (parentNodePosition - positions[positionIndex]) * (float)i / (float)segments;
			const auto sliceIndex = SelectSlice(position);
			const float currentDistance = glm::length(glm::vec2(position.x, position.z));
			if (currentDistance <= internodeGrowth.m_thickness)
			{
				for (auto& slice : m_cakeTiers[sliceIndex.x])
				{
					if (slice.m_maxDistance < currentDistance + internodeGrowth.m_thickness) slice.m_maxDistance = currentDistance + internodeGrowth.m_thickness;
				}
			}
			else if (m_cakeTiers[sliceIndex.x][sliceIndex.y].m_maxDistance < currentDistance) m_cakeTiers[sliceIndex.x][sliceIndex.y].m_maxDistance = currentDistance;
		}
		positionIndex++;
	}
	GenerateMesh();
}

void RadialBoundingVolume::OnGui()
{
	ImGui::Checkbox("Prune Buds", &m_pruneBuds);
	ImGui::Checkbox("Display bounds", &m_display);
	ImGui::ColorEdit4("Display Color", &m_displayColor.x);
	ImGui::DragFloat("Display Scale", &m_displayScale, 0.01f, 0.01f, 1.0f);
	bool edited = false;
	if (ImGui::DragInt("Layer Amount", &m_layerAmount, 1, 1, 100)) edited = true;
	if (ImGui::DragInt("Slice Amount", &m_sectorAmount, 1, 1, 100)) edited = true;
	if (ImGui::Button("Calculate Bounds") || edited) CalculateVolume();
	if (ImGui::Button("Form Entity"))
	{
		FormEntity();
	}
	FileIO::SaveFile("Save RBV", ".rbv", [this](const std::string& path)
		{
			const std::string data = Save();
			std::ofstream ofs;
			ofs.open(path.c_str(), std::ofstream::out | std::ofstream::trunc);
			ofs.write(data.c_str(), data.length());
			ofs.flush();
			ofs.close();
		}
	);
	FileIO::OpenFile("Load RBV", ".rbv", [this](const std::string& path)
		{
			Load(path);
		}
	);
	FileIO::SaveFile("Export RBV as OBJ", ".obj", [this](const std::string& path)
		{
			ExportAsObj(path);
		}
	);
	if (m_display && m_meshGenerated)
	{
		for (auto& i : m_boundMeshes) {
			RenderManager::DrawGizmoMesh(i.get(), EditorManager::GetSceneCamera().get(), m_displayColor, GetOwner().GetComponentData<GlobalTransform>().m_value);
		}
	}
}

bool RadialBoundingVolume::InVolume(const glm::vec3& position)
{
	if (glm::any(glm::isnan(position))) return true;
	if (m_meshGenerated) {
		const auto sliceIndex = SelectSlice(position);
		const float currentDistance = glm::length(glm::vec2(position.x, position.z));
		return glm::max(1.0f, m_cakeTiers[sliceIndex.x][sliceIndex.y].m_maxDistance) >= currentDistance && position.y <= m_maxHeight;
	}
	return true;
}

