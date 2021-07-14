#include <CubeVolume.hpp>
using namespace PlantFactory;
void CubeVolume::ApplyMeshRendererBounds()
{
	auto& meshRenderer = GetOwner().GetPrivateComponent<MeshRenderer>();
	const auto bound = meshRenderer.m_mesh->GetBound();
	const auto globalTransform = GetOwner().GetComponentData<GlobalTransform>();
	m_minMaxBound.m_min = globalTransform.GetPosition() + glm::vec3(glm::vec4(bound.m_min, 1.0f) * globalTransform.m_value);
	m_minMaxBound.m_max = globalTransform.GetPosition() + glm::vec3(glm::vec4(bound.m_max, 1.0f) * globalTransform.m_value);
}

CubeVolume::CubeVolume()
{
	m_minMaxBound.m_min = glm::vec3(-5, 0, -5);
	m_minMaxBound.m_max = glm::vec3(5, 10, 5);
	SetEnabled(true);
}

void CubeVolume::OnGui()
{
	ImGui::Checkbox("Obstacle", &m_asObstacle);
	ImGui::DragFloat3("Min", &m_minMaxBound.m_min.x, 0.1);
	ImGui::DragFloat3("Max", &m_minMaxBound.m_max.x, 0.1);
	ImGui::Checkbox("Display bounds", &m_displayBounds);
	if (m_displayBounds)
	{
		RenderManager::DrawGizmoMesh(DefaultResources::Primitives::Cube.get(), *RenderManager::GetMainCamera(), glm::vec4(0, 1, 0, 0.2), glm::translate(m_minMaxBound.Center()) * glm::scale(m_minMaxBound.Size()), 1);
	}
	if (GetOwner().HasPrivateComponent<MeshRenderer>()) {
		if (ImGui::Button("Apply mesh bound"))
		{
			ApplyMeshRendererBounds();
		}
	}
}

bool CubeVolume::InVolume(const glm::vec3& position)
{
	return m_minMaxBound.InBound(position);
}

glm::vec3 CubeVolume::GetRandomPoint()
{
	return glm::linearRand(m_minMaxBound.m_min, m_minMaxBound.m_max);
}
