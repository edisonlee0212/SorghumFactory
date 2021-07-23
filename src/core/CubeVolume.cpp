#include <CubeVolume.hpp>

using namespace PlantFactory;

void CubeVolume::ApplyMeshRendererBounds() {
  auto &meshRenderer = GetOwner().GetPrivateComponent<MeshRenderer>();
  m_minMaxBound = meshRenderer.m_mesh->GetBound();
}

void CubeVolume::OnCreate() {
  m_minMaxBound.m_min = glm::vec3(-5, 0, -5);
  m_minMaxBound.m_max = glm::vec3(5, 10, 5);
  SetEnabled(true);
}

void CubeVolume::OnGui() {
  ImGui::Checkbox("Obstacle", &m_asObstacle);
  ImGui::DragFloat3("Min", &m_minMaxBound.m_min.x, 0.1);
  ImGui::DragFloat3("Max", &m_minMaxBound.m_max.x, 0.1);
  ImGui::Checkbox("Display bounds", &m_displayBounds);
  if (m_displayBounds) {
    const auto globalTransform = GetOwner().GetDataComponent<GlobalTransform>();
    RenderManager::DrawGizmoMesh(
        DefaultResources::Primitives::Cube,
        glm::vec4(0, 1, 0, 0.2),
        globalTransform.m_value * glm::translate(m_minMaxBound.Center()) *
            glm::scale(m_minMaxBound.Size()),
        1);
  }
  if (GetOwner().HasPrivateComponent<MeshRenderer>()) {
    if (ImGui::Button("Apply mesh bound")) {
      ApplyMeshRendererBounds();
    }
  }
}

bool CubeVolume::InVolume(const glm::vec3 &position) {
  const auto globalTransform = GetOwner().GetDataComponent<GlobalTransform>();
  const auto &finalPos = glm::vec3(
      (glm::inverse(globalTransform.m_value) * glm::translate(position))[3]);
  return m_minMaxBound.InBound(finalPos);
}

glm::vec3 CubeVolume::GetRandomPoint() {
  return glm::linearRand(m_minMaxBound.m_min, m_minMaxBound.m_max);
}
bool CubeVolume::InVolume(const GlobalTransform &globalTransform,
                          const glm::vec3 &position) {
  const auto &finalPos = glm::vec3(
      (glm::inverse(globalTransform.m_value) * glm::translate(position))[3]);
  return m_minMaxBound.InBound(finalPos);
}