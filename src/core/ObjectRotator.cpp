//
// Created by lllll on 8/16/2021.
//

#include "ObjectRotator.hpp"
void PlantFactory::ObjectRotator::FixedUpdate() {
  auto transform = GetOwner().GetDataComponent<Transform>();
  m_rotation.y += Application::Time().FixedDeltaTime() * m_rotateSpeed;
  transform.SetEulerRotation(glm::radians(m_rotation));
  GetOwner().SetDataComponent(transform);
}
void PlantFactory::ObjectRotator::Clone(
    const std::shared_ptr<IPrivateComponent> &target) {
  *this = *std::static_pointer_cast<ObjectRotator>(target);
}
void PlantFactory::ObjectRotator::OnGui() {
  ImGui::DragFloat("Speed", &m_rotateSpeed);
  ImGui::DragFloat3("Rotation", &m_rotation.x);
}
