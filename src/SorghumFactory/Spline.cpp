//
// Created by lllll on 8/26/2021.
//

#include "Spline.hpp"
using namespace SorghumFactory;

void Spline::OnGui() {
  if (ImGui::TreeNodeEx("Curves", ImGuiTreeNodeFlags_DefaultOpen)) {
    for (int i = 0; i < m_curves.size(); i++) {
      ImGui::Text(("Curve" + std::to_string(i)).c_str());
      ImGui::InputFloat3("CP0", &m_curves[i].m_p0.x);
      ImGui::InputFloat3("CP1", &m_curves[i].m_p1.x);
      ImGui::InputFloat3("CP2", &m_curves[i].m_p2.x);
      ImGui::InputFloat3("CP3", &m_curves[i].m_p3.x);
    }
    ImGui::TreePop();
  }
}

void Spline::Clone(const std::shared_ptr<IPrivateComponent> &target) {
  *this = *std::static_pointer_cast<Spline>(target);
}
void Spline::Serialize(YAML::Emitter &out) {}
void Spline::Deserialize(const YAML::Node &in) {}

void Spline::Import(std::ifstream &stream) {
  int curveAmount;
  stream >> curveAmount;
  m_curves.clear();
  for (int i = 0; i < curveAmount; i++) {
    glm::vec3 cp[4];
    float x, y, z;
    for (auto &j : cp) {
      stream >> x >> z >> y;
      j = glm::vec3(x, y, z) * 10.0f;
    }
    m_curves.emplace_back(cp[0], cp[1], cp[2], cp[3]);
  }
}

glm::vec3 Spline::EvaluatePointFromCurve(float point) {
  const float splineU = glm::clamp(point, 0.0f, 1.0f) * float(m_curves.size());

  // Decompose the global u coordinate on the spline
  float integerPart;
  const float fractionalPart = modff(splineU, &integerPart);

  auto curveIndex = int(integerPart);
  auto curveU = fractionalPart;

  // If evaluating the very last point on the spline
  if (curveIndex == m_curves.size() && curveU <= 0.0f) {
    // Flip to the end of the last patch
    curveIndex--;
    curveU = 1.0f;
  }
  return m_curves.at(curveIndex).GetPoint(curveU);
}

glm::vec3 Spline::EvaluateAxisFromCurve(float point) {
  const float splineU = glm::clamp(point, 0.0f, 1.0f) * float(m_curves.size());

  // Decompose the global u coordinate on the spline
  float integerPart;
  const float fractionalPart = modff(splineU, &integerPart);

  auto curveIndex = int(integerPart);
  auto curveU = fractionalPart;

  // If evaluating the very last point on the spline
  if (curveIndex == m_curves.size() && curveU <= 0.0f) {
    // Flip to the end of the last patch
    curveIndex--;
    curveU = 1.0f;
  }
  return m_curves.at(curveIndex).GetAxis(curveU);
}
