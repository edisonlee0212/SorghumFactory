#include "Spline.hpp"
#include "SorghumLayer.hpp"

using namespace SorghumFactory;

void Spline::OnInspect() {
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
  static bool renderNodes = false;
  static float nodeSize = 0.1f;
  static glm::vec4 renderColor = glm::vec4(1.0f);
  ImGui::Checkbox("Render nodes", &renderNodes);
  if (renderNodes) {
    if (ImGui::TreeNodeEx("Render settings", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::DragFloat("Size", &nodeSize, 0.01f, 0.01f, 1.0f);
      ImGui::ColorEdit4("Color", &renderColor.x);
      ImGui::TreePop();
    }
    std::vector<glm::mat4> matrices;
    matrices.resize(m_nodes.size());
    for (int i = 0; i < m_nodes.size(); i++) {
      matrices[i] =
          glm::translate(m_nodes[i].m_position) * glm::scale(glm::vec3(1.0f));
    }
    Graphics::DrawGizmoMeshInstanced(
        DefaultResources::Primitives::Sphere, renderColor, matrices,
        GetOwner().GetDataComponent<GlobalTransform>().m_value, nodeSize);
  }
}

void Spline::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_left" << YAML::Value << m_left;
  out << YAML::Key << "m_curves" << YAML::BeginSeq;
  for (const auto &i : m_curves) {
    out << YAML::BeginMap;
    out << YAML::Key << "m_p0" << YAML::Value << i.m_p0;
    out << YAML::Key << "m_p1" << YAML::Value << i.m_p1;
    out << YAML::Key << "m_p2" << YAML::Value << i.m_p2;
    out << YAML::Key << "m_p3" << YAML::Value << i.m_p3;
    out << YAML::EndMap;
  }
  out << YAML::EndSeq;

  if (!m_nodes.empty()) {
    out << YAML::Key << "m_nodes" << YAML::Value
        << YAML::Binary((const unsigned char *)m_nodes.data(),
                        m_nodes.size() * sizeof(SplineNode));
  }
}
void Spline::Deserialize(const YAML::Node &in) {
  m_left = in["m_left"].as<glm::vec3>();
  if (in["m_curves"]) {
    m_curves.clear();
    for (const auto &i : in["m_curves"]) {
      m_curves.push_back(
          BezierCurve(i["m_p0"].as<glm::vec3>(), i["m_p1"].as<glm::vec3>(),
                      i["m_p2"].as<glm::vec3>(), i["m_p3"].as<glm::vec3>()));
    }
  }

  if (in["m_nodes"]) {
    YAML::Binary nodes = in["m_nodes"].as<YAML::Binary>();
    m_nodes.resize(nodes.size() / sizeof(SplineNode));
    std::memcpy(m_nodes.data(), nodes.data(), nodes.size());
  }
}

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

glm::vec3 Spline::EvaluatePointFromCurves(float point) const {
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

glm::vec3 Spline::EvaluateAxisFromCurves(float point) const {
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

void Spline::GenerateLeafGeometry(const ProceduralStemState &stemState,
                                  const ProceduralLeafState &leafState,
                                  int nodeAmount) {
  auto sorghumLayer = Application::GetLayer<SorghumLayer>();
  if (!sorghumLayer)
    return;

  m_vertices.clear();
  m_triangles.clear();
  m_segments.clear();
  float leftPeriod = 0.0f;
  float rightPeriod = 0.0f;
  int stemSegmentCount = 0;
  for (int i = 1; i < m_nodes.size(); i++) {
    auto &prev = m_nodes.at(i - 1);
    auto &curr = m_nodes.at(i);
    if (i == nodeAmount) {
      stemSegmentCount = m_segments.size();
    }
    float distance = glm::distance(prev.m_position, curr.m_position);
    BezierCurve curve = BezierCurve(
        prev.m_position, prev.m_position + distance / 5.0f * prev.m_axis,
        curr.m_position - distance / 5.0f * curr.m_axis, curr.m_position);
    for (float div = (i == 1 ? 0.0f
                             : 1.0f / static_cast<float>(
                                          sorghumLayer->m_segmentAmount));
         div <= 1.0f;
         div += 1.0f / static_cast<float>(sorghumLayer->m_segmentAmount)) {
      auto front = prev.m_axis * (1.0f - div) + curr.m_axis * div;
      auto up = glm::normalize(glm::cross(m_left, front));
      if (prev.m_isLeaf) {
        leftPeriod += leafState.m_wavinessFrequency /
                      static_cast<float>(sorghumLayer->m_segmentAmount);
        rightPeriod += leafState.m_wavinessFrequency /
                       static_cast<float>(sorghumLayer->m_segmentAmount);
      }
      auto waviness = glm::mix(prev.m_waviness, curr.m_waviness, div);
      m_segments.emplace_back(
          curve.GetPoint(div), up, front,
          glm::mix(prev.m_width, curr.m_width, div),
          glm::mix(prev.m_theta, curr.m_theta, div), curr.m_isLeaf,
          prev.m_surfacePush * glm::pow((1.0f - div), 2.0f) +
              curr.m_surfacePush * 1.0f - glm::pow((1.0f - div), 2.0f),
          1.0f + glm::sin(leftPeriod) * waviness,
          1.0f + glm::sin(rightPeriod) * waviness);
    }
  }

  const int vertexIndex = m_vertices.size();
  Vertex archetype{};
#pragma region Semantic mask color
  auto index = leafState.m_index + 1;
  m_vertexColor = glm::vec4((index % 3) * 0.5f, ((index / 3) % 3) * 0.5f,
                            ((index / 9) % 3) * 0.5f, 1.0f);
#pragma endregion
  archetype.m_color = m_vertexColor;
  const float xStep = 1.0f / sorghumLayer->m_step / 2.0f;
  const float yStemStep = 0.5f / static_cast<float>(stemSegmentCount);
  const float yLeafStep =
      0.5f / (m_segments.size() - static_cast<float>(stemSegmentCount) + 1);
  auto segmentSize = m_segments.size();

  for (int i = 0; i < segmentSize; i++) {
    auto &segment = m_segments.at(i);
    if (i <= segmentSize / 3) {
      archetype.m_color = glm::vec4(1, 0, 0, 1);
    } else if (i <= segmentSize * 2 / 3) {
      archetype.m_color = glm::vec4(0, 1, 0, 1);
    } else {
      archetype.m_color = glm::vec4(0, 0, 1, 1);
    }
    const float angleStep = segment.m_theta / sorghumLayer->m_step;
    const int vertsCount = sorghumLayer->m_step * 2 + 1;
    for (int j = 0; j < vertsCount; j++) {
      const auto position =
          segment.GetPoint((j - sorghumLayer->m_step) * angleStep);
      archetype.m_position = glm::vec3(position.x, position.y, position.z);
      float yPos = (i < stemSegmentCount)
                       ? yStemStep * i
                       : 0.5f + yLeafStep * (i - stemSegmentCount + 1);
      archetype.m_texCoords = glm::vec2(j * xStep, yPos);
      m_vertices.push_back(archetype);
    }
    if (i != 0) {
      for (int j = 0; j < vertsCount - 1; j++) {
        // Down triangle
        m_triangles.emplace_back(vertexIndex + ((i - 1) + 1) * vertsCount + j,
                                 vertexIndex + (i - 1) * vertsCount + j + 1,
                                 vertexIndex + (i - 1) * vertsCount + j);
        // Up triangle
        m_triangles.emplace_back(vertexIndex + (i - 1) * vertsCount + j + 1,
                                 vertexIndex + ((i - 1) + 1) * vertsCount + j,
                                 vertexIndex + ((i - 1) + 1) * vertsCount + j +
                                     1);
      }
    }
  }
}
void Spline::FormLeaf(const ProceduralStemState &stemState,
                      const ProceduralLeafState &leafState, int nodeAmount) {
  m_nodes.clear();
  auto startingPoint = leafState.m_distanceToRoot / stemState.m_length;
  float stemWidth = stemState.m_widthAlongStem.GetValue(startingPoint);
  float backDistance = 0.1f;
  if (startingPoint < 0.2f)
    backDistance = startingPoint / 2.0f;
  float actualStartingPoint = startingPoint - 2.0f * backDistance;
  if (leafState.m_index == 0) {
    actualStartingPoint = 0.0f;
  }
  m_nodes.emplace_back(stemState.GetPoint(actualStartingPoint), 180.0f,
                       stemWidth, 0.0f, -stemState.m_direction, false, 0.0f);
  m_nodes.emplace_back(stemState.GetPoint(startingPoint - backDistance), 180.0f,
                       stemWidth, 0.0f, -stemState.m_direction, false, 0.0f);
  m_nodes.emplace_back(stemState.GetPoint(startingPoint), 90.0f, stemWidth,
                       0.0f, -stemState.m_direction, false, 0.0f);
  glm::vec3 position = stemState.GetPoint(startingPoint);
  m_left = glm::rotate(glm::vec3(1, 0, 0), glm::radians(leafState.m_rollAngle),
                       glm::vec3(0, 1, 0));
  auto initialDirection = glm::rotate(
      glm::vec3(0, 1, 0), glm::radians(leafState.m_branchingAngle), m_left);
  glm::vec3 direction = initialDirection;
  float unitLength = leafState.m_length / nodeAmount;
  for (int i = 0; i < nodeAmount; i++) {
    position += direction * unitLength;
    m_nodes.emplace_back(
        position, glm::max(10.0f, 90.0f - (i + 1) * 30.0f),
        leafState.m_widthAlongLeaf.GetValue((float)i / (nodeAmount - 1)),
        leafState.m_wavinessAlongLeaf.GetValue((float)i / (nodeAmount - 1)),
        -direction, true, 1.0f);
    direction = glm::rotate(
        direction,
        glm::radians(leafState.m_bending.x + i * leafState.m_bending.y),
        m_left);
  }
  GenerateLeafGeometry(stemState, leafState, nodeAmount);
}
void Spline::Copy(const std::shared_ptr<Spline> &target) { *this = *target; }
void Spline::FormStem(const ProceduralStemState &stemState, int nodeAmount) {
  m_nodes.clear();
  float unitLength = stemState.m_length / nodeAmount;
  for (int i = 0; i < nodeAmount; i++) {
    float stemWidth =
        stemState.m_widthAlongStem.GetValue((float)i / (nodeAmount - 1));
    m_nodes.emplace_back(glm::normalize(stemState.m_direction) * unitLength *
                             static_cast<float>(i),
                         180.0f, stemWidth + 0.005, 0.0f, -stemState.m_direction,
                         false, 0.0f);
  }
  m_left =
      glm::rotate(glm::vec3(1, 0, 0), glm::radians(glm::linearRand(0.0f, 0.0f)),
                  stemState.m_direction);
  GenerateStemGeometry(stemState, nodeAmount);
}
void Spline::GenerateStemGeometry(const ProceduralStemState &stemState,
                                  int nodeAmount) {
  auto sorghumLayer = Application::GetLayer<SorghumLayer>();
  if (!sorghumLayer)
    return;

  m_vertices.clear();
  m_triangles.clear();
  m_segments.clear();

  for (int i = 1; i < m_nodes.size(); i++) {
    auto &prev = m_nodes.at(i - 1);
    auto &curr = m_nodes.at(i);
    float distance = glm::distance(prev.m_position, curr.m_position);
    BezierCurve curve = BezierCurve(
        prev.m_position, prev.m_position + distance / 5.0f * prev.m_axis,
        curr.m_position - distance / 5.0f * curr.m_axis, curr.m_position);
    for (float div = (i == 1 ? 0.0f
                             : 1.0f / static_cast<float>(
                                          sorghumLayer->m_segmentAmount));
         div <= 1.0f;
         div += 1.0f / static_cast<float>(sorghumLayer->m_segmentAmount)) {
      auto front = prev.m_axis * (1.0f - div) + curr.m_axis * div;
      auto up = glm::normalize(glm::cross(m_left, front));
      m_segments.emplace_back(
          curve.GetPoint(div), up, front,
          prev.m_width * (1.0f - div) + curr.m_width * div,
          prev.m_theta * (1.0f - div) + curr.m_theta * div, curr.m_isLeaf,
          prev.m_surfacePush * glm::pow((1.0f - div), 2.0f) +
              curr.m_surfacePush * 1.0f - glm::pow((1.0f - div), 2.0f),
          1.0f, 1.0f);
    }
  }
  int stemSegmentCount = m_segments.size();
  const int vertexIndex = m_vertices.size();
  Vertex archetype{};
  m_vertexColor = glm::vec4(0, 0, 0, 1);
  archetype.m_color = m_vertexColor;
  const float xStep = 1.0f / sorghumLayer->m_step / 2.0f;
  const float yStemStep = 0.5f / static_cast<float>(stemSegmentCount);
  const float yLeafStep =
      0.5f / (m_segments.size() - static_cast<float>(stemSegmentCount) + 1);
  auto segmentSize = m_segments.size();

  for (int i = 0; i < segmentSize; i++) {
    auto &segment = m_segments.at(i);
    if (i <= segmentSize / 3) {
      archetype.m_color = glm::vec4(1, 0, 0, 1);
    } else if (i <= segmentSize * 2 / 3) {
      archetype.m_color = glm::vec4(0, 1, 0, 1);
    } else {
      archetype.m_color = glm::vec4(0, 0, 1, 1);
    }
    const float angleStep = segment.m_theta / sorghumLayer->m_step;
    const int vertsCount = sorghumLayer->m_step * 2 + 1;
    for (int j = 0; j < vertsCount; j++) {
      const auto position =
          segment.GetPoint((j - sorghumLayer->m_step) * angleStep);
      archetype.m_position = glm::vec3(position.x, position.y, position.z);
      float yPos = (i < stemSegmentCount)
                       ? yStemStep * i
                       : 0.5f + yLeafStep * (i - stemSegmentCount + 1);
      archetype.m_texCoords = glm::vec2(j * xStep, yPos);
      m_vertices.push_back(archetype);
    }
    if (i != 0) {
      for (int j = 0; j < vertsCount - 1; j++) {
        // Down triangle
        m_triangles.emplace_back(vertexIndex + ((i - 1) + 1) * vertsCount + j,
                                 vertexIndex + (i - 1) * vertsCount + j + 1,
                                 vertexIndex + (i - 1) * vertsCount + j);
        // Up triangle
        m_triangles.emplace_back(vertexIndex + (i - 1) * vertsCount + j + 1,
                                 vertexIndex + ((i - 1) + 1) * vertsCount + j,
                                 vertexIndex + ((i - 1) + 1) * vertsCount + j +
                                     1);
      }
    }
  }
}

SplineNode::SplineNode() {}
SplineNode::SplineNode(glm::vec3 position, float angle, float width,
                       float waviness, glm::vec3 axis, bool isLeaf,
                       float surfacePush) {
  m_position = position;
  m_theta = angle;
  m_width = width;
  m_waviness = waviness;
  m_axis = axis;
  m_isLeaf = isLeaf;
  m_surfacePush = surfacePush;
}
