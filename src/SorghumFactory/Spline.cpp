#include "Spline.hpp"
using namespace SorghumFactory;

void Spline::OnGui() {
  
  if (ImGui::DragInt("Segment amount", &m_segmentAmount)) {
    m_segmentAmount = glm::max(2, m_segmentAmount);
  }
  if (ImGui::DragInt("Step amount", &m_step)) {
    m_step = glm::max(2, m_step);
  }

  switch (m_type) {
  case SplineType::BezierCurve: {
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
  } break;
  case SplineType::Procedural: {
    ImGui::DragFloat("Starting point", &m_startingPoint);
    ImGui::DragFloat("Unit length", &m_unitLength);
    ImGui::DragInt("Unit amount", &m_unitAmount);
    ImGui::DragFloat("Gravitropism", &m_gravitropism);
    ImGui::DragFloat("GravitropismFactor", &m_gravitropismFactor);
    ImGui::InputFloat3("Start Direction", &m_initialDirection.x);
  } break;
  }
  static bool renderNodes = true;
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
    RenderManager::DrawGizmoMeshInstanced(DefaultResources::Primitives::Sphere,
                                          renderColor, matrices,
                                          GetOwner().GetDataComponent<GlobalTransform>().m_value, nodeSize);
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
  m_type = SplineType::BezierCurve;
}

glm::vec3 Spline::EvaluatePoint(float point) {
  switch (m_type) {
  case SplineType::BezierCurve: {
    const float splineU =
        glm::clamp(point, 0.0f, 1.0f) * float(m_curves.size());

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
  case SplineType::Procedural: {
    float totalLength = m_unitLength * m_unitAmount;
    return glm::normalize(m_initialDirection) * point * totalLength;
  }
  }
}

glm::vec3 Spline::EvaluateAxis(float point) {
  switch (m_type) {
  case SplineType::BezierCurve: {
    const float splineU =
        glm::clamp(point, 0.0f, 1.0f) * float(m_curves.size());

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
  case SplineType::Procedural: {
    return m_initialDirection;
  }
  }
  throw 0;
}

void Spline::GenerateGeometry(const std::shared_ptr<Spline> &stemSpline) {

  auto stemNodeCount = FormNodes(stemSpline);

  m_vertices.clear();
  m_indices.clear();
  m_segments.clear();

  float temp = 0.0f;

  float leftPeriod = 0.0f;
  float rightPeriod = 0.0f;
  float leftFlatness = glm::gaussRand(1.75f,
                                      0.5f); // glm::linearRand(0.5f, 2.0f);
  float rightFlatness = glm::gaussRand(1.75f,
                                       0.5f); // glm::linearRand(0.5f, 2.0f);
  float leftFlatnessFactor =
      glm::gaussRand(1.25f,
                     0.2f); // glm::linearRand(1.0f, 2.5f);
  float rightFlatnessFactor =
      glm::gaussRand(1.25f,
                     0.2f); // glm::linearRand(1.0f, 2.5f);

  int stemSegmentCount = 0;
  for (int i = 1; i < m_nodes.size(); i++) {
    auto &prev = m_nodes.at(i - 1);
    auto &curr = m_nodes.at(i);
    if (i == stemNodeCount) {
      stemSegmentCount = m_segments.size();
    }
    float distance = glm::distance(prev.m_position, curr.m_position);
    BezierCurve curve = BezierCurve(
        prev.m_position, prev.m_position + distance / 5.0f * prev.m_axis,
        curr.m_position - distance / 5.0f * curr.m_axis, curr.m_position);
    for (float div = 1.0f / static_cast<float>(m_segmentAmount); div <= 1.0f;
         div += 1.0f / static_cast<float>(m_segmentAmount)) {
      auto front = prev.m_axis * (1.0f - div) + curr.m_axis * div;
      auto up = glm::normalize(glm::cross(m_left, front));
      if (prev.m_isLeaf) {
        leftPeriod += glm::gaussRand(1.25f, 0.5f) / static_cast<float>(m_segmentAmount);
        rightPeriod += glm::gaussRand(1.25f, 0.5f) / static_cast<float>(m_segmentAmount);
      }
      m_segments.emplace_back(curve.GetPoint(div), up, front,
                              prev.m_width * (1.0f - div) + curr.m_width * div,
                              prev.m_theta * (1.0f - div) + curr.m_theta * div,
                              curr.m_isLeaf,
                              prev.m_surfacePush * glm::pow((1.0f - div), 2.0f) + curr.m_surfacePush * 1.0f - glm::pow((1.0f - div), 2.0f),
                              glm::sin(leftPeriod) * leftFlatness,
                              glm::sin(rightPeriod) * rightFlatness,
                              leftFlatnessFactor,
                              rightFlatnessFactor);
    }
  }

  const int vertexIndex = m_vertices.size();
  Vertex archetype;
  const float xStep = 1.0f / m_step / 2.0f;
  const float yStemStep = 0.5f / static_cast<float>(stemSegmentCount);
  const float yLeafStep =
      0.5f / (m_segments.size() - static_cast<float>(stemSegmentCount) + 1);
  for (int i = 0; i < m_segments.size(); i++) {
    auto &segment = m_segments.at(i);
    const float angleStep = segment.m_theta / m_step;
    const int vertsCount = m_step * 2 + 1;
    for (int j = 0; j < vertsCount; j++) {
      const auto position = segment.GetPoint((j - m_step) * angleStep);
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
        m_indices.push_back(vertexIndex + ((i - 1) + 1) * vertsCount + j);
        m_indices.push_back(vertexIndex + (i - 1) * vertsCount + j + 1);
        m_indices.push_back(vertexIndex + (i - 1) * vertsCount + j);
        // Up triangle
        m_indices.push_back(vertexIndex + (i - 1) * vertsCount + j + 1);
        m_indices.push_back(vertexIndex + ((i - 1) + 1) * vertsCount + j);
        m_indices.push_back(vertexIndex + ((i - 1) + 1) * vertsCount + j + 1);
      }
    }
  }
}
int Spline::FormNodes(const std::shared_ptr<Spline> &stemSpline) {
  m_nodes.clear();
  int stemNodeCount = 0;
  switch (m_type) {
  case SplineType::BezierCurve: {
    if (m_startingPoint != -1) {
      float width = 0.1f - m_startingPoint * 0.05f;
      for (float i = glm::max(0.0f, m_startingPoint - 0.3f);
           i < m_startingPoint - 0.05f; i += 0.05f) {
        m_nodes.emplace_back(stemSpline->EvaluatePoint(i), 180.0f, width,
                             stemSpline->EvaluateAxis(i), false, 0.0f);
      }
      stemNodeCount = m_nodes.size();
      for (float i = 0.05f; i <= 1.0f; i += 0.05f) {
        float w = 0.2f;
        if (i > 0.75f)
          w -= (i - 0.75f) * 0.75f;
        m_nodes.emplace_back(EvaluatePoint(i), i == 0.05f ? 60.0f : 10.0f, w,
                             EvaluateAxis(i), true, 1.0f);
      }
    } else {
      for (float i = 0.0f; i <= 1.0f; i += 0.05f) {
        m_nodes.emplace_back(EvaluatePoint(i), 180.0f, 0.04f, EvaluateAxis(i),
                             false, 0.0f);
      }
      auto endPoint = EvaluatePoint(1.0f);
      auto endAxis = EvaluateAxis(1.0f);
      m_nodes.emplace_back(endPoint + endAxis * 0.05f, 10.0f, 0.001f, endAxis,
                           false, 0.0f);
      stemNodeCount = m_nodes.size();
    }
  } break;
  case SplineType::Procedural: {
    if (m_startingPoint != -1) {
      float width = 0.1f - m_startingPoint * 0.05f;
      float backDistance = 0.1f;
      if(m_startingPoint < 0.2f) backDistance = m_startingPoint / 2.0f;
      float startingPoint = m_startingPoint - 2.0f * backDistance;
      if (m_order == 0) {
        startingPoint = 0.0f;
      }
      m_nodes.emplace_back(stemSpline->EvaluatePoint(startingPoint), 180.0f, width,
                           -stemSpline->EvaluateAxis(startingPoint), false, 0.0f);
      m_nodes.emplace_back(stemSpline->EvaluatePoint(m_startingPoint - backDistance), 180.0f, width,
                           -stemSpline->EvaluateAxis(m_startingPoint - backDistance), false, 0.0f);
      m_nodes.emplace_back(stemSpline->EvaluatePoint(m_startingPoint), 90.0f, width,
                           -stemSpline->EvaluateAxis(m_startingPoint), false, 0.0f);
      stemNodeCount = m_nodes.size();
      glm::vec3 position = stemSpline->EvaluatePoint(m_startingPoint);
      glm::vec3 direction = m_initialDirection;
      for (int i = 0; i < m_unitAmount; i++) {
        position += direction * m_unitLength;
        float w = 0.2f;
        if (i > m_unitAmount - 4)
          w -= (i - m_unitAmount + 4) * 0.06f;
        m_nodes.emplace_back(position, glm::max(10.0f, 90.0f - (i + 1) * 30.0f), w,
                             -direction, true, 1.0f);
        direction = glm::rotate(
            direction, glm::radians(m_gravitropism + i * m_gravitropismFactor),
            m_left);
      }
    } else {
      for (int i = 0; i < m_unitAmount; i++) {
        m_nodes.emplace_back(glm::normalize(m_initialDirection) * m_unitLength *
                                 static_cast<float>(i),
                             180.0f, 0.04f, -m_initialDirection, false, 0.0f);
      }
      stemNodeCount = m_nodes.size();
    }
  } break;
  default:
    UNIENGINE_ERROR("Unknown type!");
    break;
  }
  return stemNodeCount;
}
void Spline::Copy(const std::shared_ptr<Spline> &target) {
  m_type = target->m_type;
  m_left = target->m_left;
  m_startingPoint = target->m_startingPoint;
  m_order = target->m_order;
  m_unitAmount = target->m_unitAmount;
  m_unitLength = target->m_unitLength;
  m_gravitropismFactor = target->m_gravitropismFactor;
  m_gravitropism = target->m_gravitropism;
  m_initialDirection = target->m_initialDirection;
  m_curves = target->m_curves;
  m_nodes = target->m_nodes;
  m_vertices = target->m_vertices;
  m_indices = target->m_indices;
}
SplineNode::SplineNode(glm::vec3 position, float angle, float width, glm::vec3 axis,
                       bool isLeaf, float surfacePush) {
  m_position = position;
  m_theta = angle;
  m_width = width;
  m_axis = axis;
  m_isLeaf = isLeaf;
  m_surfacePush = surfacePush;
}