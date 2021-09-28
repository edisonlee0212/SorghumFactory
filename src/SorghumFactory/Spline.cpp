#include "Spline.hpp"
using namespace SorghumFactory;

void Spline::OnInspect() {

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
    ImGui::Text(("Order: " + std::to_string(m_order)).c_str());
    ImGui::Text(("Unit length: " + std::to_string(m_unitLength)).c_str());
    ImGui::Text(("Unit amount: " + std::to_string(m_unitAmount)).c_str());
    ImGui::Text(("Bending: " + std::to_string(m_gravitropism)).c_str());
    ImGui::Text(
        ("Bending increase: " + std::to_string(m_gravitropismFactor)).c_str());
    ImGui::Text(("Direction: [" + std::to_string(m_initialDirection.x) + ", " +
                 std::to_string(m_initialDirection.y) + ", " +
                 std::to_string(m_initialDirection.z) + "]")
                    .c_str());
    ImGui::Text(("Stem width: " + std::to_string(m_stemWidth)).c_str());
    ImGui::Text(("Leaf width: " + std::to_string(m_leafMaxWidth)).c_str());
    ImGui::Text(
        ("Width decrease start: " + std::to_string(m_leafWidthDecreaseStart))
            .c_str());
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
    RenderManager::DrawGizmoMeshInstanced(
        DefaultResources::Primitives::Sphere, renderColor, matrices,
        GetOwner().GetDataComponent<GlobalTransform>().m_value, nodeSize);
  }
}

void Spline::Clone(const std::shared_ptr<IPrivateComponent> &target) {
  *this = *std::static_pointer_cast<Spline>(target);
}
void Spline::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_type" << YAML::Value << (unsigned)m_type;
  out << YAML::Key << "m_left" << YAML::Value << m_left;
  out << YAML::Key << "m_startingPoint" << YAML::Value << m_startingPoint;

  out << YAML::Key << "m_wavinessPeriod" << YAML::Value << m_wavinessPeriod;
  out << YAML::Key << "m_waviness" << YAML::Value << m_waviness;
  out << YAML::Key << "m_wavinessFactor" << YAML::Value << m_wavinessFactor;

  out << YAML::Key << "m_order" << YAML::Value << m_order;
  out << YAML::Key << "m_unitLength" << YAML::Value << m_unitLength;
  out << YAML::Key << "m_unitAmount" << YAML::Value << m_unitAmount;
  out << YAML::Key << "m_gravitropism" << YAML::Value << m_gravitropism;
  out << YAML::Key << "m_gravitropismFactor" << YAML::Value
      << m_gravitropismFactor;
  out << YAML::Key << "m_initialDirection" << YAML::Value << m_initialDirection;
  out << YAML::Key << "m_stemWidth" << YAML::Value << m_stemWidth;
  out << YAML::Key << "m_leafMaxWidth" << YAML::Value << m_leafMaxWidth;
  out << YAML::Key << "m_leafWidthDecreaseStart" << YAML::Value
      << m_leafWidthDecreaseStart;
  out << YAML::Key << "m_segmentAmount" << YAML::Value << m_segmentAmount;
  out << YAML::Key << "m_step" << YAML::Value << m_step;

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

  m_type = (SplineType)in["m_type"].as<unsigned>();
  m_left = in["m_left"].as<glm::vec3>();
  m_startingPoint = in["m_startingPoint"].as<float>();

  if(in["m_wavinessPeriod"]) m_wavinessPeriod = in["m_wavinessPeriod"].as<float>();
  if(in["m_waviness"]) m_waviness = in["m_waviness"].as<float>();
  if(in["m_wavinessFactor"]) m_wavinessFactor = in["m_wavinessFactor"].as<float>();

  m_order = in["m_order"].as<int>();
  m_unitLength = in["m_unitLength"].as<float>();
  m_unitAmount = in["m_unitAmount"].as<int>();
  m_gravitropism = in["m_gravitropism"].as<float>();
  m_gravitropismFactor = in["m_gravitropismFactor"].as<float>();
  m_initialDirection = in["m_initialDirection"].as<glm::vec3>();
  m_stemWidth = in["m_stemWidth"].as<float>();
  m_leafMaxWidth = in["m_leafMaxWidth"].as<float>();
  m_leafWidthDecreaseStart = in["m_leafWidthDecreaseStart"].as<float>();
  if (in["m_curves"]) {
    m_curves.clear();
    for (const auto &i : in["m_curves"]) {
      m_curves.push_back(
          BezierCurve(i["m_p0"].as<glm::vec3>(), i["m_p1"].as<glm::vec3>(),
                      i["m_p2"].as<glm::vec3>(), i["m_p3"].as<glm::vec3>()));
    }
  }
  m_segmentAmount = in["m_segmentAmount"].as<float>();
  m_step = in["m_step"].as<float>();

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
  float leftFlatness = m_waviness;              // glm::linearRand(0.5f, 2.0f);
  float rightFlatness = m_waviness;             // glm::linearRand(0.5f, 2.0f);
  float leftFlatnessFactor = m_wavinessFactor;  // glm::linearRand(1.0f, 2.5f);
  float rightFlatnessFactor = m_wavinessFactor; // glm::linearRand(1.0f, 2.5f);

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
        leftPeriod += m_wavinessPeriod / static_cast<float>(m_segmentAmount);
        rightPeriod += m_wavinessPeriod / static_cast<float>(m_segmentAmount);
      }
      m_segments.emplace_back(
          curve.GetPoint(div), up, front,
          prev.m_width * (1.0f - div) + curr.m_width * div,
          prev.m_theta * (1.0f - div) + curr.m_theta * div, curr.m_isLeaf,
          prev.m_surfacePush * glm::pow((1.0f - div), 2.0f) +
              curr.m_surfacePush * 1.0f - glm::pow((1.0f - div), 2.0f),
          1.0f + glm::sin(leftPeriod) * leftFlatness,
          1.0f + glm::sin(rightPeriod) * rightFlatness);
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
      float width = m_stemWidth;
      for (float i = glm::max(0.0f, m_startingPoint - 0.3f);
           i < m_startingPoint - 0.05f; i += 0.05f) {
        m_nodes.emplace_back(stemSpline->EvaluatePoint(i), 180.0f, width,
                             stemSpline->EvaluateAxis(i), false, 0.0f);
      }

      float w = m_leafMaxWidth;
      float lengthDecrease =
          (m_leafMaxWidth - 0.02f) /
          (m_unitAmount - m_leafWidthDecreaseStart * m_unitAmount);
      for (int i = 0; i < m_unitAmount; i++) {
        if (i > m_leafWidthDecreaseStart * m_unitAmount)
          w -= lengthDecrease;
        float factor = (float)i / m_unitAmount;
        m_nodes.emplace_back(EvaluatePoint(factor),
                             factor <= 0.05f ? 60.0f : 10.0f, w,
                             EvaluateAxis(factor), true, 1.0f);
      }

      for (float i = 0.05f; i <= 1.0f; i += 0.05f) {
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
      float width = m_stemWidth;
      float backDistance = 0.1f;
      if (m_startingPoint < 0.2f)
        backDistance = m_startingPoint / 2.0f;
      float startingPoint = m_startingPoint - 2.0f * backDistance;
      if (m_order == 0) {
        startingPoint = 0.0f;
      }
      m_nodes.emplace_back(stemSpline->EvaluatePoint(startingPoint), 180.0f,
                           width, -stemSpline->EvaluateAxis(startingPoint),
                           false, 0.0f);

      m_nodes.emplace_back(
          stemSpline->EvaluatePoint(startingPoint + 0.01), 180.0f, width,
          -stemSpline->EvaluateAxis(startingPoint), false, 0.0f);
      m_nodes.emplace_back(
          stemSpline->EvaluatePoint(m_startingPoint - backDistance), 180.0f,
          width, -stemSpline->EvaluateAxis(m_startingPoint - backDistance),
          false, 0.0f);
      m_nodes.emplace_back(stemSpline->EvaluatePoint(m_startingPoint), 90.0f,
                           width, -stemSpline->EvaluateAxis(m_startingPoint),
                           false, 0.0f);
      stemNodeCount = m_nodes.size();
      glm::vec3 position = stemSpline->EvaluatePoint(m_startingPoint);
      glm::vec3 direction = m_initialDirection;
      float w = m_leafMaxWidth;
      float lengthDecrease =
          (m_leafMaxWidth - 0.02f) /
          (m_unitAmount - m_leafWidthDecreaseStart * m_unitAmount);
      for (int i = 0; i < m_unitAmount; i++) {
        position += direction * m_unitLength;
        if (i > m_leafWidthDecreaseStart * m_unitAmount)
          w -= lengthDecrease;
        m_nodes.emplace_back(position, glm::max(10.0f, 90.0f - (i + 1) * 30.0f),
                             w, -direction, true, 1.0f);
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
void Spline::Copy(const std::shared_ptr<Spline> &target) { *this = *target; }
SplineNode::SplineNode(glm::vec3 position, float angle, float width,
                       glm::vec3 axis, bool isLeaf, float surfacePush) {
  m_position = position;
  m_theta = angle;
  m_width = width;
  m_axis = axis;
  m_isLeaf = isLeaf;
  m_surfacePush = surfacePush;
}
SplineNode::SplineNode() {}
