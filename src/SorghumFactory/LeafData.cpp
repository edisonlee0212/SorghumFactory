//
// Created by lllll on 3/13/2022.
//

#include "LeafData.hpp"
#include "SorghumLayer.hpp"
using namespace SorghumFactory;
void LeafData::OnInspect() {
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
void LeafData::OnDestroy() {
  m_curves.clear();
  m_nodes.clear();
  m_segments.clear();
  m_vertices.clear();
  m_triangles.clear();
}
void LeafData::Serialize(YAML::Emitter &out) {
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
void LeafData::Deserialize(const YAML::Node &in) {
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
void LeafData::GenerateLeafGeometry(const SorghumStatePair &sorghumStatePair) {
  auto sorghumLayer = Application::GetLayer<SorghumLayer>();
  if (!sorghumLayer)
    return;
  auto leafIndex = GetOwner().GetDataComponent<LeafTag>().m_index;

  int previousLeafSize = sorghumStatePair.m_left.m_leaves.size();
  int nextLeafSize = sorghumStatePair.m_right.m_leaves.size();
  int completedLeafSize = sorghumStatePair.m_left.m_leaves.size() + glm::floor((sorghumStatePair.m_right.m_leaves.size() - sorghumStatePair.m_left.m_leaves.size()) * sorghumStatePair.m_a);
  ProceduralLeafState actualLeft, actualRight;
  float actualA = previousLeafSize == nextLeafSize ? sorghumStatePair.m_a : sorghumStatePair.m_a * (nextLeafSize - previousLeafSize) -
                                                                                (completedLeafSize - previousLeafSize);
  actualRight = sorghumStatePair.m_right.m_leaves[leafIndex];
  if (sorghumStatePair.GetLeafSize() == previousLeafSize) {
    actualLeft = sorghumStatePair.m_left.m_leaves[leafIndex];
  } else if(leafIndex >= completedLeafSize){
    actualLeft = actualRight;
    actualLeft.m_distanceToRoot = actualRight.m_distanceToRoot;
    actualLeft.m_curling = 90.0f;
    actualLeft.m_length = 0.0f;
    actualLeft.m_widthAlongLeaf.m_minValue = actualLeft.m_widthAlongLeaf.m_maxValue = 0.0f;
    actualLeft.m_wavinessAlongLeaf.m_minValue = actualLeft.m_wavinessAlongLeaf.m_maxValue = 0.0f;
  }else{
    actualA = 1.0f;
    actualLeft = actualRight;
  }

  m_vertices.clear();
  m_triangles.clear();
  m_segments.clear();

  float leftFreq = glm::mix(actualLeft.m_wavinessFrequency.x,
                            actualRight.m_wavinessFrequency.x, actualA);
  float rightFreq = glm::mix(actualLeft.m_wavinessFrequency.y,
                             actualRight.m_wavinessFrequency.y, actualA);

  for (int i = 1; i < m_nodes.size(); i++) {
    auto &prev = m_nodes.at(i - 1);
    auto &curr = m_nodes.at(i);
    float distance = glm::distance(prev.m_position, curr.m_position);
    BezierCurve curve = BezierCurve(
        prev.m_position, prev.m_position + distance / 5.0f * prev.m_axis,
        curr.m_position - distance / 5.0f * curr.m_axis, curr.m_position);

    for (float div = (i == 1 ? 0.0f : 0.5f); div <= 1.0f; div += 0.5f) {
      float leftPeriod =
          glm::mix(actualLeft.m_wavinessPeriodStart.x,
                   actualRight.m_wavinessPeriodStart.x, actualA) +
          glm::mix(prev.m_range, curr.m_range, div) * leftFreq;
      float rightPeriod =
          glm::mix(actualLeft.m_wavinessPeriodStart.x,
                   actualRight.m_wavinessPeriodStart.x, actualA) +
          glm::mix(prev.m_range, curr.m_range, div) * rightFreq;

      auto front = prev.m_axis * (1.0f - div) + curr.m_axis * div;
      auto up = glm::normalize(glm::cross(m_left, front));
      auto waviness = glm::mix(prev.m_waviness, curr.m_waviness, div);
      m_segments.emplace_back(
          curve.GetPoint(div), up, front,
          glm::mix(prev.m_width, curr.m_width, div),
          glm::mix(prev.m_theta, curr.m_theta, div), curr.m_isLeaf,
          glm::mix(prev.m_surfacePush, curr.m_surfacePush, div),
          glm::sin(leftPeriod) * waviness, glm::sin(rightPeriod) * waviness);
    }
  }

  const int vertexIndex = m_vertices.size();
  Vertex archetype{};
#pragma region Semantic mask color
  auto index = leafIndex + 1;
  m_vertexColor = glm::vec4((index % 3) * 0.5f, ((index / 3) % 3) * 0.5f,
                            ((index / 9) % 3) * 0.5f, 1.0f);
#pragma endregion
  archetype.m_color = m_vertexColor;

  const float xStep = 1.0f / sorghumLayer->m_horizontalSubdivisionStep / 2.0f;
  auto segmentSize = m_segments.size();
  const float yLeafStep = 0.5f / segmentSize;

  for (int i = 0; i < segmentSize; i++) {
    auto &segment = m_segments.at(i);
    if (i <= segmentSize / 3) {
      archetype.m_color = glm::vec4(1, 0, 0, 1);
    } else if (i <= segmentSize * 2 / 3) {
      archetype.m_color = glm::vec4(0, 1, 0, 1);
    } else {
      archetype.m_color = glm::vec4(0, 0, 1, 1);
    }
    const float angleStep =
        segment.m_theta / sorghumLayer->m_horizontalSubdivisionStep;
    const int vertsCount = sorghumLayer->m_horizontalSubdivisionStep * 2 + 1;
    for (int j = 0; j < vertsCount; j++) {
      const auto position = segment.GetPoint(
          (j - sorghumLayer->m_horizontalSubdivisionStep) * angleStep);
      archetype.m_position = glm::vec3(position.x, position.y, position.z);
      float yPos = 0.5f + yLeafStep * i;
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
void LeafData::FormLeaf(const SorghumStatePair &sorghumStatePair) {
  auto leafIndex = GetOwner().GetDataComponent<LeafTag>().m_index;

  int previousLeafSize = sorghumStatePair.m_left.m_leaves.size();
  int nextLeafSize = sorghumStatePair.m_right.m_leaves.size();
  int completedLeafSize = sorghumStatePair.m_left.m_leaves.size() + glm::floor((sorghumStatePair.m_right.m_leaves.size() - sorghumStatePair.m_left.m_leaves.size()) * sorghumStatePair.m_a);
  ProceduralLeafState actualLeft, actualRight;
  float actualA = previousLeafSize == nextLeafSize ? sorghumStatePair.m_a : sorghumStatePair.m_a * (nextLeafSize - previousLeafSize) -
                                                                                (completedLeafSize - previousLeafSize);
  actualRight = sorghumStatePair.m_right.m_leaves[leafIndex];
  if (sorghumStatePair.GetLeafSize() == previousLeafSize) {
    actualLeft = sorghumStatePair.m_left.m_leaves[leafIndex];
  } else if(leafIndex >= completedLeafSize){
    actualLeft = actualRight;
    actualLeft.m_distanceToRoot = actualRight.m_distanceToRoot;
    actualLeft.m_curling = 90.0f;
    actualLeft.m_length = 0.0f;
    actualLeft.m_widthAlongLeaf.m_minValue = actualLeft.m_widthAlongLeaf.m_maxValue = 0.0f;
    actualLeft.m_wavinessAlongLeaf.m_minValue = actualLeft.m_wavinessAlongLeaf.m_maxValue = 0.0f;
  }else{
    actualA = 1.0f;
    actualLeft = actualRight;
  }

  float stemLength = sorghumStatePair.GetStemLength();
  auto stemDirection = sorghumStatePair.GetStemDirection();
  auto sorghumLayer = Application::GetLayer<SorghumLayer>();
  m_nodes.clear();
  auto startingPoint = glm::mix(actualLeft.m_distanceToRoot,
                                actualRight.m_distanceToRoot, actualA) /
                       stemLength;
  float stemWidth = glm::mix(
      sorghumStatePair.m_left.m_stem.m_widthAlongStem.GetValue(startingPoint),
      sorghumStatePair.m_right.m_stem.m_widthAlongStem.GetValue(startingPoint),
      sorghumStatePair.m_a);
  float backDistance = 0.05f;
  if (startingPoint < backDistance)
    backDistance = startingPoint;
  float sheathPoint = startingPoint - backDistance;

  int nodeForSheath =
      glm::max(2.0f, stemLength * backDistance /
                         sorghumLayer->m_verticalSubdivisionMaxUnitLength);
  for (int i = 0; i <= nodeForSheath; i++) {
    float currentPoint = (float)i / nodeForSheath * backDistance;
    m_nodes.emplace_back(
        sorghumStatePair.GetStemPoint(sheathPoint + currentPoint),
        180.0f - 90.0f * (float)i / nodeForSheath,
        stemWidth + 0.002f * (float)i / nodeForSheath, 0.0f, -stemDirection,
        false, 0.0f, 0.0f);
  }
  glm::vec3 position = sorghumStatePair.GetStemPoint(startingPoint);
  m_left = glm::rotate(glm::vec3(1, 0, 0),
                       glm::radians(glm::mix(actualLeft.m_rollAngle,
                                             actualRight.m_rollAngle, actualA)),
                       glm::vec3(0, 1, 0));
  auto initialDirection =
      glm::rotate(glm::vec3(0, 1, 0),
                  glm::radians(glm::mix(actualLeft.m_branchingAngle,
                                        actualRight.m_branchingAngle, actualA)),
                  m_left);
  glm::vec3 direction = initialDirection;
  auto leafLength =
      glm::mix(actualLeft.m_length, actualRight.m_length, actualA);
  auto leafBending =
      glm::mix(actualLeft.m_bending, actualRight.m_bending, actualA);

  int nodeAmount = glm::max(
      4.0f, leafLength / sorghumLayer->m_verticalSubdivisionMaxUnitLength);
  float unitLength = leafLength / nodeAmount;
  float expandAngle =
      glm::mix(actualLeft.m_curling, actualRight.m_curling, actualA);
  int nodeToFullExpand =
      glm::max(2.0f, 0.05f * leafLength /
                         sorghumLayer->m_verticalSubdivisionMaxUnitLength);
  for (int i = 0; i <= nodeAmount; i++) {
    const float factor = (float)i / nodeAmount;
    position += direction * unitLength;
    float collarFactor = glm::min(1.0f, (float)i / nodeToFullExpand);
    float wavinessAlongLeaf =
        glm::mix(actualLeft.m_wavinessAlongLeaf.GetValue(factor),
                 actualRight.m_wavinessAlongLeaf.GetValue(factor), actualA);
    float width = glm::mix(stemWidth + 0.002f, glm::mix(actualLeft.m_widthAlongLeaf.GetValue(factor),
                                                        actualRight.m_widthAlongLeaf.GetValue(factor), actualA), collarFactor);
    float angle = 90.0f - (90.0f - expandAngle) * glm::pow(collarFactor, 2.0f);
    m_nodes.emplace_back(position, angle, width, wavinessAlongLeaf, -direction,
                         true, 0.0f, factor);
    direction = glm::rotate(
        direction,
        glm::radians(leafBending.x + factor * leafBending.y) / nodeAmount,
        m_left);
  }
  GenerateLeafGeometry(sorghumStatePair);
}
void LeafData::Copy(const std::shared_ptr<LeafData> &target) {
  *this = *target;
}
