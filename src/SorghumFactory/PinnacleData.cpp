//
// Created by lllll on 3/13/2022.
//

#include "PinnacleData.hpp"
#include "IVolume.hpp"
using namespace SorghumFactory;
void PinnacleData::OnInspect() {

}
void PinnacleData::OnDestroy() {
  m_vertices.clear();
  m_triangles.clear();
}
void PinnacleData::Serialize(YAML::Emitter &out) {
  ISerializable::Serialize(out);
}
void PinnacleData::Deserialize(const YAML::Node &in) {
  ISerializable::Deserialize(in);
}
void PinnacleData::FormPinnacle(const SorghumStatePair &sorghumStatePair) {
  m_vertices.clear();
  m_triangles.clear();
  auto pinnacleSize = glm::mix(sorghumStatePair.m_left.m_pinnacle.m_pinnacleSize, sorghumStatePair.m_right.m_pinnacle.m_pinnacleSize, sorghumStatePair.m_a);
  auto seedAmount = glm::mix(sorghumStatePair.m_left.m_pinnacle.m_seedAmount, sorghumStatePair.m_right.m_pinnacle.m_seedAmount, sorghumStatePair.m_a);
  auto seedRadius = glm::mix(sorghumStatePair.m_left.m_pinnacle.m_seedRadius, sorghumStatePair.m_right.m_pinnacle.m_seedRadius, sorghumStatePair.m_a);
  std::vector<glm::vec3> icosahedronVertices;
  std::vector<glm::uvec3> icosahedronTriangles;
  SphereMeshGenerator::Icosahedron(icosahedronVertices, icosahedronTriangles);
  int offset = 0;
  UniEngine::Vertex archetype = {};
  SphericalVolume volume;
  volume.m_radius = pinnacleSize;
  for (int seedIndex = 0;
       seedIndex < seedAmount;
       seedIndex++) {
    glm::vec3 positionOffset = volume.GetRandomPoint();
    for (const auto position : icosahedronVertices) {
      archetype.m_position =
          position * seedRadius +
          positionOffset + sorghumStatePair.GetStemPoint(1.0f);
      m_vertices.push_back(archetype);
    }
    for (const auto triangle : icosahedronTriangles) {
      glm::uvec3 actualTriangle = triangle + glm::uvec3(offset);
      m_triangles.push_back(actualTriangle);
    }
    offset += icosahedronVertices.size();
  }
}