//
// Created by lllll on 3/3/2022.
//

#include "FieldGround.hpp"
#include "glm/gtc/noise.hpp"

using namespace PlantArchitect;
void FieldGround::GenerateMesh(float overrideDepth) {
  std::vector<Vertex> vertices;
  std::vector<glm::uvec3> triangles;
  Vertex archetype;
  glm::vec3 randomPositionOffset =
      glm::linearRand(glm::vec3(0.0f), glm::vec3(10000.0f));
  for (int i = -m_size.x; i <= m_size.x; i++) {
    for (int j = -m_size.y; j <= m_size.y; j++) {
      archetype.m_position.x = m_scale.x * i;
      archetype.m_position.z = m_scale.y * j;
      archetype.m_position.y =
          glm::min(0.0f, (overrideDepth < 0.0f ? m_alleyDepth : overrideDepth) *
                             glm::cos(archetype.m_position.z * m_rowWidth));

      float noise = glm::simplex(m_noiseScale * archetype.m_position +
                                 randomPositionOffset) *
                    m_noiseIntensity;
      archetype.m_position.y += noise;
      archetype.m_texCoords = glm::vec2((float)i / (2 * m_size.x + 1),
                                        (float)j / (2 * m_size.y + 1));
      vertices.push_back(archetype);
    }
  }

  for (int i = 0; i < 2 * m_size.x; i++) {
    for (int j = 0; j < 2 * m_size.y; j++) {
      int n = 2 * m_size.x + 1;
      triangles.emplace_back(i + j * n, i + 1 + j * n, i + (j + 1) * n);
      triangles.emplace_back(i + 1 + (j + 1) * n, i + (j + 1) * n,
                             i + 1 + j * n);
    }
  }
  auto owner = GetOwner();
  auto scene = GetScene();
  auto meshRenderer =
      scene->GetOrSetPrivateComponent<MeshRenderer>(owner).lock();
  auto mesh = ProjectManager::CreateTemporaryAsset<Mesh>();
  auto material = ProjectManager::CreateTemporaryAsset<Material>();
  mesh->SetVertices(17, vertices, triangles);
  meshRenderer->m_mesh = mesh;
  meshRenderer->m_material = material;
}
void FieldGround::OnInspect() {
  static bool autoRefresh = false;
  ImGui::Checkbox("Auto refresh", &autoRefresh);
  bool changed = ImGui::DragFloat2("Scale", &m_scale.x);
  changed = changed || ImGui::DragInt2("Size", &m_size.x);
  changed = changed || ImGui::DragFloat("Row Width", &m_rowWidth);
  changed = changed || ImGui::DragFloat("Alley Depth", &m_alleyDepth);
  changed = changed || ImGui::DragFloat("Noise Scale", &m_noiseScale);
  changed = changed || ImGui::DragFloat("Noise Intensity", &m_noiseIntensity);

  if (ImGui::Button("Apply") || (changed && autoRefresh)) {
    GenerateMesh();
  }
}
void FieldGround::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_scale" << YAML::Value << m_scale;
  out << YAML::Key << "m_size" << YAML::Value << m_size;
  out << YAML::Key << "m_rowWidth" << YAML::Value << m_rowWidth;
  out << YAML::Key << "m_alleyDepth" << YAML::Value << m_alleyDepth;
  out << YAML::Key << "m_noiseScale" << YAML::Value << m_noiseScale;
  out << YAML::Key << "m_noiseIntensity" << YAML::Value << m_noiseIntensity;
}
void FieldGround::Deserialize(const YAML::Node &in) {
  if (in["m_scale"])
    m_scale = in["m_scale"].as<glm::vec2>();
  if (in["m_size"])
    m_size = in["m_size"].as<glm::ivec2>();
  if (in["m_rowWidth"])
    m_rowWidth = in["m_rowWidth"].as<float>();
  if (in["m_alleyDepth"])
    m_alleyDepth = in["m_alleyDepth"].as<float>();
  if (in["m_noiseScale"])
    m_noiseScale = in["m_noiseScale"].as<float>();
  if (in["m_noiseIntensity"])
    m_noiseIntensity = in["m_noiseIntensity"].as<float>();


}
void FieldGround::OnCreate() {
  m_scale = glm::vec2(0.02f);
  m_size = glm::ivec2(150);
  m_rowWidth = 8.25f;
  m_alleyDepth = 0.15f;

  m_noiseScale = 5.0f;
  m_noiseIntensity = 0.025f;
}
