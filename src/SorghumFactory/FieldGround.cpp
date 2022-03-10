//
// Created by lllll on 3/3/2022.
//

#include "FieldGround.hpp"
#include "glm/gtc/noise.hpp"
void SorghumFactory::FieldGround::GenerateMesh(float overrideDepth) {
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
  auto meshRenderer =
      GetOwner().GetOrSetPrivateComponent<MeshRenderer>().lock();
  auto mesh = AssetManager::CreateAsset<Mesh>();
  auto material = AssetManager::CreateAsset<Material>();
  mesh->SetVertices(17, vertices, triangles);
  meshRenderer->m_mesh = mesh;
  meshRenderer->m_material = material;
}
void SorghumFactory::FieldGround::OnInspect() {
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
void SorghumFactory::FieldGround::Serialize(YAML::Emitter &out) {}
void SorghumFactory::FieldGround::Deserialize(const YAML::Node &in) {}
void SorghumFactory::FieldGround::OnCreate() {
  m_scale = glm::vec2(0.02f);
  m_size = glm::ivec2(150);
  m_rowWidth = 8.25f;
  m_alleyDepth = 0.15f;

  m_noiseScale = 5.0f;
  m_noiseIntensity = 0.025f;
}
