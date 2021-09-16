//
// Created by lllll on 9/16/2021.
//

#include "SorghumField.hpp"
#include "SorghumProceduralDescriptor.hpp"
#include "SorghumSystem.hpp"
#include <SorghumField.hpp>

void SorghumFactory::RectangularSorghumFieldPattern::GenerateField(
    std::vector<std::vector<glm::mat4>> &matricesList) {
  const int size = matricesList.size();
  glm::vec2 center = glm::vec2(m_distances.x * (m_size.x - 1),
                               m_distances.y * (m_size.y - 1)) /
                     2.0f;
  for (int xi = 0; xi < m_size.x; xi++) {
    for (int yi = 0; yi < m_size.y; yi++) {
      const auto selectedIndex = glm::linearRand(0, size - 1);
      matricesList[selectedIndex].push_back(
          glm::translate(glm::vec3(xi * m_distances.x - center.x, 0.0f,
                                   yi * m_distances.y - center.y)) *
          glm::mat4_cast(glm::quat(glm::radians(
              glm::vec3(glm::gaussRand(0.0f, m_rotationVariation.x),
                        glm::gaussRand(0.0f, m_rotationVariation.y),
                        glm::gaussRand(0.0f, m_rotationVariation.z))))) *
          glm::scale(glm::vec3(1.0f)));
    }
  }
}
void SorghumFactory::SorghumField::OnInspect() {
  if (ImGui::BeginMenu("Settings")) {
    static float distance = 3;
    static float variance = 0.2;
    static float yAxisVar = 30.0f;
    static float xzAxisVar = 3.0f;
    static int expand = 3;
    if (ImGui::BeginMenu("Create field...")) {
      ImGui::DragFloat("Avg. Y axis rotation", &yAxisVar, 0.01f, 0.0f, 180.0f);
      ImGui::DragFloat("Avg. XZ axis rotation", &xzAxisVar, 0.01f, 0.0f, 90.0f);
      ImGui::DragFloat("Avg. Distance", &distance, 0.01f);
      ImGui::DragFloat("Position variance", &variance, 0.01f);
      ImGui::DragInt("Expand", &expand, 1, 0, 3);
      if (ImGui::Button("Apply")) {
        m_newSorghumAmount = (2 * expand + 1) * (2 * expand + 1);
        m_newSorghumPositions.resize(m_newSorghumAmount);
        m_newSorghumRotations.resize(m_newSorghumAmount);
        const auto currentSize = m_newSorghumParameters.size();
        m_newSorghumParameters.resize(m_newSorghumAmount);
        for (auto i = currentSize; i < m_newSorghumAmount; i++) {
          m_newSorghumParameters[i] = m_newSorghumParameters[0];
        }
        int index = 0;
        for (int i = -expand; i <= expand; i++) {
          for (int j = -expand; j <= expand; j++) {
            glm::vec3 value = glm::vec3(i * distance, 0, j * distance);
            value.x += glm::linearRand(-variance, variance);
            value.z += glm::linearRand(-variance, variance);
            m_newSorghumPositions[index] = value;
            value = glm::vec3(glm::linearRand(-xzAxisVar, xzAxisVar),
                              glm::linearRand(-yAxisVar, yAxisVar),
                              glm::linearRand(-xzAxisVar, xzAxisVar));
            m_newSorghumRotations[index] = value;
            index++;
          }
        }
      }
      ImGui::EndMenu();
    }
    ImGui::InputInt("New sorghum amount", &m_newSorghumAmount);
    if (m_newSorghumAmount < 1)
      m_newSorghumAmount = 1;

    ImGui::EndMenu();
  }
  ImGui::Separator();
  ImGui::Spacing();
  ImGui::Spacing();
  ImGui::Spacing();

  static AssetRef tempSorghumDescriptor;
  EditorManager::DragAndDropButton<SorghumProceduralDescriptor>(
      tempSorghumDescriptor, "Apply to all");
  if (tempSorghumDescriptor.Get<SorghumProceduralDescriptor>()) {
    for (auto &i : m_newSorghumParameters) {
      i = tempSorghumDescriptor;
    }
    tempSorghumDescriptor.Clear();
  }
  if (m_newSorghumPositions.size() < m_newSorghumAmount) {
    const auto currentSize = m_newSorghumPositions.size();
    m_newSorghumParameters.resize(m_newSorghumAmount);
    for (auto i = currentSize; i < m_newSorghumAmount; i++) {
      m_newSorghumParameters[i] = m_newSorghumParameters[0];
    }
    m_newSorghumPositions.resize(m_newSorghumAmount);
    m_newSorghumRotations.resize(m_newSorghumAmount);
  }
  if(ImGui::CollapsingHeader("Details")) {
    for (auto i = 0; i < m_newSorghumAmount; i++) {
      std::string title = "New Sorghum No.";
      title += std::to_string(i);
      if (ImGui::TreeNode(title.c_str())) {
        EditorManager::DragAndDropButton<SorghumProceduralDescriptor>(
            m_newSorghumParameters[i], "Descriptor");
        ImGui::InputFloat3(("Position##" + std::to_string(i)).c_str(),
                           &m_newSorghumPositions[i].x);
        ImGui::TreePop();
      }
    }
  }
}
void SorghumFactory::SorghumField::Serialize(YAML::Emitter &out) {}
void SorghumFactory::SorghumField::Deserialize(const YAML::Node &in) {}
