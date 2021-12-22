//
// Created by lllll on 9/16/2021.
//

#include "SorghumField.hpp"
#include "SorghumData.hpp"
#include "SorghumLayer.hpp"
#include "SorghumProceduralDescriptor.hpp"
#include "TransformLayer.hpp"
#include <SorghumField.hpp>
using namespace SorghumFactory;
void RectangularSorghumFieldPattern::GenerateField(
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
void SorghumField::OnInspect() {
  ImGui::DragInt("Size limit", &m_sizeLimit, 1, 0, 10000);
  ImGui::DragFloat("Sorghum size", &m_sorghumSize, 0.01f, 0, 10);
  if (ImGui::Button("Refresh matrices")) {
    GenerateMatrices();
  }
  if (ImGui::Button("Instantiate")) {
    InstantiateField(false);
  }
  if (ImGui::Button("Instantiate (mask)")) {
    InstantiateField(true);
  }

  ImGui::Text("Matrices count: %d", (int)m_newSorghums.size());
}
void SorghumField::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_sizeLimit" << YAML::Value << m_sizeLimit;
  out << YAML::Key << "m_sorghumSize" << YAML::Value << m_sorghumSize;
  out << YAML::Key << "m_newSorghums" << YAML::Value << YAML::BeginSeq;
  for (auto &i : m_newSorghums) {
    out << YAML::BeginMap;
    i.first.Save("SPD", out);
    out << YAML::Key << "Transform" << YAML::Value << i.second;
    out << YAML::EndMap;
  }
  out << YAML::EndSeq;
}
void SorghumField::Deserialize(const YAML::Node &in) {
  if (in["m_sizeLimit"])
    m_sizeLimit = in["m_sizeLimit"].as<int>();
  if (in["m_sorghumSize"])
    m_sorghumSize = in["m_sorghumSize"].as<float>();
  m_newSorghums.clear();
  if (in["m_newSorghums"]) {
    for (const auto &i : in["m_newSorghums"]) {
      AssetRef spd;
      spd.Load("SPD", i);
      m_newSorghums.emplace_back(spd, i["Transform"].as<glm::mat4>());
    }
  }
}
void SorghumField::CollectAssetRef(std::vector<AssetRef> &list) {
  for (auto &i : m_newSorghums) {
    list.push_back(i.first);
  }
}
void SorghumField::InstantiateField(bool semanticMask) {
  if (m_newSorghums.empty())
    GenerateMatrices();
  if (m_newSorghums.empty()) {
    UNIENGINE_ERROR("No matrices generated!");
    return;
  }
  auto sorghumLayer = Application::GetLayer<SorghumLayer>();
  if (sorghumLayer) {
    auto fieldAsset = AssetManager::Get<SorghumField>(GetHandle());
    auto field =
        EntityManager::CreateEntity(EntityManager::GetCurrentScene(), "Field");
    // Create sorghums here.
    int size = 0;
    for (auto &newSorghum : fieldAsset->m_newSorghums) {
      Entity sorghumEntity = sorghumLayer->CreateSorghum();
      auto sorghumTransform = sorghumEntity.GetDataComponent<Transform>();
      sorghumTransform.m_value = newSorghum.second;
      sorghumTransform.SetScale(glm::vec3(m_sorghumSize));
      sorghumEntity.SetDataComponent(sorghumTransform);
      auto sorghumData =
          sorghumEntity.GetOrSetPrivateComponent<SorghumData>().lock();
      sorghumData->m_parameters = newSorghum.first;
      sorghumData->ApplyParameters();
      if (semanticMask)
        sorghumData->GenerateGeometrySeperated(semanticMask);
      else
        sorghumData->GenerateGeometry(true);
      sorghumEntity.SetParent(field);
      size++;
      if (size >= m_sizeLimit)
        break;
    }

    Application::GetLayer<TransformLayer>()
        ->CalculateTransformGraphForDescendents(
            EntityManager::GetCurrentScene(), field);
    field.SetStatic(true);
  } else {
    UNIENGINE_ERROR("No sorghum layer!");
  }
}

void RectangularSorghumField::GenerateMatrices() {
  if (!m_spd.Get<SorghumProceduralDescriptor>())
    return;
  m_newSorghums.clear();
  for (int xi = 0; xi < m_size.x; xi++) {
    for (int yi = 0; yi < m_size.y; yi++) {
      auto position =
          glm::gaussRand(glm::vec3(0.0f), glm::vec3(m_distanceVariance.x, 0.0f,
                                                    m_distanceVariance.y)) +
          glm::vec3(xi * m_distance.x, 0.0f, yi * m_distance.y);
      auto rotation = glm::quat(glm::radians(
          glm::vec3(glm::gaussRand(glm::vec3(0.0f), m_rotationVariance))));
      m_newSorghums.emplace_back(m_spd, glm::translate(position) *
                                            glm::mat4_cast(rotation) *
                                            glm::scale(glm::vec3(1.0f)));
    }
  }
}
void RectangularSorghumField::OnInspect() {
  SorghumField::OnInspect();
  EditorManager::DragAndDropButton<SorghumProceduralDescriptor>(m_spd, "SPD");
  ImGui::DragFloat4("Distance mean/var", &m_distance.x, 0.01f);
  ImGui::DragFloat3("Rotation variance", &m_rotationVariance.x, 0.01f, 0.0f,
                    180.0f);
  ImGui::DragInt2("Size", &m_size.x, 1, 0, 3);
}
void RectangularSorghumField::Serialize(YAML::Emitter &out) {
  m_spd.Save("SPD", out);

  out << YAML::Key << "m_distance" << YAML::Value << m_distance;
  out << YAML::Key << "m_distanceVariance" << YAML::Value << m_distanceVariance;
  out << YAML::Key << "m_rotationVariance" << YAML::Value << m_rotationVariance;
  out << YAML::Key << "m_size" << YAML::Value << m_size;

  SorghumField::Serialize(out);
}
void RectangularSorghumField::Deserialize(const YAML::Node &in) {
  m_spd.Load("SPD", in);

  m_distance = in["m_distance"].as<glm::vec2>();
  m_distanceVariance = in["m_distanceVariance"].as<glm::vec2>();
  m_rotationVariance = in["m_rotationVariance"].as<glm::vec3>();
  m_size = in["m_size"].as<glm::vec2>();

  SorghumField::Deserialize(in);
}
void RectangularSorghumField::CollectAssetRef(std::vector<AssetRef> &list) {
  SorghumField::CollectAssetRef(list);
  list.push_back(m_spd);
}

void PositionsField::GenerateMatrices() {
  if (!m_spd.Get<SorghumProceduralDescriptor>())
    return;
  m_newSorghums.clear();
  for (auto & position : m_positions) {
    if(position.x < m_sampleX.x || position.y < m_sampleY.x || position.x > m_sampleX.y || position.y > m_sampleY.y) continue;
    auto pos = glm::vec3(position.x - m_sampleX.x, 0, position.y - m_sampleY.x) * m_factor;
    auto rotation = glm::quat(glm::radians(
        glm::vec3(glm::gaussRand(glm::vec3(0.0f), m_rotationVariance))));
    m_newSorghums.emplace_back(m_spd, glm::translate(pos) *
                                          glm::mat4_cast(rotation) *
                                          glm::scale(glm::vec3(1.0f)));
  }
}
void PositionsField::OnInspect() {
  SorghumField::OnInspect();
  EditorManager::DragAndDropButton<SorghumProceduralDescriptor>(m_spd, "SPD");
  ImGui::Text("Available count: %d", m_positions.size());
  ImGui::DragFloat("Distance factor", &m_factor, 0.01f, 0.0f, 20.0f);
  ImGui::DragFloat3("Rotation variance", &m_rotationVariance.x, 0.01f, 0.0f,
                    180.0f);

  ImGui::Text("X range: [%.3f, %.3f]", m_xRange.x, m_xRange.y);
  ImGui::Text("Y Range: [%.3f, %.3f]", m_yRange.x, m_yRange.y);

  if (ImGui::DragScalarN("Width range", ImGuiDataType_Double, &m_sampleX.x, 2, 0.1f)) {
    m_sampleX.x = glm::min(m_sampleX.x, m_sampleX.y);
    m_sampleX.y = glm::max(m_sampleX.x, m_sampleX.y);
  }
  if (ImGui::DragScalarN("Length Range", ImGuiDataType_Double, &m_sampleY.x, 2, 0.1f)) {
    m_sampleY.x = glm::min(m_sampleY.x, m_sampleY.y);
    m_sampleY.y = glm::max(m_sampleY.x, m_sampleY.y);
  }
  FileUtils::OpenFile(
      "Load Positions", "Position list", {".txt"},
      [this](const std::filesystem::path &path) { ImportFromFile(path); },
      false);
}
void PositionsField::Serialize(YAML::Emitter &out) {
  m_spd.Save("SPD", out);
  out << YAML::Key << "m_rotationVariance" << YAML::Value << m_rotationVariance;
  out << YAML::Key << "m_sampleX" << YAML::Value << m_sampleX;
  out << YAML::Key << "m_sampleY" << YAML::Value << m_sampleY;
  out << YAML::Key << "m_xRange" << YAML::Value << m_xRange;
  out << YAML::Key << "m_yRange" << YAML::Value << m_yRange;
  out << YAML::Key << "m_factor" << YAML::Value << m_factor;
  SaveListAsBinary<glm::dvec2>("m_positions", m_positions, out);
  SorghumField::Serialize(out);
}
void PositionsField::Deserialize(const YAML::Node &in) {
  m_spd.Load("SPD", in);
  m_rotationVariance = in["m_rotationVariance"].as<glm::vec3>();
  if(in["m_sampleX"]) m_sampleX = in["m_sampleX"].as<glm::dvec2>();
  if(in["m_sampleY"]) m_sampleY = in["m_sampleY"].as<glm::dvec2>();
  if(in["m_xRange"]) m_xRange = in["m_xRange"].as<glm::dvec2>();
  if(in["m_yRange"]) m_yRange = in["m_yRange"].as<glm::dvec2>();
  m_factor = in["m_factor"].as<float>();
  LoadListFromBinary<glm::dvec2>("m_positions", m_positions, in);
  SorghumField::Deserialize(in);
}
void PositionsField::CollectAssetRef(std::vector<AssetRef> &list) {
  SorghumField::CollectAssetRef(list);
  list.push_back(m_spd);
}
void PositionsField::ImportFromFile(const std::filesystem::path &path) {
  std::ifstream ifs;
  ifs.open(path.c_str());
  UNIENGINE_LOG("Loading from " + path.string());
  if (ifs.is_open()) {
    int amount;
    ifs >> amount;
    m_positions.resize(amount);
    m_xRange = glm::vec2(99999999, -99999999);
    m_yRange = glm::vec2(99999999, -99999999);
    for (auto &position : m_positions) {
      ifs >> position.x >> position.y;
      m_xRange.x = glm::min(position.x, m_xRange.x);
      m_xRange.y = glm::max(position.x, m_xRange.y);
      m_yRange.x = glm::min(position.y, m_yRange.x);
      m_yRange.y = glm::max(position.y, m_yRange.y);
    }
  }
}
