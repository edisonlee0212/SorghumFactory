//
// Created by lllll on 1/1/2022.
//

#include "PointCloudCapture.hpp"
void Scripts::PointCloudCapture::OnIdle(
    Scripts::AutoSorghumGenerationPipeline &pipeline) {
  if (m_positionsField.Get<PositionsField>() && m_currentIndex != -1) {
    pipeline.m_status = AutoSorghumGenerationPipelineStatus::BeforeGrowth;
  }
}
void Scripts::PointCloudCapture::OnBeforeGrowth(
    Scripts::AutoSorghumGenerationPipeline &pipeline) {
  auto positionsField = m_positionsField.Get<PositionsField>();
  if (!positionsField) {
    pipeline.m_status = AutoSorghumGenerationPipelineStatus::Idle;
    Reset();
    return;
  }
  auto result = positionsField->InstantiateAroundIndex(m_currentIndex, 2.5f);
  m_currentSorghum = result.first;
  m_currentSorghumField = result.second;
  if (!m_currentSorghum.IsValid() || !m_currentSorghumField.IsValid()) {
    pipeline.m_status = AutoSorghumGenerationPipelineStatus::Idle;
    Reset();
    return;
  }
  pipeline.m_status = AutoSorghumGenerationPipelineStatus::Growth;
}
void Scripts::PointCloudCapture::OnGrowth(
    Scripts::AutoSorghumGenerationPipeline &pipeline) {
  pipeline.m_status = AutoSorghumGenerationPipelineStatus::AfterGrowth;
}
void Scripts::PointCloudCapture::OnAfterGrowth(
    Scripts::AutoSorghumGenerationPipeline &pipeline) {
  Application::GetLayer<SorghumLayer>()->ScanPointCloudLabeled(
      std::filesystem::absolute(
          ProjectManager::GetProjectPath().parent_path() /
          m_currentExportFolder / "PointCloud" /
          (std::to_string(m_currentIndex) + std::string(".ply"))),
      m_currentSorghum, m_boundingBoxRadius, m_boundingBoxHeightRange, m_pointDistance, m_angle);

  Entities::DeleteEntity(Entities::GetCurrentScene(), m_currentSorghumField);
  m_currentSorghum = m_currentSorghumField = Entity();
  m_currentIndex += 1;
  if (m_currentIndex > m_endIndex) {
    Reset();
    pipeline.m_status = AutoSorghumGenerationPipelineStatus::Idle;
    return;
  } else {
    pipeline.m_status = AutoSorghumGenerationPipelineStatus::BeforeGrowth;
  }
}
void Scripts::PointCloudCapture::OnInspect() {
  Editor::DragAndDropButton<PositionsField>(m_positionsField, "Position Field");
  ImGui::DragInt("Start Index", &m_startIndex, 1, 0, m_endIndex);
  ImGui::DragInt("End Index", &m_endIndex, 1, m_startIndex, 99999);
  ImGui::DragFloat2("Point distance", &m_pointDistance.x, 0.0001f);
  ImGui::DragFloat("Scanner angle", &m_angle, 0.5f);
  ImGui::DragFloat("Bounding box radius", &m_boundingBoxRadius, 0.01f);
  ImGui::DragFloat2("Bounding box height range", &m_boundingBoxHeightRange.x, 0.01f);
  if (!Application::IsPlaying()) {
    ImGui::Text("Application not Playing!");
  } else if (!m_positionsField.Get<PositionsField>()) {
    ImGui::Text("Position Field Missing");
  } else if (m_currentIndex != -1) {
    ImGui::Text("Busy...");
  } else {
    if (ImGui::Button("Start")) {
      std::filesystem::create_directories(std::filesystem::absolute(
          ProjectManager::GetProjectPath().parent_path() /
          m_currentExportFolder / "PointCloud"));
      m_currentIndex = 0;
    }
  }
}
void Scripts::PointCloudCapture::Reset() {
  m_currentIndex = -1;
  if (m_currentSorghumField.IsValid())
    Entities::DeleteEntity(Entities::GetCurrentScene(), m_currentSorghumField);
  m_currentSorghum = m_currentSorghumField = Entity();
}
