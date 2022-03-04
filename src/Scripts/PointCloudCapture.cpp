//
// Created by lllll on 1/1/2022.
//
#include "FieldGround.hpp"
#include "PointCloudCapture.hpp"
void Scripts::PointCloudCapture::OnBeforeGrowth(
    Scripts::AutoSorghumGenerationPipeline &pipeline) {
  auto positionsField = m_positionsField.Get<PositionsField>();
  if (!positionsField) {
    UNIENGINE_ERROR("Invalid position field");
    Reset(pipeline);
    return;
  }
  auto fieldGround = m_ground.GetOrSetPrivateComponent<FieldGround>().lock();
  fieldGround->GenerateMesh(glm::linearRand(0.12f, 0.17f) );
  auto result = positionsField->InstantiateAroundIndex(pipeline.m_currentIndex % positionsField->m_positions.size(), 2.5f);
  pipeline.m_currentGrowingSorghum = result.first;
  m_currentSorghumField = result.second;
  if (!pipeline.m_currentGrowingSorghum.IsValid() ||
      !m_currentSorghumField.IsValid()) {
    Reset(pipeline);
    UNIENGINE_ERROR("Invalid sorghum/field");
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
      pipeline.m_currentGrowingSorghum, m_currentSorghumField,
      std::filesystem::absolute(
          ProjectManager::GetProjectPath().parent_path().parent_path() /
          m_currentExportFolder / m_name / "PointCloud" /
          (std::to_string(pipeline.m_currentIndex) + std::string(".ply"))),
      m_settings);

  Entities::DeleteEntity(Entities::GetCurrentScene(), m_currentSorghumField);
  pipeline.m_currentGrowingSorghum = m_currentSorghumField = Entity();
  pipeline.m_currentIndex++;
  pipeline.m_status = AutoSorghumGenerationPipelineStatus::BeforeGrowth;
}
void Scripts::PointCloudCapture::OnInspect() {
  Editor::DragAndDropButton<PositionsField>(m_positionsField, "Position Field");
  m_settings.OnInspect();
  if (m_positionsField.Get<PositionsField>()) {
    if (ImGui::Button("Instantiate pipeline")) {
      Instantiate();
    }
  } else {
    ImGui::Text("PositionsField Missing!");
  }
}

void Scripts::PointCloudCapture::Start(
    Scripts::AutoSorghumGenerationPipeline &pipeline) {
  std::filesystem::create_directories(
      std::filesystem::absolute(ProjectManager::GetProjectPath().parent_path().parent_path() /
                                m_currentExportFolder / m_name / "PointCloud"));
  m_ground = Entities::CreateEntity(Entities::GetCurrentScene(), "Ground");
  auto fieldGround = m_ground.GetOrSetPrivateComponent<FieldGround>().lock();
  m_settings.m_ground = m_ground;
}
void Scripts::PointCloudCapture::End(
    Scripts::AutoSorghumGenerationPipeline &pipeline) {
  if (m_currentSorghumField.IsValid())
    Entities::DeleteEntity(Entities::GetCurrentScene(), m_currentSorghumField);
  pipeline.m_currentGrowingSorghum = m_currentSorghumField = {};

  Entities::DeleteEntity(Entities::GetCurrentScene(), m_ground);
}
bool Scripts::PointCloudCapture::IsReady() {
  return m_positionsField.Get<PositionsField>().get();
}
void Scripts::PointCloudCapture::Reset(
    Scripts::AutoSorghumGenerationPipeline &pipeline) {
  if (m_currentSorghumField.IsValid())
    Entities::DeleteEntity(Entities::GetCurrentScene(), m_currentSorghumField);
  pipeline.m_currentGrowingSorghum = m_currentSorghumField = Entity();
}
void Scripts::PointCloudCapture::Instantiate() {
  auto pointCloudCaptureEntity = Entities::CreateEntity(
      Entities::GetCurrentScene(), "PointCloudPipeline");
  auto pointCloudPipeline =
      pointCloudCaptureEntity
          .GetOrSetPrivateComponent<AutoSorghumGenerationPipeline>()
          .lock();
  pointCloudPipeline->m_pipelineBehaviour = AssetManager::Get<PointCloudCapture>(GetHandle());
}
void Scripts::PointCloudCapture::CollectAssetRef(std::vector<AssetRef> &list) {
  list.push_back(m_positionsField);
}
void Scripts::PointCloudCapture::Serialize(YAML::Emitter &out) {
  m_positionsField.Save("m_positionsField", out);
  out << YAML::Key << "m_currentExportFolder" << YAML::Value << m_currentExportFolder.string();
  m_settings.Serialize("m_settings", out);
}
void Scripts::PointCloudCapture::Deserialize(const YAML::Node &in) {
  m_positionsField.Load("m_positionsField", in);
  if(in["m_currentExportFolder"]) m_currentExportFolder = in["m_currentExportFolder"].as<std::string>();
  m_settings.Deserialize("m_settings", in);
}
