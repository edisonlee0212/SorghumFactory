//
// Created by lllll on 1/1/2022.
//
#include "FieldGround.hpp"
#include "PointCloudCapture.hpp"
using namespace Scripts;
void PointCloudCapture::OnBeforeGrowth(
    AutoSorghumGenerationPipeline &pipeline) {
  auto positionsField = m_positionsField.Get<PositionsField>();
  if (!positionsField) {
    UNIENGINE_ERROR("Invalid position field");
    Reset(pipeline);
    return;
  }
  auto scene = pipeline.GetScene();
  auto fieldGround = scene->GetOrSetPrivateComponent<FieldGround>(m_ground).lock();
  fieldGround->GenerateMesh(glm::linearRand(0.12f, 0.17f) );
  auto result = positionsField->InstantiateAroundIndex(pipeline.GetSeed() % positionsField->m_positions.size(), 2.5f);
  pipeline.m_currentGrowingSorghum = result.first;
  m_currentSorghumField = result.second;
  if (!scene->IsEntityValid(pipeline.m_currentGrowingSorghum) ||
      !scene->IsEntityValid(m_currentSorghumField)) {
    Reset(pipeline);
    UNIENGINE_ERROR("Invalid sorghum/field");
    return;
  }
  pipeline.m_status = AutoSorghumGenerationPipelineStatus::Growth;
}
void PointCloudCapture::OnGrowth(
    AutoSorghumGenerationPipeline &pipeline) {
  pipeline.m_status = AutoSorghumGenerationPipelineStatus::AfterGrowth;
}
void PointCloudCapture::OnAfterGrowth(
    AutoSorghumGenerationPipeline &pipeline) {
  Application::GetLayer<SorghumLayer>()->ScanPointCloudLabeled(
      pipeline.m_currentGrowingSorghum, m_currentSorghumField,
      m_currentExportFolder / GetAssetRecord().lock()->GetAssetFileName() / "PointCloud" /
          (pipeline.m_prefix + std::string(".ply")),
      m_settings);
  auto scene = pipeline.GetScene();
  scene->DeleteEntity(m_currentSorghumField);
  pipeline.m_currentGrowingSorghum = m_currentSorghumField = Entity();
  pipeline.m_status = AutoSorghumGenerationPipelineStatus::Idle;
}
void PointCloudCapture::OnInspect() {
  if (m_positionsField.Get<PositionsField>()) {
    if (ImGui::Button("Instantiate pipeline")) {
      Instantiate();
    }
  } else {
    ImGui::Text("PositionsField Missing!");
  }

  ImGui::Text("Current output folder: %s",
              m_currentExportFolder.string().c_str());
  FileUtils::OpenFolder(
      "Choose output folder...",
      [&](const std::filesystem::path &path) {
        m_currentExportFolder = std::filesystem::absolute(path);
      },
      false);

  Editor::DragAndDropButton<PositionsField>(m_positionsField, "Position Field");
  m_settings.OnInspect();

}

void PointCloudCapture::OnStart(
    AutoSorghumGenerationPipeline &pipeline) {
  std::filesystem::create_directories(
      m_currentExportFolder / GetAssetRecord().lock()->GetAssetFileName() / "PointCloud");
  auto scene = pipeline.GetScene();

  m_ground = scene->CreateEntity("Ground");
  auto fieldGround = scene->GetOrSetPrivateComponent<FieldGround>(m_ground).lock();
  m_settings.m_ground = m_ground;
}
void PointCloudCapture::OnEnd(
    AutoSorghumGenerationPipeline &pipeline) {
  auto scene = pipeline.GetScene();
  if (scene->IsEntityValid(m_currentSorghumField))
    scene->DeleteEntity(m_currentSorghumField);
  pipeline.m_currentGrowingSorghum = m_currentSorghumField = {};

  scene->DeleteEntity(m_ground);
}
void PointCloudCapture::Reset(
    AutoSorghumGenerationPipeline &pipeline) {
  auto scene = pipeline.GetScene();
  if (scene->IsEntityValid(m_currentSorghumField))
    scene->DeleteEntity(m_currentSorghumField);
  pipeline.m_currentGrowingSorghum = m_currentSorghumField = Entity();
}
void PointCloudCapture::Instantiate() {
  auto scene = Application::GetActiveScene();
  auto pointCloudCaptureEntity = scene->CreateEntity("PointCloudPipeline");
  auto pointCloudPipeline =
      scene->GetOrSetPrivateComponent<AutoSorghumGenerationPipeline>(pointCloudCaptureEntity)
          .lock();
  pointCloudPipeline->m_pipelineBehaviour = std::dynamic_pointer_cast<PointCloudCapture>(m_self.lock());
}
void PointCloudCapture::CollectAssetRef(std::vector<AssetRef> &list) {
  list.push_back(m_positionsField);
}
void PointCloudCapture::Serialize(YAML::Emitter &out) {
  m_positionsField.Save("m_positionsField", out);
  out << YAML::Key << "m_currentExportFolder" << YAML::Value << m_currentExportFolder.string();
  m_settings.Serialize("m_settings", out);
}
void PointCloudCapture::Deserialize(const YAML::Node &in) {
  m_positionsField.Load("m_positionsField", in);
  if(in["m_currentExportFolder"]) m_currentExportFolder = in["m_currentExportFolder"].as<std::string>();
  m_settings.Deserialize("m_settings", in);
}
