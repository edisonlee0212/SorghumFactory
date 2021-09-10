//
// Created by lllll on 9/9/2021.
//

#include "SDFDataCapture.hpp"
#include "DepthCamera.hpp"
void Scripts::SDFDataCapture::OnInspect() {
  if(!Application::IsPlaying()){
    ImGui::Text("Start Engine first!");
    return;
  }

  EditorManager::DragAndDropButton(m_cameraEntity, "Camera Entity");

  ImGui::Text("Sorghum");
  ImGui::Checkbox("Enable mask", &m_segmentedMask);

  ImGui::Text("Camera Positioning");
  ImGui::DragFloat3("Focus point", &m_focusPoint.x, 0.1f);
  ImGui::DragFloat3("Pitch Angle From/Step/End", &m_pitchAngleStart, 1);
  ImGui::DragFloat("Turn Angle Step", &m_turnAngleStep, 1);
  ImGui::DragFloat("Distance to plant", &m_distance, 0.1);

  ImGui::Text("Camera Settings");
  ImGui::DragFloat("Camera FOV", &m_fov);
  ImGui::DragInt2("Camera Resolution", &m_resolution.x);
  ImGui::DragFloat2("Camera near/far", &m_cameraMin);
  ImGui::Checkbox("Use clear color", &m_useClearColor);
  ImGui::ColorEdit3("Camera Clear Color", &m_backgroundColor.x);

  auto cameraEntity = m_cameraEntity.Get();
  if (!cameraEntity.IsNull()) {
    if (ImGui::Button("Start")) {
      m_pitchAngle = m_turnAngle = 0;
      auto sorghumSystem = EntityManager::GetSystem<SorghumSystem>();
      m_currentGrowingSorghum = sorghumSystem->ImportPlant(std::filesystem::path("../Resources") /
                                                                                           "Sorghum/skeleton_procedural_4.txt",
                                                                                       "Sorghum 4", m_segmentedMask);
      sorghumSystem->GenerateMeshForAllSorghums();
    }
  }
}

void Scripts::SDFDataCapture::OnIdle(
    Scripts::AutoSorghumGenerationPipeline &pipeline) {
  if (m_pitchAngle == 0 && m_turnAngle == 0 && !m_cameraEntity.Get().IsNull()) {
    pipeline.m_status = AutoSorghumGenerationPipelineStatus::BeforeGrowth;
  } else {
    m_pitchAngle = m_turnAngle = -1;
  }
}
void Scripts::SDFDataCapture::OnBeforeGrowth(
    Scripts::AutoSorghumGenerationPipeline &pipeline) {
  auto cameraEntity = m_cameraEntity.Get();
  if (cameraEntity.IsNull()) {
    pipeline.m_status = AutoSorghumGenerationPipelineStatus::Idle;
    m_pitchAngle = m_turnAngle = -1;
    return;
  }
  if (m_turnAngle > 360) {
    pipeline.m_status = AutoSorghumGenerationPipelineStatus::Idle;
    m_pitchAngle = m_turnAngle = -1;
    if(!m_currentGrowingSorghum.IsNull()) EntityManager::DeleteEntity(m_currentGrowingSorghum);
    return;
  }
  auto height = m_distance * glm::sin(glm::radians((float)m_pitchAngle));
  auto groundDistance =
      m_distance * glm::cos(glm::radians((float)m_pitchAngle));
  glm::vec3 cameraPosition =
      m_distance *
      glm::vec3(glm::sin(glm::radians((float)m_turnAngle)) * groundDistance,
                height,
                glm::cos(glm::radians((float)m_turnAngle)) * groundDistance);
  GlobalTransform cameraGlobalTransform;
  cameraGlobalTransform.SetPosition(cameraPosition + m_focusPoint);
  cameraGlobalTransform.SetRotation(
      glm::quatLookAt(glm::normalize(-cameraPosition), glm::vec3(0, 1, 0)));

  cameraEntity.SetDataComponent(cameraGlobalTransform);
  auto camera = cameraEntity.GetOrSetPrivateComponent<Camera>().lock();
  auto postProcessing = cameraEntity.GetOrSetPrivateComponent<PostProcessing>().lock();
  auto depthCamera =
      cameraEntity.GetOrSetPrivateComponent<DepthCamera>().lock();
  postProcessing->SetEnabled(false);
  camera->m_fov = m_fov;
  camera->m_allowAutoResize = false;
  camera->m_farDistance = m_cameraMax;
  camera->m_nearDistance = m_cameraMin;
  camera->ResizeResolution(m_resolution.x, m_resolution.y);
  camera->m_clearColor = m_backgroundColor;
  camera->m_useClearColor = m_useClearColor;

  depthCamera->m_factor = m_cameraMax - m_cameraMin;
  depthCamera->m_useCameraResolution = true;
  pipeline.m_status = AutoSorghumGenerationPipelineStatus::Growth;


}
void Scripts::SDFDataCapture::OnGrowth(
    Scripts::AutoSorghumGenerationPipeline &pipeline) {
  auto cameraEntity = m_cameraEntity.Get();
  if (cameraEntity.IsNull()) {
    pipeline.m_status = AutoSorghumGenerationPipelineStatus::Idle;
    m_pitchAngle = m_turnAngle = -1;
    return;
  }
  auto camera = cameraEntity.GetOrSetPrivateComponent<Camera>().lock();
  auto depthCamera =
      cameraEntity.GetOrSetPrivateComponent<DepthCamera>().lock();

  pipeline.m_status = AutoSorghumGenerationPipelineStatus::AfterGrowth;
}

void Scripts::SDFDataCapture::OnAfterGrowth(
    Scripts::AutoSorghumGenerationPipeline &pipeline) {
  auto cameraEntity = m_cameraEntity.Get();
  if (cameraEntity.IsNull()) {
    pipeline.m_status = AutoSorghumGenerationPipelineStatus::Idle;
    m_pitchAngle = m_turnAngle = -1;
    return;
  }
  auto camera = cameraEntity.GetOrSetPrivateComponent<Camera>().lock();
  auto depthCamera =
      cameraEntity.GetOrSetPrivateComponent<DepthCamera>().lock();

  std::filesystem::create_directories(m_currentExportFolder);
  camera->GetTexture()->Save(m_currentExportFolder /
                             (std::to_string(m_pitchAngle) + "_" +
                              std::to_string(m_turnAngle) + (m_segmentedMask ? "_m.png" : ".png")));
  depthCamera->m_colorTexture->Save(m_currentExportFolder /
                                    (std::to_string(m_pitchAngle) + "_" +
                                     std::to_string(m_turnAngle) + "_d.png"));
  m_pitchAngle += m_pitchAngleStep;


  if (m_pitchAngle > m_pitchAngleEnd) {
    m_pitchAngle = 0;
    m_turnAngle += m_turnAngleStep;
  }
  pipeline.m_status = AutoSorghumGenerationPipelineStatus::BeforeGrowth;
}
