//
// Created by lllll on 9/9/2021.
//

#include "GeneralDataCapture.hpp"
#include "DepthCamera.hpp"
#include <SorghumData.hpp>
#include <SorghumStateGenerator.hpp>
#ifdef RAYTRACERFACILITY
#include "RayTracerCamera.hpp"
#include "RayTracerLayer.hpp"
using namespace RayTracerFacility;
#endif
using namespace Scripts;
void GeneralDataCapture::OnInspect() {
  Editor::DragAndDropButton<RayTracerCamera>(m_rayTracerCamera,
                                             "Ray Tracer Camera");
  Editor::DragAndDropButton<SorghumStateGenerator>(m_parameters,
                                                   "SorghumStateGenerator");
  ImGui::DragInt("Instance Count", &m_generationAmount, 1, 0);
  if (ImGui::TreeNode("Data selection")) {
    ImGui::Checkbox("Capture image", &m_captureImage);
    ImGui::Checkbox("Capture mask", &m_captureMask);
    ImGui::Checkbox("Capture mesh", &m_captureMesh);
    ImGui::TreePop();
  }
  if (ImGui::TreeNode("Camera Settings")) {
    ImGui::DragInt3("Pitch Angle From/Step/End", &m_pitchAngleStart, 1);
    ImGui::DragInt3("Turn Angle From/Step/End", &m_turnAngleStart, 1);
    ImGui::Separator();
    ImGui::DragFloat("Camera FOV", &m_fov);
    ImGui::DragInt2("Camera Resolution", &m_resolution.x);
    ImGui::DragFloat2("Camera near/far", &m_cameraMin);
    ImGui::Checkbox("Use clear color", &m_useClearColor);
    ImGui::ColorEdit3("Camera Clear Color", &m_backgroundColor.x);
    ImGui::TreePop();
  }
  auto rayTracerCamera = m_rayTracerCamera.Get<RayTracerCamera>();
  if (!Application::IsPlaying()) {
    ImGui::Text("Application not Playing!");
  } else if (!m_parameters.Get<SorghumStateGenerator>()) {
    ImGui::Text("SorghumStateGenerator Missing");
  } else if (!rayTracerCamera) {
    ImGui::Text("Camera Missing");
  } else if (m_remainingInstanceAmount != 0) {
    ImGui::Text("Busy...");
  } else {
    if (ImGui::Button("Start")) {
      m_remainingInstanceAmount = m_generationAmount;
      CalculateMatrices();
      m_sorghumInfos.clear();
      std::filesystem::create_directories(
          ProjectManager::GetProjectPath().parent_path() /
          m_currentExportFolder);
      if (m_captureImage) {
        std::filesystem::create_directories(
            ProjectManager::GetProjectPath().parent_path() /
            m_currentExportFolder / "Image");
      }
      if (m_captureMask) {
        std::filesystem::create_directories(
            ProjectManager::GetProjectPath().parent_path() /
            m_currentExportFolder / "Mask");
      }
      if (m_captureMesh) {
        std::filesystem::create_directories(
            ProjectManager::GetProjectPath().parent_path() /
            m_currentExportFolder / "Mesh");
      }
    }
  }
}

void GeneralDataCapture::OnIdle(AutoSorghumGenerationPipeline &pipeline) {
  auto rayTracerCamera = m_rayTracerCamera.Get<RayTracerCamera>();
  if (!rayTracerCamera) {
    m_generationAmount = 0;
    return;
  }
  if (m_remainingInstanceAmount > 0) {
    pipeline.m_status = AutoSorghumGenerationPipelineStatus::BeforeGrowth;
  }
}
void GeneralDataCapture::OnBeforeGrowth(AutoSorghumGenerationPipeline &pipeline) {
  if (!SetUpCamera()) {
    pipeline.m_status = AutoSorghumGenerationPipelineStatus::Idle;
    m_generationAmount = 0;
    return;
  }
  auto descriptor = m_parameters.Get<SorghumStateGenerator>();
  m_currentGrowingSorghum =
      Application::GetLayer<SorghumLayer>()->CreateSorghum(descriptor);
  auto sorghumData =
      m_currentGrowingSorghum.GetOrSetPrivateComponent<SorghumData>().lock();
  sorghumData->m_seed = m_generationAmount - m_remainingInstanceAmount;
  sorghumData->Apply();
  sorghumData->GenerateGeometry();
  sorghumData->ApplyGeometry(true, true, false);
  pipeline.m_status = AutoSorghumGenerationPipelineStatus::Growth;
}
void GeneralDataCapture::OnGrowth(AutoSorghumGenerationPipeline &pipeline) {
  pipeline.m_status = AutoSorghumGenerationPipelineStatus::AfterGrowth;
}

void GeneralDataCapture::OnAfterGrowth(AutoSorghumGenerationPipeline &pipeline) {
  auto rayTracerCamera = m_rayTracerCamera.Get<RayTracerCamera>();
  auto rayTracerCameraEntity = rayTracerCamera->GetOwner();
  auto prefix =
      m_parameters.Get<SorghumStateGenerator>()->GetPath().stem().string() +
      "_" + std::to_string(m_generationAmount - m_remainingInstanceAmount);
  m_sorghumInfos.push_back({GlobalTransform(), prefix});
  switch (m_captureStatus) {
  case MultipleAngleCaptureStatus::Info: {
    if (m_captureMesh) {
      m_currentGrowingSorghum.GetOrSetPrivateComponent<SorghumData>()
          .lock()
          ->ExportModel((ProjectManager::GetProjectPath().parent_path() /
                         m_currentExportFolder / "Mesh" / (prefix + ".obj"))
                            .string());
    }
    if (m_captureImage) {
      Application::GetLayer<RayTracerLayer>()
          ->m_environmentProperties.m_environmentalLightingType =
          RayTracerFacility::EnvironmentalLightingType::Color;
      Application::GetLayer<RayTracerLayer>()
          ->m_environmentProperties.m_sunColor = glm::vec3(1.0f);
      RayProperties rayProperties;
      rayProperties.m_samples = 1000;
      rayTracerCamera->SetOutputType(OutputType::Color);
      rayTracerCamera->SetDenoiserStrength(m_denoiserStrength);
      for (const auto &i : m_cameraMatrices) {
        rayTracerCameraEntity.SetDataComponent(i.m_camera);
        rayTracerCamera->Render(rayProperties);
        rayTracerCamera->m_colorTexture->SetPathAndSave(
            m_currentExportFolder / "Image" /
            (prefix + i.m_postFix + "_image.png"));
      }
    }
    if (m_captureMask) {
      m_captureStatus = MultipleAngleCaptureStatus::Mask;
      m_currentGrowingSorghum.GetOrSetPrivateComponent<SorghumData>()
          .lock()
          ->ApplyGeometry(true, true, true);
    } else {
      m_captureStatus = MultipleAngleCaptureStatus::Angles;
    }
  } break;
  case MultipleAngleCaptureStatus::Mask: {
    Application::GetLayer<RayTracerLayer>()
        ->m_environmentProperties.m_environmentalLightingType =
        RayTracerFacility::EnvironmentalLightingType::Color;
    Application::GetLayer<RayTracerLayer>()
        ->m_environmentProperties.m_sunColor = glm::vec3(1.0f);
    RayProperties rayProperties;
    rayProperties.m_samples = 1;
    rayTracerCamera->SetDenoiserStrength(0.0f);
    rayTracerCamera->SetOutputType(OutputType::Albedo);
    for (const auto &i : m_cameraMatrices) {
      rayTracerCameraEntity.SetDataComponent(i.m_camera);
      rayTracerCamera->Render();
      rayTracerCamera->m_colorTexture->SetPathAndSave(
          m_currentExportFolder / "Mask" / (prefix + i.m_postFix + "_mask.png"));
    }
    m_captureStatus = MultipleAngleCaptureStatus::Angles;
  } break;
  case MultipleAngleCaptureStatus::Angles: {
    m_captureStatus = MultipleAngleCaptureStatus::Info;
    m_remainingInstanceAmount--;
    if (m_remainingInstanceAmount == 0) {
      ExportMatrices(ProjectManager::GetProjectPath().parent_path() /
                     m_currentExportFolder / ("camera_matrices.yml"));
      ProjectManager::ScanProjectFolder(true);
      pipeline.m_status = AutoSorghumGenerationPipelineStatus::Idle;
    } else {
      pipeline.m_status = AutoSorghumGenerationPipelineStatus::BeforeGrowth;
    }
    Entities::DeleteEntity(Entities::GetCurrentScene(),
                           m_currentGrowingSorghum);

  } break;
  }
}
bool GeneralDataCapture::SetUpCamera() {
  auto rayTracerCamera = m_rayTracerCamera.Get<RayTracerCamera>();
  auto cameraEntity = rayTracerCamera->GetOwner();
  if (cameraEntity.IsNull()) {
    UNIENGINE_ERROR("Camera entity missing!");
    return false;
  }
  Entities::GetCurrentScene()->m_environmentSettings.m_environmentType =
      UniEngine::EnvironmentType::Color;
  Entities::GetCurrentScene()->m_environmentSettings.m_backgroundColor =
      glm::vec3(1.0f);
  Entities::GetCurrentScene()->m_environmentSettings.m_ambientLightIntensity =
      1.0f;
  rayTracerCamera->SetFov(m_fov);
  rayTracerCamera->m_allowAutoResize = false;
  rayTracerCamera->m_frameSize = m_resolution;
  auto depthCamera =
      cameraEntity.GetOrSetPrivateComponent<DepthCamera>().lock();
  depthCamera->m_useCameraResolution = true;
  return true;
}
void GeneralDataCapture::ExportMatrices(const std::filesystem::path &path) {
  YAML::Emitter out;
  out << YAML::BeginMap;
  out << YAML::Key << "Capture Info" << YAML::BeginSeq;
  for (int i = 0; i < m_sorghumInfos.size(); i++) {
    for (int j = 0; j < m_cameraMatrices.size(); j++) {
      out << YAML::BeginMap;
      out << YAML::Key << "File Name" << YAML::Value
          << m_sorghumInfos[i].m_name + m_cameraMatrices[j].m_postFix;
      out << YAML::Key << "Projection" << YAML::Value
          << m_cameraMatrices[j].m_projection;
      out << YAML::Key << "View" << YAML::Value << m_cameraMatrices[j].m_view;
      out << YAML::Key << "Camera Model" << YAML::Value
          << m_cameraMatrices[j].m_camera.m_value;
      out << YAML::Key << "Sorghum Model" << YAML::Value
          << m_sorghumInfos[i].m_sorghum.m_value;
      out << YAML::EndMap;
    }
  }
  out << YAML::EndSeq;
  out << YAML::EndMap;
  std::ofstream fout(path.string());
  fout << out.c_str();
  fout.flush();
}
void GeneralDataCapture::CalculateMatrices() {
  m_cameraMatrices.clear();
  auto parameter = m_parameters.Get<SorghumStateGenerator>();
  for (int pitch = m_pitchAngleStart; pitch <= m_pitchAngleEnd;
       pitch += m_pitchAngleStep) {
    for (int turn = m_turnAngleStart; turn <= m_turnAngleEnd;
         turn += m_turnAngleStep) {
      auto distance = parameter->m_stemLength.m_mean * 5.0f;
      auto height = distance * glm::sin(glm::radians((float)pitch));
      auto groundDistance = distance * glm::cos(glm::radians((float)pitch));
      glm::vec3 cameraPosition = glm::vec3(
          glm::sin(glm::radians((float)turn)) * groundDistance, height,
          glm::cos(glm::radians((float)turn)) * groundDistance);

      auto position = cameraPosition + glm::vec3(0, parameter->m_stemLength.m_mean, 0);
      auto rotation =
          glm::quatLookAt(glm::normalize(-cameraPosition), glm::vec3(0, 1, 0));
      GlobalTransform cameraGlobalTransform;
      cameraGlobalTransform.SetPosition(position);
      cameraGlobalTransform.SetRotation(rotation);
      CameraMatricesCollection collection;
      collection.m_camera = cameraGlobalTransform;
      const glm::vec3 front = rotation * glm::vec3(0, 0, -1);
      const glm::vec3 up = rotation * glm::vec3(0, 1, 0);
      collection.m_projection = glm::perspective(
          glm::radians(m_fov * 0.5f), (float)m_resolution.x / m_resolution.y,
          m_cameraMin, m_cameraMax);
      collection.m_view = glm::lookAt(position, position + front, up);
      collection.m_postFix = "_pitch" + std::to_string((int)pitch) + "_turn" +
                             std::to_string((int)turn);
      m_cameraMatrices.push_back(collection);
    }
  }
}
