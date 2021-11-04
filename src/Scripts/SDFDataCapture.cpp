//
// Created by lllll on 9/9/2021.
//

#include "SDFDataCapture.hpp"
#include "DepthCamera.hpp"
#include <SorghumData.hpp>
#include <SorghumProceduralDescriptor.hpp>
using namespace Scripts;
void SDFDataCapture::OnInspect() {
  EditorManager::DragAndDropButton(m_cameraEntity, "Camera Entity");
  EditorManager::DragAndDropButton<SorghumProceduralDescriptor>(
      m_parameters, "Sorghum Descriptors");
  ImGui::DragInt("Instance Count", &m_generationAmount, 1, 0);
  if (ImGui::TreeNode("Data selection")) {
    ImGui::Checkbox("Capture image", &m_captureImage);
    ImGui::Checkbox("Capture mask", &m_captureMask);
    ImGui::Checkbox("Capture depth", &m_captureDepth);
    ImGui::Checkbox("Capture mesh", &m_captureMesh);
    ImGui::TreePop();
  }
  if (ImGui::TreeNode("Camera Settings")) {
    ImGui::DragFloat3("Focus point", &m_focusPoint.x, 0.1f);
    ImGui::DragFloat3("Pitch Angle From/Step/End", &m_pitchAngleStart, 1);
    ImGui::DragFloat("Turn Angle From/Step/End", &m_turnAngleStart, 1);
    ImGui::DragFloat("Distance to plant", &m_distance, 0.1);
    ImGui::Separator();
    ImGui::DragFloat("Camera FOV", &m_fov);
    ImGui::DragInt2("Camera Resolution", &m_resolution.x);
    ImGui::DragFloat2("Camera near/far", &m_cameraMin);
    ImGui::Checkbox("Use clear color", &m_useClearColor);
    ImGui::ColorEdit3("Camera Clear Color", &m_backgroundColor.x);
    ImGui::TreePop();
  }
  auto cameraEntity = m_cameraEntity.Get();
  if (!Application::IsPlaying()) {
    ImGui::Text("Application not Playing!");
  } else if (!m_parameters.Get<SorghumProceduralDescriptor>()) {
    ImGui::Text("SPD Missing");
  } else if (cameraEntity.IsNull()) {
    ImGui::Text("Camera Missing");
  } else if (m_remainingInstanceAmount != 0) {
    ImGui::Text("Busy...");
  } else {
    if (ImGui::Button("Start")) {
      m_pitchAngle = m_turnAngle = 0;
      m_remainingInstanceAmount = m_generationAmount;
      m_projections.clear();
      m_views.clear();
      m_names.clear();
      m_cameraModels.clear();
      m_sorghumModels.clear();

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
      if (m_captureDepth) {
        std::filesystem::create_directories(
            ProjectManager::GetProjectPath().parent_path() /
            m_currentExportFolder / "Depth");
      }
      if (m_captureMesh) {
        std::filesystem::create_directories(
            ProjectManager::GetProjectPath().parent_path() /
            m_currentExportFolder / "Mesh");
      }
    }
  }
}

void SDFDataCapture::OnIdle(AutoSorghumGenerationPipeline &pipeline) {
  if (m_cameraEntity.Get().IsNull()) {
    m_pitchAngle = m_pitchAngleStart;
    m_turnAngle = -1;
    m_generationAmount = 0;
    return;
  }
  if (m_pitchAngle == m_pitchAngleStart && m_turnAngle == 0 &&
      m_remainingInstanceAmount > 0) {
    pipeline.m_status = AutoSorghumGenerationPipelineStatus::BeforeGrowth;
  }
}
void SDFDataCapture::OnBeforeGrowth(AutoSorghumGenerationPipeline &pipeline) {
  if (!SetUpCamera()) {
    pipeline.m_status = AutoSorghumGenerationPipelineStatus::Idle;
    m_generationAmount = 0;
    m_pitchAngle = m_turnAngle = -1;
    return;
  }
  auto descriptor = m_parameters.Get<SorghumProceduralDescriptor>();
  m_currentGrowingSorghum =
      Application::GetLayer<SorghumLayer>()->CreateSorghum(descriptor);
  pipeline.m_status = AutoSorghumGenerationPipelineStatus::Growth;
}
void SDFDataCapture::OnGrowth(AutoSorghumGenerationPipeline &pipeline) {
  pipeline.m_status = AutoSorghumGenerationPipelineStatus::AfterGrowth;
  m_skipCurrentFrame = true;
}

void SDFDataCapture::OnAfterGrowth(AutoSorghumGenerationPipeline &pipeline) {
  if (m_skipCurrentFrame) {
    if (!SetUpCamera()) {
      pipeline.m_status = AutoSorghumGenerationPipelineStatus::Idle;
      m_generationAmount = 0;
      m_pitchAngle = m_turnAngle = -1;
      return;
    }
    m_skipCurrentFrame = false;
  } else {
    auto cameraEntity = m_cameraEntity.Get();
    auto prefix =
        m_parameters.Get<SorghumProceduralDescriptor>()
            ->GetPath()
            .stem()
            .string() +
        "_" + std::to_string(m_generationAmount - m_remainingInstanceAmount) +
        "_" + std::to_string(m_pitchAngle) + "_" + std::to_string(m_turnAngle);
    switch (m_captureStatus) {
    case MultipleAngleCaptureStatus::Info: {
      m_cameraModels.push_back(
          cameraEntity.GetDataComponent<GlobalTransform>().m_value);
      m_sorghumModels.push_back(
          m_currentGrowingSorghum.GetDataComponent<GlobalTransform>().m_value);
      m_projections.push_back(Camera::m_cameraInfoBlock.m_projection);
      m_views.push_back(Camera::m_cameraInfoBlock.m_view);
      m_names.push_back(prefix);
      if (m_captureImage) {
        auto camera = cameraEntity.GetOrSetPrivateComponent<Camera>().lock();
        camera->GetTexture()->SetPathAndSave(m_currentExportFolder / "Image" /
                                             (prefix + "_image.png"));
      }
      if (m_captureDepth) {
        auto depthCamera =
            cameraEntity.GetOrSetPrivateComponent<DepthCamera>().lock();
        depthCamera->m_colorTexture->SetPathAndSave(
            m_currentExportFolder / "Depth" / (prefix + "_depth.png"));
      }
      if (m_captureMask) {
        m_captureStatus = MultipleAngleCaptureStatus::Mask;
        m_currentGrowingSorghum.GetOrSetPrivateComponent<SorghumData>()
            .lock()
            ->GenerateGeometry(true);
        m_skipCurrentFrame = true;
      } else {
        m_captureStatus = MultipleAngleCaptureStatus::Angles;
      }
    } break;
    case MultipleAngleCaptureStatus::Mask: {
      auto camera = cameraEntity.GetOrSetPrivateComponent<Camera>().lock();
      camera->GetTexture()->SetPathAndSave(m_currentExportFolder / "Mask" /
                                           (prefix + "_mask.png"));
      m_currentGrowingSorghum.GetOrSetPrivateComponent<SorghumData>()
          .lock()
          ->GenerateGeometry(false);
      m_captureStatus = MultipleAngleCaptureStatus::Angles;
    } break;
    case MultipleAngleCaptureStatus::Angles: {
      m_captureStatus = MultipleAngleCaptureStatus::Info;
      if ((m_pitchAngle + m_pitchAngleStep) <= m_pitchAngleEnd) {
        m_pitchAngle += m_pitchAngleStep;
        SetUpCamera();
        m_skipCurrentFrame = true;
      } else if ((m_turnAngle + m_turnAngleStep) < 360.0f || (m_turnAngle + m_turnAngleEnd) < 360.0f) {
        m_turnAngle += m_turnAngleStep;
        m_pitchAngle = m_pitchAngleStart;
        SetUpCamera();
        m_skipCurrentFrame = true;
      } else {
        if (m_captureMesh) {
          m_currentGrowingSorghum.GetOrSetPrivateComponent<SorghumData>()
              .lock()
              ->ExportModel((ProjectManager::GetProjectPath().parent_path() /
                             m_currentExportFolder / "Mesh" /
                             (m_parameters.Get<SorghumProceduralDescriptor>()
                                  ->GetPath()
                                  .stem()
                                  .string() +
                              "_" +
                              std::to_string(m_generationAmount -
                                             m_remainingInstanceAmount) +
                              ".obj"))
                                .string());
        }
        m_parameters.Get<SorghumProceduralDescriptor>()->Export(
            std::filesystem::absolute(
                ProjectManager::GetProjectPath().parent_path()) /
            m_currentExportFolder /
            m_parameters.Get<SorghumProceduralDescriptor>()
                ->GetPath()
                .filename());
        m_remainingInstanceAmount--;
        m_pitchAngle = m_pitchAngleStart;
        m_turnAngle = 0;
        if (m_remainingInstanceAmount == 0) {
          ExportMatrices(ProjectManager::GetProjectPath().parent_path() /
                         m_currentExportFolder / ("camera_matrices.yml"));
          ProjectManager::ScanProjectFolder(true);
          pipeline.m_status = AutoSorghumGenerationPipelineStatus::Idle;
        } else {
          pipeline.m_status = AutoSorghumGenerationPipelineStatus::BeforeGrowth;
        }

        EntityManager::DeleteEntity(EntityManager::GetCurrentScene(),
                                    m_currentGrowingSorghum);
      }
    } break;
    }
  }
}
bool SDFDataCapture::SetUpCamera() {
  auto cameraEntity = m_cameraEntity.Get();
  if (cameraEntity.IsNull()) {
    m_pitchAngle = m_pitchAngleStart;
    m_turnAngle = m_remainingInstanceAmount = 0;
    UNIENGINE_ERROR("Camera entity missing!");
    return false;
  }
  EntityManager::GetCurrentScene()->m_environmentSettings.m_environmentType =
      UniEngine::EnvironmentType::Color;
  EntityManager::GetCurrentScene()->m_environmentSettings.m_backgroundColor =
      glm::vec3(1.0f);
  EntityManager::GetCurrentScene()
      ->m_environmentSettings.m_ambientLightIntensity = 1.0f;
  auto height = m_distance * glm::sin(glm::radians((float)m_pitchAngle));
  auto groundDistance =
      m_distance * glm::cos(glm::radians((float)m_pitchAngle));
  glm::vec3 cameraPosition = glm::vec3(
      glm::sin(glm::radians((float)m_turnAngle)) * groundDistance, height,
      glm::cos(glm::radians((float)m_turnAngle)) * groundDistance);
  m_cameraPosition = cameraPosition + m_focusPoint;
  m_cameraRotation =
      glm::quatLookAt(glm::normalize(-cameraPosition), glm::vec3(0, 1, 0));

  GlobalTransform cameraGlobalTransform;
  cameraGlobalTransform.SetPosition(m_cameraPosition);
  cameraGlobalTransform.SetRotation(m_cameraRotation);
  cameraEntity.SetDataComponent(cameraGlobalTransform);

  auto camera = cameraEntity.GetOrSetPrivateComponent<Camera>().lock();
  camera->m_fov = m_fov;
  camera->m_allowAutoResize = false;
  camera->m_farDistance = m_cameraMax;
  camera->m_nearDistance = m_cameraMin;
  camera->ResizeResolution(m_resolution.x, m_resolution.y);
  camera->m_clearColor = m_backgroundColor;
  camera->m_useClearColor = m_useClearColor;

  auto depthCamera =
      cameraEntity.GetOrSetPrivateComponent<DepthCamera>().lock();
  depthCamera->m_useCameraResolution = true;

  if (cameraEntity.HasPrivateComponent<PostProcessing>()) {
    auto postProcessing =
        cameraEntity.GetOrSetPrivateComponent<PostProcessing>().lock();
    postProcessing->SetEnabled(false);
  }
  return true;
}
void SDFDataCapture::ExportMatrices(const std::filesystem::path &path) {
  YAML::Emitter out;
  out << YAML::BeginMap;
  out << YAML::Key << "Capture Info" << YAML::BeginSeq;
  for (int i = 0; i < m_projections.size(); i++) {
    out << YAML::BeginMap;
    out << YAML::Key << "File Prefix" << YAML::Value << m_names[i];
    out << YAML::Key << "Projection" << YAML::Value << m_projections[i];
    out << YAML::Key << "View" << YAML::Value << m_views[i];
    out << YAML::Key << "Camera Model" << YAML::Value << m_cameraModels[i];
    out << YAML::Key << "Sorghum Model" << YAML::Value << m_sorghumModels[i];
    out << YAML::EndMap;
  }
  out << YAML::EndSeq;
  out << YAML::EndMap;
  std::ofstream fout(path.string());
  fout << out.c_str();
  fout.flush();
}
