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
  Editor::DragAndDropButton<SorghumStateGenerator>(m_parameters,
                                                   "SorghumStateGenerator");
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
  if (m_parameters.Get<SorghumStateGenerator>()) {
    if (ImGui::Button("Instantiate pipeline")) {
      Instantiate();
    }
  } else {
    ImGui::Text("SorghumStateGenerator Missing!");
  }
}

void GeneralDataCapture::OnBeforeGrowth(
    AutoSorghumGenerationPipeline &pipeline) {
  if (!SetUpCamera(pipeline)) {
    pipeline.m_currentIndex = -1;
    return;
  }
  auto descriptor = m_parameters.Get<SorghumStateGenerator>();
  pipeline.m_currentGrowingSorghum =
      Application::GetLayer<SorghumLayer>()->CreateSorghum(descriptor);
  auto sorghumData =
      pipeline.m_currentGrowingSorghum.GetOrSetPrivateComponent<SorghumData>()
          .lock();
  sorghumData->m_seed = pipeline.m_currentIndex;
  sorghumData->Apply();
  sorghumData->GenerateGeometry();
  sorghumData->ApplyGeometry(true, true, false);
  pipeline.m_status = AutoSorghumGenerationPipelineStatus::Growth;
}
void GeneralDataCapture::OnGrowth(AutoSorghumGenerationPipeline &pipeline) {
  pipeline.m_status = AutoSorghumGenerationPipelineStatus::AfterGrowth;
}

void GeneralDataCapture::OnAfterGrowth(
    AutoSorghumGenerationPipeline &pipeline) {
  auto rayTracerCamera = m_rayTracerCamera.GetOrSetPrivateComponent<RayTracerCamera>().lock();
  auto prefix =
      m_parameters.Get<SorghumStateGenerator>()->GetPath().stem().string() +
      "_" + std::to_string(pipeline.m_currentIndex);
  m_sorghumInfos.push_back({GlobalTransform(), prefix});
  switch (m_captureStatus) {
  case MultipleAngleCaptureStatus::Info: {
    if (m_captureMesh) {
      pipeline.m_currentGrowingSorghum.GetOrSetPrivateComponent<SorghumData>()
          .lock()
          ->ExportModel((ProjectManager::GetProjectPath().parent_path() /
                         m_currentExportFolder / m_name / "Mesh" / (prefix + ".obj"))
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
        m_rayTracerCamera.SetDataComponent(i.m_camera);
        rayTracerCamera->Render(rayProperties);
        rayTracerCamera->m_colorTexture->SetPathAndSave(
            m_currentExportFolder / m_name / "Image" /
            (prefix + i.m_postFix + "_image.png"));
      }
    }
    if (m_captureMask) {
      m_captureStatus = MultipleAngleCaptureStatus::Mask;
      pipeline.m_currentGrowingSorghum.GetOrSetPrivateComponent<SorghumData>()
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
      m_rayTracerCamera.SetDataComponent(i.m_camera);
      rayTracerCamera->Render();
      rayTracerCamera->m_colorTexture->SetPathAndSave(
          m_currentExportFolder / m_name / "Mask" /
          (prefix + i.m_postFix + "_mask.png"));
    }
    m_captureStatus = MultipleAngleCaptureStatus::Angles;
  } break;
  case MultipleAngleCaptureStatus::Angles: {
    m_captureStatus = MultipleAngleCaptureStatus::Info;
    Entities::DeleteEntity(Entities::GetCurrentScene(),
                           pipeline.m_currentGrowingSorghum);
    pipeline.m_currentGrowingSorghum = {};
    pipeline.m_currentIndex++;
    pipeline.m_status = AutoSorghumGenerationPipelineStatus::BeforeGrowth;
  } break;
  }
}
bool GeneralDataCapture::SetUpCamera(AutoSorghumGenerationPipeline &pipeline) {
  auto rayTracerCamera = m_rayTracerCamera.GetOrSetPrivateComponent<RayTracerCamera>().lock();
  if (m_rayTracerCamera.IsNull()) {
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
      m_rayTracerCamera.GetOrSetPrivateComponent<DepthCamera>().lock();
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

      auto position =
          cameraPosition + glm::vec3(0, parameter->m_stemLength.m_mean, 0);
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
void GeneralDataCapture::CollectAssetRef(std::vector<AssetRef> &list) {
  list.push_back(m_parameters);
}
void GeneralDataCapture::Serialize(YAML::Emitter &out) {
  m_parameters.Save("m_parameters", out);
  out << YAML::Key << "m_captureImage" << YAML::Value << m_captureImage;
  out << YAML::Key << "m_captureMask" << YAML::Value << m_captureMask;
  out << YAML::Key << "m_captureMesh" << YAML::Value << m_captureMesh;
  out << YAML::Key << "m_currentExportFolder" << YAML::Value << m_currentExportFolder.string();
  out << YAML::Key << "m_pitchAngleStart" << YAML::Value << m_pitchAngleStart;
  out << YAML::Key << "m_pitchAngleStep" << YAML::Value << m_pitchAngleStep;
  out << YAML::Key << "m_pitchAngleEnd" << YAML::Value << m_pitchAngleEnd;
  out << YAML::Key << "m_turnAngleStart" << YAML::Value << m_turnAngleStart;
  out << YAML::Key << "m_turnAngleStep" << YAML::Value << m_turnAngleStep;
  out << YAML::Key << "m_turnAngleEnd" << YAML::Value << m_turnAngleEnd;
  out << YAML::Key << "m_fov" << YAML::Value << m_fov;
  out << YAML::Key << "m_denoiserStrength" << YAML::Value << m_denoiserStrength;
  out << YAML::Key << "m_resolution" << YAML::Value << m_resolution;
  out << YAML::Key << "m_useClearColor" << YAML::Value << m_useClearColor;
  out << YAML::Key << "m_backgroundColor" << YAML::Value << m_backgroundColor;
  out << YAML::Key << "m_cameraMin" << YAML::Value << m_cameraMin;
  out << YAML::Key << "m_cameraMax" << YAML::Value << m_cameraMax;
}
void GeneralDataCapture::Deserialize(const YAML::Node &in) {
  m_parameters.Load("m_parameters", in);
  if(in["m_captureImage"]) m_captureImage = in["m_captureImage"].as<bool>();
  if(in["m_captureMask"]) m_captureMask = in["m_captureMask"].as<bool>();
  if(in["m_captureMesh"]) m_captureMesh = in["m_captureMesh"].as<bool>();
  if(in["m_currentExportFolder"]) m_currentExportFolder = in["m_currentExportFolder"].as<std::string>();
  if(in["m_pitchAngleStart"]) m_pitchAngleStart = in["m_pitchAngleStart"].as<int>();
  if(in["m_pitchAngleStep"]) m_pitchAngleStep = in["m_pitchAngleStep"].as<int>();
  if(in["m_pitchAngleEnd"]) m_pitchAngleEnd = in["m_pitchAngleEnd"].as<int>();
  if(in["m_turnAngleStart"]) m_turnAngleStart = in["m_turnAngleStart"].as<int>();
  if(in["m_turnAngleStep"]) m_turnAngleStep = in["m_turnAngleStep"].as<int>();
  if(in["m_turnAngleEnd"]) m_turnAngleEnd = in["m_turnAngleEnd"].as<int>();
  if(in["m_fov"]) m_fov = in["m_fov"].as<float>();
  if(in["m_denoiserStrength"]) m_denoiserStrength = in["m_denoiserStrength"].as<float>();
  if(in["m_resolution"]) m_resolution = in["m_resolution"].as<glm::ivec2>();
  if(in["m_useClearColor"]) m_useClearColor = in["m_useClearColor"].as<bool>();
  if(in["m_backgroundColor"]) m_backgroundColor = in["m_backgroundColor"].as<glm::vec3>();
  if(in["m_cameraMin"]) m_cameraMin = in["m_cameraMin"].as<float>();
  if(in["m_cameraMax"]) m_cameraMax = in["m_cameraMax"].as<float>();
}
void GeneralDataCapture::Instantiate() {
  auto pipelineEntity = Entities::CreateEntity(Entities::GetCurrentScene(),
                                               "GeneralDataPipeline");
  auto pipeline =
      pipelineEntity.GetOrSetPrivateComponent<AutoSorghumGenerationPipeline>()
          .lock();
  pipeline->m_pipelineBehaviour =
      AssetManager::Get<GeneralDataCapture>(GetHandle());

}
bool GeneralDataCapture::IsReady() {
  return m_parameters.Get<SorghumStateGenerator>().get();
}
void GeneralDataCapture::Start(AutoSorghumGenerationPipeline &pipeline) {
  m_captureStatus = MultipleAngleCaptureStatus::Info;
  m_rayTracerCamera = Entities::CreateEntity(Entities::GetCurrentScene(), "Ray tracer");
  m_rayTracerCamera.SetParent(pipeline.GetOwner());

  auto rayTracerCamera =
      m_rayTracerCamera.GetOrSetPrivateComponent<RayTracerCamera>().lock();
  rayTracerCamera->SetMainCamera(true);

  CalculateMatrices();
  m_sorghumInfos.clear();
  std::filesystem::create_directories(
      ProjectManager::GetProjectPath().parent_path() / m_currentExportFolder / m_name);
  if (m_captureImage) {
    std::filesystem::create_directories(
        ProjectManager::GetProjectPath().parent_path() / m_currentExportFolder / m_name /
        "Image");
  }
  if (m_captureMask) {
    std::filesystem::create_directories(
        ProjectManager::GetProjectPath().parent_path() / m_currentExportFolder / m_name /
        "Mask");
  }
  if (m_captureMesh) {
    std::filesystem::create_directories(
        ProjectManager::GetProjectPath().parent_path() / m_currentExportFolder / m_name /
        "Mesh");
  }
}
void GeneralDataCapture::End(AutoSorghumGenerationPipeline &pipeline) {
  ExportMatrices(ProjectManager::GetProjectPath().parent_path() /
                 m_currentExportFolder / m_name / ("camera_matrices.yml"));
  ProjectManager::ScanProjectFolder(true);

  Entities::DeleteEntity(Entities::GetCurrentScene(), m_rayTracerCamera);
  m_rayTracerCamera = {};
}
