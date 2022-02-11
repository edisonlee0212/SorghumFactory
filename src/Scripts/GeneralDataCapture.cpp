//
// Created by lllll on 9/9/2021.
//

#include "GeneralDataCapture.hpp"
#include "DepthCamera.hpp"
#include "TransformLayer.hpp"
#include <SorghumData.hpp>
#include <SorghumStateGenerator.hpp>
using namespace Scripts;
void GeneralDataCapture::OnInspect() {
  Editor::DragAndDropButton<SorghumStateGenerator>(m_parameters,
                                                   "SorghumStateGenerator");

  Editor::DragAndDropButton<Prefab>(m_labPrefab, "Lab");
  Editor::DragAndDropButton<Prefab>(m_dirtPrefab, "Dirt");
  if (ImGui::TreeNode("Data selection")) {
    ImGui::Checkbox("Capture image", &m_captureImage);
    ImGui::Checkbox("Capture mask", &m_captureMask);
    ImGui::Checkbox("Capture mesh", &m_captureMesh);
    ImGui::TreePop();
  }
  if (ImGui::TreeNode("Camera Settings")) {
    ImGui::DragInt("Bounces", &m_rayProperties.m_bounces, 1, 1, 16);
    ImGui::DragInt("Samples", &m_rayProperties.m_samples, 1, 1, 2048);
    ImGui::DragFloat("Denoiser strength", &m_denoiserStrength, 0.01f);
    ImGui::DragFloat("Distance to center", &m_distanceToCenter, 0.01f);
    ImGui::DragFloat("Height", &m_height, 0.01f);
    ImGui::DragInt3("Turn angle Start/Step/End", &m_turnAngleStart, 0.01f);
    ImGui::Separator();
    ImGui::DragFloat("Camera FOV", &m_fov);
    ImGui::DragFloat("Camera gamma", &m_gamma, 0.01f);
    ImGui::DragInt2("Camera Resolution", &m_resolution.x);
    ImGui::DragFloat2("Camera near/far", &m_cameraMin);
    ImGui::Checkbox("Use clear color", &m_useClearColor);
    ImGui::ColorEdit3("Env lighting color", &m_backgroundColor.x);
    ImGui::DragFloat("Env lighting intensity", &m_backgroundColorIntensity);
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
  if (m_lab.IsValid())
    m_lab.SetEnabled(true);
  if (m_dirt.IsValid()) {
    m_dirt.SetEnabled(true);
    auto dirtGT = m_dirt.GetDataComponent<GlobalTransform>();
    dirtGT.SetRotation(dirtGT.GetRotation() * glm::quat(glm::vec3(0, glm::linearRand(0.0f, 2.0f * glm::pi<float>()), 0)));
    m_dirt.SetDataComponent(dirtGT);
  }
}
void GeneralDataCapture::OnGrowth(AutoSorghumGenerationPipeline &pipeline) {
  pipeline.m_status = AutoSorghumGenerationPipelineStatus::AfterGrowth;
}

void GeneralDataCapture::OnAfterGrowth(
    AutoSorghumGenerationPipeline &pipeline) {
  auto rayTracerCamera =
      m_rayTracerCamera.GetOrSetPrivateComponent<RayTracerCamera>().lock();
  auto prefix =
      m_parameters.Get<SorghumStateGenerator>()->GetPath().stem().string() +
      "_" + std::to_string(pipeline.m_currentIndex);
  m_sorghumInfos.push_back({GlobalTransform(), prefix});
  switch (m_captureStatus) {
  case MultipleAngleCaptureStatus::Info: {
    if (m_captureMesh) {
      pipeline.m_currentGrowingSorghum.GetOrSetPrivateComponent<SorghumData>()
          .lock()
          ->ExportModel((ProjectManager::GetProjectPath().parent_path().parent_path() /
                         m_currentExportFolder / m_name / "Mesh" /
                         (prefix + ".obj"))
                            .string());
    }
    if (m_captureImage) {
      Application::GetLayer<RayTracerLayer>()
          ->m_environmentProperties.m_environmentalLightingType =
          RayTracerFacility::EnvironmentalLightingType::Color;
      Application::GetLayer<RayTracerLayer>()
          ->m_environmentProperties.m_sunColor = glm::vec3(1.0f);
      rayTracerCamera->SetOutputType(OutputType::Color);
      rayTracerCamera->SetDenoiserStrength(m_denoiserStrength);
      GlobalTransform cameraGT;
      cameraGT.SetPosition(glm::vec3(0, m_height, m_distanceToCenter));
      cameraGT.SetRotation(glm::vec3(0, 0, 0));
      m_rayTracerCamera.SetDataComponent(cameraGT);
      for (int turnAngle = m_turnAngleStart; turnAngle <= m_turnAngleEnd;
           turnAngle += m_turnAngleStep) {
        auto sorghumGT = pipeline.m_currentGrowingSorghum
                             .GetDataComponent<GlobalTransform>();
        sorghumGT.SetRotation(glm::vec3(0, glm::radians((float)turnAngle), 0));
        pipeline.m_currentGrowingSorghum.SetDataComponent(sorghumGT);
        Application::GetLayer<TransformLayer>()->CalculateTransformGraphs(
            Entities::GetCurrentScene());
        Application::GetLayer<RayTracerLayer>()->UpdateScene();
        rayTracerCamera->Render(m_rayProperties);
        rayTracerCamera->m_colorTexture->Export(ProjectManager::GetProjectPath().parent_path().parent_path() /
            m_currentExportFolder / m_name / "Image" /
            (prefix + "_" + std::to_string(turnAngle) + "_image.png"));
      }
      cameraGT.SetPosition(glm::vec3(0, m_distanceToCenter, 0));
      cameraGT.SetRotation(glm::vec3(glm::radians(-90.0f), 0, 0));
      m_rayTracerCamera.SetDataComponent(cameraGT);
      for (int turnAngle = m_turnAngleStart; turnAngle <= m_turnAngleEnd;
           turnAngle += m_turnAngleStep) {
        auto sorghumGT = pipeline.m_currentGrowingSorghum
                             .GetDataComponent<GlobalTransform>();
        sorghumGT.SetRotation(glm::vec3(0, glm::radians((float)turnAngle), 0));
        pipeline.m_currentGrowingSorghum.SetDataComponent(sorghumGT);
        Application::GetLayer<TransformLayer>()->CalculateTransformGraphs(
            Entities::GetCurrentScene());
        Application::GetLayer<RayTracerLayer>()->UpdateScene();
        rayTracerCamera->Render(m_rayProperties);
        rayTracerCamera->m_colorTexture->Export(ProjectManager::GetProjectPath().parent_path().parent_path() /
            m_currentExportFolder / m_name / "Image" /
            (prefix + "_" + std::to_string(turnAngle) + "_top_image.png"));
      }
    }
    if (m_captureMask) {
      if (m_lab.IsValid())
        m_lab.SetEnabled(false);
      if (m_dirt.IsValid())
        m_dirt.SetEnabled(false);
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
    rayProperties.m_bounces = 1;
    rayTracerCamera->SetDenoiserStrength(0.0f);
    rayTracerCamera->SetOutputType(OutputType::Albedo);
    GlobalTransform cameraGT;
    cameraGT.SetPosition(glm::vec3(0, m_height, m_distanceToCenter));
    cameraGT.SetRotation(glm::vec3(0, 0, 0));
    m_rayTracerCamera.SetDataComponent(cameraGT);
    for (int turnAngle = m_turnAngleStart; turnAngle <= m_turnAngleEnd;
         turnAngle += m_turnAngleStep) {
      auto sorghumGT =
          pipeline.m_currentGrowingSorghum.GetDataComponent<GlobalTransform>();
      sorghumGT.SetRotation(glm::vec3(0, glm::radians((float)turnAngle), 0));
      pipeline.m_currentGrowingSorghum.SetDataComponent(sorghumGT);
      Application::GetLayer<TransformLayer>()->CalculateTransformGraphs(
          Entities::GetCurrentScene());
      Application::GetLayer<RayTracerLayer>()->UpdateScene();
      rayTracerCamera->Render(rayProperties);
      rayTracerCamera->m_colorTexture->Export(ProjectManager::GetProjectPath().parent_path().parent_path() /
          m_currentExportFolder / m_name / "Mask" /
          (prefix + "_" + std::to_string(turnAngle) + "_mask.png"));
    }
    cameraGT.SetPosition(glm::vec3(0, m_distanceToCenter, 0));
    cameraGT.SetRotation(glm::vec3(glm::radians(-90.0f), 0, 0));
    m_rayTracerCamera.SetDataComponent(cameraGT);
    for (int turnAngle = m_turnAngleStart; turnAngle <= m_turnAngleEnd;
         turnAngle += m_turnAngleStep) {
      auto sorghumGT =
          pipeline.m_currentGrowingSorghum.GetDataComponent<GlobalTransform>();
      sorghumGT.SetRotation(glm::vec3(0, glm::radians((float)turnAngle), 0));
      pipeline.m_currentGrowingSorghum.SetDataComponent(sorghumGT);
      Application::GetLayer<TransformLayer>()->CalculateTransformGraphs(
          Entities::GetCurrentScene());
      Application::GetLayer<RayTracerLayer>()->UpdateScene();
      rayTracerCamera->Render(rayProperties);
      rayTracerCamera->m_colorTexture->Export(ProjectManager::GetProjectPath().parent_path().parent_path() /
          m_currentExportFolder / m_name / "Mask" /
          (prefix + "_" + std::to_string(turnAngle) + "_top_mask.png"));
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
  auto rayTracerCamera =
      m_rayTracerCamera.GetOrSetPrivateComponent<RayTracerCamera>().lock();
  if (m_rayTracerCamera.IsNull()) {
    UNIENGINE_ERROR("Camera entity missing!");
    return false;
  }
  Entities::GetCurrentScene()->m_environmentSettings.m_environmentType =
      UniEngine::EnvironmentType::Color;
  Entities::GetCurrentScene()->m_environmentSettings.m_backgroundColor =
      m_backgroundColor;
  Entities::GetCurrentScene()->m_environmentSettings.m_ambientLightIntensity =
      m_backgroundColorIntensity;
  auto& envProp = Application::GetLayer<RayTracerLayer>()->m_environmentProperties;
  envProp.m_environmentalLightingType = RayTracerFacility::EnvironmentalLightingType::Color;
  envProp.m_sunColor = m_backgroundColor;
  envProp.m_skylightIntensity = m_backgroundColorIntensity;
  rayTracerCamera->SetFov(m_fov);
  rayTracerCamera->SetGamma(m_gamma);
  rayTracerCamera->SetMainCamera(false);
  rayTracerCamera->m_allowAutoResize = false;
  rayTracerCamera->m_frameSize = m_resolution;
  auto depthCamera =
      m_rayTracerCamera.GetOrSetPrivateComponent<DepthCamera>().lock();
  depthCamera->m_useCameraResolution = true;
  return true;
}
void GeneralDataCapture::CollectAssetRef(std::vector<AssetRef> &list) {
  list.push_back(m_parameters);
  list.push_back(m_labPrefab);
  list.push_back(m_dirtPrefab);
}
void GeneralDataCapture::Serialize(YAML::Emitter &out) {
  m_parameters.Save("m_parameters", out);
  m_labPrefab.Save("m_labPrefab", out);
  m_dirtPrefab.Save("m_dirtPrefab", out);
  out << YAML::Key << "m_rayProperties.m_samples" << YAML::Value
      << m_rayProperties.m_samples;
  out << YAML::Key << "m_rayProperties.m_bounces" << YAML::Value
      << m_rayProperties.m_bounces;

  out << YAML::Key << "m_captureImage" << YAML::Value << m_captureImage;
  out << YAML::Key << "m_captureMask" << YAML::Value << m_captureMask;
  out << YAML::Key << "m_captureMesh" << YAML::Value << m_captureMesh;
  out << YAML::Key << "m_currentExportFolder" << YAML::Value
      << m_currentExportFolder.string();
  out << YAML::Key << "m_distanceToCenter" << YAML::Value << m_distanceToCenter;
  out << YAML::Key << "m_height" << YAML::Value << m_height;
  out << YAML::Key << "m_turnAngleStart" << YAML::Value << m_turnAngleStart;
  out << YAML::Key << "m_turnAngleStep" << YAML::Value << m_turnAngleStep;
  out << YAML::Key << "m_turnAngleEnd" << YAML::Value << m_turnAngleEnd;
  out << YAML::Key << "m_fov" << YAML::Value << m_fov;
  out << YAML::Key << "m_gamma" << YAML::Value << m_gamma;
  out << YAML::Key << "m_denoiserStrength" << YAML::Value << m_denoiserStrength;
  out << YAML::Key << "m_resolution" << YAML::Value << m_resolution;
  out << YAML::Key << "m_useClearColor" << YAML::Value << m_useClearColor;
  out << YAML::Key << "m_backgroundColor" << YAML::Value << m_backgroundColor;
  out << YAML::Key << "m_backgroundColorIntensity" << YAML::Value << m_backgroundColorIntensity;
  out << YAML::Key << "m_cameraMin" << YAML::Value << m_cameraMin;
  out << YAML::Key << "m_cameraMax" << YAML::Value << m_cameraMax;
}
void GeneralDataCapture::Deserialize(const YAML::Node &in) {
  m_parameters.Load("m_parameters", in);
  m_labPrefab.Load("m_labPrefab", in);
  m_dirtPrefab.Load("m_dirtPrefab", in);
  if (in["m_rayProperties.m_samples"])
    m_rayProperties.m_samples = in["m_rayProperties.m_samples"].as<float>();
  if (in["m_rayProperties.m_bounces"])
    m_rayProperties.m_bounces = in["m_rayProperties.m_bounces"].as<float>();

  if (in["m_captureImage"])
    m_captureImage = in["m_captureImage"].as<bool>();
  if (in["m_captureMask"])
    m_captureMask = in["m_captureMask"].as<bool>();
  if (in["m_captureMesh"])
    m_captureMesh = in["m_captureMesh"].as<bool>();
  if (in["m_currentExportFolder"])
    m_currentExportFolder = in["m_currentExportFolder"].as<std::string>();
  if (in["m_distanceToCenter"])
    m_distanceToCenter = in["m_distanceToCenter"].as<float>();
  if (in["m_height"])
    m_height = in["m_height"].as<float>();
  if (in["m_turnAngleStart"])
    m_turnAngleStart = in["m_turnAngleStart"].as<int>();
  if (in["m_turnAngleStep"])
    m_turnAngleStep = in["m_turnAngleStep"].as<int>();
  if (in["m_turnAngleEnd"])
    m_turnAngleEnd = in["m_turnAngleEnd"].as<int>();
  if (in["m_gamma"])
    m_gamma = in["m_gamma"].as<float>();
  if (in["m_fov"])
    m_fov = in["m_fov"].as<float>();
  if (in["m_denoiserStrength"])
    m_denoiserStrength = in["m_denoiserStrength"].as<float>();
  if (in["m_resolution"])
    m_resolution = in["m_resolution"].as<glm::ivec2>();
  if (in["m_useClearColor"])
    m_useClearColor = in["m_useClearColor"].as<bool>();
  if (in["m_backgroundColor"])
    m_backgroundColor = in["m_backgroundColor"].as<glm::vec3>();
  if (in["m_backgroundColorIntensity"])
    m_backgroundColorIntensity = in["m_backgroundColorIntensity"].as<float>();
  if (in["m_cameraMin"])
    m_cameraMin = in["m_cameraMin"].as<float>();
  if (in["m_cameraMax"])
    m_cameraMax = in["m_cameraMax"].as<float>();
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
  if (m_labPrefab.Get<Prefab>()) {
    m_lab = m_labPrefab.Get<Prefab>()->ToEntity();
  }
  if (m_dirtPrefab.Get<Prefab>()) {
    m_dirt = m_dirtPrefab.Get<Prefab>()->ToEntity();
  }
  m_captureStatus = MultipleAngleCaptureStatus::Info;
  m_rayTracerCamera =
      Entities::CreateEntity(Entities::GetCurrentScene(), "Ray tracer");
  m_rayTracerCamera.SetParent(pipeline.GetOwner());

  auto rayTracerCamera =
      m_rayTracerCamera.GetOrSetPrivateComponent<RayTracerCamera>().lock();
  rayTracerCamera->SetMainCamera(true);

  m_sorghumInfos.clear();
  std::filesystem::create_directories(
      ProjectManager::GetProjectPath().parent_path().parent_path() / m_currentExportFolder /
      m_name);
  if (m_captureImage) {
    std::filesystem::create_directories(
        ProjectManager::GetProjectPath().parent_path().parent_path() / m_currentExportFolder /
        m_name / "Image");
  }
  if (m_captureMask) {
    std::filesystem::create_directories(
        ProjectManager::GetProjectPath().parent_path().parent_path() / m_currentExportFolder /
        m_name / "Mask");
  }
  if (m_captureMesh) {
    std::filesystem::create_directories(
        ProjectManager::GetProjectPath().parent_path().parent_path() / m_currentExportFolder /
        m_name / "Mesh");
  }
}
void GeneralDataCapture::End(AutoSorghumGenerationPipeline &pipeline) {
  ProjectManager::ScanProjectFolder(true);
  if (m_lab.IsValid())
    Entities::DeleteEntity(Entities::GetCurrentScene(), m_lab);
  if (m_dirt.IsValid())
    Entities::DeleteEntity(Entities::GetCurrentScene(), m_dirt);
  Entities::DeleteEntity(Entities::GetCurrentScene(), m_rayTracerCamera);
  m_rayTracerCamera = {};
}
