//
// Created by lllll on 9/9/2021.
//
#ifdef RAYTRACERFACILITY
#include "GeneralDataCapture.hpp"
#include "Prefab.hpp"
#include "TransformLayer.hpp"
#include <SorghumData.hpp>
#include <SorghumStateGenerator.hpp>
using namespace Scripts;
void GeneralDataCapture::OnInspect() {
  if (ImGui::Button("Instantiate pipeline")) {
    Instantiate();
  }
  ImGui::Text("Current output folder: %s",
              m_currentExportFolder.string().c_str());
  FileUtils::OpenFolder(
      "Choose output folder...",
      [&](const std::filesystem::path &path) {
        m_currentExportFolder = std::filesystem::absolute(path);
      },
      false);
  Editor::DragAndDropButton<Prefab>(m_labPrefab, "Lab");
  Editor::DragAndDropButton<Prefab>(m_dirtPrefab, "Dirt");
  if (ImGui::TreeNode("Data selection")) {
    ImGui::Checkbox("Capture image", &m_captureImage);
    ImGui::Checkbox("Capture mask", &m_captureMask);
    ImGui::Checkbox("Capture mesh", &m_captureMesh);
    ImGui::Checkbox("Capture depth", &m_captureDepth);
    ImGui::Checkbox("Export matrices", &m_exportMatrices);
    ImGui::TreePop();
  }
  if (ImGui::TreeNode("Camera Settings")) {
    ImGui::DragInt("Bounces", &m_rayProperties.m_bounces, 1, 1, 16);
    ImGui::DragInt("Samples", &m_rayProperties.m_samples, 1, 1, 2048);
    ImGui::DragFloat("Denoiser strength", &m_denoiserStrength, 0.01f);
    ImGui::DragFloat("Distance to center", &m_distanceToCenter, 0.01f);
    ImGui::DragFloat("Height", &m_height, 0.01f);
    ImGui::DragFloat("Top distance to center", &m_topDistanceToCenter, 0.01f);
    ImGui::DragInt3("Turn angle Start/Step/End", &m_turnAngleStart);
    ImGui::DragInt3("Top turn angle Start/Step/End", &m_topTurnAngleStart);
    ImGui::Separator();
    ImGui::DragFloat("Camera FOV", &m_fov);
    ImGui::DragFloat("Camera gamma", &m_gamma, 0.01f);
    ImGui::DragInt2("Camera Resolution", &m_resolution.x);
    ImGui::DragFloat("Camera max distance", &m_cameraMax);
    ImGui::Checkbox("Use clear color", &m_useClearColor);
    ImGui::ColorEdit3("Env lighting color", &m_backgroundColor.x);
    ImGui::DragFloat("Env lighting intensity", &m_backgroundColorIntensity);
    ImGui::TreePop();
  }
}

void GeneralDataCapture::OnBeforeGrowth(
    AutoSorghumGenerationPipeline &pipeline) {
  SetUpCamera(pipeline);
  pipeline.m_currentGrowingSorghum =
      Application::GetLayer<SorghumLayer>()->CreateSorghum(
          pipeline.m_currentUsingDescriptor.Get<SorghumStateGenerator>());
  auto scene = pipeline.GetScene();
  auto sorghumData = scene
                         ->GetOrSetPrivateComponent<SorghumData>(
                             pipeline.m_currentGrowingSorghum)
                         .lock();
  sorghumData->m_seed = pipeline.GetSeed();
  pipeline.m_status = AutoSorghumGenerationPipelineStatus::Growth;
}
void GeneralDataCapture::OnGrowth(AutoSorghumGenerationPipeline &pipeline) {
  pipeline.m_status = AutoSorghumGenerationPipelineStatus::AfterGrowth;
}

void GeneralDataCapture::OnAfterGrowth(
    AutoSorghumGenerationPipeline &pipeline) {
  auto scene = pipeline.GetScene();
  auto rayTracerCamera =
      scene->GetOrSetPrivateComponent<RayTracerCamera>(pipeline.GetOwner())
          .lock();
  m_sorghumInfos.push_back({GlobalTransform(), pipeline.m_prefix});

  if (m_captureMask) {
    if (scene->IsEntityValid(m_lab))
      scene->SetEnable(m_lab, false);
    if (scene->IsEntityValid(m_dirt))
      scene->SetEnable(m_dirt, false);
    auto sorghumData = scene
                           ->GetOrSetPrivateComponent<SorghumData>(
                               pipeline.m_currentGrowingSorghum)
                           .lock();
    sorghumData->m_seperated = true;
    sorghumData->m_includeStem = true;
    sorghumData->m_segmentedMask = true;
    sorghumData->GenerateGeometry();
    sorghumData->ApplyGeometry();
    Application::GetLayer<RayTracerLayer>()
        ->m_environmentProperties.m_environmentalLightingType =
        RayTracerFacility::EnvironmentalLightingType::Scene;
    scene->m_environmentSettings.m_backgroundColor = glm::vec3(1.0f);
    scene->m_environmentSettings.m_ambientLightIntensity = 1.0f;
    RayProperties rayProperties;
    rayProperties.m_samples = 1;
    rayProperties.m_bounces = 1;
    rayTracerCamera->SetDenoiserStrength(0.0f);
    rayTracerCamera->SetOutputType(OutputType::Albedo);
    GlobalTransform cameraGT;
    cameraGT.SetPosition(glm::vec3(0, m_height, m_distanceToCenter));
    cameraGT.SetRotation(glm::vec3(0, 0, 0));
    scene->SetDataComponent(pipeline.GetOwner(), cameraGT);

    for (int turnAngle = m_turnAngleStart; turnAngle <= m_turnAngleEnd;
         turnAngle += m_turnAngleStep) {
      auto sorghumGT = scene->GetDataComponent<GlobalTransform>(
          pipeline.m_currentGrowingSorghum);
      sorghumGT.SetRotation(glm::vec3(0, glm::radians((float)turnAngle), 0));
      scene->SetDataComponent(pipeline.m_currentGrowingSorghum, sorghumGT);
      Application::GetLayer<TransformLayer>()->CalculateTransformGraphs(scene);
      Application::GetLayer<RayTracerLayer>()->UpdateScene();
      rayTracerCamera->Render(rayProperties);
      rayTracerCamera->m_colorTexture->Export(
          m_currentExportFolder / GetAssetRecord().lock()->GetAssetFileName() /
          "Mask" /
          ("side_" + pipeline.m_prefix + "_" + std::to_string(turnAngle) +
           "_mask.png"));
    }
    cameraGT.SetPosition(glm::vec3(0, m_topDistanceToCenter, 0));
    cameraGT.SetRotation(glm::vec3(glm::radians(-90.0f), 0, 0));
    scene->SetDataComponent(pipeline.GetOwner(), cameraGT);
    for (int turnAngle = m_topTurnAngleStart; turnAngle <= m_topTurnAngleEnd;
         turnAngle += m_topTurnAngleStep) {
      auto sorghumGT = scene->GetDataComponent<GlobalTransform>(
          pipeline.m_currentGrowingSorghum);
      sorghumGT.SetRotation(glm::vec3(0, glm::radians((float)turnAngle), 0));
      scene->SetDataComponent(pipeline.m_currentGrowingSorghum, sorghumGT);
      Application::GetLayer<TransformLayer>()->CalculateTransformGraphs(scene);
      Application::GetLayer<RayTracerLayer>()->UpdateScene();
      rayTracerCamera->Render(rayProperties);
      rayTracerCamera->m_colorTexture->Export(
          m_currentExportFolder / GetAssetRecord().lock()->GetAssetFileName() /
          "Mask" /
          ("top_" + pipeline.m_prefix + "_" + std::to_string(turnAngle) +
           "_mask.png"));
    }
  }

  if (m_captureDepth) {
    if (scene->IsEntityValid(m_lab))
      scene->SetEnable(m_lab, true);
    if (scene->IsEntityValid(m_dirt)) {
      scene->SetEnable(m_dirt, true);
      auto dirtGT = scene->GetDataComponent<GlobalTransform>(m_dirt);
      dirtGT.SetRotation(
          dirtGT.GetRotation() *
          glm::quat(
              glm::vec3(0, glm::linearRand(0.0f, 2.0f * glm::pi<float>()), 0)));
      scene->SetDataComponent(m_dirt, dirtGT);
    }
    auto sorghumData = scene
                           ->GetOrSetPrivateComponent<SorghumData>(
                               pipeline.m_currentGrowingSorghum)
                           .lock();
    if (!m_captureMask) {
      sorghumData->m_seperated = true;
      sorghumData->m_includeStem = true;
      sorghumData->m_segmentedMask = false;
      sorghumData->GenerateGeometry();
      sorghumData->ApplyGeometry();
    }
    RayProperties rayProperties;
    rayProperties.m_samples = 1;
    rayProperties.m_bounces = 1;
    Application::GetLayer<RayTracerLayer>()
        ->m_environmentProperties.m_environmentalLightingType =
        RayTracerFacility::EnvironmentalLightingType::Scene;
    scene->m_environmentSettings.m_backgroundColor = m_backgroundColor;
    scene->m_environmentSettings.m_ambientLightIntensity =
        m_backgroundColorIntensity;
    rayTracerCamera->SetOutputType(OutputType::Depth);
    rayTracerCamera->SetDenoiserStrength(0.0f);
    rayTracerCamera->SetMaxDistance(m_cameraMax);
    GlobalTransform cameraGT;
    cameraGT.SetPosition(glm::vec3(0, m_height, m_distanceToCenter));
    cameraGT.SetRotation(glm::vec3(0, 0, 0));
    scene->SetDataComponent(pipeline.GetOwner(), cameraGT);
    for (int turnAngle = m_turnAngleStart; turnAngle <= m_turnAngleEnd;
         turnAngle += m_turnAngleStep) {
      auto sorghumGT = scene->GetDataComponent<GlobalTransform>(
          pipeline.m_currentGrowingSorghum);
      sorghumGT.SetRotation(glm::vec3(0, glm::radians((float)turnAngle), 0));
      scene->SetDataComponent(pipeline.m_currentGrowingSorghum, sorghumGT);
      Application::GetLayer<TransformLayer>()->CalculateTransformGraphs(scene);
      Application::GetLayer<RayTracerLayer>()->UpdateScene();
      rayTracerCamera->Render(rayProperties);
      rayTracerCamera->m_colorTexture->Export(
          m_currentExportFolder / GetAssetRecord().lock()->GetAssetFileName() /
          "Depth" /
          ("side_" + pipeline.m_prefix + "_" + std::to_string(turnAngle) +
           "_depth.hdr"));
    }
    cameraGT.SetPosition(glm::vec3(0, m_topDistanceToCenter, 0));
    cameraGT.SetRotation(glm::vec3(glm::radians(-90.0f), 0, 0));
    scene->SetDataComponent(pipeline.GetOwner(), cameraGT);
    for (int turnAngle = m_topTurnAngleStart; turnAngle <= m_topTurnAngleEnd;
         turnAngle += m_topTurnAngleStep) {
      auto sorghumGT = scene->GetDataComponent<GlobalTransform>(
          pipeline.m_currentGrowingSorghum);
      sorghumGT.SetRotation(glm::vec3(0, glm::radians((float)turnAngle), 0));
      scene->SetDataComponent(pipeline.m_currentGrowingSorghum, sorghumGT);
      Application::GetLayer<TransformLayer>()->CalculateTransformGraphs(scene);
      Application::GetLayer<RayTracerLayer>()->UpdateScene();
      rayTracerCamera->Render(rayProperties);
      rayTracerCamera->m_colorTexture->Export(
          m_currentExportFolder / GetAssetRecord().lock()->GetAssetFileName() /
          "Depth" /
          ("top_" + pipeline.m_prefix + "_" + std::to_string(turnAngle) +
           "_depth.hdr"));
    }
  }
  if (m_captureImage) {
    if (scene->IsEntityValid(m_lab))
      scene->SetEnable(m_lab, true);
    if (scene->IsEntityValid(m_dirt)) {
      scene->SetEnable(m_dirt, true);
      auto dirtGT = scene->GetDataComponent<GlobalTransform>(m_dirt);
      dirtGT.SetRotation(
          dirtGT.GetRotation() *
          glm::quat(
              glm::vec3(0, glm::linearRand(0.0f, 2.0f * glm::pi<float>()), 0)));
      scene->SetDataComponent(m_dirt, dirtGT);
    }

    auto sorghumData = scene
                           ->GetOrSetPrivateComponent<SorghumData>(
                               pipeline.m_currentGrowingSorghum)
                           .lock();
    sorghumData->m_seperated = true;
    sorghumData->m_includeStem = true;
    sorghumData->m_segmentedMask = false;
    sorghumData->GenerateGeometry();
    sorghumData->ApplyGeometry();

    Application::GetLayer<RayTracerLayer>()
        ->m_environmentProperties.m_environmentalLightingType =
        RayTracerFacility::EnvironmentalLightingType::Scene;
    scene->m_environmentSettings.m_backgroundColor = m_backgroundColor;
    scene->m_environmentSettings.m_ambientLightIntensity =
        m_backgroundColorIntensity;
    rayTracerCamera->SetOutputType(OutputType::Color);
    rayTracerCamera->SetDenoiserStrength(m_denoiserStrength);
    GlobalTransform cameraGT;
    cameraGT.SetPosition(glm::vec3(0, m_height, m_distanceToCenter));
    cameraGT.SetRotation(glm::vec3(0, 0, 0));
    scene->SetDataComponent(pipeline.GetOwner(), cameraGT);
    for (int turnAngle = m_turnAngleStart; turnAngle <= m_turnAngleEnd;
         turnAngle += m_turnAngleStep) {
      auto sorghumGT = scene->GetDataComponent<GlobalTransform>(
          pipeline.m_currentGrowingSorghum);
      sorghumGT.SetRotation(glm::vec3(0, glm::radians((float)turnAngle), 0));
      scene->SetDataComponent(pipeline.m_currentGrowingSorghum, sorghumGT);
      Application::GetLayer<TransformLayer>()->CalculateTransformGraphs(scene);
      Application::GetLayer<RayTracerLayer>()->UpdateScene();
      rayTracerCamera->Render(m_rayProperties);
      rayTracerCamera->m_colorTexture->Export(
          m_currentExportFolder / GetAssetRecord().lock()->GetAssetFileName() /
          "Image" /
          ("side_" + pipeline.m_prefix + "_" + std::to_string(turnAngle) +
           "_image.png"));
    }
    cameraGT.SetPosition(glm::vec3(0, m_topDistanceToCenter, 0));
    cameraGT.SetRotation(glm::vec3(glm::radians(-90.0f), 0, 0));
    scene->SetDataComponent(pipeline.GetOwner(), cameraGT);
    for (int turnAngle = m_topTurnAngleStart; turnAngle <= m_topTurnAngleEnd;
         turnAngle += m_topTurnAngleStep) {
      auto sorghumGT = scene->GetDataComponent<GlobalTransform>(
          pipeline.m_currentGrowingSorghum);
      sorghumGT.SetRotation(glm::vec3(0, glm::radians((float)turnAngle), 0));
      scene->SetDataComponent(pipeline.m_currentGrowingSorghum, sorghumGT);
      Application::GetLayer<TransformLayer>()->CalculateTransformGraphs(scene);
      Application::GetLayer<RayTracerLayer>()->UpdateScene();
      rayTracerCamera->Render(m_rayProperties);
      rayTracerCamera->m_colorTexture->Export(
          m_currentExportFolder / GetAssetRecord().lock()->GetAssetFileName() /
          "Image" /
          ("top_" + pipeline.m_prefix + "_" + std::to_string(turnAngle) +
           "_image.png"));
    }
  }
  if (m_captureMesh) {
    auto sorghumData =
        scene->GetOrSetPrivateComponent<SorghumData>(pipeline.m_currentGrowingSorghum)
            .lock();
    if (!m_captureDepth && !m_captureMask && !m_captureImage) {
      sorghumData->m_seperated = true;
      sorghumData->m_includeStem = true;
      sorghumData->m_segmentedMask = false;
      sorghumData->GenerateGeometry();
      sorghumData->ApplyGeometry();
    }
    sorghumData->ExportModel(
        (m_currentExportFolder / GetAssetRecord().lock()->GetAssetFileName() /
         "Mesh" / (pipeline.m_prefix + ".obj"))
            .string());
  }

  if ((m_captureImage || m_captureMask || m_captureDepth) && m_exportMatrices) {
    GlobalTransform cameraGT;
    cameraGT.SetPosition(glm::vec3(0, m_height, m_distanceToCenter));
    cameraGT.SetRotation(glm::vec3(0, 0, 0));
    for (int turnAngle = m_turnAngleStart; turnAngle <= m_turnAngleEnd;
         turnAngle += m_turnAngleStep) {
      auto sorghumGT =
          scene->GetDataComponent<GlobalTransform>(pipeline.m_currentGrowingSorghum);
      sorghumGT.SetRotation(glm::vec3(0, glm::radians((float)turnAngle), 0));
      sorghumGT.SetRotation(glm::vec3(0, glm::radians((float)turnAngle), 0));
      m_cameraModels.push_back(cameraGT.m_value);
      m_treeModels.push_back(sorghumGT.m_value);
      m_projections.push_back(Camera::m_cameraInfoBlock.m_projection);
      m_views.push_back(Camera::m_cameraInfoBlock.m_view);
      m_names.push_back("side_" + pipeline.m_prefix + "_" +
                        std::to_string(turnAngle));
    }

    cameraGT.SetPosition(glm::vec3(0, m_topDistanceToCenter, 0));
    cameraGT.SetRotation(glm::vec3(glm::radians(-90.0f), 0, 0));
    for (int turnAngle = m_topTurnAngleStart; turnAngle <= m_topTurnAngleEnd;
         turnAngle += m_topTurnAngleStep) {
      auto sorghumGT =
          scene->GetDataComponent<GlobalTransform>(pipeline.m_currentGrowingSorghum);
      sorghumGT.SetRotation(glm::vec3(0, glm::radians((float)turnAngle), 0));
      m_cameraModels.push_back(cameraGT.m_value);
      m_treeModels.push_back(sorghumGT.m_value);
      m_projections.push_back(Camera::m_cameraInfoBlock.m_projection);
      m_views.push_back(Camera::m_cameraInfoBlock.m_view);
      m_names.push_back("top_" + pipeline.m_prefix + "_" +
                        std::to_string(turnAngle));
    }
  }

  if (scene->IsEntityValid(pipeline.m_currentGrowingSorghum))
    scene->DeleteEntity(pipeline.m_currentGrowingSorghum);
  pipeline.m_currentGrowingSorghum = {};
  pipeline.m_status = AutoSorghumGenerationPipelineStatus::Idle;
}
void GeneralDataCapture::SetUpCamera(AutoSorghumGenerationPipeline &pipeline) {
  auto scene = pipeline.GetScene();
  auto rayTracerCamera =
      scene->GetOrSetPrivateComponent<RayTracerCamera>(pipeline.GetOwner()).lock();

  Application::GetLayer<RayTracerLayer>()
      ->m_environmentProperties.m_environmentalLightingType =
      RayTracerFacility::EnvironmentalLightingType::Scene;
  scene->m_environmentSettings.m_environmentType =
      UniEngine::EnvironmentType::Color;
  scene->m_environmentSettings.m_backgroundColor = m_backgroundColor;
  scene->m_environmentSettings.m_ambientLightIntensity =
      m_backgroundColorIntensity;
  rayTracerCamera->SetFov(m_fov);
  rayTracerCamera->SetGamma(m_gamma);
  rayTracerCamera->m_allowAutoResize = false;
  rayTracerCamera->m_frameSize = m_resolution;
  rayTracerCamera->SetAccumulate(false);
}
void GeneralDataCapture::CollectAssetRef(std::vector<AssetRef> &list) {
  list.push_back(m_labPrefab);
  list.push_back(m_dirtPrefab);
}
void GeneralDataCapture::Serialize(YAML::Emitter &out) {
  m_labPrefab.Save("m_labPrefab", out);
  m_dirtPrefab.Save("m_dirtPrefab", out);
  out << YAML::Key << "m_rayProperties.m_samples" << YAML::Value
      << m_rayProperties.m_samples;
  out << YAML::Key << "m_rayProperties.m_bounces" << YAML::Value
      << m_rayProperties.m_bounces;
  out << YAML::Key << "m_captureDepth" << YAML::Value << m_captureDepth;
  out << YAML::Key << "m_captureImage" << YAML::Value << m_captureImage;
  out << YAML::Key << "m_captureMask" << YAML::Value << m_captureMask;
  out << YAML::Key << "m_captureMesh" << YAML::Value << m_captureMesh;
  out << YAML::Key << "m_exportMatrices" << YAML::Value << m_exportMatrices;
  out << YAML::Key << "m_currentExportFolder" << YAML::Value
      << m_currentExportFolder.string();
  out << YAML::Key << "m_distanceToCenter" << YAML::Value << m_distanceToCenter;
  out << YAML::Key << "m_height" << YAML::Value << m_height;
  out << YAML::Key << "m_topDistanceToCenter" << YAML::Value
      << m_topDistanceToCenter;
  out << YAML::Key << "m_turnAngleStart" << YAML::Value << m_turnAngleStart;
  out << YAML::Key << "m_turnAngleStep" << YAML::Value << m_turnAngleStep;
  out << YAML::Key << "m_turnAngleEnd" << YAML::Value << m_turnAngleEnd;

  out << YAML::Key << "m_topTurnAngleStart" << YAML::Value
      << m_topTurnAngleStart;
  out << YAML::Key << "m_topTurnAngleStep" << YAML::Value << m_topTurnAngleStep;
  out << YAML::Key << "m_topTurnAngleEnd" << YAML::Value << m_topTurnAngleEnd;

  out << YAML::Key << "m_fov" << YAML::Value << m_fov;
  out << YAML::Key << "m_gamma" << YAML::Value << m_gamma;
  out << YAML::Key << "m_denoiserStrength" << YAML::Value << m_denoiserStrength;
  out << YAML::Key << "m_resolution" << YAML::Value << m_resolution;
  out << YAML::Key << "m_useClearColor" << YAML::Value << m_useClearColor;
  out << YAML::Key << "m_backgroundColor" << YAML::Value << m_backgroundColor;
  out << YAML::Key << "m_backgroundColorIntensity" << YAML::Value
      << m_backgroundColorIntensity;
  out << YAML::Key << "m_cameraMax" << YAML::Value << m_cameraMax;
}
void GeneralDataCapture::Deserialize(const YAML::Node &in) {
  m_labPrefab.Load("m_labPrefab", in);
  m_dirtPrefab.Load("m_dirtPrefab", in);
  if (in["m_rayProperties.m_samples"])
    m_rayProperties.m_samples = in["m_rayProperties.m_samples"].as<float>();
  if (in["m_rayProperties.m_bounces"])
    m_rayProperties.m_bounces = in["m_rayProperties.m_bounces"].as<float>();
  if (in["m_captureDepth"])
    m_captureDepth = in["m_captureDepth"].as<bool>();
  if (in["m_captureImage"])
    m_captureImage = in["m_captureImage"].as<bool>();
  if (in["m_captureMask"])
    m_captureMask = in["m_captureMask"].as<bool>();
  if (in["m_captureMesh"])
    m_captureMesh = in["m_captureMesh"].as<bool>();
  if (in["m_exportMatrices"])
    m_exportMatrices = in["m_exportMatrices"].as<bool>();

  if (in["m_currentExportFolder"])
    m_currentExportFolder = in["m_currentExportFolder"].as<std::string>();
  if (in["m_distanceToCenter"])
    m_distanceToCenter = in["m_distanceToCenter"].as<float>();

  if (in["m_topDistanceToCenter"])
    m_topDistanceToCenter = in["m_topDistanceToCenter"].as<float>();

  if (in["m_height"])
    m_height = in["m_height"].as<float>();
  if (in["m_turnAngleStart"])
    m_turnAngleStart = in["m_turnAngleStart"].as<int>();
  if (in["m_turnAngleStep"])
    m_turnAngleStep = in["m_turnAngleStep"].as<int>();
  if (in["m_turnAngleEnd"])
    m_turnAngleEnd = in["m_turnAngleEnd"].as<int>();

  if (in["m_topTurnAngleStart"])
    m_topTurnAngleStart = in["m_topTurnAngleStart"].as<int>();
  if (in["m_topTurnAngleStep"])
    m_topTurnAngleStep = in["m_topTurnAngleStep"].as<int>();
  if (in["m_topTurnAngleEnd"])
    m_topTurnAngleEnd = in["m_topTurnAngleEnd"].as<int>();

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
  if (in["m_cameraMax"])
    m_cameraMax = in["m_cameraMax"].as<float>();
}
void GeneralDataCapture::Instantiate() {
  auto scene = Application::GetActiveScene();
  auto pipelineEntity = scene->CreateEntity(GetAssetRecord().lock()->GetAssetFileName());
  auto pipeline =
      scene->GetOrSetPrivateComponent<AutoSorghumGenerationPipeline>(pipelineEntity)
          .lock();
  pipeline->m_pipelineBehaviour =
      std::dynamic_pointer_cast<GeneralDataCapture>(m_self.lock());
}

void GeneralDataCapture::OnStart(AutoSorghumGenerationPipeline &pipeline) {
  auto scene = pipeline.GetScene();
  if (m_labPrefab.Get<Prefab>()) {
    m_lab = m_labPrefab.Get<Prefab>()->ToEntity(scene);
  }
  if (m_dirtPrefab.Get<Prefab>()) {
    m_dirt = m_dirtPrefab.Get<Prefab>()->ToEntity(scene);
  }
  auto rayTracerCamera =
      scene->GetOrSetPrivateComponent<RayTracerCamera>(pipeline.GetOwner()).lock();
  rayTracerCamera->SetMainCamera(true);

  m_sorghumInfos.clear();
  std::filesystem::create_directories(
      m_currentExportFolder / GetAssetRecord().lock()->GetAssetFileName());
  if (m_captureImage) {
    std::filesystem::create_directories(
        m_currentExportFolder / GetAssetRecord().lock()->GetAssetFileName() /
        "Image");
  }
  if (m_captureMask) {
    std::filesystem::create_directories(
        m_currentExportFolder / GetAssetRecord().lock()->GetAssetFileName() /
        "Mask");
  }
  if (m_captureMesh) {
    std::filesystem::create_directories(
        m_currentExportFolder / GetAssetRecord().lock()->GetAssetFileName() /
        "Mesh");
  }
  if (m_captureDepth) {
    std::filesystem::create_directories(
        m_currentExportFolder / GetAssetRecord().lock()->GetAssetFileName() /
        "Depth");
  }
}
void GeneralDataCapture::OnEnd(AutoSorghumGenerationPipeline &pipeline) {
  auto scene = pipeline.GetScene();
  if (scene->IsEntityValid(m_lab))
    scene->DeleteEntity(m_lab);
  if (scene->IsEntityValid(m_dirt))
    scene->DeleteEntity(m_dirt);

  if ((m_captureImage || m_captureMask || m_captureDepth) && m_exportMatrices)
    ExportMatrices(m_currentExportFolder /
                   GetAssetRecord().lock()->GetAssetFileName() /
                   "matrices.yml");
}

void GeneralDataCapture::ExportMatrices(const std::filesystem::path &path) {
  YAML::Emitter out;
  out << YAML::BeginMap;
  out << YAML::Key << "Capture Info" << YAML::BeginSeq;
  for (int i = 0; i < m_projections.size(); i++) {
    out << YAML::BeginMap;
    out << YAML::Key << "File Prefix" << YAML::Value << m_names[i];
    out << YAML::Key << "Projection" << YAML::Value << m_projections[i];
    out << YAML::Key << "View" << YAML::Value << m_views[i];
    out << YAML::Key << "Camera Transform" << YAML::Value << m_cameraModels[i];
    out << YAML::Key << "Plant Transform" << YAML::Value << m_treeModels[i];
    out << YAML::EndMap;
  }
  out << YAML::EndSeq;
  out << YAML::EndMap;
  std::ofstream fout(path.string());
  fout << out.c_str();
  fout.flush();
}

#endif