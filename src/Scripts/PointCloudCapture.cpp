//
// Created by lllll on 1/1/2022.
//
#include "PointCloudCapture.hpp"
#include "FieldGround.hpp"
#include "LeafData.hpp"
#include <Tinyply.hpp>
#ifdef RAYTRACERFACILITY
#include "PARSensorGroup.hpp"
#include "RayTracerLayer.hpp"
#include "BTFMeshRenderer.hpp"
#include <TriangleIlluminationEstimator.hpp>
using namespace RayTracerFacility;
#endif

using namespace Scripts;
using namespace tinyply;

void PointCloudCapture::OnBeforeGrowth(
    AutoSorghumGenerationPipeline &pipeline) {
  auto positionsField = m_positionsField.Get<PositionsField>();
  if (!positionsField) {
    UNIENGINE_ERROR("Invalid position field");
    Reset(pipeline);
    return;
  }
  auto scene = pipeline.GetScene();
  auto positionField = m_positionsField.Get<PositionsField>();
  positionField->m_sorghumStateGenerator = pipeline.m_currentUsingDescriptor.Get<SorghumStateGenerator>();
  positionField->m_seperated = true;
  auto fieldGround =
      scene->GetOrSetPrivateComponent<FieldGround>(m_ground).lock();
  fieldGround->m_scale = m_settings.m_scale;
  fieldGround->m_size = m_settings.m_size;
  fieldGround->m_rowWidth = m_settings.m_rowWidth;
  fieldGround->m_alleyDepth = m_settings.m_alleyDepth;
  fieldGround->m_noiseScale = m_settings.m_noiseScale;
  fieldGround->m_noiseIntensity = m_settings.m_noiseIntensity;
  Transform fieldGroundTransform;
  fieldGroundTransform.SetPosition(glm::vec3(0, glm::linearRand(0.0f, 0.15f), 0));
  scene->SetDataComponent(m_ground, fieldGroundTransform);
  fieldGround->GenerateMesh(glm::linearRand(0.12f, 0.17f));
  auto result = positionsField->InstantiateAroundIndex(
      pipeline.GetSeed() % positionsField->m_positions.size(), 2.5f, m_currentCenter, m_settings.m_positionVariance);
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
void PointCloudCapture::OnGrowth(AutoSorghumGenerationPipeline &pipeline) {
  pipeline.m_status = AutoSorghumGenerationPipelineStatus::AfterGrowth;
}
void PointCloudCapture::OnAfterGrowth(AutoSorghumGenerationPipeline &pipeline) {
  ScanPointCloudLabeled(
      pipeline,
      m_currentExportFolder / GetAssetRecord().lock()->GetAssetFileName() /
          "PointCloud" / (pipeline.m_prefix + std::string(".ply")),
      m_settings);
  ExportCSV(
      pipeline,
      m_currentExportFolder / GetAssetRecord().lock()->GetAssetFileName() /
          "CSV" / (pipeline.m_prefix + std::string(".csv")));

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

void PointCloudCapture::OnStart(AutoSorghumGenerationPipeline &pipeline) {
  std::filesystem::create_directories(
      m_currentExportFolder / GetAssetRecord().lock()->GetAssetFileName() /
      "PointCloud");
  std::filesystem::create_directories(
      m_currentExportFolder / GetAssetRecord().lock()->GetAssetFileName() /
      "CSV");
  auto scene = pipeline.GetScene();

  m_ground = scene->CreateEntity("Ground");
  auto fieldGround =
      scene->GetOrSetPrivateComponent<FieldGround>(m_ground).lock();
}
void PointCloudCapture::OnEnd(AutoSorghumGenerationPipeline &pipeline) {
  auto scene = pipeline.GetScene();
  if (scene->IsEntityValid(m_currentSorghumField))
    scene->DeleteEntity(m_currentSorghumField);
  pipeline.m_currentGrowingSorghum = m_currentSorghumField = {};

  scene->DeleteEntity(m_ground);
}
void PointCloudCapture::Reset(AutoSorghumGenerationPipeline &pipeline) {
  auto scene = pipeline.GetScene();
  if (scene->IsEntityValid(m_currentSorghumField))
    scene->DeleteEntity(m_currentSorghumField);
  pipeline.m_currentGrowingSorghum = m_currentSorghumField = Entity();
}
void PointCloudCapture::Instantiate() {
  auto scene = Application::GetActiveScene();
  auto pointCloudCaptureEntity = scene->CreateEntity("PointCloudPipeline");
  auto pointCloudPipeline =
      scene
          ->GetOrSetPrivateComponent<AutoSorghumGenerationPipeline>(
              pointCloudCaptureEntity)
          .lock();
  pointCloudPipeline->m_pipelineBehaviour =
      std::dynamic_pointer_cast<PointCloudCapture>(m_self.lock());
}
void PointCloudCapture::CollectAssetRef(std::vector<AssetRef> &list) {
  list.push_back(m_positionsField);
}
void PointCloudCapture::Serialize(YAML::Emitter &out) {
  m_positionsField.Save("m_positionsField", out);
  out << YAML::Key << "m_currentExportFolder" << YAML::Value
      << m_currentExportFolder.string();
  m_settings.Serialize("m_settings", out);
}
void PointCloudCapture::Deserialize(const YAML::Node &in) {
  m_positionsField.Load("m_positionsField", in);
  if (in["m_currentExportFolder"])
    m_currentExportFolder = in["m_currentExportFolder"].as<std::string>();
  m_settings.Deserialize("m_settings", in);
}

void PointCloudSampleSettings::OnInspect() {
  ImGui::DragFloat2("Point distance", &m_pointDistance.x, 0.0001f);
  ImGui::DragFloat("Scanner angle", &m_scannerAngle, 0.5f);

  ImGui::DragFloat2("Bounding box height range", &m_boundingBoxHeightRange.x,
                    0.01f);

  ImGui::Checkbox("Auto adjust bounding box", &m_adjustBoundingBox);
  if (m_adjustBoundingBox) {
    ImGui::DragFloat("Adjustment factor", &m_adjustmentFactor, 0.01f, 0.0f,
                     2.0f);
  } else {
    ImGui::DragFloat("Bounding box radius", &m_boundingBoxRadius, 0.01f);
  }
  ImGui::DragFloat("Position Variance", &m_positionVariance);
  if(ImGui::TreeNodeEx("Ground settings", ImGuiTreeNodeFlags_DefaultOpen)){
    ImGui::DragInt2("Size", &m_size.x);
    ImGui::DragFloat("Row Width", &m_rowWidth);
    ImGui::DragFloat("Alley Depth", &m_alleyDepth);
    ImGui::DragFloat("Noise Scale", &m_noiseScale);
    ImGui::DragFloat("Noise Intensity", &m_noiseIntensity);
    ImGui::TreePop();
  }
}
void PointCloudSampleSettings::Serialize(const std::string &name,
                                         YAML::Emitter &out) const {
  out << YAML::Key << name << YAML::Value << YAML::BeginMap;

  out << YAML::Key << "m_boundingBoxHeightRange" << YAML::Value
      << m_boundingBoxHeightRange;
  out << YAML::Key << "m_pointDistance" << YAML::Value << m_pointDistance;
  out << YAML::Key << "m_scannerAngle" << YAML::Value << m_scannerAngle;
  out << YAML::Key << "m_adjustBoundingBox" << YAML::Value
      << m_adjustBoundingBox;
  out << YAML::Key << "m_boundingBoxRadius" << YAML::Value
      << m_boundingBoxRadius;
  out << YAML::Key << "m_adjustmentFactor" << YAML::Value << m_adjustmentFactor;
  out << YAML::Key << "m_segmentAmount" << YAML::Value << m_segmentAmount;


  out << YAML::Key << "m_scale" << YAML::Value << m_scale;
  out << YAML::Key << "m_size" << YAML::Value << m_size;
  out << YAML::Key << "m_rowWidth" << YAML::Value << m_rowWidth;
  out << YAML::Key << "m_alleyDepth" << YAML::Value << m_alleyDepth;
  out << YAML::Key << "m_noiseScale" << YAML::Value << m_noiseScale;
  out << YAML::Key << "m_noiseIntensity" << YAML::Value << m_noiseIntensity;

  out << YAML::Key << "m_positionVariance" << YAML::Value << m_positionVariance;
  out << YAML::EndMap;
}
void PointCloudSampleSettings::Deserialize(const std::string &name,
                                           const YAML::Node &in) {
  if (in[name]) {
    auto &cd = in[name];
    if (cd["m_boundingBoxHeightRange"])
      m_boundingBoxHeightRange = cd["m_boundingBoxHeightRange"].as<glm::vec2>();
    if (cd["m_pointDistance"])
      m_pointDistance = cd["m_pointDistance"].as<glm::vec2>();
    if (cd["m_scannerAngle"])
      m_scannerAngle = cd["m_scannerAngle"].as<float>();
    if (cd["m_adjustBoundingBox"])
      m_adjustBoundingBox = cd["m_adjustBoundingBox"].as<bool>();
    if (cd["m_boundingBoxRadius"])
      m_boundingBoxRadius = cd["m_boundingBoxRadius"].as<float>();
    if (cd["m_adjustmentFactor"])
      m_adjustmentFactor = cd["m_adjustmentFactor"].as<float>();
    if (cd["m_segmentAmount"])
      m_segmentAmount = cd["m_segmentAmount"].as<int>();

    if (cd["m_scale"])
      m_scale = cd["m_scale"].as<glm::vec2>();
    if (cd["m_size"])
      m_size = cd["m_size"].as<glm::ivec2>();
    if (cd["m_rowWidth"])
      m_rowWidth = cd["m_rowWidth"].as<float>();
    if (cd["m_alleyDepth"])
      m_alleyDepth = cd["m_alleyDepth"].as<float>();
    if (cd["m_noiseScale"])
      m_noiseScale = cd["m_noiseScale"].as<float>();
    if (cd["m_noiseIntensity"])
      m_noiseIntensity = cd["m_noiseIntensity"].as<float>();

    if (cd["m_positionVariance"])
      m_positionVariance = cd["m_positionVariance"].as<float>();
  }
}

void PointCloudCapture::ScanPointCloudLabeled(
    AutoSorghumGenerationPipeline &pipeline, const std::filesystem::path &savePath,
    const PointCloudSampleSettings &settings) {
#ifdef RAYTRACERFACILITY
  auto boundingBoxHeight =
      settings.m_boundingBoxHeightRange.y - settings.m_boundingBoxHeightRange.x;
  auto planeSize = glm::vec2(
      settings.m_boundingBoxRadius * 2.0f +
          boundingBoxHeight / glm::cos(glm::radians(settings.m_scannerAngle)),
      settings.m_boundingBoxRadius * 2.0f);
  auto boundingBoxCenter = (settings.m_boundingBoxHeightRange.y +
                            settings.m_boundingBoxHeightRange.x) /
                           2.0f;
  std::vector<int> isGround;
  std::vector<int> leafIndex;
  std::vector<int> leafPartIndex;
  std::vector<int> plantIndex;
  std::vector<int> isMainPlant;
  std::vector<uint64_t> meshRendererHandles;
  std::vector<glm::dvec3> points;
  std::vector<glm::vec3> colors;
  auto scene = pipeline.GetScene();
  const auto column = unsigned(planeSize.x / settings.m_pointDistance.x);
  const int columnStart = -(int)(column / 2);
  const auto row = unsigned(planeSize.y / settings.m_pointDistance.y);
  const int rowStart = -(int)(row / 2);
  const auto size = column * row;
  auto gt = scene->GetDataComponent<GlobalTransform>(pipeline.m_currentGrowingSorghum);

  glm::vec3 front = glm::vec3(0, -1, 0);
  glm::vec3 up = glm::vec3(0, 0, -1);
  glm::vec3 left = glm::vec3(1, 0, 0);
  glm::vec3 actualVector = glm::normalize(
      glm::rotate(front, glm::radians(settings.m_scannerAngle), up));
  glm::vec3 center =
      gt.GetPosition() + glm::vec3(0, boundingBoxCenter, 0) -
      actualVector * (settings.m_boundingBoxHeightRange.y / 2.0f /
                      glm::cos(glm::radians(settings.m_scannerAngle)));

  std::vector<PointCloudSample> pcSamples;
  pcSamples.resize(size * 2);
  std::vector<std::shared_future<void>> results;
  Jobs::ParallelFor(
      size,
      [&](unsigned i) {
        const int columnIndex = (int)i / row;
        const int rowIndex = (int)i % row;
        const auto position =
            center +
            left * (float)(columnStart + columnIndex) *
                settings.m_pointDistance.x +
            up * (float)(rowStart + rowIndex) * settings.m_pointDistance.y;
        pcSamples[i].m_start = position;
        pcSamples[i].m_direction = actualVector;
      },
      results);
  for (const auto &i : results)
    i.wait();
  auto plantPosition = gt.GetPosition();
  actualVector = glm::normalize(
      glm::rotate(front, glm::radians(-settings.m_scannerAngle), up));
  center = gt.GetPosition() + glm::vec3(0, boundingBoxCenter, 0) -
           actualVector * (boundingBoxHeight / 2.0f /
                           glm::cos(glm::radians(settings.m_scannerAngle)));

  std::vector<std::shared_future<void>> results2;
  Jobs::ParallelFor(
      size,
      [&](unsigned i) {
        const int columnIndex = (int)i / row;
        const int rowIndex = (int)i % row;
        const auto position =
            center +
            left * (float)(columnStart + columnIndex) *
                settings.m_pointDistance.x +
            up * (float)(rowStart + rowIndex) * settings.m_pointDistance.y;
        pcSamples[i + size].m_start = position;
        pcSamples[i + size].m_direction = actualVector;
      },
      results2);
  for (const auto &i : results2)
    i.wait();

  Handle groundHandle =
      scene->GetOrSetPrivateComponent<MeshRenderer>(m_ground)
          .lock()
          ->GetHandle();
  std::vector<std::pair<Handle, int>> mainPlantHandles = {};
  std::vector<std::vector<std::pair<Handle, int>>> plantHandles = {};
  mainPlantHandles.emplace_back(
      scene->GetOrSetPrivateComponent<MeshRenderer>(pipeline.m_currentGrowingSorghum)
          .lock()
          ->GetHandle(),
      0);
  scene->ForEachChild(pipeline.m_currentGrowingSorghum, [&](Entity child) {
    auto meshRenderer = scene->GetOrSetPrivateComponent<MeshRenderer>(child);
    if (meshRenderer.expired() || !scene->HasPrivateComponent<LeafData>(child))
      return;
    auto handle = meshRenderer.lock()->GetHandle();
    auto index =
        scene->GetOrSetPrivateComponent<LeafData>(child).lock()->m_index + 1;
    mainPlantHandles.emplace_back(handle, index);
  });

  for (const auto &i : scene->GetChildren(m_currentSorghumField)) {
    if (i.GetIndex() == pipeline.m_currentGrowingSorghum.GetIndex())
      continue;
    plantHandles.emplace_back();
    plantHandles.back().emplace_back(
        scene->GetOrSetPrivateComponent<MeshRenderer>(i).lock()->GetHandle(),
        0);
    scene->ForEachChild(i, [&](Entity child) {
      auto meshRenderer = scene->GetOrSetPrivateComponent<MeshRenderer>(child);
      if (meshRenderer.expired() || !scene->HasPrivateComponent<LeafData>(child))
        return;
      auto handle = meshRenderer.lock()->GetHandle();
      auto index =
          scene->GetOrSetPrivateComponent<LeafData>(child).lock()->m_index + 1;
      plantHandles.back().emplace_back(handle, index);
    });
  }

  CudaModule::SamplePointCloud(
      Application::GetLayer<RayTracerLayer>()->m_environmentProperties,
      pcSamples);
  if (settings.m_adjustBoundingBox) {
    std::mutex minX, minZ, maxX, maxZ;
    float minXVal, minZVal, maxXVal, maxZVal;
    minXVal = minZVal = 999999;
    maxXVal = maxZVal = -999999;
    std::vector<std::shared_future<void>> scanResults;
    Jobs::ParallelFor(
        pcSamples.size(),
        [&](unsigned i) {
          auto &sample = pcSamples[i];
          if (!sample.m_hit) {
            return;
          }
          for (const auto &pair : mainPlantHandles) {
            if (pair.first.GetValue() == sample.m_handle) {
              if (sample.m_end.x < minXVal) {
                std::lock_guard<std::mutex> lock(minX);
                minXVal = sample.m_end.x;
              } else if (sample.m_end.x > maxXVal) {
                std::lock_guard<std::mutex> lock(maxX);
                maxXVal = sample.m_end.x;
              }
              if (sample.m_end.z < minZVal) {
                std::lock_guard<std::mutex> lock(minZ);
                minZVal = sample.m_end.z;
              } else if (sample.m_end.z > maxZVal) {
                std::lock_guard<std::mutex> lock(maxZ);
                maxZVal = sample.m_end.z;
              }
              return;
            }
          }
        },
        scanResults);
    for (const auto &i : scanResults)
      i.wait();
    float xCenter = (minXVal + maxXVal) / 2.0f;
    float zCenter = (minZVal + maxZVal) / 2.0f;
    minXVal = xCenter + settings.m_adjustmentFactor * (minXVal - xCenter);
    maxXVal = xCenter + settings.m_adjustmentFactor * (maxXVal - xCenter);
    minZVal = zCenter + settings.m_adjustmentFactor * (minZVal - zCenter);
    maxZVal = zCenter + settings.m_adjustmentFactor * (maxZVal - zCenter);

    for (const auto &sample : pcSamples) {
      if (!sample.m_hit) {
        continue;
      }
      auto position = sample.m_end;
      if (position.x < minXVal || position.x > maxXVal ||
          position.z < minZVal || position.z > maxZVal ||
          position.y - plantPosition.y < settings.m_boundingBoxHeightRange.x ||
          position.y - plantPosition.y > settings.m_boundingBoxHeightRange.y) {
        continue;
      }
      points.push_back(sample.m_end);
      colors.push_back(sample.m_albedo);
      meshRendererHandles.push_back(sample.m_handle);
    }
  } else {
    for (const auto &sample : pcSamples) {
      if (!sample.m_hit) {
        continue;
      }
      auto position = sample.m_end;
      if (glm::abs(position.x - plantPosition.x) >
              settings.m_boundingBoxRadius ||
          glm::abs(position.z - plantPosition.z) >
              settings.m_boundingBoxRadius ||
          position.y - plantPosition.y < settings.m_boundingBoxHeightRange.x ||
          position.y - plantPosition.y > settings.m_boundingBoxHeightRange.y) {
        continue;
      }
      points.emplace_back(sample.m_end.x, sample.m_end.z, sample.m_end.y);
      colors.push_back(sample.m_albedo);
      meshRendererHandles.push_back(sample.m_handle);
    }
  }
  isGround.resize(points.size());
  leafIndex.resize(points.size());
  leafPartIndex.resize(points.size());
  isMainPlant.resize(points.size());
  plantIndex.resize(points.size());
  std::vector<std::shared_future<void>> results3;
  Jobs::ParallelFor(
      points.size(),
      [&](unsigned i) {
        if (colors[i].x != 0) {
          leafPartIndex[i] = 1;
        } else if (colors[i].y != 0) {
          leafPartIndex[i] = 2;
        } else {
          leafPartIndex[i] = 3;
        }
        isMainPlant[i] = 0;
        plantIndex[i] = 0;

        if (meshRendererHandles[i] == groundHandle)
          isGround[i] = 1;
        else
          isGround[i] = 0;

        for (const auto &pair : mainPlantHandles) {
          if (pair.first.GetValue() == meshRendererHandles[i]) {
            leafIndex[i] = pair.second;
            isMainPlant[i] = 1;
            return;
          }
        }

        int j = 0;
        for (const auto &leafPairs : plantHandles) {
          j++;
          for (const auto &pair : leafPairs) {
            if (pair.first.GetValue() == meshRendererHandles[i]) {
              leafIndex[i] = pair.second;
              plantIndex[i] = j;
              return;
            }
          }
        }
      },
      results3);
  for (const auto &i : results3)
    i.wait();

  std::filebuf fb_binary;
  fb_binary.open(savePath.string(), std::ios::out | std::ios::binary);
  std::ostream outstream_binary(&fb_binary);
  if (outstream_binary.fail())
    throw std::runtime_error("failed to open " + savePath.string());
  /*
  std::filebuf fb_ascii;
  fb_ascii.open(filename + "-ascii.ply", std::ios::out);
  std::ostream outstream_ascii(&fb_ascii);
  if (outstream_ascii.fail()) throw std::runtime_error("failed to open " +
  filename);
  */


  std::vector<std::shared_future<void>> results4;
  Jobs::ParallelFor(
      points.size(),
      [&](unsigned i) {
        points[i].x += m_currentCenter.x;
        points[i].y += m_currentCenter.y;
      },
      results4);
  for (const auto &i : results4)
    i.wait();

  PlyFile cube_file;


  cube_file.add_properties_to_element(
      "vertex", {"x", "y", "z"}, Type::FLOAT64, points.size(),
      reinterpret_cast<uint8_t *>(points.data()), Type::INVALID, 0);
  cube_file.add_properties_to_element(
      "color", {"red", "green", "blue"}, Type::FLOAT32, colors.size(),
      reinterpret_cast<uint8_t *>(colors.data()), Type::INVALID, 0);
  cube_file.add_properties_to_element(
      "leafIndex", {"value"}, Type::INT32, leafIndex.size(),
      reinterpret_cast<uint8_t *>(leafIndex.data()), Type::INVALID, 0);
  cube_file.add_properties_to_element(
      "leafPartIndex", {"value"}, Type::INT32, leafPartIndex.size(),
      reinterpret_cast<uint8_t *>(leafPartIndex.data()), Type::INVALID, 0);
  cube_file.add_properties_to_element(
      "isMainPlant", {"value"}, Type::INT32, isMainPlant.size(),
      reinterpret_cast<uint8_t *>(isMainPlant.data()), Type::INVALID, 0);
  cube_file.add_properties_to_element(
      "plantIndex", {"value"}, Type::INT32, plantIndex.size(),
      reinterpret_cast<uint8_t *>(plantIndex.data()), Type::INVALID, 0);
  cube_file.add_properties_to_element(
      "isGround", {"value"}, Type::INT32, isGround.size(),
      reinterpret_cast<uint8_t *>(isGround.data()), Type::INVALID, 0);
  // Write a binary file
  cube_file.write(outstream_binary, true);
#else
  UNIENGINE_ERROR("Ray tracer disabled!");
#endif
}
void PointCloudCapture::ExportCSV(AutoSorghumGenerationPipeline &pipeline,
                                  const std::filesystem::path &path) {
  auto scene = pipeline.GetScene();
  std::ofstream ofs;
  ofs.open(path.c_str(), std::ofstream::out | std::ofstream::trunc);
  if (ofs.is_open()) {
    std::string output;
    auto scene = pipeline.GetScene();
    std::map<int, std::shared_ptr<LeafData>> leafDataList;
    auto children = scene->GetChildren(pipeline.m_currentGrowingSorghum);
    for(const auto & i : children){
      if(scene->HasPrivateComponent<LeafData>(i)){
        auto leafData = scene->GetOrSetPrivateComponent<LeafData>(i).lock();
        leafDataList[leafData->m_index] = leafData;
      }
    }
    output += "leaf_index,sheath_pos_x,sheath_pos_y,sheath_pos_z,tip_pos_x,tip_pos_y,tip_pos_z,branching_angle,roll_angle\n";
    for(const auto& i : leafDataList){
      output += std::to_string(i.second->m_index) + ",";

      output += std::to_string(i.second->m_leafSheath.x) + ",";
      output += std::to_string(i.second->m_leafSheath.y) + ",";
      output += std::to_string(i.second->m_leafSheath.z) + ",";

      output += std::to_string(i.second->m_leafTip.x) + ",";
      output += std::to_string(i.second->m_leafTip.y) + ",";
      output += std::to_string(i.second->m_leafTip.z) + ",";

      output += std::to_string(i.second->m_branchingAngle) + ",";
      output += std::to_string(i.second->m_rollAngle) + "\n";
    }

    ofs.write(output.c_str(), output.size());
    ofs.flush();
    ofs.close();
  } else {
    UNIENGINE_ERROR("Can't open file!");
  }
}
