//
// Created by lllll on 2/23/2022.
//
#ifdef RAYTRACERFACILITY
#include "IlluminationEstimation.hpp"
#include "PARSensorGroup.hpp"
#include "RayTracerLayer.hpp"
#include "SkyIlluminance.hpp"

using namespace Scripts;
void IlluminationEstimationPipeline::OnInspect() {
  if(ImGui::Button("Instantiate pipeline")){

  }

  if (!m_results.empty()) {
    FileUtils::SaveFile(
        "Export results", "CSV", {".csv"},
        [&](const std::filesystem::path &path) { ExportCSV(path); }, false);
  }

  Editor::DragAndDropButton<SkyIlluminance>(m_skyIlluminance,
                                            "Sky Illuminance");
  m_rayProperties.OnInspect();
  ImGui::DragFloat("Interval", &m_timeInterval, 0.1f, 0.01f, 100.0f);
}
void IlluminationEstimationPipeline::OnBeforeProcessing(
    GeneralAutomatedPipeline &pipeline) {
}
void IlluminationEstimationPipeline::OnAfterProcessing(
    GeneralAutomatedPipeline &pipeline) {
}
void IlluminationEstimationPipeline::OnProcessing(
    GeneralAutomatedPipeline &pipeline) {
  auto skyIlluminance = m_skyIlluminance.Get<SkyIlluminance>();
  if (!skyIlluminance) {
    m_currentTime = 0;
    pipeline.m_status = GeneralAutomatedPipelineStatus::Idle;
    return;
  }
  if (m_currentTime > skyIlluminance->m_maxTime) {
    m_currentTime = 0;
    return;
  }
  auto snapshot = skyIlluminance->Get(m_currentTime);
  Application::GetLayer<RayTracerLayer>()
      ->m_environmentProperties.m_sunDirection = snapshot.GetSunDirection();

  Application::GetLayer<RayTracerLayer>()
      ->m_environmentProperties.m_skylightIntensity =
      snapshot.GetSunIntensity();

  m_results.emplace_back(m_currentTime, std::vector<float>());
  for (auto sensorGroupRef : m_sensorGroups) {
    auto sensorGroup = sensorGroupRef.Get<PARSensorGroup>();
    if (sensorGroup && !sensorGroup->m_samplers.empty()) {
      sensorGroup->CalculateIllumination(m_rayProperties, 0, 0.0f);
      float sum = 0;
      for (const auto &i : sensorGroup->m_samplers) {
        sum += i.m_energy;
      }
      sum /= sensorGroup->m_samplers.size();
      m_results.back().second.push_back(sum);
    }
  }
  m_currentTime += m_timeInterval;
}
void IlluminationEstimationPipeline::CollectAssetRef(
    std::vector<AssetRef> &list) {
  list.push_back(m_skyIlluminance);
  for (const auto &i : m_sensorGroups)
    list.push_back(i);
}
void IlluminationEstimationPipeline::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_timeInterval" << YAML::Value << m_timeInterval;
  out << YAML::Key << "m_rayProperties.m_bounces" << YAML::Value
      << m_rayProperties.m_bounces;
  out << YAML::Key << "m_rayProperties.m_samples" << YAML::Value
      << m_rayProperties.m_samples;
  m_skyIlluminance.Save("m_skyIlluminance", out);

  if (!m_sensorGroups.empty()) {
    out << YAML::Key << "m_sensorGroups" << YAML::Value << YAML::BeginSeq;
    for (const auto &i : m_sensorGroups) {
      out << YAML::BeginMap;
      i.Serialize(out);
      out << YAML::EndMap;
    }
    out << YAML::EndSeq;
  }
}
void IlluminationEstimationPipeline::Deserialize(const YAML::Node &in) {
  m_timeInterval = in["m_timeInterval"].as<float>();
  m_rayProperties.m_bounces = in["m_rayProperties.m_bounces"].as<float>();
  m_rayProperties.m_samples = in["m_rayProperties.m_samples"].as<float>();
  m_skyIlluminance.Load("m_skyIlluminance", in);
  m_sensorGroups.clear();
  if (in["m_sensorGroups"]) {
    for (const auto &i : in["m_sensorGroups"]) {
      AssetRef sensor;
      sensor.Deserialize(i);
      if (sensor.Get<PARSensorGroup>())
        m_sensorGroups.push_back(sensor);
    }
  }
}

void IlluminationEstimationPipeline::ExportCSV(
    const std::filesystem::path &path) {
  std::ofstream ofs;
  ofs.open(path.c_str(), std::ofstream::out | std::ofstream::trunc);
  if (ofs.is_open()) {
    std::string output;
    output += "time,";
    for (int i = 0; i < m_sensorGroups.size(); i++) {
      output += m_sensorGroups[i]
                    .Get<PARSensorGroup>()
                    ->GetAssetRecord()
                    .lock()
                    ->GetAssetFileName();
      if (i < m_sensorGroups.size() - 1)
        output += ",";
    }
    output += "\n";

    for (const auto &i : m_results) {
      output += std::to_string(i.first) + ",";
      for (int j = 0; j < i.second.size(); j++) {
        output += std::to_string(i.second[j]);
        if (j < i.second.size() - 1)
          output += ",";
      }
      output += "\n";
    }
    ofs.write(output.c_str(), output.size());
    ofs.flush();
    ofs.close();
  } else {
    UNIENGINE_ERROR("Can't open file!");
  }
}
void IlluminationEstimationPipeline::OnStart(
    GeneralAutomatedPipeline &pipeline) {
  m_results.clear();
  m_results.resize(m_sensorGroups.size());
}


#endif