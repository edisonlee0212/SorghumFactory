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

  if (!m_results.empty()) {
    FileUtils::SaveFile(
        "Export results", "CSV", {".csv"},
        [&](const std::filesystem::path &path) { ExportCSV(path); }, false);
  }

  Editor::DragAndDropButton<SkyIlluminance>(m_skyIlluminance,
                                            "Sky Illuminance");

  auto skyIlluminance = m_skyIlluminance.Get<SkyIlluminance>();
  if (!skyIlluminance) {
    ImGui::Text("Sky illuminance missing!");
    return;
  } else {
    m_rayProperties.OnInspect();
    ImGui::DragFloat("Interval", &m_timeInterval, 0.1f, 0.01f, 100.0f);
    if (ImGui::Button("Start")) {
      Clear();
      m_currentTime = skyIlluminance->m_minTime;
      m_running = true;
    }
  }
}
void IlluminationEstimationPipeline::OnProcessing(
    GeneralAutomatedPipeline &pipeline) {
  if (!m_running)
    return;
  auto skyIlluminance = m_skyIlluminance.Get<SkyIlluminance>();
  if (!skyIlluminance) {
    m_running = false;
    m_currentTime = 0;
    return;
  }
  if (m_currentTime > skyIlluminance->m_maxTime) {
    m_running = false;
    m_currentTime = 0;
    return;
  }
  auto snapshot = skyIlluminance->Get(m_currentTime);
  Application::GetLayer<RayTracerLayer>()
      ->m_environmentProperties.m_sunDirection = snapshot.GetSunDirection();

  Application::GetLayer<RayTracerLayer>()
      ->m_environmentProperties.m_skylightIntensity =
      snapshot.GetSunIntensity();

  auto par1 = m_PARSensorGroup1.Get<PARSensorGroup>();
  auto par2 = m_PARSensorGroup2.Get<PARSensorGroup>();
  auto par3 = m_PARSensorGroup3.Get<PARSensorGroup>();

  if (par1 && !par1->m_samplers.empty()) {
    par1->CalculateIllumination(m_rayProperties, 0, 0.0f);
    float sum = 0;
    for (const auto &i : par1->m_samplers) {
      sum += i.m_energy;
    }
    sum /= par1->m_samplers.size();
    m_PAR1Result.push_back(sum);
  }
  if (par2 && !par2->m_samplers.empty()) {
    par2->CalculateIllumination(m_rayProperties, 0, 0.0f);
    float sum = 0;
    for (const auto &i : par2->m_samplers) {
      sum += i.m_energy;
    }
    sum /= par2->m_samplers.size();
    m_PAR2Result.push_back(sum);
  }
  if (par3 && !par3->m_samplers.empty()) {
    par3->CalculateIllumination(m_rayProperties, 0, 0.0f);
    float sum = 0;
    for (const auto &i : par3->m_samplers) {
      sum += i.m_energy;
    }
    sum /= par3->m_samplers.size();
    m_PAR3Result.push_back(sum);
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
  if(in["m_sensorGroups"]){
    for(const auto& i : in["m_sensorGroups"]){
      AssetRef sensor;
      sensor.Deserialize(i);
      if(sensor.Get<PARSensorGroup>()) m_sensorGroups.push_back(sensor);
    }
  }
}
void IlluminationEstimationPipeline::Clear() {
  m_running = false;
  m_currentTime = 0;
  m_results.clear();
}
void IlluminationEstimationPipeline::ExportCSV(
    const std::filesystem::path &path) const {
  std::ofstream ofs;
  ofs.open(path.c_str(), std::ofstream::out | std::ofstream::trunc);
  if (ofs.is_open()) {
    std::string output;
    output += "Illumination\n";
    for (const auto &i : results) {
      output += std::to_string(i) + "\n";
    }
    ofs.write(output.c_str(), output.size());
    ofs.flush();
    ofs.close();
  } else {
    UNIENGINE_ERROR("Can't open file!");
  }
}
#endif