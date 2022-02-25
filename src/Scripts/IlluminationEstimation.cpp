//
// Created by lllll on 2/23/2022.
//

#include "IlluminationEstimation.hpp"
#include "PARSensorGroup.hpp"
#include "RayTracerLayer.hpp"
#include "SkyIlluminance.hpp"
void Scripts::IlluminationEstimation::OnInspect() {
  ImGui::Text("PAR1 Result size: %llu", m_PAR1Result.size());
  ImGui::Text("PAR2 Result size: %llu", m_PAR2Result.size());
  ImGui::Text("PAR3 Result size: %llu", m_PAR3Result.size());
  if (m_running) {
    ImGui::Text("Running!");
    if (ImGui::Button("Force stop")) {
      Clear();
    }
    return;
  }
  if (!m_PAR1Result.empty()) {
    FileUtils::SaveFile(
        "Export PAR1 Results", "CSV", {".csv"},
        [&](const std::filesystem::path &path) {
          ExportCSV(path, m_PAR1Result);
        },
        false);
  }
  if (!m_PAR2Result.empty()) {
    FileUtils::SaveFile(
        "Export PAR2 Results", "CSV", {".csv"},
        [&](const std::filesystem::path &path) {
          ExportCSV(path, m_PAR2Result);
        },
        false);
  }
  if (!m_PAR3Result.empty()) {
    FileUtils::SaveFile(
        "Export PAR3 Results", "CSV", {".csv"},
        [&](const std::filesystem::path &path) {
          ExportCSV(path, m_PAR3Result);
        },
        false);
  }

  Editor::DragAndDropButton<SkyIlluminance>(m_skyIlluminance,
                                            "Sky Illuminance");
  Editor::DragAndDropButton<PARSensorGroup>(m_PARSensorGroup1, "PAR Group 1");
  Editor::DragAndDropButton<PARSensorGroup>(m_PARSensorGroup2, "PAR Group 2");
  Editor::DragAndDropButton<PARSensorGroup>(m_PARSensorGroup3, "PAR Group 3");
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
void Scripts::IlluminationEstimation::Update() {
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
void Scripts::IlluminationEstimation::CollectAssetRef(
    std::vector<AssetRef> &list) {
  list.push_back(m_skyIlluminance);
  list.push_back(m_PARSensorGroup1);
  list.push_back(m_PARSensorGroup2);
  list.push_back(m_PARSensorGroup3);
}
void Scripts::IlluminationEstimation::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_timeInterval" << YAML::Value << m_timeInterval;
  out << YAML::Key << "m_rayProperties.m_bounces" << YAML::Value << m_rayProperties.m_bounces;
  out << YAML::Key << "m_rayProperties.m_samples" << YAML::Value << m_rayProperties.m_samples;
  m_skyIlluminance.Save("m_skyIlluminance", out);
  m_PARSensorGroup1.Save("m_PARSensorGroup1", out);
  m_PARSensorGroup2.Save("m_PARSensorGroup2", out);
  m_PARSensorGroup3.Save("m_PARSensorGroup3", out);

}
void Scripts::IlluminationEstimation::Deserialize(const YAML::Node &in) {
  m_timeInterval = in["m_timeInterval"].as<float>();
  m_rayProperties.m_bounces = in["m_rayProperties.m_bounces"].as<float>();
  m_rayProperties.m_samples = in["m_rayProperties.m_samples"].as<float>();
  m_skyIlluminance.Load("m_skyIlluminance", in);
  m_PARSensorGroup1.Load("m_PARSensorGroup1", in);
  m_PARSensorGroup2.Load("m_PARSensorGroup2", in);
  m_PARSensorGroup3.Load("m_PARSensorGroup3", in);
}
void Scripts::IlluminationEstimation::Clear() {
  m_running = false;
  m_currentTime = 0;
  m_PAR1Result.clear();
  m_PAR2Result.clear();
  m_PAR3Result.clear();
}
void Scripts::IlluminationEstimation::ExportCSV(
    const std::filesystem::path &path,
    const std::vector<float> &results) const {
  std::ofstream ofs;
  ofs.open(path.c_str(), std::ofstream::out | std::ofstream::trunc);
  if (ofs.is_open()) {
    std::string output;
    output += "Illumination\n";
    for(const auto& i : results){
      output += std::to_string(i) + "\n";
    }
    ofs.write(output.c_str(), output.size());
    ofs.flush();
    ofs.close();
  }else {
    UNIENGINE_ERROR("Can't open file!");
  }
}
