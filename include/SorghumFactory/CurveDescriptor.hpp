#pragma once
#include <sorghum_factory_export.h>

using namespace UniEngine;
namespace PlantArchitect {
struct CurveDescriptorSettings {
  float m_speed = 0.01f;
  float m_minMaxControl = true;
  float m_endAdjustment = true;
  std::string m_tip = "";
};

template <class T> struct CurveDescriptor {
  T m_minValue = 0;
  T m_maxValue = 1;
  UniEngine::Curve m_curve = UniEngine::Curve(0.5f, 0.5f, {0, 0}, {1, 1});
  CurveDescriptor();
  CurveDescriptor(T min, T max,
                  UniEngine::Curve curve = UniEngine::Curve(0.5f, 0.5f, {0, 0},
                                                            {1, 1}));
  bool OnInspect(const std::string &name,
                 const CurveDescriptorSettings &settings = {});
  void Serialize(const std::string &name, YAML::Emitter &out);
  void Deserialize(const std::string &name, const YAML::Node &in);

  [[nodiscard]] T GetValue(float t) const;
};
template <class T> struct SingleDistribution {
  T m_mean;
  float m_deviation = 0.0f;
  bool OnInspect(const std::string &name, float speed = 0.01f,
                 const std::string &tip = "");
  void Serialize(const std::string &name, YAML::Emitter &out);
  void Deserialize(const std::string &name, const YAML::Node &in);
  [[nodiscard]] T GetValue() const;
};

struct MixedDistributionSettings {
  float m_speed = 0.01f;
  CurveDescriptorSettings m_meanSettings;
  CurveDescriptorSettings m_devSettings;
  std::string m_tip = "";
};

template <class T> struct MixedDistribution {
  CurveDescriptor<T> m_mean;
  CurveDescriptor<float> m_deviation;
  bool OnInspect(const std::string &name,
                 const MixedDistributionSettings &settings = {});
  void Serialize(const std::string &name, YAML::Emitter &out);
  void Deserialize(const std::string &name, const YAML::Node &in);
  T GetValue(float t) const;
};

template <class T>
void SingleDistribution<T>::Serialize(const std::string &name,
                                      YAML::Emitter &out) {
  out << YAML::Key << name << YAML::Value << YAML::BeginMap;
  {
    out << YAML::Key << "m_mean" << YAML::Value << m_mean;
    out << YAML::Key << "m_deviation" << YAML::Value << m_deviation;
  }
  out << YAML::EndMap;
}
template <class T>
void SingleDistribution<T>::Deserialize(const std::string &name,
                                        const YAML::Node &in) {
  if (in[name]) {
    const auto &cd = in[name];
    m_mean = cd["m_mean"].as<T>();
    m_deviation = cd["m_deviation"].as<float>();
  }
}
template <class T>
bool SingleDistribution<T>::OnInspect(const std::string &name, float speed,
                                      const std::string &tip) {
  bool changed = false;
  if (ImGui::TreeNode(name.c_str())) {
    if (!tip.empty() && ImGui::IsItemHovered()) {
      ImGui::BeginTooltip();
      ImGui::TextUnformatted(tip.c_str());
      ImGui::EndTooltip();
    }
    if (typeid(T).hash_code() == typeid(float).hash_code()) {
      changed = ImGui::DragFloat("Mean", (float *)&m_mean, speed);
    } else if (typeid(T).hash_code() == typeid(glm::vec2).hash_code()) {
      changed = ImGui::DragFloat2("Mean", (float *)&m_mean, speed);
    } else if (typeid(T).hash_code() == typeid(glm::vec3).hash_code()) {
      changed = ImGui::DragFloat3("Mean", (float *)&m_mean, speed);
    }
    if (ImGui::DragFloat("Deviation", &m_deviation, speed))
      changed = true;
    ImGui::TreePop();
  }
  return changed;
}
template <class T> T SingleDistribution<T>::GetValue() const {
  return glm::gaussRand(m_mean, T(m_deviation));
}

template <class T>
bool MixedDistribution<T>::OnInspect(
    const std::string &name, const MixedDistributionSettings &settings) {
  bool changed = false;
  if (ImGui::TreeNode(name.c_str())) {
    if (!settings.m_tip.empty() && ImGui::IsItemHovered()) {
      ImGui::BeginTooltip();
      ImGui::TextUnformatted(settings.m_tip.c_str());
      ImGui::EndTooltip();
    }
    auto meanTitle = name + "(mean)";
    auto devTitle = name + "(deviation)";
    changed = m_mean.OnInspect(meanTitle, settings.m_meanSettings);
    if (m_deviation.OnInspect(devTitle, settings.m_devSettings))
      changed = true;
    ImGui::TreePop();
  }
  return changed;
}

template <class T>
void MixedDistribution<T>::Serialize(const std::string &name,
                                     YAML::Emitter &out) {
  out << YAML::Key << name << YAML::Value << YAML::BeginMap;
  {
    m_mean.Serialize("m_mean", out);
    m_deviation.Serialize("m_deviation", out);
  }
  out << YAML::EndMap;
}
template <class T>
void MixedDistribution<T>::Deserialize(const std::string &name,
                                       const YAML::Node &in) {
  if (in[name]) {
    const auto &cd = in[name];
    m_mean.Deserialize("m_mean", cd);
    m_deviation.Deserialize("m_deviation", cd);
  }
}
template <class T> T MixedDistribution<T>::GetValue(float t) const {
  return glm::gaussRand(m_mean.GetValue(t), T(m_deviation.GetValue(t)));
}

template <class T>
bool CurveDescriptor<T>::OnInspect(const std::string &name,
                                   const CurveDescriptorSettings &settings) {
  bool changed = false;
  if (ImGui::TreeNode(name.c_str())) {
    if (!settings.m_tip.empty() && ImGui::IsItemHovered()) {
      ImGui::BeginTooltip();
      ImGui::TextUnformatted(settings.m_tip.c_str());
      ImGui::EndTooltip();
    }
    if (settings.m_minMaxControl) {
      if (typeid(T).hash_code() == typeid(float).hash_code()) {
        changed = ImGui::DragFloat(("Min##" + name).c_str(),
                                   (float *)&m_minValue, settings.m_speed);
        if (ImGui::DragFloat(("Max##" + name).c_str(), (float *)&m_maxValue,
                             settings.m_speed))
          changed = true;
      } else if (typeid(T).hash_code() == typeid(glm::vec2).hash_code()) {
        changed = ImGui::DragFloat2(("Min##" + name).c_str(),
                                    (float *)&m_minValue, settings.m_speed);
        if (ImGui::DragFloat2(("Max##" + name).c_str(), (float *)&m_maxValue,
                              settings.m_speed))
          changed = true;
      } else if (typeid(T).hash_code() == typeid(glm::vec3).hash_code()) {
        changed = ImGui::DragFloat3(("Min##" + name).c_str(),
                                    (float *)&m_minValue, settings.m_speed);
        if (ImGui::DragFloat3(("Max##" + name).c_str(), (float *)&m_maxValue,
                              settings.m_speed))
          changed = true;
      }
    }
    auto flag = settings.m_endAdjustment
                    ? (unsigned)CurveEditorFlags::ALLOW_RESIZE |
                          (unsigned)CurveEditorFlags::SHOW_GRID
                    : (unsigned)CurveEditorFlags::ALLOW_RESIZE |
                          (unsigned)CurveEditorFlags::SHOW_GRID |
                          (unsigned)CurveEditorFlags::DISABLE_START_END_Y;
    if (m_curve.OnInspect(("Curve##" + name).c_str(), ImVec2(-1, -1), flag)) {
      changed = true;
    }

    ImGui::TreePop();
  }
  return changed;
}
template <class T>
void CurveDescriptor<T>::Serialize(const std::string &name,
                                   YAML::Emitter &out) {
  out << YAML::Key << name << YAML::Value << YAML::BeginMap;
  {
    out << YAML::Key << "m_minValue" << YAML::Value << m_minValue;
    out << YAML::Key << "m_maxValue" << YAML::Value << m_maxValue;
    out << YAML::Key << "m_curve" << YAML::Value << YAML::BeginMap;
    m_curve.Serialize(out);
    out << YAML::EndMap;
  }
  out << YAML::EndMap;
}
template <class T>
void CurveDescriptor<T>::Deserialize(const std::string &name,
                                     const YAML::Node &in) {
  if (in[name]) {
    const auto &cd = in[name];
    if (cd["m_minValue"])
      m_minValue = cd["m_minValue"].as<T>();
    if (cd["m_maxValue"])
      m_maxValue = cd["m_maxValue"].as<T>();
    if (cd["m_curve"])
      m_curve.Deserialize(cd["m_curve"]);
  }
}
template <class T> CurveDescriptor<T>::CurveDescriptor() {
  m_curve = UniEngine::Curve(0.5f, 0.5f, {0, 0}, {1, 1});
}
template <class T>
CurveDescriptor<T>::CurveDescriptor(T min, T max, UniEngine::Curve curve) {
  m_minValue = min;
  m_maxValue = max;
  m_curve = curve;
}
template <class T> T CurveDescriptor<T>::GetValue(float t) const {
  return glm::mix(m_minValue, m_maxValue,
                  glm::clamp(m_curve.GetValue(t), 0.0f, 1.0f));
}
}; // namespace PlantArchitect