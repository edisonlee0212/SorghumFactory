#pragma once
#include <sorghum_factory_export.h>

using namespace UniEngine;
namespace SorghumFactory {
template <class T> struct CurveDescriptor {
  T m_minValue;
  T m_maxValue;
  UniEngine::Curve m_curve = UniEngine::Curve(0.5f, 0.5f, {0, 0}, {1, 1});
  CurveDescriptor();
  CurveDescriptor(T min, T max,
                  UniEngine::Curve curve = UniEngine::Curve(0.5f, 0.5f, {0, 0},
                                                            {1, 1}));
  bool OnInspect(const std::string &name, bool minmaxControl = true, float speed = 0.01f, const std::string& tip = "");
  void Serialize(const std::string &name, YAML::Emitter &out);
  void Deserialize(const std::string &name, const YAML::Node &in);

  [[nodiscard]] T GetValue(float t) const;
};
template <class T> struct SingleDistribution {
  T m_mean;
  float m_deviation = 0.0f;
  bool OnInspect(const std::string &name, float speed = 0.01f, const std::string& tip = "");
  void Serialize(const std::string &name, YAML::Emitter &out);
  void Deserialize(const std::string &name, const YAML::Node &in);
  [[nodiscard]] T GetValue() const;
};

template <class T> struct MixedDistribution {
  CurveDescriptor<T> m_mean;
  CurveDescriptor<float> m_deviation;
  bool OnInspect(const std::string &name, float speed = 0.01f, const std::string& tip = "");
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
bool SingleDistribution<T>::OnInspect(const std::string &name, float speed, const std::string& tip) {
  bool changed = false;
  if (ImGui::TreeNode(name.c_str())) {
    if (!tip.empty() && ImGui::IsItemHovered())
    {
      ImGui::BeginTooltip();
      ImGui::TextUnformatted(tip.c_str());
      ImGui::EndTooltip();
    }
    if (typeid(T).hash_code() == typeid(float).hash_code()) {
      changed = ImGui::DragFloat("Mean", (float*)&m_mean, speed);
    } else if (typeid(T).hash_code() == typeid(glm::vec2).hash_code()) {
      changed = ImGui::DragFloat2("Mean", (float*)&m_mean, speed);
    } else if (typeid(T).hash_code() == typeid(glm::vec3).hash_code()) {
      changed = ImGui::DragFloat3("Mean", (float*)&m_mean, speed);
    }
    if(ImGui::DragFloat("Deviation", &m_deviation, speed)) changed = true;
    ImGui::TreePop();
  }
  return changed;
}
template <class T> T SingleDistribution<T>::GetValue() const {
  return glm::gaussRand(m_mean, T(m_deviation));
}

template <class T>
bool MixedDistribution<T>::OnInspect(const std::string &name, float speed, const std::string& tip) {
  bool changed = false;
  if (ImGui::TreeNode(name.c_str())) {
    if (!tip.empty() && ImGui::IsItemHovered())
    {
      ImGui::BeginTooltip();
      ImGui::TextUnformatted(tip.c_str());
      ImGui::EndTooltip();
    }
    auto meanTitle = name + "(mean)";
    auto devTitle = name + "(deviation)";
    changed = m_mean.OnInspect(meanTitle, speed);
    if(m_deviation.OnInspect(devTitle, speed)) changed = true;
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
  return glm::gaussRand(m_mean.GetValue(t) , T(m_deviation.GetValue(t)));
}

template <class T> bool CurveDescriptor<T>::OnInspect(const std::string &name, bool minmaxControl, float speed, const std::string& tip) {
  bool changed = false;
  if (ImGui::TreeNode(name.c_str())) {
    if (!tip.empty() && ImGui::IsItemHovered())
    {
      ImGui::BeginTooltip();
      ImGui::TextUnformatted(tip.c_str());
      ImGui::EndTooltip();
    }
    if(minmaxControl) {
      if (typeid(T).hash_code() == typeid(float).hash_code()) {
        changed = ImGui::DragFloat("Min", (float *)&m_minValue, speed);
        if (ImGui::DragFloat("Max", (float *)&m_maxValue, speed))
          changed = true;
      } else if (typeid(T).hash_code() == typeid(glm::vec2).hash_code()) {
        changed = ImGui::DragFloat2("Min", (float *)&m_minValue, speed);
        if (ImGui::DragFloat2("Max", (float *)&m_maxValue, speed))
          changed = true;
      } else if (typeid(T).hash_code() == typeid(glm::vec3).hash_code()) {
        changed = ImGui::DragFloat3("Min", (float *)&m_minValue, speed);
        if (ImGui::DragFloat3("Max", (float *)&m_maxValue, speed))
          changed = true;
      }
    }
    if(m_curve.OnInspect("Curve")) changed = true;
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
}; // namespace SorghumFactory