//
// Created by lllll on 9/1/2021.
//

#include "AutoSorghumGenerationPipeline.hpp"
#include "Editor.hpp"
#include "GeneralDataCapture.hpp"
#include <SorghumData.hpp>
#include <SorghumStateGenerator.hpp>
#ifdef RAYTRACERFACILITY
#include "RayTracerCamera.hpp"
#include "RayTracerLayer.hpp"
using namespace RayTracerFacility;
#endif
using namespace Scripts;

void AutoSorghumGenerationPipeline::Update() {
  auto behaviour =
      m_pipelineBehaviour.Get<IAutoSorghumGenerationPipelineBehaviour>();
  if (behaviour) {
    switch (m_status) {
    case AutoSorghumGenerationPipelineStatus::Idle: {
      if (!m_busy) {
        break;
      } else if (m_remainingInstanceAmount > 0) {
        m_remainingInstanceAmount--;
        m_status = AutoSorghumGenerationPipelineStatus::BeforeGrowth;
      } else if (!m_descriptors.empty()) {
        m_currentUsingDescriptor = m_descriptors.back();
        m_descriptors.pop_back();
        m_remainingInstanceAmount = m_generationAmount;
      } else {
        UNIENGINE_LOG("Finished!");
        behaviour->OnEnd(*this);
        m_busy = false;
      }
      break;
    }
    case AutoSorghumGenerationPipelineStatus::BeforeGrowth: {
      m_prefix = m_currentUsingDescriptor.Get<SorghumStateGenerator>()
                     ->GetAssetRecord()
                     .lock()
                     ->GetAssetFileName() +
                 "_" +
                 std::to_string(GetSeed());

      behaviour->OnBeforeGrowth(*this);
      if (m_status != AutoSorghumGenerationPipelineStatus::BeforeGrowth) {
        if (!m_currentGrowingSorghum.IsValid()) {
          UNIENGINE_ERROR("No sorghum created or wrongly created!");
          m_status = AutoSorghumGenerationPipelineStatus::BeforeGrowth;
        }
      }
      break;
    }
    case AutoSorghumGenerationPipelineStatus::Growth:
      behaviour->OnGrowth(*this);
      m_status = AutoSorghumGenerationPipelineStatus::AfterGrowth;
      break;
    case AutoSorghumGenerationPipelineStatus::AfterGrowth:
      behaviour->OnAfterGrowth(*this);
      break;
    }
  }
}

void AutoSorghumGenerationPipeline::OnInspect() {
  ImGui::DragInt("Start Index", &m_startIndex);
  ImGui::DragInt("Amount per descriptor", &m_generationAmount);

  auto behaviour =
      m_pipelineBehaviour.Get<IAutoSorghumGenerationPipelineBehaviour>();
  if (!behaviour) {
    ImGui::Text("Behaviour missing!");
  } else if (m_busy) {
    ImGui::Text("Task dispatched...");
    ImGui::Text(
        ("Remaining descriptors: " + std::to_string(m_descriptors.size()))
            .c_str());
    ImGui::Text(("Total: " + std::to_string(m_generationAmount) +
                 ", Remaining: " + std::to_string(m_remainingInstanceAmount))
                    .c_str());
    if (ImGui::Button("Force stop")) {
      m_remainingInstanceAmount = 0;
      m_descriptors.clear();
      m_busy = false;
    }
  } else {
    ImGui::Text(("Loaded descriptors: " + std::to_string(m_descriptors.size()))
                    .c_str());
    FileUtils::OpenFolder(
        "Collect descriptors", [&](const std::filesystem::path &path) {
          m_descriptors.clear();
          m_currentUsingDescriptor.Clear();
          auto &projectManager = ProjectManager::GetInstance();
          if (std::filesystem::exists(path) &&
              std::filesystem::is_directory(path)) {
            for (const auto &entry :
                 std::filesystem::recursive_directory_iterator(path)) {
              if (!std::filesystem::is_directory(entry.path())) {
                auto relativePath =
                    ProjectManager::GetPathRelativeToProject(entry.path());
                if (entry.path().extension() == ".sorghumstategenerator") {
                  auto descriptor =
                      std::dynamic_pointer_cast<SorghumStateGenerator>(
                          ProjectManager::GetOrCreateAsset(relativePath));
                  m_descriptors.emplace_back(descriptor);
                }
              }
            }
          }
        });
    if (m_descriptors.empty()) {
      ImGui::Text("No descriptors!");
    } else if (Application::IsPlaying()) {
      if (ImGui::Button("Start")) {
        m_busy = true;
        behaviour->OnStart(*this);
        m_status = AutoSorghumGenerationPipelineStatus::Idle;
      }
    } else {
      ImGui::Text("Start Engine first!");
    }
  }
}

void AutoSorghumGenerationPipeline::CollectAssetRef(
    std::vector<AssetRef> &list) {
  list.push_back(m_pipelineBehaviour);
}
void AutoSorghumGenerationPipeline::Serialize(YAML::Emitter &out) {
  m_pipelineBehaviour.Save("m_pipelineBehaviour", out);
}
void AutoSorghumGenerationPipeline::Deserialize(const YAML::Node &in) {
  m_pipelineBehaviour.Load("m_pipelineBehaviour", in);
}
int AutoSorghumGenerationPipeline::GetSeed() const {
  return m_generationAmount - m_remainingInstanceAmount + m_startIndex + 1;
}
void AutoSorghumGenerationPipeline::OnDestroy() {
  m_descriptors.clear();
  m_currentUsingDescriptor.Clear();
  m_pipelineBehaviour.Clear();
}

void IAutoSorghumGenerationPipelineBehaviour::OnBeforeGrowth(
    AutoSorghumGenerationPipeline &pipeline) {}

void IAutoSorghumGenerationPipelineBehaviour::OnGrowth(
    AutoSorghumGenerationPipeline &pipeline) {}

void IAutoSorghumGenerationPipelineBehaviour::OnAfterGrowth(
    AutoSorghumGenerationPipeline &pipeline) {}
