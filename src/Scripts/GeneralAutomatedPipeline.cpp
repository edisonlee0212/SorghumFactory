//
// Created by lllll on 4/29/2022.
//

#include "GeneralAutomatedPipeline.hpp"

using namespace Scripts;

void GeneralAutomatedPipeline::OnDestroy() { m_pipelineBehaviour.Clear(); }
void GeneralAutomatedPipeline::Update() {
  auto behaviour =
      m_pipelineBehaviour.Get<IGeneralAutomatedPipelineBehaviour>();
  if (behaviour) {
    switch (m_status) {
    case GeneralAutomatedPipelineStatus::Idle: {
      if (!m_busy) {
        break;
      } else if (m_remainingTaskAmount > 0) {
        m_remainingTaskAmount--;
        m_status = GeneralAutomatedPipelineStatus::BeforeProcessing;
      } else {
        UNIENGINE_LOG("Finished!");
        behaviour->OnEnd(*this);
        m_busy = false;
      }
      break;
    }
    case GeneralAutomatedPipelineStatus::BeforeProcessing: {
      behaviour->OnBeforeProcessing(*this);
      break;
    }
    case GeneralAutomatedPipelineStatus::Processing:
      behaviour->OnProcessing(*this);
      break;
    case GeneralAutomatedPipelineStatus::AfterProcessing:
      behaviour->OnAfterProcessing(*this);
      break;
    }
  }
}
void GeneralAutomatedPipeline::OnInspect() {
  ImGui::DragInt("Start Index", &m_startIndex);
  ImGui::DragInt("Amount per descriptor", &m_taskAmount);

  auto behaviour =
      m_pipelineBehaviour.Get<IGeneralAutomatedPipelineBehaviour>();
  if (!behaviour) {
    ImGui::Text("Behaviour missing!");
  } else if (m_busy) {
    ImGui::Text("Task dispatched...");
    ImGui::Text(("Total: " + std::to_string(m_taskAmount) +
                 ", Remaining: " + std::to_string(m_remainingTaskAmount))
                    .c_str());
    if (ImGui::Button("Force stop")) {
      m_remainingTaskAmount = 0;
      m_busy = false;
    }
  }
}
void GeneralAutomatedPipeline::CollectAssetRef(std::vector<AssetRef> &list) {
  list.push_back(m_pipelineBehaviour);
}
void GeneralAutomatedPipeline::Serialize(YAML::Emitter &out) {
  m_pipelineBehaviour.Save("m_pipelineBehaviour", out);
}
void GeneralAutomatedPipeline::Deserialize(const YAML::Node &in) {
  m_pipelineBehaviour.Load("m_pipelineBehaviour", in);
}
