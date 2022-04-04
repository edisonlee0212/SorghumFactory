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
  if(m_currentIndex == -1) {
    m_status = AutoSorghumGenerationPipelineStatus::Idle;
    return;
  }
  auto behaviour =
      m_pipelineBehaviour.Get<IAutoSorghumGenerationPipelineBehaviour>();
  if (behaviour) {
    switch (m_status) {
    case AutoSorghumGenerationPipelineStatus::BeforeGrowth:
      behaviour->OnBeforeGrowth(*this);
      break;
    case AutoSorghumGenerationPipelineStatus::Growth:
      behaviour->OnGrowth(*this);
      break;
    case AutoSorghumGenerationPipelineStatus::AfterGrowth:
      behaviour->OnAfterGrowth(*this);
      if (m_currentIndex > m_endIndex) {
        m_status = AutoSorghumGenerationPipelineStatus::Idle;
        m_currentIndex = -1;
        UNIENGINE_LOG("Finished!");
        behaviour->End(*this);
      }
      break;
    }
  }
}

void AutoSorghumGenerationPipeline::OnInspect() {
  DropBehaviourButton();
  auto behaviour =
      m_pipelineBehaviour.Get<IAutoSorghumGenerationPipelineBehaviour>();
  if (behaviour) {
    ImGui::DragInt("Start Index", &m_startIndex, 1, 0, m_endIndex);
    ImGui::DragInt("End Index", &m_endIndex, 1, m_startIndex, 999999);

    if (!Application::IsPlaying()) {
      ImGui::Text("Application not Playing!");
    } else if (!behaviour->IsReady()) {
      ImGui::Text("Pipeline is not ready");
    } else if (m_status != AutoSorghumGenerationPipelineStatus::Idle) {
      ImGui::Text("Busy... (Current: %d, total: %d)", m_currentIndex - m_startIndex, m_endIndex - m_startIndex);
    } else {
      if (ImGui::Button("Start")) {
        m_currentIndex = m_startIndex;
        m_status = AutoSorghumGenerationPipelineStatus::BeforeGrowth;
        behaviour->Start(*this);
      }
    }
  } else {
    ImGui::Text("Pipeline behaviour missing!");
  }
}

void AutoSorghumGenerationPipeline::DropBehaviourButton() {
  if (m_pipelineBehaviour.Get<IAutoSorghumGenerationPipelineBehaviour>()) {
    auto behaviour =
        m_pipelineBehaviour.Get<IAutoSorghumGenerationPipelineBehaviour>();
    ImGui::Text("Current attached behaviour: ");
    ImGui::Button(behaviour->GetTitle().c_str());
    if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
      Editor::GetInstance().m_inspectingAsset = behaviour;
    }
    const std::string tag =
        "##" + behaviour->GetTypeName() +
        (behaviour ? std::to_string(behaviour->GetHandle()) : "");
    if (ImGui::BeginPopupContextItem(tag.c_str())) {
      if (ImGui::Button(("Remove" + tag).c_str())) {
        m_pipelineBehaviour.Clear();
      }
      ImGui::EndPopup();
    }
  } else {
    ImGui::Text("Drop Behaviour");
    ImGui::SameLine();
    ImGui::Button("Here");
    if (ImGui::BeginDragDropTarget()) {
      if (const ImGuiPayload *payload =
              ImGui::AcceptDragDropPayload("GeneralDataCapture")) {
        IM_ASSERT(payload->DataSize == sizeof(std::shared_ptr<IAsset>));
        std::shared_ptr<IAutoSorghumGenerationPipelineBehaviour> payload_n =
            std::dynamic_pointer_cast<IAutoSorghumGenerationPipelineBehaviour>(
                *static_cast<std::shared_ptr<IAsset> *>(payload->Data));
        m_pipelineBehaviour = payload_n;
      }
      ImGui::EndDragDropTarget();
    }
  }
}
void AutoSorghumGenerationPipeline::CollectAssetRef(
    std::vector<AssetRef> &list) {
  IPrivateComponent::CollectAssetRef(list);
}
void AutoSorghumGenerationPipeline::Serialize(YAML::Emitter &out) {
  ISerializable::Serialize(out);
}
void AutoSorghumGenerationPipeline::Deserialize(const YAML::Node &in) {
  ISerializable::Deserialize(in);
}

void IAutoSorghumGenerationPipelineBehaviour::OnBeforeGrowth(
    AutoSorghumGenerationPipeline &pipeline) {}

void IAutoSorghumGenerationPipelineBehaviour::OnGrowth(
    AutoSorghumGenerationPipeline &pipeline) {}

void IAutoSorghumGenerationPipelineBehaviour::OnAfterGrowth(
    AutoSorghumGenerationPipeline &pipeline) {}
