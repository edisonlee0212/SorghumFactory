//
// Created by lllll on 9/1/2021.
//

#include "AutoSorghumGenerationPipeline.hpp"
#include "Editor.hpp"
using namespace Scripts;

void AutoSorghumGenerationPipeline::Update() {
    auto behaviour = m_pipelineBehaviour.Get<IAutoSorghumGenerationPipelineBehaviour>();
    if(behaviour){
        switch (m_status) {
            case AutoSorghumGenerationPipelineStatus::Idle:
                behaviour->OnIdle(*this);
                break;
            case AutoSorghumGenerationPipelineStatus::BeforeGrowth:
                behaviour->OnBeforeGrowth(*this);
                break;
            case AutoSorghumGenerationPipelineStatus::Growth:
                behaviour->OnGrowth(*this);
                break;
            case AutoSorghumGenerationPipelineStatus::AfterGrowth:
                behaviour->OnAfterGrowth(*this);
                break;
        }
    }
}

void AutoSorghumGenerationPipeline::OnInspect() {
    DropBehaviourButton();
}

void AutoSorghumGenerationPipeline::DropBehaviourButton() {
    if(m_pipelineBehaviour.Get<IAutoSorghumGenerationPipelineBehaviour>()){
        auto behaviour = m_pipelineBehaviour.Get<IAutoSorghumGenerationPipelineBehaviour>();
        ImGui::Text("Current attached behaviour: ");
        ImGui::Button((behaviour->m_name).c_str());
        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0))
        {
            Editor::GetInstance().m_inspectingAsset = behaviour;
        }
        const std::string tag = "##" + behaviour->GetTypeName() + (behaviour ? std::to_string(behaviour->GetHandle()) : "");
        if (ImGui::BeginPopupContextItem(tag.c_str()))
        {
            if (ImGui::Button(("Remove" + tag).c_str()))
            {
                m_pipelineBehaviour.Clear();
            }
            ImGui::EndPopup();
        }
    }else {
        ImGui::Text("Drop Behaviour");
        ImGui::SameLine();
        ImGui::Button("Here");
        if (ImGui::BeginDragDropTarget()) {
            if (const ImGuiPayload *payload = ImGui::AcceptDragDropPayload("GeneralDataCapture")) {
                IM_ASSERT(payload->DataSize == sizeof(std::shared_ptr<IAsset>));
                std::shared_ptr<IAutoSorghumGenerationPipelineBehaviour> payload_n =
                        std::dynamic_pointer_cast<
                        IAutoSorghumGenerationPipelineBehaviour>(
                                *static_cast<std::shared_ptr<IAsset> *>(payload->Data));
                m_pipelineBehaviour = payload_n;
            }
            ImGui::EndDragDropTarget();
        }
    }
}

void IAutoSorghumGenerationPipelineBehaviour::OnBeforeGrowth(
    AutoSorghumGenerationPipeline & pipeline) {

}

void IAutoSorghumGenerationPipelineBehaviour::OnGrowth(
    AutoSorghumGenerationPipeline & pipeline) {

}

void IAutoSorghumGenerationPipelineBehaviour::OnAfterGrowth(
    AutoSorghumGenerationPipeline & pipeline) {

}

void IAutoSorghumGenerationPipelineBehaviour::OnIdle(
    AutoSorghumGenerationPipeline &pipeline) {

}
