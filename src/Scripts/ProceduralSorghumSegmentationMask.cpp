//
// Created by lllll on 9/1/2021.
//

#include "ProceduralSorghumSegmentationMask.hpp"
#include "AssetManager.hpp"
#include "EntityManager.hpp"

using namespace Scripts;

void ProceduralSorghumSegmentationMask::OnBeforeGrowth(
    AutoSorghumGenerationPipeline & pipeline) {
    if(m_remainingInstanceAmount <= 0){
        m_remainingInstanceAmount = 0;
        pipeline.m_status = AutoSorghumGenerationPipelineStatus::Idle;
        return;
    }
    pipeline.m_status = AutoSorghumGenerationPipelineStatus::Growth;

}

void ProceduralSorghumSegmentationMask::OnGrowth(
    AutoSorghumGenerationPipeline & pipeline) {
    if(m_remainingInstanceAmount == 0){
        pipeline.m_status = AutoSorghumGenerationPipelineStatus::Idle;
        return;
    }

}

void ProceduralSorghumSegmentationMask::OnAfterGrowth(
    AutoSorghumGenerationPipeline & pipeline) {

    m_remainingInstanceAmount--;
    if(m_remainingInstanceAmount == 0){
        pipeline.m_status = AutoSorghumGenerationPipelineStatus::Idle;
    }else{
        pipeline.m_status = AutoSorghumGenerationPipelineStatus::BeforeGrowth;
    }
}

void ProceduralSorghumSegmentationMask::OnInspect() {
    ImGui::Text("Space colonization parameters");
    m_parameters.OnInspect();
    ImGui::Text("Pipeline Settings:");
    ImGui::DragInt("Generation Amount", &m_generationAmount);
    ImGui::DragInt("Growth iteration", &m_perTreeGrowthIteration);
    ImGui::DragInt("Attraction point per plant", &m_attractionPointAmount);
    if(m_remainingInstanceAmount == 0) {
        if (ImGui::Button("Start")) {
            std::filesystem::create_directories(m_currentExportFolder);
            m_remainingInstanceAmount = m_generationAmount;
        }
    }else{
        ImGui::Text("Task dispatched...");
        ImGui::Text(("Total: " + std::to_string(m_generationAmount) + ", Remaining: " + std::to_string(m_remainingInstanceAmount)).c_str());
    }
}

void ProceduralSorghumSegmentationMask::OnIdle(
    AutoSorghumGenerationPipeline & pipeline) {
    if(m_remainingInstanceAmount > 0){
        pipeline.m_status = AutoSorghumGenerationPipelineStatus::BeforeGrowth;
        return;
    }
}

