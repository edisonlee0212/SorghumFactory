#pragma once
#include <AutoSorghumGenerationPipeline.hpp>
#include <SorghumLayer.hpp>
#include <SorghumProceduralDescriptor.hpp>
using namespace SorghumFactory;
namespace Scripts {
    class ProceduralSorghumSegmentationMask
    : public IAutoSorghumGenerationPipelineBehaviour {
        int m_remainingInstanceAmount = 0;
        Entity m_currentGrowingTree;
    public:
      SorghumProceduralDescriptor m_parameters;
        int m_generationAmount = 10;
        std::filesystem::path m_currentExportFolder = "./export/";
        int m_perTreeGrowthIteration = 40;
        int m_attractionPointAmount = 8000;
        void OnIdle(AutoSorghumGenerationPipeline & pipeline) override;
        void OnBeforeGrowth(AutoSorghumGenerationPipeline & pipeline) override;
        void OnGrowth(AutoSorghumGenerationPipeline & pipeline) override;
        void OnAfterGrowth(AutoSorghumGenerationPipeline & pipeline) override;
        void OnInspect() override;


    };
}