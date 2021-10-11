#pragma once
#include <SorghumSystem.hpp>

using namespace SorghumFactory;

namespace Scripts {
    enum class AutoSorghumGenerationPipelineStatus{
        Idle,
        BeforeGrowth,
        Growth,
        AfterGrowth
    };
    class AutoSorghumGenerationPipeline : public IPrivateComponent {
        void DropBehaviourButton();

    public:
        AutoSorghumGenerationPipelineStatus m_status = AutoSorghumGenerationPipelineStatus::Idle;
        AssetRef m_pipelineBehaviour;
        void Update() override;
        void OnInspect() override;
    };

    class IAutoSorghumGenerationPipelineBehaviour : public IAsset{
    public:
        virtual void OnIdle(AutoSorghumGenerationPipeline & pipeline);
        virtual void OnBeforeGrowth(AutoSorghumGenerationPipeline & pipeline);
        virtual void OnGrowth(AutoSorghumGenerationPipeline & pipeline);
        virtual void OnAfterGrowth(AutoSorghumGenerationPipeline & pipeline);
    };
}