#pragma once
#include <AutoSorghumGenerationPipeline.hpp>
#include <SorghumLayer.hpp>
using namespace EcoSysLab;
namespace Scripts {
	struct PointCloudSampleSettings {
		glm::vec2 m_scannerBoundingBoxHeightRange = glm::vec2(-1, 2);
		glm::vec2 m_scannerPointDistance = glm::vec2(0.0075f);
		float m_scannerAngle = 30.0f;
		bool m_adjustBoundingBox = true;
		float m_scannerBoundingBoxRadius = 1.0f;
		float m_outputAdjustmentFactor = 1.5f;
		float m_minOutputRadius = 1.5f;
		int m_segmentAmount = 3;

		float m_positionVariance = 0.4f;
		void OnInspect();
		void Serialize(const std::string& name, YAML::Emitter& out) const;
		void Deserialize(const std::string& name, const YAML::Node& in);
	};
	class PointCloudCapture : public IAutoSorghumGenerationPipelineBehaviour {
		Entity m_currentSorghumField;
		Entity m_ground;
		glm::dvec2 m_currentCenter;
		void Instantiate();
	public:
		PointCloudSampleSettings m_settings;
		void Reset(AutoSorghumGenerationPipeline& pipeline);
		AssetRef m_positionsField;
		AssetRef m_fieldGround;
		std::filesystem::path m_currentExportFolder;
		void OnBeforeGrowth(AutoSorghumGenerationPipeline& pipeline) override;
		void OnGrowth(AutoSorghumGenerationPipeline& pipeline) override;
		void OnAfterGrowth(AutoSorghumGenerationPipeline& pipeline) override;
		void OnInspect() override;
		void OnStart(AutoSorghumGenerationPipeline& pipeline) override;
		void OnEnd(AutoSorghumGenerationPipeline& pipeline) override;

		void CollectAssetRef(std::vector<AssetRef>& list) override;
		void Serialize(YAML::Emitter& out) override;
		void Deserialize(const YAML::Node& in) override;

		void ScanPointCloudLabeled(AutoSorghumGenerationPipeline& pipeline,
			const std::filesystem::path& savePath,
			const PointCloudSampleSettings& settings);

		static void ExportCSV(AutoSorghumGenerationPipeline& pipeline, const std::filesystem::path& path);
	};
}