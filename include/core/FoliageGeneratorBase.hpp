#pragma once
#include <InternodeRingSegment.hpp>
using namespace UniEngine;
namespace PlantFactory {
	struct Branchlet
	{
		std::vector<InternodeRingSegment> m_rings;
		std::vector<glm::mat4> m_leafLocalTransforms;
		glm::vec3 m_normal;
	};
	class FoliageGeneratorBase : public PrivateComponentBase
	{
	public:
		virtual void Generate() = 0;
	};

	struct DefaultFoliageInfo : ComponentDataBase
	{
		glm::vec2 m_leafSize = glm::vec2(0.1f);
		float m_leafIlluminationLimit = 0;
		float m_leafInhibitorFactor = 0;
		bool m_isBothSide = true;
		int m_sideLeafAmount = 1;
		float m_startBendingAngle = 45;
		float m_bendingAngleIncrement = 0;
		float m_leafPhotoTropism = 999.0f;
		float m_leafGravitropism = 1.0f;
		float m_leafDistance = 0;
	};

	class DefaultFoliageGenerator : public FoliageGeneratorBase
	{
		friend class TreeReconstructionSystem;
		DefaultFoliageInfo m_defaultFoliageInfo;
		EntityArchetype m_archetype;
		static std::shared_ptr<Texture2D> m_leafSurfaceTex;
		std::shared_ptr<Material> m_leafMaterial;
		void GenerateLeaves(Entity& internode, glm::mat4& treeTransform, std::vector<glm::mat4>& leafTransforms, bool isLeft);
	public:
		DefaultFoliageGenerator();
		void Generate() override;
		void OnGui() override;
	};
}