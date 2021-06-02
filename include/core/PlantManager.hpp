#pragma once
#include <CUDAModule.hpp>
#include <InternodeRingSegment.hpp>
#include <TreeData.hpp>
#include <VoxelSpace.hpp>
#include <Volume.hpp>
using namespace UniEngine;
namespace PlantFactory {
	enum class PlantType
	{
		GeneralTree,
		Sorghum
	};
#pragma region Tree
	struct PlantInfo : ComponentDataBase
	{
		PlantType m_plantType;
		float m_startTime;
		float m_age;
	};
#pragma endregion	
#pragma region Internode
	struct BranchCylinder : ComponentDataBase {
		glm::mat4 m_value;
		bool operator ==(const BranchCylinder& other) const {
			return other.m_value == m_value;
		}
	};
	struct BranchCylinderWidth : ComponentDataBase {
		float m_value;
		bool operator ==(const BranchCylinderWidth& other) const {
			return other.m_value == m_value;
		}
	};
	struct BranchPointer : ComponentDataBase {
		glm::mat4 m_value;
		bool operator ==(const BranchCylinder& other) const {
			return other.m_value == m_value;
		}
	};

	struct Illumination : ComponentDataBase {
		float m_currentIntensity = 0;
		glm::vec3 m_accumulatedDirection = glm::vec3(0.0f);
	};

	struct BranchColor : ComponentDataBase
	{
		glm::vec4 m_value;
	};
	
	struct InternodeInfo : ComponentDataBase
	{
		PlantType m_plantType;
		Entity m_plant = Entity();
		float m_usedResource;
		bool m_activated = true;
		float m_startAge = 0;
		float m_startGlobalTime = 0;
		int m_order = 1;
		int m_level = 1;
	};
	struct InternodeGrowth : ComponentDataBase
	{
		float m_inhibitor = 0;
		float m_inhibitorTransmitFactor = 1;
		int m_distanceToRoot = 0; //Ok
		float m_internodeLength = 0.0f;
		
		glm::vec3 m_branchEndPosition = glm::vec3(0);
		glm::vec3 m_branchStartPosition = glm::vec3(0);
		glm::vec3 m_childrenTotalTorque = glm::vec3(0.0f);
		glm::vec3 m_childMeanPosition = glm::vec3(0.0f);
		float m_MassOfChildren = 0;
		float m_sagging = 0.0f;
		
		float m_thickness = 0.02f; //Ok
		glm::quat m_desiredLocalRotation = glm::quat(glm::vec3(0.0f));
		glm::quat m_desiredGlobalRotation = glm::quat(glm::vec3(0.0f));
		//Will be used to calculate the gravity bending.
		glm::vec3 m_desiredGlobalPosition = glm::vec3(0.0f);
		Entity m_thickestChild;
	};
	struct InternodeStatistics : ComponentDataBase
	{
		int m_childrenEndNodeAmount = 0; //Ok
		bool m_isMaxChild = false; //Ok
		bool m_isEndNode = false; //Ok
		int m_maxChildOrder = 0; //Ok
		int m_maxChildLevel = 0; //Ok
		int m_distanceToBranchEnd = 0; //Ok The no-branching chain distance.
		int m_longestDistanceToAnyEndNode = 0; //Ok
		int m_totalLength = 0; //Ok
		int m_distanceToBranchStart = 0; //Ok
	};

#pragma endregion
#pragma region Bud
	class Bud;
	struct InternodeCandidate
	{
		Entity m_owner;
		Entity m_parent;
		std::vector<Bud> m_buds;
		GlobalTransform m_globalTransform;
		Transform m_transform;
		InternodeInfo m_info = InternodeInfo();
		InternodeGrowth m_growth = InternodeGrowth();
		InternodeStatistics m_statistics = InternodeStatistics();
	};

	struct ResourceParcel
	{
		float m_nutrient;
		float m_carbon;
		float m_globalTime = 0;
		ResourceParcel();
		ResourceParcel(const float& water, const float& carbon);
		ResourceParcel& operator+=(const ResourceParcel& value);
		[[nodiscard]] bool IsEnough() const;
		void OnGui() const;
	};

	class Bud {
	public:
		bool m_enoughForGrowth = false;
		float m_resourceWeight = 1.0f;
		ResourceParcel m_currentResource;
		std::vector<ResourceParcel> m_resourceLog;
		float m_deathGlobalTime = -1;
		float m_avoidanceAngle;
		bool m_active = true;
		bool m_isApical = false;
		float m_mainAngle = 0;
	};
#pragma endregion
	class InternodeData : public PrivateComponentBase {
	public:
		Entity m_owner;
		std::vector<InternodeCandidate> m_spaceColonizationGrowthModelCandidates;
		std::vector<glm::mat4> m_leavesTransforms;
		std::mutex m_internodeLock;
		std::vector<glm::vec3> m_points;
		std::vector<Bud> m_buds;
		std::vector<InternodeRingSegment> m_rings;
		glm::vec3 m_normalDir;
		int m_step;
		void OnGui() override;
	};
#pragma region Enums
	enum class BranchRenderType
	{
		Illumination,
		Sagging,
		Inhibitor,
		InhibitorTransmitFactor,
		ResourceToGrow,
		Order,
		MaxChildOrder,
		Level,
		MaxChildLevel,
		IsMaxChild,
		ChildrenEndNodeAmount,
		IsEndNode,
		DistanceToBranchEnd,
		DistanceToBranchStart,
		TotalLength,
		LongestDistanceToAnyEndNode,
		
	};

	enum class PointerRenderType
	{
		Illumination,
		Bending
	};
#pragma endregion
	class PlantManager
	{
	protected:
#pragma region Class related
		PlantManager() = default;
		PlantManager(PlantManager&&) = default;
		PlantManager(const PlantManager&) = default;
		PlantManager& operator=(PlantManager&&) = default;
		PlantManager& operator=(const PlantManager&) = default;
#pragma endregion
	public:
		std::map<PlantType, std::function<void(PlantManager& manager, std::vector<ResourceParcel>& resources)>>
			m_plantResourceAllocators;
		std::map<PlantType, std::function<void(PlantManager& manager, std::vector<InternodeCandidate>& candidates)>>
			m_plantGrowthModels;
		std::map<PlantType, std::function<void(PlantManager& manager, std::vector<Volume*>& obstacles)>>
			m_plantInternodePruners;
		std::map<PlantType, std::function<void(PlantManager& manager)>>
			m_plantMetaDataCalculators;
		std::map<PlantType, std::function<void(PlantManager& manager)>>
			m_plantFoliageGenerators;
		std::map<PlantType, std::function<void(PlantManager& manager)>>
			m_plantMeshGenerators;
#pragma region Growth
		static bool GrowAllPlants();
		static bool GrowAllPlants(const unsigned& iterations);
		static bool GrowCandidates(std::vector<InternodeCandidate>& candidates);
		static void CalculateIlluminationForInternodes(PlantManager& manager);
		static void CollectNutrient(std::vector<Entity>& trees, std::vector<ResourceParcel>& totalNutrients, std::vector<ResourceParcel>& nutrientsAvailable);
		static void ApplyTropism(const glm::vec3& targetDir, float tropism, glm::vec3& front, glm::vec3& up);
#pragma endregion
#pragma region Members
		Entity m_ground;
		
		/**
		 * \brief The period of time for each iteration. Must be smaller than 1.0f.
		 */
		float m_deltaTime = 1.0f;
		/**
		 * \brief The current global time.
		 */
		float m_globalTime;
		/**
		 * \brief Whether the PlanetManager is initialized.
		 */
		bool m_ready;
		float m_illuminationFactor = 0.002f;
		float m_illuminationAngleFactor = 2.0f;
		
		
		
		EntityArchetype m_internodeArchetype;
		EntityArchetype m_plantArchetype;
		EntityQuery m_plantQuery;
		EntityQuery m_internodeQuery;
		
		std::vector<Entity> m_plants;
		std::vector<Entity> m_internodes;
		std::vector<GlobalTransform> m_internodeTransforms;
		
		
		std::unique_ptr<OpenGLUtils::GLTexture2D> m_rayTracerTestOutput;
		glm::ivec2 m_rayTracerTestOutputSize = glm::ivec2(1024, 1024);
		bool m_rendered = false;
		std::shared_ptr<Cubemap> m_environmentalMap;
#pragma region Timers
		float m_foliageGenerationTimer = 0;
		float m_meshGenerationTimer = 0;
		float m_resourceAllocationTimer = 0;
		float m_internodeFormTimer = 0;
		float m_internodeCreateTimer = 0;
		float m_illuminationCalculationTimer = 0;
		float m_pruningTimer = 0;
		float m_metaDataTimer = 0;
#pragma endregion
#pragma region GUI Settings
		int m_iterationsToGrow = 0;
		bool m_endUpdate = false;
		
		
#pragma endregion
#pragma region Rendering
		RayMLVQ::DebugRenderingProperties m_properties;
		float m_cameraFov = 60;
		bool m_rayTracerDebugRenderingEnabled = true;
		

		float m_lastX = 0;
		float m_lastY = 0;
		float m_lastScrollY = 0;
		bool m_startMouse = false;
		bool m_startScroll = false;
		bool m_rightMouseButtonHold = false;
#pragma endregion
#pragma endregion
#pragma region Helpers
		static Entity CreateCubeObstacle();
		static void UpdateDebugRenderOutputScene();
		
		static void RenderRayTracerDebugOutput();
		
		static void DeleteAllPlants();
		static Entity CreatePlant(const PlantType& type, const Transform& transform);
		static Entity CreateInternode(const PlantType& type, const Entity& parentEntity);
		
		
#pragma endregion
#pragma region Runtime
		static void OnGui();
		static PlantManager& GetInstance();
		static void Init();
		static void Update();
		static void Refresh();
		static void End();
#pragma endregion
	};
}

