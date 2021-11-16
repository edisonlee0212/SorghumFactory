// PlantFactory.cpp : This file contains the 'main' function. Program execution
// begins and ends there.
//
#include <Application.hpp>
#ifdef RAYTRACERFACILITY
#include <CUDAModule.hpp>
#include <MLVQRenderer.hpp>
#include <RayTracerManager.hpp>
#endif
#include <EditorManager.hpp>
#include <PhysicsLayer.hpp>
#include <PostProcessing.hpp>
#include <ProjectManager.hpp>
#include <Utilities.hpp>

#include <ClassRegistry.hpp>
#include <ObjectRotator.hpp>
#include <SorghumData.hpp>
#include <SorghumLayer.hpp>
#include <TriangleIlluminationEstimator.hpp>

#include <AutoSorghumGenerationPipeline.hpp>
#include <DepthCamera.hpp>
#include <ProceduralSorghumSegmentationMask.hpp>
#include <SorghumField.hpp>
#include <SorghumProceduralDescriptor.hpp>
using namespace Scripts;
using namespace SorghumFactory;
#ifdef RAYTRACERFACILITY
using namespace RayTracerFacility;
#endif
void EngineSetup();

int main() {
  ClassRegistry::RegisterDataComponent<LeafTag>("LeafTag");
  ClassRegistry::RegisterDataComponent<SorghumTag>("SorghumTag");

  ClassRegistry::RegisterPrivateComponent<DepthCamera>("DepthCamera");
  ClassRegistry::RegisterPrivateComponent<ObjectRotator>("ObjectRotator");
  ClassRegistry::RegisterPrivateComponent<Spline>("Spline");
  ClassRegistry::RegisterPrivateComponent<SorghumData>("SorghumData");

  ClassRegistry::RegisterAsset<SorghumProceduralDescriptor>(
      "SorghumProceduralDescriptor", ".spd");
  ClassRegistry::RegisterAsset<SorghumField>("SorghumField", ".sorghumfield");
  const bool enableRayTracing = true;
  ApplicationConfigs applicationConfigs;
  Application::Create(applicationConfigs);
#ifdef RAYTRACERFACILITY
  if (enableRayTracing)
    Application::PushLayer<RayTracerManager>();
#endif
  Application::PushLayer<SorghumLayer>();
#pragma region Engine Loop
  Application::Start();
#pragma endregion

  Application::End();
}

