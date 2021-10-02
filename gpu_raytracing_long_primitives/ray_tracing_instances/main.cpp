/*
 * Copyright (c) 2014-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2014-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


// ImGui - standalone example application for Glfw + Vulkan, using programmable
// pipeline If you are new to ImGui, see examples/README.txt and documentation
// at the top of imgui.cpp.

#include <array>
#include <random>

#include "backends/imgui_impl_glfw.h"
#include "imgui.h"

#include "hello_vulkan.h"
#include "imgui/imgui_camera_widget.h"
#include "nvh/cameramanipulator.hpp"
#include "nvh/fileoperations.hpp"
#include "nvpsystem.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/context_vk.hpp"


// Utility to time the execution of something resetting the timer
// on each elapse call
// Usage:
// {
//   MilliTimer timer;
//   ... stuff ...
//   double time_elapse = timer.elapse();
// }
#include <chrono>
#include <fstream>
#include <filesystem>
#include <iostream>

struct MilliTimer
{
  MilliTimer() { reset(); }
  void   reset() { startTime = std::chrono::high_resolution_clock::now(); }
  double elapse()
  {
    auto now  = std::chrono::high_resolution_clock::now();
    auto t    = std::chrono::duration_cast<std::chrono::microseconds>(now - startTime).count() / 1000.0;
    startTime = now;
    return t;
  }
  void print() { LOGI(" --> (%5.3f ms)\n", elapse()); }

  std::chrono::high_resolution_clock::time_point startTime;
};


//////////////////////////////////////////////////////////////////////////
#define UNUSED(x) (void)(x)
//////////////////////////////////////////////////////////////////////////

// Default search path for shaders
std::vector<std::string> defaultSearchPaths;


// GLFW Callback functions
static void onErrorCallback(int error, const char* description)
{
  fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

// Extra UI
void renderUI(HelloVulkan& helloVk)
{
  ImGuiH::CameraWidget();
  if(ImGui::CollapsingHeader("Light"))
  {
    ImGui::RadioButton("Point", &helloVk.m_pushConstant.lightType, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Infinite", &helloVk.m_pushConstant.lightType, 1);

    ImGui::SliderFloat3("Position", &helloVk.m_pushConstant.lightPosition.x, -20.f, 20.f);
    ImGui::SliderFloat("Intensity", &helloVk.m_pushConstant.lightIntensity, 0.f, 150.f);
  }
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
static int const SAMPLE_WIDTH  = 1280;
static int const SAMPLE_HEIGHT = 720;

#define EVALUATION_LOGGING

int run(const std::string& logPathName, std::ofstream& summaryFile)
{
    std::filesystem::path logPath(logPathName);
    if (!std::filesystem::exists(logPath))
    {
        std::filesystem::create_directories(logPath);
    }

    // Setup GLFW window
    glfwSetErrorCallback(onErrorCallback);
    if(!glfwInit())
    {
        return 1;
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(SAMPLE_WIDTH, SAMPLE_HEIGHT, PROJECT_NAME, nullptr, nullptr);

    // Setup camera
    CameraManip.setWindowSize(SAMPLE_WIDTH, SAMPLE_HEIGHT);
    CameraManip.setLookat(nvmath::vec3f(5, 4, -4), nvmath::vec3f(0, 1, 0), nvmath::vec3f(0, 1, 0));

    // Setup Vulkan
    if(!glfwVulkanSupported())
    {
        printf("GLFW: Vulkan Not Supported\n");
        return 1;
    }

    // setup some basic things for the sample, logging file for example
    NVPSystem system(PROJECT_NAME);

    // Search path for shaders and other media
    defaultSearchPaths = {
            NVPSystem::exePath() + PROJECT_RELDIRECTORY,
            NVPSystem::exePath() + PROJECT_RELDIRECTORY "..",
            std::string(PROJECT_NAME),
    };

    // Vulkan required extensions
    assert(glfwVulkanSupported() == 1);
    uint32_t count{0};
    auto     reqExtensions = glfwGetRequiredInstanceExtensions(&count);

    // Requesting Vulkan extensions and layers
    nvvk::ContextCreateInfo contextInfo;
    contextInfo.setVersion(1, 2);                       // Using Vulkan 1.2
    for(uint32_t ext_id = 0; ext_id < count; ext_id++)  // Adding required extensions (surface, win32, linux, ..)
        contextInfo.addInstanceExtension(reqExtensions[ext_id]);
    contextInfo.addInstanceLayer("VK_LAYER_LUNARG_monitor", true);              // FPS in titlebar
    contextInfo.addInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME, true);  // Allow debug names
    contextInfo.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);            // Enabling ability to present rendering

    // #VKRay: Activate the ray tracing extension
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    contextInfo.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &accelFeature);  // To build acceleration structures
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
    contextInfo.addDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, false, &rtPipelineFeature);  // To use vkCmdTraceRaysKHR
    contextInfo.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);  // Required by ray tracing pipeline

    // Creating Vulkan base application
    nvvk::Context vkctx{};
    vkctx.initInstance(contextInfo);
    // Find all compatible devices
    auto compatibleDevices = vkctx.getCompatibleDevices(contextInfo);
    assert(!compatibleDevices.empty());
    // Use a compatible device
    vkctx.initDevice(compatibleDevices[0], contextInfo);

    // Create example
    HelloVulkan helloVk;

    // Window need to be opened to get the surface on which to draw
    const VkSurfaceKHR surface = helloVk.getVkSurface(vkctx.m_instance, window);
    vkctx.setGCTQueueWithPresent(surface);

    helloVk.setup(vkctx.m_instance, vkctx.m_device, vkctx.m_physicalDevice, vkctx.m_queueGCT.familyIndex);
    helloVk.createSwapchain(surface, SAMPLE_WIDTH, SAMPLE_HEIGHT);
    helloVk.createDepthBuffer();
    helloVk.createRenderPass();
    helloVk.createFrameBuffers();

    // Setup Imgui
    helloVk.initGUI(0);  // Using sub-pass 0

    MilliTimer timer;

    helloVk.loadModel(nvh::findFile("media/scenes/cube_multi.obj", defaultSearchPaths, true));

#if 0
    // Creation of the example
  std::random_device              rd;         //Will be used to obtain a seed for the random number engine
  std::mt19937                    gen(rd());  //Standard mersenne_twister_engine seeded with rd()
  std::normal_distribution<float> dis(1.0f, 1.0f);
  std::normal_distribution<float> disn(0.05f, 0.05f);

  for(int n = 0; n < 2000; ++n)
  {
    helloVk.loadModel(nvh::findFile("media/scenes/cube_multi.obj", defaultSearchPaths, true));
    HelloVulkan::ObjInstance& inst = helloVk.m_objInstance.back();

    float         scale = fabsf(disn(gen));
    nvmath::mat4f mat   = nvmath::translation_mat4(nvmath::vec3f{dis(gen), 2.0f + dis(gen), dis(gen)});
    mat                 = mat * nvmath::rotation_mat4_x(dis(gen));
    mat                 = mat * nvmath::scale_mat4(nvmath::vec3f(scale));
    inst.transform      = mat;
    inst.transformIT    = nvmath::transpose(nvmath::invert((inst.transform)));
  }
#endif

    //helloVk.loadModel(nvh::findFile("media/scenes/plane.obj", defaultSearchPaths, true));

    cyHairFile myHairFile;
    const char* filename = "../media/scenes/dark.hair";
    helloVk.loadHairModel(filename, myHairFile);

//    std::ofstream infoFile;
//    infoFile.open(logPathName + "info.log", std::ios::trunc);

    double time_elapse = timer.elapse();
//  LOGI(" --> (%f)", time_elapse);

    helloVk.createOffscreenRender();
    helloVk.createDescriptorSetLayout();
    helloVk.createGraphicsPipeline();
    helloVk.createUniformBuffer();
    helloVk.createSceneDescriptionBuffer();

    // #VKRay
    helloVk.initRayTracing();
    helloVk.createBottomLevelAS(summaryFile);
    helloVk.updateDescriptorSet();
    helloVk.createTopLevelAS(summaryFile);
    helloVk.createRtDescriptorSet();
    helloVk.createRtPipeline();

    helloVk.createPostDescriptor();
    helloVk.createPostPipeline();
    helloVk.updatePostDescriptorSet();


    nvmath::vec4f clearColor   = nvmath::vec4f(1, 1, 1, 1.00f);
    bool          useRaytracer = true;
    // keep track of camera positions
    uint32_t fileNumber = 1;
    bool isLogging = false;
    bool startLogging = false;
    float logTimer = 0.0f;
    float waitTime = 20.0f;
    uint32_t totalFramesPerPos = 0;
    float totalFrametimePerPos = 0.0f;
    float totalFrametime = 0.0f;
    // automatic evaluation uses these positions in each run
    std::vector<std::pair<nvmath::vec3, nvmath::vec3>> cameraPositions;
    cameraPositions.emplace_back(std::pair(nvmath::vec3(0.000f, 4.001f, -8.012f), nvmath::vec3(0.000f, 0.412f, -1.000f)));
    cameraPositions.emplace_back(std::pair(nvmath::vec3(0.611f, 10.012f, 3.512f), nvmath::vec3(0.000f, 2.811f, 3.513f)));
    cameraPositions.emplace_back(std::pair(nvmath::vec3(9.941f, 2.731f, 0.366f), nvmath::vec3(-0.124f, -1.242f, 1.017f)));
    cameraPositions.emplace_back(std::pair(nvmath::vec3(0.873f, -9.116f, 12.627f), nvmath::vec3(0.574f, -6.739f, 9.909f)));
    cameraPositions.emplace_back(std::pair(nvmath::vec3(-9.699f, -7.202f, 6.284f), nvmath::vec3(-6.783f, -5.240f, 5.400f)));
    cameraPositions.emplace_back(std::pair(nvmath::vec3(2.591f, 1.724f, -3.551f), nvmath::vec3(0.674f, -1.606f, 3.551f)));
    cameraPositions.emplace_back(std::pair(nvmath::vec3(5.528f, 1.787f, 9.538f), nvmath::vec3(2.663f, 0.580f, 7.677f)));
    cameraPositions.emplace_back(std::pair(nvmath::vec3(-5.285f, -6.440f, -6.023f), nvmath::vec3(-3.897f, -4.378f, -3.386f)));

    const uint32_t cameraPosCount = cameraPositions.size() + 1; // +1 cause of the initial position

    helloVk.setupGlfwCallbacks(window);
    ImGui_ImplGlfw_InitForVulkan(window, true);

    // Log file for frametime
    std::ofstream timeFile;

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        if (helloVk.isMinimized())
            continue;

        // Start the Dear ImGui frame
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Show UI window.
        if (helloVk.showGui())
        {
            float framerate = ImGui::GetIO().Framerate;
            ImGuiH::Panel::Begin();
            ImGui::Checkbox("Log", &startLogging);
            ImGui::ColorEdit3("Clear color", reinterpret_cast<float*>(&clearColor));
            ImGui::Checkbox("Ray Tracer mode", &useRaytracer);  // Switch between raster and ray tracing

            renderUI(helloVk);
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / framerate, framerate);
            ImGuiH::Control::Info("", "", "(F10) Toggle Pane", ImGuiH::Control::Flags::Disabled);
            ImGuiH::Panel::End();

#ifdef EVALUATION_LOGGING
            logTimer += 1.0f / framerate;
            // timer is up, move on, either start or stop logging
            if (logTimer > waitTime)
            {
                logTimer = 0.0f;
                // stop logging and make transition to next camera position
                if (isLogging)
                {
                    timeFile.close();
                    isLogging = false;
                    CameraManip.setAnimationDuration(1.0);
                    summaryFile << "Avg. frametime: " + std::to_string(totalFrametimePerPos / totalFramesPerPos) << std::endl;
                    totalFrametime += totalFrametimePerPos / totalFramesPerPos;
                    totalFrametimePerPos = 0.0f;
                    totalFramesPerPos = 0.0f;
                    if (!cameraPositions.empty())
                    {
                        auto pos = cameraPositions.back();
                        cameraPositions.pop_back();
                        CameraManip.setLookat(pos.first, pos.second, nvmath::vec3(0.0f, 1.0f, 0.0f), false);
                        CameraManip.updateAnim();
                        std::cout << "End logging and move camera..." << std::endl;
                    }
                    else
                    {
                        std::cout << "No Positions left, move to next parameters..." << std::endl;
                        break;
                    }
                    // wait till frametime stabilizes
                    waitTime = 20.0f;
                }
                // start logging for current position
                else
                {
                    timeFile.open(logPathName + std::to_string(fileNumber) + "timeFile.log", std::ios::trunc);
                    ++fileNumber;
                    isLogging = true;
                    waitTime = 20.0f;
                    std::cout << "Logging started..." << std::endl;
                }
            }
#endif
            if (isLogging)
            {
                // Log file for frametime
                totalFrametimePerPos += 1000.0f / framerate;
                ++totalFramesPerPos;
                timeFile << 1000.0f / framerate << std::endl;
            }
#ifndef EVALUATION_LOGGING
            if (startLogging && !isLogging)
          {
              timeFile.open(logPathName + std::to_string(fileNumber) + "timeFile.log", std::ios::trunc);
              ++fileNumber;
              isLogging = true;
          }
          if (!startLogging && isLogging)
          {
              timeFile.close();
              isLogging = false;
          }
#endif
        }

        // Start rendering the scene
        helloVk.prepareFrame();

        // Start command buffer of this frame
        auto                   curFrame = helloVk.getCurFrame();
        const VkCommandBuffer& cmdBuf = helloVk.getCommandBuffers()[curFrame];

        VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmdBuf, &beginInfo);
        // Updating camera buffer
        helloVk.updateUniformBuffer(cmdBuf);

        // Clearing screen
        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = { {clearColor[0], clearColor[1], clearColor[2], clearColor[3]} };
        clearValues[1].depthStencil = { 1.0f, 0 };

        // Offscreen render pass
        {
            VkRenderPassBeginInfo offscreenRenderPassBeginInfo{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
            offscreenRenderPassBeginInfo.clearValueCount = 2;
            offscreenRenderPassBeginInfo.pClearValues = clearValues.data();
            offscreenRenderPassBeginInfo.renderPass = helloVk.m_offscreenRenderPass;
            offscreenRenderPassBeginInfo.framebuffer = helloVk.m_offscreenFramebuffer;
            offscreenRenderPassBeginInfo.renderArea = { {0, 0}, helloVk.getSize() };

            // Rendering Scene
            if (useRaytracer)
            {
                helloVk.raytrace(cmdBuf, clearColor);
            }
            else
            {
                vkCmdBeginRenderPass(cmdBuf, &offscreenRenderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
                helloVk.rasterize(cmdBuf);
                vkCmdEndRenderPass(cmdBuf);
            }
        }

        // 2nd rendering pass: tone mapper, UI
        {
            VkRenderPassBeginInfo postRenderPassBeginInfo{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
            postRenderPassBeginInfo.clearValueCount = 2;
            postRenderPassBeginInfo.pClearValues = clearValues.data();
            postRenderPassBeginInfo.renderPass = helloVk.getRenderPass();
            postRenderPassBeginInfo.framebuffer = helloVk.getFramebuffers()[curFrame];
            postRenderPassBeginInfo.renderArea = { {0, 0}, helloVk.getSize() };

            // Rendering tonemapper
            vkCmdBeginRenderPass(cmdBuf, &postRenderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
            helloVk.drawPost(cmdBuf);
            // Rendering UI
            ImGui::Render();
            ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmdBuf);
            vkCmdEndRenderPass(cmdBuf);
        }

        // Submit for display
        vkEndCommandBuffer(cmdBuf);
        helloVk.submitFrame();
    }

    summaryFile << "Total Avg. frametime: " << std::to_string(totalFrametime / cameraPosCount) << std::endl;
    // Cleanup
    vkDeviceWaitIdle(helloVk.getDevice());

    helloVk.destroyResources();
    helloVk.destroy();
    vkctx.deinit();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

//--------------------------------------------------------------------------------------------------
// Application Entry
//
int main(int argc, char** argv)
{

    std::filesystem::path logPath("../media/LogData/baseline/");
    if (!std::filesystem::exists(logPath))
    {
        std::filesystem::create_directories(logPath);
    }
    std::ofstream summaryFile;
    summaryFile.open("../media/LogData/baseline/summary.log", std::ios::trunc);
#ifdef EVALUATION_LOGGING
            std::string logPathName = "../media/LogData/baseline/";
            // start run with current parameters
            if (run(logPathName, summaryFile) != 0)
            {
                summaryFile.close();
                return -1;
            }
            summaryFile << "------------------------------" << std::endl;
    int status = 0;
#else
      int status = run("../media/LogData/baseline/", summaryFile);
#endif
    summaryFile.close();
    return status;
}
