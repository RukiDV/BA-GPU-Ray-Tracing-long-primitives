#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#include "raycommon.glsl"
#include "wavefront.glsl"

hitAttributeEXT vec2 attribs;

// clang-format off
layout(location = 0) rayPayloadInEXT hitPayload prd;
layout(location = 1) rayPayloadEXT bool isShadowed;

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;

//layout(binding = 2, set = 1, scalar) buffer ScnDesc { sceneDesc i[]; } scnDesc;
//layout(binding = 5, set = 1, scalar) buffer Vertices { Vertex v[]; } vertices[];
//layout(binding = 6, set = 1) buffer Indices { uint i[]; } indices[];

//layout(binding = 1, set = 1, scalar) buffer MatColorBufferObject { WaveFrontMaterial m[]; } materials[];
//layout(binding = 3, set = 1) uniform sampler2D textureSamplers[];
//layout(binding = 4, set = 1)  buffer MatIndexColorBuffer { int i[]; } matIndex[];
layout(binding = 9, set = 1, scalar, std140) buffer allHairs_ {Hair i[];} allHairs;
// clang-format on

layout(push_constant) uniform Constants
{
  vec4  clearColor;
  vec3  lightPosition;
  float lightIntensity;
  int   lightType;
}
pushC;


void main()
{
//prd.hitValue = vec3(allHairs.i[gl_PrimitiveID].c0);
//return;
  vec3 worldPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

 if (gl_HitTEXT < 9999)
 {
   Hair instanceHair = allHairs.i[gl_PrimitiveID];

   vec3 normal = instanceHair.n1;
   vec3 color = instanceHair.c0;
   // Vector toward the light
   vec3  L;
   float lightIntensity = pushC.lightIntensity;
   float lightDistance  = 100000.0;
   // Point light
   if (pushC.lightType == 0)
   {
     vec3 lDir      = pushC.lightPosition - worldPos;
     lightDistance  = length(lDir);
     lightIntensity = pushC.lightIntensity / (lightDistance * lightDistance);
     L              = normalize(lDir);
   }
   else // Directional light
   {
     L = normalize(pushC.lightPosition - vec3(0));
   }

   // Diffuse
   vec3  diffuse     = computeDiffuse(color, L, normal);
   vec3  specular    = vec3(0);
   float attenuation = 0.3;

   // Tracing shadow ray only if the light is visible from the surface
   if (dot(normal, L) > 0)
   {
     float tMin   = 0.001;
     float tMax   = 100.0;
     vec3  origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
     vec3  rayDir = L;
     uint  flags  = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT
     | gl_RayFlagsSkipClosestHitShaderEXT;
     isShadowed = true;
     traceRayEXT(topLevelAS, // acceleration structure
     flags, // rayFlags
     0xFF, // cullMask
     0, // sbtRecordOffset
     0, // sbtRecordStride
     1, // missIndex
     origin, // ray origin
     tMin, // ray min range
     rayDir, // ray direction
     tMax, // ray max range
     1// payload (location = 1)
     );

     if (isShadowed)
     {
       attenuation = 0.3;
     }
     else
     {
       attenuation = 1;
       // Specular
       //specular = computeSpecular(gl_WorldRayDirectionEXT, L, normal);
     }
   }

   prd.hitValue = vec3(lightIntensity * (attenuation * color + diffuse));
 }
  else
 {
   prd.hitValue = vec3(1.0, 0.0, 1.0);
 }
}
