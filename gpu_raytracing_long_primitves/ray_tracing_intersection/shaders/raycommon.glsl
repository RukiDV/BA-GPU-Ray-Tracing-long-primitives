struct hitPayload
{
  vec3 hitValue;
};

struct Sphere
{
  vec3  center;
  float radius;
};

struct Bezier
{
  vec3 p0;
  vec3 p1;
  vec3 p2;
  vec3 p3;
  float thickness;
};

struct Hair
{
  vec3 p0;
  vec3  p1;
  vec3  c0;
  vec3  c1;
  vec3  n0;
  vec3  n1;
  float thickness;
};

struct Aabb
{
  vec3 minimum;
  vec3 maximum;
};

#define KIND_SPHERE 0
#define KIND_CUBE 1
#define KIND_BEZIER 2
#define KIND_HAIR 3
