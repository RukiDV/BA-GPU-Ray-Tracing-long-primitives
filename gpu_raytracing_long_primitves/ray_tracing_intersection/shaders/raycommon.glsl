struct hitPayload
{
  vec3 hitValue;
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

#define KIND_HAIR 0
