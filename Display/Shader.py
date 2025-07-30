from OpenGL.GL import *
from OpenGL.GLU import *

vertex = """
varying vec3 color;
varying vec3 pos;
varying vec3 mpos;

uniform mat4 view;

void main(void)
{   
  gl_Position = gl_ProjectionMatrix * view * gl_ModelViewMatrix * gl_Vertex;
  color = gl_Color.rgb;
  pos = (view * gl_ModelViewMatrix * gl_Vertex).xyz;
  mpos = gl_Vertex.xyz;
}
"""

fragment = """
varying vec3 color;
varying vec3 pos;
varying vec3 mpos;
uniform mat4 view;

float getColor(float i) {
  return float(sin(i * 25.0) > 0.5) * 1.0;
}

float getCombinedColor() {
  float x = getColor(mpos.x);
  float y = getColor(mpos.y);
  float z = getColor(mpos.z);

  return max(x, max(y, z)) + min(x, min(y, z));
}

void main () {
  vec4 raw_dir = view * vec4(-1, 0, 1, 0.0);
  vec3 direction = normalize(raw_dir.xyz);

  float texture = max(0.25, getCombinedColor());
  vec3 color = color * texture;

  vec3 py = normalize(dFdy(pos));
  vec3 px = normalize(dFdx(pos));
  vec3 norm = normalize(cross(py, px));

  vec3 view = normalize(pos);
  vec3 reflectDir = reflect(-direction, norm);  

  float spec = max(dot(view, reflectDir), 0.0) * texture;
  float diff = (dot(direction, norm) + 1.0) * 0.5f;

  gl_FragColor = vec4(diff * color + pow(spec, 64.0) * vec3(1.0, 1.0, 1.0), 1.0);
}
"""

def load_program(vertex_source, fragment_source):
    vertex_shader = load_shader(GL_VERTEX_SHADER, vertex_source)
    if vertex_shader == 0:
        return 0

    fragment_shader = load_shader(GL_FRAGMENT_SHADER, fragment_source)
    if fragment_shader == 0:
        return 0

    program = glCreateProgram()

    if program == 0:
        return 0

    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)

    glLinkProgram(program)

    if glGetProgramiv(program, GL_LINK_STATUS, None) == GL_FALSE:
        glDeleteProgram(program)
        return 0

    return program

def load_shader(shader_type, source):
    shader = glCreateShader(shader_type)

    if shader == 0:
        return 0

    glShaderSource(shader, source)
    glCompileShader(shader)

    if glGetShaderiv(shader, GL_COMPILE_STATUS, None) == GL_FALSE:
        info_log = glGetShaderInfoLog(shader)
        print(info_log)
        glDeleteProgram(shader)
        return 0

    return shader
