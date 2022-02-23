#version 300 es
precision highp float;
in vec2 v_texCoord;
layout(location = 0) out vec4 outColor;
uniform sampler2D s_Texture;
uniform vec2 u_TexSize;
uniform bool u_Mirror;
uniform float u_Time;

void main() {
    vec2 xy = gl_FragCoord.xy / u_TexSize.yy;

    float amplitud = 0.03;
    float frecuencia = 10.0;
    float gris = 1.0;
    float divisor = 4.8 / u_TexSize.y;
    float grosorInicial = divisor * 0.2;

    const int kNumPatrones = 6;

    vec3 datosPatron[kNumPatrones];
    datosPatron[0] = vec3(-0.7071, 0.7071, 3.0); // -45
    datosPatron[1] = vec3(0.0, 1.0, 0.6); // 0
    datosPatron[2] = vec3(0.0, 1.0, 0.5); // 0
    datosPatron[3] = vec3(1.0, 0.0, 0.4); // 90
    datosPatron[4] = vec3(1.0, 0.0, 0.3); // 90
    datosPatron[5] = vec3(0.0, 1.0, 0.2); // 0

    vec2 uv = vec2(gl_FragCoord.x / u_TexSize.x, xy.y);
    if (u_Mirror) uv.x = 1.0-uv.x;
    uv.y = 1.0-uv.y;
    vec4 color = texture(s_Texture, uv);

    for(int i = 0; i < kNumPatrones; i++) {
        float coseno = datosPatron[i].x;
        float seno = datosPatron[i].y;

        vec2 punto = vec2(
            xy.x * coseno - xy.y * seno,
            xy.x * seno + xy.y * coseno
        );

        float grosor = grosorInicial * float(i + 1);
        float dist = mod(punto.y + grosor * 0.5 - sin(punto.x * frecuencia) * amplitud, divisor);
        float brillo = 0.3 * color.r + 0.4 * color.g + 0.3 * color.b;

        if(dist < grosor && brillo < 0.75 - 0.12 * float(i)) {
            float k = datosPatron[i].z;
            float x = (grosor - dist) / grosor;
            float fx = abs((x - 0.5) / k) - (0.5 - k) / k;
            gris = min(fx, gris);
        }
    }

    outColor = vec4(gris, gris, gris, 1.0);
}