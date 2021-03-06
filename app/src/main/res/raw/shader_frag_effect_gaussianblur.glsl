#version 300 es
precision highp float;
in vec2 v_texCoord;
layout(location = 0) out vec4 outColor;
uniform sampler2D s_Texture;
uniform vec2 u_TexSize;
uniform bool u_Mirror;
uniform float u_Time;

#ifdef GL_ES
precision mediump float;
#endif

float normpdf(in float x, in float sigma) {
    return 0.39894*exp(-0.5*x*x/(sigma*sigma))/sigma;
}

void main() {
    vec2 uv = gl_FragCoord.xy / u_TexSize.xy;
    if (u_Mirror) uv.x = 1.0-uv.x;
    uv.y = 1.0-uv.y;
    vec3 c = texture(s_Texture, uv).rgb;
    //declare stuff
    const int mSize = 11;
    const int kSize = (mSize-1)/2;
    float kernel[mSize];
    vec3 final_colour = vec3(0.0);
    //create the 1-D kernel
    float sigma = 7.0;
    float Z = 0.0;
    for (int j = 0; j <= kSize; ++j) {
        kernel[kSize+j] = kernel[kSize-j] = normpdf(float(j), sigma);
    }
    //get the normalization factor (as the gaussian has been clamped)
    for (int j = 0; j < mSize; ++j) {
        Z += kernel[j];
    }
    //read out the texels
    for (int i=-kSize; i <= kSize; ++i) {
        for (int j=-kSize; j <= kSize; ++j) {
            vec2 iuv = (gl_FragCoord.xy+vec2(float(i),float(j))) / u_TexSize.xy;
            if (u_Mirror) iuv.x = 1.0-iuv.x;
            iuv.y = 1.0-iuv.y;
            final_colour += kernel[kSize+j]*kernel[kSize+i]*texture(s_Texture, iuv).rgb;
        }
    }
    outColor = vec4(final_colour/(Z*Z), 1.0);
}