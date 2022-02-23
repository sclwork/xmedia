package com.scliang.x.media;

import android.app.Activity;
import android.content.Context;
import android.opengl.GLSurfaceView;
import android.text.TextUtils;
import android.view.Window;
import android.view.WindowManager;

import androidx.annotation.NonNull;
import androidx.annotation.RawRes;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.ref.SoftReference;
import java.util.ArrayList;
import java.util.List;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

public class MediaManager {
    /*
     *
     */
    public static void init(Context context, GLSurfaceView glView) {
        MediaManager mm = SingletonHolder.INSTANCE;
        mm.mCtx = new SoftReference<>(context);
        mm.initGlslResource(context);
        if (glView != null) {
            glView.setEGLContextClientVersion(3);
            glView.setRenderer(new MediaRenderer());
            glView.setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);
            glView.queueEvent(()->mm.jniInit(mm.getFileRootPath(context)));
            GlView = new SoftReference<>(glView);
        }
    }

    public static void start() {
        MediaManager mm = SingletonHolder.INSTANCE;
        mm.acquireScreen();
        if (GlView != null && GlView.get() != null) {
            GlView.get().queueEvent(mm::jniResume);
            GlView.get().onResume();
        }
    }

    public static void stop() {
        MediaManager mm = SingletonHolder.INSTANCE;
        mm.releaseScreen();
        if (GlView != null && GlView.get() != null) {
            GlView.get().queueEvent(mm::jniPause);
            GlView.get().onPause();
        }
    }

    public static void release() {
        MediaManager mm = SingletonHolder.INSTANCE;
        if (GlView != null && GlView.get() != null) {
            GlView.get().queueEvent(mm::jniRelease);
        }
        GlView = null;
    }


    /*
     *
     */
    public static List<String> getSupportedEffectPaints() {
        return new ArrayList<>(effectNames);
    }

    public static void updateEffectPaint(String name) {
        MediaManager mm = SingletonHolder.INSTANCE;
        if (GlView != null && GlView.get() != null) {
            GlView.get().queueEvent(()->mm.jniUpdatePaint(name));
        }
    }


    /*
     *
     */
    public static void preview(int merge, String... cameras) {
        if (cameras.length > 0) {
            StringBuilder sb = new StringBuilder();
            for (String c : cameras) sb.append(",").append(c);
            MediaManager mm = SingletonHolder.INSTANCE;
            if (GlView != null && GlView.get() != null) {
                GlView.get().queueEvent(()->mm.jniPreview(sb.substring(1), merge));
            }
        }
    }

    public static void stopPreview() {
        MediaManager mm = SingletonHolder.INSTANCE;
        if (GlView != null && GlView.get() != null) {
            GlView.get().queueEvent(()->mm.jniPreview("", -1));
        }
    }


    /*
     *
     */
    public static void setCameraAWB(String id, int awb, OnSetCameraAWBListener listener) {
        MediaManager mm = SingletonHolder.INSTANCE;
        if (GlView != null && GlView.get() != null) {
            GlView.get().queueEvent(()->{
                boolean res = mm.jniSetCameraAWB(id, awb);
                GlView.get().post(()->{
                    if (listener != null) listener.onSetCameraAWBCompleted(id, awb, res);
                });
            });
        }
    }
    public interface OnSetCameraAWBListener {
        void onSetCameraAWBCompleted(String id, int awb, boolean result);
    }


    /*
     *
     */
    public static boolean recording() {
        MediaManager mm = SingletonHolder.INSTANCE;
        return mm.jniRecording();
    }

    public static void startRecord(@NonNull String name) {
        MediaManager mm = SingletonHolder.INSTANCE;
        if (GlView != null && GlView.get() != null) {
            GlView.get().queueEvent(()->mm.jniRecordStart(name));
        }
    }

    public static void stopRecord() {
        MediaManager mm = SingletonHolder.INSTANCE;
        if (GlView != null && GlView.get() != null) {
            GlView.get().queueEvent(mm::jniRecordStop);
        }
    }


    /*
     *
     */
    public static boolean playing() {
        MediaManager mm = SingletonHolder.INSTANCE;
        return mm.jniPlaying();
    }

    public static void startPlay(@NonNull String name) {
        MediaManager mm = SingletonHolder.INSTANCE;
        if (GlView != null && GlView.get() != null) {
            GlView.get().queueEvent(()->mm.jniPlayStart(name));
        }
    }

    public static void stopPlay() {
        MediaManager mm = SingletonHolder.INSTANCE;
        if (GlView != null && GlView.get() != null) {
            GlView.get().queueEvent(mm::jniPlayStop);
        }
    }


    /*
     *
     */
    private void initGlslResource(Context context) {
        effectNames.clear();
        setupGlslFiles(context);
        setupErrorTipFile(context);
        setupMnnResFile(context);
    }


    /*
     *
     */
    private String getFileRootPath(Context context) {
        try {
            File dir = context.getDir("files", Context.MODE_PRIVATE);
            return dir.getAbsolutePath();
        } catch (Exception e) {
            e.printStackTrace();
            return "";
        }
    }

    private void setupErrorTipFile(Context context) {
        InputStream is = null;
        FileOutputStream os = null;
        try {
            is = context.getResources().openRawResource(R.raw.ic_vid_file_not_exists);
            File dir = context.getDir("files", Context.MODE_PRIVATE);
            File file = new File(dir, "ic_vid_file_not_exists.png");
            if (file.exists()) file.delete();
            os = new FileOutputStream(file);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1)
                os.write(buffer, 0, bytesRead);

            file.getAbsolutePath();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try { if (is != null) is.close();
            } catch (IOException ignored) { }
            try { if (os != null) os.close();
            } catch (IOException ignored) { }
        }
    }

    private void getShaderFile(Context context, @RawRes int raw, String name) {
        InputStream is = null;
        FileOutputStream os = null;
        try {
            if (TextUtils.isEmpty(name)) {
                return;
            }

            if (name.contains("shader_frag_effect_")) {
                effectNames.add(
                        name.replace("shader_frag_effect_", "")
                                .replace(".glsl", "")
                                .toUpperCase());
            }

            is = context.getResources().openRawResource(raw);
            File dir = context.getDir("files", Context.MODE_PRIVATE);
            File file = new File(dir, name);
            if (file.exists()) file.delete();
            os = new FileOutputStream(file);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1)
                os.write(buffer, 0, bytesRead);

            file.getAbsolutePath();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try { if (is != null) is.close();
            } catch (IOException ignored) { }
            try { if (os != null) os.close();
            } catch (IOException ignored) { }
        }
    }

    private void setupGlslFiles(Context context) {
        getShaderFile(context,
                R.raw.shader_frag_none,
                "shader_frag_none.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_none,
                "shader_frag_effect_none.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_face,
                "shader_frag_effect_face.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_ripple,
                "shader_frag_effect_ripple.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_distortedtv,
                "shader_frag_effect_distortedtv.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_distortedtv_box,
                "shader_frag_effect_distortedtv_box.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_distortedtv_glitch,
                "shader_frag_effect_distortedtv_glitch.glsl");
//        getShaderFile(context,
//                R.raw.shader_frag_effect_distortedtv_crt,
//                "shader_frag_effect_distortedtv_crt.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_floyd,
                "shader_frag_effect_floyd.glsl");
//        getShaderFile(context,
//                R.raw.shader_frag_effect_3basic,
//                "shader_frag_effect_3basic.glsl");
//        getShaderFile(context,
//                R.raw.shader_frag_effect_3floyd,
//                "shader_frag_effect_3floyd.glsl");
//        getShaderFile(context,
//                R.raw.shader_frag_effect_pagecurl,
//                "shader_frag_effect_pagecurl.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_old_video,
                "shader_frag_effect_old_video.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_crosshatch,
                "shader_frag_effect_crosshatch.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_cmyk,
                "shader_frag_effect_cmyk.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_drawing,
                "shader_frag_effect_drawing.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_neon,
                "shader_frag_effect_neon.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_fisheye,
                "shader_frag_effect_fisheye.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_barrelblur,
                "shader_frag_effect_barrelblur.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_fastblur,
                "shader_frag_effect_fastblur.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_gaussianblur,
                "shader_frag_effect_gaussianblur.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_illustration,
                "shader_frag_effect_illustration.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_hexagon,
                "shader_frag_effect_hexagon.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_sobel,
                "shader_frag_effect_sobel.glsl");
//        getShaderFile(context,
//                R.raw.shader_frag_effect_lens,
//                "shader_frag_effect_lens.glsl");
        getShaderFile(context,
                R.raw.shader_frag_effect_float_camera,
                "shader_frag_effect_float_camera.glsl");
        getShaderFile(context,
                R.raw.shader_vert_none,
                "shader_vert_none.glsl");
        getShaderFile(context,
                R.raw.shader_vert_effect_none,
                "shader_vert_effect_none.glsl");
    }

    private void setupMnnResFile(Context context) {
        InputStream is = null;
        FileOutputStream os = null;
        try {
            is = context.getResources().openRawResource(R.raw.blazeface);
            File dir = context.getDir("files", Context.MODE_PRIVATE);
            File file = new File(dir, "blazeface.mnn");
            if (file.exists()) file.delete();
            os = new FileOutputStream(file);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1)
                os.write(buffer, 0, bytesRead);

            file.getAbsolutePath();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try { if (is != null) is.close();
            } catch (IOException ignored) { }
            try { if (os != null) os.close();
            } catch (IOException ignored) { }
        }
    }


    /*
     *
     */
    private static class MediaRenderer implements GLSurfaceView.Renderer {
        private final MediaManager mm = SingletonHolder.INSTANCE;
        @Override
        public void onSurfaceCreated(GL10 gl, EGLConfig config){
            mm.jniSurfaceCreated();}
        @Override
        public void onSurfaceChanged(GL10 gl, int width, int height){
            mm.jniSurfaceChanged(width, height);}
        @Override
        public void onDrawFrame(GL10 gl){
            mm.jniDrawFrame();}
    }
    /*
     *
     */
    private void acquireScreen() {
        Context ctx = mCtx == null ? null : mCtx.get();
        if (GlView != null && GlView.get() != null && ctx instanceof Activity) {
            GlView.get().post(() -> {
                Window window = ((Activity)ctx).getWindow();
                if (window != null) {
                    window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
                }
            });
        }
    }
    private void releaseScreen() {
        Context ctx = mCtx == null ? null : mCtx.get();
        if (GlView != null && GlView.get() != null && ctx instanceof Activity) {
            GlView.get().post(() -> {
                Window window = ((Activity)ctx).getWindow();
                if (window != null) {
                    window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
                }
            });
        }
    }
    /*
     *
     */
    private SoftReference<Context> mCtx;
    private static SoftReference<GLSurfaceView> GlView;
    /*
     *
     */
    private static void requestRender(int code) {
        if(GlView !=null&& GlView.get()!=null) GlView.get().requestRender(); }
    /*
     *
     */
    static { System.loadLibrary("xmedia-lib"); }
    /*
     *
     */
    private native int jniInit(@NonNull String fileRoot);
    private native int jniResume();
    private native int jniPause();
    private native int jniRelease();
    /*
     *
     */
    private native int jniSurfaceCreated();
    private native int jniSurfaceChanged(int width, int height);
    private native int jniUpdatePaint(@NonNull String name);
    private native int jniDrawFrame();
    private native int jniPreview(@NonNull String cameras, int merge);
    /*
     *
     */
    private native boolean jniSetCameraAWB(String id, int awb);
    /*
     *
     */
    private native boolean jniRecording();
    private native int     jniRecordStart(@NonNull String name);
    private native int     jniRecordStop();
    /*
     *
     */
    private native boolean jniPlaying();
    private native int     jniPlayStart(@NonNull String name);
    private native int     jniPlayStop();
    /*
     *
     */
    private final static List<String> effectNames = new ArrayList<>();
    /*
     *
     */
    private static class SingletonHolder { private static final MediaManager INSTANCE = new MediaManager(); }
    private MediaManager() {}
}
